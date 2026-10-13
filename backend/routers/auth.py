"""LaborLens – Auth Router"""
from email.message import EmailMessage
import logging
import smtplib
import asyncio

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from pymysql.err import IntegrityError
import bcrypt
import hashlib
import hmac
import httpx
import jwt
import re
import secrets
from urllib.parse import urlencode, urlparse
from datetime import datetime, timedelta

from routers.services.database import execute_query, execute_one, execute_insert, execute_affected
from config import settings
from routers.services import oauth_state

router   = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)
ACCOUNT_LANGUAGE_PREFERENCES = {"auto", "en", "fil", "hil"}
FIREBASE_JWKS_URL = "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"
firebase_jwks_client = jwt.PyJWKClient(FIREBASE_JWKS_URL, cache_keys=True)


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    language_preference: str = "auto"
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class FirebaseLoginRequest(BaseModel):
    id_token: str = Field(min_length=100, max_length=8192)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class PreferenceRequest(BaseModel):
    language_preference: str


class ProfileUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)


class AccountUpdateRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1024)
    email: EmailStr
    new_password: str | None = Field(default=None, min_length=8, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: str
    language_preference: str = "auto"
    display_name: str | None = None


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("Password must be no more than 72 UTF-8 bytes.")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    encoded = plain.encode("utf-8")
    if len(encoded) > 72:
        return False
    return bcrypt.checkpw(encoded, hashed.encode())


def password_token_version(user_id: int, password_hash: str) -> str:
    """Opaque, keyed credential version; never put the password hash in a JWT."""
    return hmac.new(
        settings.JWT_SECRET.encode(),
        f"laborlens-login-version:{user_id}:{password_hash}".encode(),
        hashlib.sha256,
    ).hexdigest()


def create_token(user_id: int, username: str, role: str, password_hash: str) -> str:
    expire  = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "username": username, "role": role, "exp": expire,
               "credential_version": password_token_version(user_id, password_hash)}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def verify_firebase_id_token(id_token: str) -> dict:
    """Verify a Firebase client ID token and require Google as its provider."""
    project_id = str(settings.FIREBASE_PROJECT_ID or "").strip()
    if not project_id:
        raise RuntimeError("FIREBASE_PROJECT_ID is not configured")

    signing_key = firebase_jwks_client.get_signing_key_from_jwt(id_token)
    payload = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        audience=project_id,
        issuer=f"https://securetoken.google.com/{project_id}",
        leeway=30,
        options={"require": ["exp", "iat", "aud", "iss", "sub"]},
    )
    firebase_claim = payload.get("firebase")
    if not isinstance(firebase_claim, dict) or firebase_claim.get("sign_in_provider") != "google.com":
        raise jwt.InvalidTokenError("Firebase token was not issued for Google sign-in")
    if payload.get("email_verified") is not True:
        raise jwt.InvalidTokenError("Firebase Google account email is not verified")
    email = payload.get("email")
    if not isinstance(email, str) or "@" not in email:
        raise jwt.InvalidTokenError("Firebase token does not contain a valid email")
    return payload


def google_is_configured() -> bool:
    return bool(
        settings.GOOGLE_CLIENT_ID
        and settings.GOOGLE_CLIENT_SECRET
    )


def resolve_google_redirect_uri(request: Request) -> str:
    # A local API must receive its own callback and browser-binding cookie.
    if request.url.hostname in {"localhost", "127.0.0.1"}:
        return str(request.url_for("google_callback"))
    configured = (settings.GOOGLE_REDIRECT_URI or "").strip()
    if configured and configured.rstrip("/").endswith("/api/auth/google/callback"):
        return configured
    return str(request.url_for("google_callback"))


def normalize_frontend_url(value: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("/") and not raw.startswith("//"):
        return raw
    if "://" not in raw:
        raw = f"https://{raw.lstrip('/')}"
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return parsed.geturl()


def safe_frontend_url(candidate: str = "") -> str:
    fallback = normalize_frontend_url(settings.FRONTEND_URL) or "/"
    parsed_fallback = urlparse(fallback)
    allowed_hosts = {"localhost", "127.0.0.1"}
    if parsed_fallback.hostname:
        allowed_hosts.add(parsed_fallback.hostname.lower())

    normalized_candidate = normalize_frontend_url(candidate)
    parsed_candidate = urlparse(normalized_candidate)
    if (
        parsed_candidate.scheme in {"http", "https"}
        and parsed_candidate.hostname
        and parsed_candidate.hostname.lower() in allowed_hosts
    ):
        return normalized_candidate
    return fallback


def google_redirect(frontend_url: str, params: dict) -> RedirectResponse:
    separator = "&" if "#" in frontend_url else "#"
    response = RedirectResponse(f"{frontend_url}{separator}{urlencode(params)}")
    response.delete_cookie(oauth_state.COOKIE, path=oauth_state.COOKIE_PATH)
    response.headers["Cache-Control"] = "no-store"
    return response


def google_auth_url(request: Request, state: str) -> str:
    redirect_uri = resolve_google_redirect_uri(request)
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
        "state": state,
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def password_reset_url(token: str) -> str:
    frontend_url = safe_frontend_url(settings.FRONTEND_URL)
    separator = "&" if "#" in frontend_url else "#"
    return f"{frontend_url}{separator}{urlencode({'reset_token': token})}"


def smtp_is_configured() -> bool:
    return bool(
        str(settings.SMTP_HOST or "").strip()
        and str(settings.SMTP_USERNAME or "").strip()
        and str(settings.SMTP_PASSWORD or "").strip()
    )


def smtp_password_value() -> str:
    password = str(settings.SMTP_PASSWORD or "").strip()
    if str(settings.SMTP_HOST or "").strip().lower() in {"smtp.gmail.com", "gmail.com"}:
        return password.replace(" ", "")
    return password


def smtp_failure_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "application-specific password" in lowered or "bad credentials" in lowered or "535" in lowered:
        return (
            "Email was not sent because Gmail rejected the SMTP login. "
            "Use a Gmail App Password, not your normal Google password, then restart FastAPI."
        )
    if "connection" in lowered or "timed out" in lowered:
        return "Email was not sent because the backend could not connect to the SMTP server."
    return f"Email was not sent because the SMTP server returned an error: {message[:300]}"


def send_password_reset_email(to_email: str, reset_link: str) -> None:
    if not smtp_is_configured():
        raise RuntimeError("SMTP is not configured")

    host = str(settings.SMTP_HOST or "").strip()
    port = int(settings.SMTP_PORT or 587)
    username = str(settings.SMTP_USERNAME or "").strip()
    password = smtp_password_value()
    from_email = str(settings.SMTP_FROM_EMAIL or username).strip() or username
    from_name = str(settings.SMTP_FROM_NAME or "LaborLens").strip() or "LaborLens"
    message = EmailMessage()
    message["Subject"] = "Reset your LaborLens password"
    message["From"] = f"{from_name} <{from_email}>"
    message["To"] = to_email
    message.set_content(
        "Hello,\n\n"
        "We received a request to reset your LaborLens password.\n\n"
        f"Open this link to set a new password:\n{reset_link}\n\n"
        "This link expires in 1 hour. If you did not request this, you can ignore this email.\n\n"
        "LaborLens"
    )
    message.add_alternative(
        f"""
        <div style="font-family:Arial,sans-serif;line-height:1.6;color:#172033;">
          <h2 style="color:#10233f;">Reset your LaborLens password</h2>
          <p>We received a request to reset your LaborLens password.</p>
          <p><a href="{reset_link}" style="display:inline-block;background:#10233f;color:#fff;padding:12px 18px;border-radius:8px;text-decoration:none;font-weight:700;">Set a new password</a></p>
          <p style="font-size:13px;color:#667085;">This link expires in 1 hour. If you did not request this, you can ignore this email.</p>
        </div>
        """,
        subtype="html",
    )

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=20) as server:
            server.login(username, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=20) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(username, password)
        server.send_message(message)


async def unique_username(name: str, email: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or email.split("@")[0])).strip("_")
    base = (base or "google_user")[:80]
    candidate = base
    suffix = 1
    while await execute_one("SELECT id FROM users WHERE username = %s", (candidate,)):
        suffix += 1
        candidate = f"{base[:70]}_{suffix}"
    return candidate


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user = await execute_one(
            "SELECT * FROM users WHERE id = %s AND is_active = 1",
            (int(payload["sub"]),),
        )
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        version = payload.get("credential_version")
        expected = password_token_version(user["id"], user["password_hash"])
        if not isinstance(version, str) or not hmac.compare_digest(version, expected):
            raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
        return user
    except (KeyError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def require_admin(current_user=Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    username = req.username.strip()
    language = req.language_preference if req.language_preference in ACCOUNT_LANGUAGE_PREFERENCES else "auto"
    if not username:
        raise HTTPException(status_code=400, detail="Full name is required")
    if len(req.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be no more than 72 bytes. Symbols and non-English characters may use multiple bytes.")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    existing = await execute_one(
        "SELECT id FROM users WHERE email = %s OR username = %s",
        (req.email, username),
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email or username already exists")
    pwd_hash = hash_password(req.password)
    user_id  = await execute_insert(
        "INSERT INTO users (username, display_name, email, password_hash, language_preference) VALUES (%s,%s,%s,%s,%s)",
        (username, username, req.email, pwd_hash, language),
    )
    token = create_token(user_id, username, "user", pwd_hash)
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        username=username,
        role="user",
        language_preference=language,
        display_name=username,
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = await execute_one(
        "SELECT * FROM users WHERE email = %s AND is_active = 1", (req.email,)
    )
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"], user["username"], user["role"], user["password_hash"])
    return TokenResponse(
        access_token=token,
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
        language_preference=user.get("language_preference") or "auto",
        display_name=user.get("display_name") or user["username"],
    )


@router.post("/firebase", response_model=TokenResponse)
async def firebase_login(req: FirebaseLoginRequest):
    if not str(settings.FIREBASE_PROJECT_ID or "").strip():
        raise HTTPException(status_code=503, detail="Firebase sign-in is not configured on this server.")
    try:
        claims = await asyncio.to_thread(verify_firebase_id_token, req.id_token)
    except Exception as exc:
        # Never log the submitted bearer token or expose verifier internals.
        logger.info("Firebase sign-in token rejected (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=401,
            detail="Google sign-in could not be verified. Please try again.",
        ) from None

    email = claims["email"].strip().lower()
    user = await execute_one(
        "SELECT * FROM users WHERE LOWER(email) = %s LIMIT 1",
        (email,),
    )
    if user and not user.get("is_active", 1):
        raise HTTPException(status_code=403, detail="This LaborLens account is disabled.")

    if not user:
        display_name = str(claims.get("name") or claims.get("display_name") or "").strip()
        username = await unique_username(display_name, email)
        password_hash = hash_password(secrets.token_urlsafe(32))
        try:
            user_id = await execute_insert(
                "INSERT INTO users (username, display_name, email, password_hash, language_preference) VALUES (%s,%s,%s,%s,%s)",
                (username, display_name or username, email, password_hash, "auto"),
            )
        except IntegrityError:
            # A simultaneous first sign-in may have created this email already.
            user = await execute_one(
                "SELECT * FROM users WHERE LOWER(email) = %s LIMIT 1",
                (email,),
            )
            if not user:
                raise HTTPException(status_code=409, detail="Could not create the LaborLens account. Please retry sign-in.") from None
            if not user.get("is_active", 1):
                raise HTTPException(status_code=403, detail="This LaborLens account is disabled.") from None
        else:
            user = {
                "id": user_id,
                "password_hash": password_hash,
                "username": username,
                "display_name": display_name or username,
                "role": "user",
                "language_preference": "auto",
            }

    jwt_token = create_token(user["id"], user["username"], user["role"], user["password_hash"])
    return TokenResponse(
        access_token=jwt_token,
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
        language_preference=user.get("language_preference") or "auto",
        display_name=user.get("display_name") or user["username"],
    )


@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    user = await execute_one(
        "SELECT id FROM users WHERE email = %s AND is_active = 1",
        (req.email,),
    )
    response = {
        "message": "If that email exists, password reset instructions have been sent."
    }
    if not user:
        return response

    raw_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)
    await execute_insert(
        "INSERT INTO password_reset_tokens (user_id, token_hash, expires_at) VALUES (%s,%s,%s)",
        (user["id"], token_hash(raw_token), expires_at),
    )
    reset_link = password_reset_url(raw_token)
    try:
        send_password_reset_email(req.email, reset_link)
    except Exception as exc:
        # SMTP exceptions may contain the email body or credentials. Log only
        # the error type, never the reset URL/token or raw exception text.
        logger.warning("Password reset email delivery failed (%s)", type(exc).__name__)
        await execute_affected(
            "DELETE FROM password_reset_tokens WHERE token_hash = %s",
            (token_hash(raw_token),),
        )
    # Keep the public response identical for unknown accounts, sent emails,
    # and delivery failures, including when DEBUG is enabled.
    return response


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    if len(req.new_password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="Password must be no more than 72 bytes. Symbols and non-English characters may use multiple bytes.")
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    reset = await execute_one(
        """
        SELECT id, user_id FROM password_reset_tokens
        WHERE token_hash = %s AND used_at IS NULL AND expires_at > UTC_TIMESTAMP()
        """,
        (token_hash(req.token),),
    )
    if not reset:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    await execute_insert(
        "UPDATE users SET password_hash = %s WHERE id = %s",
        (hash_password(req.new_password), reset["user_id"]),
    )
    await execute_insert(
        "UPDATE password_reset_tokens SET used_at = UTC_TIMESTAMP() WHERE id = %s",
        (reset["id"],),
    )
    return {"message": "Password updated. You can now sign in."}


@router.get("/password-reset-email-status")
async def password_reset_email_status():
    host = str(settings.SMTP_HOST or "").strip()
    username = str(settings.SMTP_USERNAME or "").strip()
    from_email = str(settings.SMTP_FROM_EMAIL or username).strip()
    return {
        "smtp_configured": smtp_is_configured(),
        "host": host,
        "port": int(settings.SMTP_PORT or 587),
        "username_present": bool(username),
        "password_present": bool(str(settings.SMTP_PASSWORD or "").strip()),
        "from_email": from_email,
        "note": (
            "Configured. Restart FastAPI after .env changes."
            if smtp_is_configured()
            else "Missing SMTP_HOST, SMTP_USERNAME, or SMTP_PASSWORD."
        ),
    }


@router.get("/google/status")
async def google_status():
    return {"configured": google_is_configured()}


@router.get("/google/start")
async def google_start(request: Request, frontend_url: str = ""):
    if not google_is_configured():
        raise HTTPException(
            status_code=501,
            detail=(
                "Google sign-in is not configured yet. Add GOOGLE_CLIENT_ID, "
                "and GOOGLE_CLIENT_SECRET in the backend .env."
            ),
        )
    state, browser = await asyncio.to_thread(
        oauth_state.issue, settings.OAUTH_STATE_DB_PATH, safe_frontend_url(frontend_url)
    )
    response = RedirectResponse(google_auth_url(request, state), status_code=302)
    response.set_cookie(
        oauth_state.COOKIE, browser, max_age=oauth_state.TTL,
        httponly=True, secure=resolve_google_redirect_uri(request).startswith("https://"),
        samesite="lax", path=oauth_state.COOKIE_PATH,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/google/callback")
async def google_callback(request: Request, code: str = "", error: str = "", state: str = ""):
    frontend_url = await asyncio.to_thread(
        oauth_state.consume, settings.OAUTH_STATE_DB_PATH,
        state, request.cookies.get(oauth_state.COOKIE, ""),
    )
    if frontend_url is None:
        raise HTTPException(status_code=400, detail="Google sign-in expired or could not be verified. Please start sign-in again.")
    # Consumed before any Google exchange, including denied or failed sign-ins.
    frontend_url = frontend_url.rstrip("/") or "/"
    redirect_uri = resolve_google_redirect_uri(request)
    if error:
        return google_redirect(frontend_url, {"google_error": error})
    if not code or not google_is_configured():
        return google_redirect(frontend_url, {"google_error": "Google sign-in failed"})

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            return google_redirect(
                frontend_url,
                {
                    "google_error": (
                        "Google token exchange failed. Check that the Authorized redirect URI "
                        f"in Google Cloud exactly matches: {redirect_uri}"
                    )
                },
            )

        access_token = token_resp.json().get("access_token")
        user_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_resp.status_code != 200:
            return google_redirect(frontend_url, {"google_error": "Google profile lookup failed"})

    google_user = user_resp.json()
    email = google_user.get("email")
    if not email or not google_user.get("email_verified", False):
        return google_redirect(frontend_url, {"google_error": "Google email is not verified"})

    email = email.strip().lower()
    user = await execute_one(
        "SELECT * FROM users WHERE LOWER(email) = %s LIMIT 1",
        (email,),
    )
    if user and not user.get("is_active", 1):
        return google_redirect(frontend_url, {"google_error": "This LaborLens account is disabled."})
    if not user:
        username = await unique_username(google_user.get("name", ""), email)
        password_hash = hash_password(secrets.token_urlsafe(32))
        user_id = await execute_insert(
            "INSERT INTO users (username, display_name, email, password_hash, language_preference) VALUES (%s,%s,%s,%s,%s)",
            (username, google_user.get("name") or username, email, password_hash, "auto"),
        )
        user = {
            "id": user_id,
            "password_hash": password_hash,
            "username": username,
            "display_name": google_user.get("name") or username,
            "role": "user",
            "language_preference": "auto",
        }

    jwt_token = create_token(user["id"], user["username"], user["role"], user["password_hash"])
    fragment = urlencode({
        "google_token": jwt_token,
        "google_user_id": user["id"],
        "google_username": user["username"],
        "google_display_name": user.get("display_name") or user["username"],
        "google_role": user["role"],
        "google_language_preference": user.get("language_preference") or "auto",
    })
    response = RedirectResponse(f"{frontend_url}#{fragment}")
    response.delete_cookie(oauth_state.COOKIE, path=oauth_state.COOKIE_PATH)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/me")
async def get_me(current_user=Depends(get_current_user)):
    return {
        "id":                  current_user["id"],
        "username":            current_user["username"],
        "display_name":        current_user.get("display_name") or current_user["username"],
        "email":               current_user["email"],
        "role":                current_user["role"],
        "language_preference": current_user.get("language_preference") or "auto",
    }


@router.patch("/me/profile")
async def update_profile(req: ProfileUpdateRequest, current_user=Depends(get_current_user)):
    """Update the account display name without changing the login username."""
    display_name = " ".join(str(req.display_name).split())
    if not display_name:
        raise HTTPException(status_code=400, detail="Display name is required.")
    if len(display_name) > 100:
        raise HTTPException(status_code=400, detail="Display name must be 100 characters or fewer.")
    await execute_affected(
        "UPDATE users SET display_name = %s WHERE id = %s AND is_active = 1",
        (display_name, current_user["id"]),
    )
    return {"message": "Profile updated.", "display_name": display_name}


@router.patch("/me/account")
async def update_account(req: AccountUpdateRequest, current_user=Depends(get_current_user)):
    """Update only the authenticated account, with a fresh password check."""
    email = str(req.email).strip().lower()
    if len(email) > 255:
        raise HTTPException(status_code=400, detail="Email is too long.")
    if req.new_password is not None and len(req.new_password.encode("utf-8")) > 72:
        raise HTTPException(status_code=400, detail="New password must be no more than 72 bytes.")
    try:
        valid = await asyncio.to_thread(verify_password, req.current_password, current_user["password_hash"])
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    existing = await execute_one(
        "SELECT id FROM users WHERE email = %s AND id <> %s",
        (email, current_user["id"]),
    )
    if existing:
        raise HTTPException(status_code=409, detail="This email is already used by another account.")
    if email == current_user["email"] and req.new_password is None:
        return {"message": "No changes to save.", "email": email}
    new_hash = (await asyncio.to_thread(hash_password, req.new_password)
                if req.new_password is not None else current_user["password_hash"])
    try:
        changed = await execute_affected(
            "UPDATE users SET email = %s, password_hash = %s "
            "WHERE id = %s AND password_hash = %s AND email = %s AND is_active = 1",
            (email, new_hash, current_user["id"], current_user["password_hash"], current_user["email"]),
        )
    except IntegrityError as exc:
        if exc.args[0] == 1062:
            raise HTTPException(status_code=409, detail="This email is already used by another account.") from exc
        raise
    if not changed:
        raise HTTPException(status_code=409, detail="Your account changed. Refresh this page and try again.")
    if req.new_password is not None:
        return {"message": "Password updated. Please sign in again on all devices.",
                "email": email, "reauthenticate": True}
    return {"message": "Account updated. Use your updated details the next time you sign in.", "email": email}


@router.patch("/me/preferences")
async def update_preferences(
    req: PreferenceRequest,
    current_user=Depends(get_current_user),
):
    if req.language_preference not in ACCOUNT_LANGUAGE_PREFERENCES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    await execute_insert(
        "UPDATE users SET language_preference = %s WHERE id = %s",
        (req.language_preference, current_user["id"]),
    )
    return {
        "message": "Preferences updated",
        "language_preference": req.language_preference,
    }
