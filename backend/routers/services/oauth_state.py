"""Persistent one-use Google OAuth state, bound to a separate browser secret."""
from contextlib import closing
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
import time

TTL = 600
COOKIE = "ll_google_oauth"
COOKIE_PATH = "/api/auth/google"

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def connection(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute("CREATE TABLE IF NOT EXISTS oauth_states (state_hash TEXT PRIMARY KEY, browser_hash TEXT NOT NULL, frontend TEXT NOT NULL, expires REAL NOT NULL)")
    return db

def issue(path, frontend, now=None):
    now = time.time() if now is None else now
    state, browser = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    with closing(connection(path)) as db:
        with db:
            db.execute("DELETE FROM oauth_states WHERE expires <= ?", (now,))
            db.execute("INSERT INTO oauth_states VALUES (?, ?, ?, ?)", (digest(state), digest(browser), frontend, now + TTL))
    return state, browser

def consume(path, state, browser, now=None):
    if not state or not browser or len(state) > 128 or len(browser) > 128:
        return None
    now = time.time() if now is None else now
    with closing(connection(path)) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT browser_hash, frontend, expires FROM oauth_states WHERE state_hash = ?", (digest(state),)).fetchone()
            if not row or row[2] <= now or not hmac.compare_digest(row[0], digest(browser)):
                return None
            db.execute("DELETE FROM oauth_states WHERE state_hash = ?", (digest(state),))
            return row[1]
