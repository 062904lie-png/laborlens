"""Durable, atomic anonymous preview budgets shared by local API workers."""
import asyncio
from contextlib import closing
import hashlib
import hmac
import ipaddress
from pathlib import Path
import sqlite3
import time

from fastapi import HTTPException, Request
from config import settings

MESSAGE_LIMIT = 5
WINDOW_SECONDS = 24 * 60 * 60


def client_key(request: Request) -> str:
    # Trust only the address resolved by Uvicorn's configured trusted proxies,
    # never an arbitrary X-Forwarded-For header supplied by a caller.
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        host = str(address)
    except ValueError:
        raise HTTPException(503, "Guest access is temporarily unavailable. Please sign in.")
    return hmac.new(settings.JWT_SECRET.encode(),
                    ("guest-preview:" + host).encode(), hashlib.sha256).hexdigest()


def read_or_reserve(path: str, key: str, *, consume: bool, now: float | None = None) -> dict:
    """Reserve before model work, atomically across processes.

    Attempts count even when downstream generation fails, preventing repeated
    failures from exhausting provider resources outside the budget.
    """
    now = time.time() if now is None else now
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path, timeout=10)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS guest_preview_usage "
                           "(client_key TEXT PRIMARY KEY, used INTEGER NOT NULL, resets_at REAL NOT NULL)")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM guest_preview_usage WHERE resets_at <= ?", (now,))
        row = connection.execute(
            "SELECT used, resets_at FROM guest_preview_usage WHERE client_key = ?", (key,)
        ).fetchone()
        used, resets_at = row if row else (0, now + WINDOW_SECONDS)
        allowed = used < MESSAGE_LIMIT
        if consume and allowed:
            used += 1
            connection.execute(
                "INSERT INTO guest_preview_usage VALUES (?, ?, ?) "
                "ON CONFLICT(client_key) DO UPDATE SET used = excluded.used, resets_at = excluded.resets_at",
                (key, used, resets_at),
            )
        connection.commit()
    return {"allowed": allowed, "used": used, "limit": MESSAGE_LIMIT,
            "remaining": max(0, MESSAGE_LIMIT - used), "resets_at": resets_at}


async def guest_usage(request: Request, *, consume: bool = False) -> dict:
    key = client_key(request)
    try:
        return await asyncio.to_thread(read_or_reserve, settings.GUEST_USAGE_DB_PATH, key, consume=consume)
    except (sqlite3.Error, OSError):
        raise HTTPException(503, "Guest access is temporarily unavailable. Please sign in.")
