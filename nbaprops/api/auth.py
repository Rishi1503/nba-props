"""Single-user login for the dashboard (plan, Stage 10: "one hashed password checked by the
API"), standard library only.

  DASHBOARD_PASSWORD_HASH  scrypt hash of your password: `python -m nbaprops.api.auth`
                           prompts for it and prints the value to store as a secret
  DASHBOARD_SECRET         a long random string that signs session tokens

Login returns a bearer token (HMAC-SHA256 signed, 30-day expiry) that the website sends as
`Authorization: Bearer ...`. Bearer tokens rather than cookies because the site and the API
live on different domains, where browsers increasingly block cookies. Comparisons are
constant-time, and an address is locked out for 15 minutes after 5 failed logins.
"""

import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque

TOKEN_DAYS = 30
MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60
SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, digest = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), dklen=32, n=int(n), r=int(r), p=int(p)
        )
        return hmac.compare_digest(got.hex(), digest)
    except (ValueError, TypeError):
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(secret: str, now: float | None = None, days: int = TOKEN_DAYS) -> tuple[str, int]:
    exp = int((now or time.time()) + days * 86400)
    payload = _b64(json.dumps({"exp": exp}).encode())
    sig = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}", exp


def check_token(token: str, secret: str, now: float | None = None) -> bool:
    try:
        payload, sig = token.split(".")
        want = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, want):
            return False
        return json.loads(_unb64(payload))["exp"] > (now or time.time())
    except (ValueError, KeyError, json.JSONDecodeError):
        return False


class Throttle:
    """Failed logins per address in a sliding window (one API instance: memory is enough)."""

    def __init__(self, limit: int = MAX_FAILURES, window: int = LOCKOUT_SECONDS):
        self.limit, self.window = limit, window
        self.failures: dict[str, deque] = defaultdict(deque)

    def locked(self, who: str, now: float | None = None) -> bool:
        now = now or time.time()
        q = self.failures[who]
        while q and q[0] < now - self.window:
            q.popleft()
        return len(q) >= self.limit

    def fail(self, who: str, now: float | None = None) -> None:
        self.failures[who].append(now or time.time())

    def clear(self, who: str) -> None:
        self.failures.pop(who, None)


def settings_from_env() -> tuple[str, str]:
    """(password hash, token secret); both required for the API to accept any login."""
    return os.environ.get("DASHBOARD_PASSWORD_HASH", ""), os.environ.get("DASHBOARD_SECRET", "")


if __name__ == "__main__":
    pw = getpass.getpass("Dashboard password: ")
    if len(pw) < 12:
        raise SystemExit("Use at least 12 characters.")
    if pw != getpass.getpass("Again: "):
        raise SystemExit("They didn't match.")
    # single-quoted: the hash contains "$", which a shell sourcing .env would expand
    print(f"DASHBOARD_PASSWORD_HASH='{hash_password(pw)}'")
    print(f"DASHBOARD_SECRET='{secrets.token_urlsafe(48)}'")
