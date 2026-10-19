"""Alerts to your phone (plan, Stage 8): a Discord webhook or a Telegram bot.

Configured by environment variables (GitHub secrets on the scheduled runner):
  DISCORD_WEBHOOK_URL                      or
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
With neither, alerts go to the log. Sending never raises: an alert outage must not stop a job.
`once` sends a given problem a single time (a stale board would otherwise alert every tick).
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import requests
from sqlalchemy.dialects.postgresql import insert

from nbaprops.db.models import AlertLog

log = logging.getLogger(__name__)


def run_link() -> str:
    """Link to this GitHub Actions run's log, when running there."""
    server, repo, run = (
        os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    )
    return f"{server}/{repo}/actions/runs/{run}" if server and repo and run else ""


def _chunks(text: str, size: int) -> list[str]:
    """Split on line breaks into messages under `size` characters."""
    out, cur = [], ""
    for line in text.splitlines() or [""]:
        while len(line) > size:  # a single very long line
            out.append(line[:size])
            line = line[size:]
        if len(cur) + len(line) + 1 > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        out.append(cur)
    return out


class Notifier:
    def send(self, text: str) -> bool:
        raise NotImplementedError


class LogNotifier(Notifier):
    def send(self, text: str) -> bool:
        log.warning("ALERT: %s", text)
        return True


@dataclass
class DiscordNotifier(Notifier):
    url: str
    limit: int = 1900  # Discord's cap is 2,000 characters

    def send(self, text: str) -> bool:
        ok = True
        for part in _chunks(text, self.limit):
            try:
                r = requests.post(self.url, json={"content": part}, timeout=10)
                ok &= r.status_code < 300
            except requests.RequestException as exc:
                log.error("Discord alert failed: %s", exc)
                ok = False
        return ok


@dataclass
class TelegramNotifier(Notifier):
    token: str
    chat_id: str
    limit: int = 4000  # Telegram's cap is 4,096

    def send(self, text: str) -> bool:
        ok = True
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for part in _chunks(text, self.limit):
            try:
                r = requests.post(url, json={"chat_id": self.chat_id, "text": part}, timeout=10)
                ok &= r.status_code < 300
            except requests.RequestException as exc:
                log.error("Telegram alert failed: %s", exc)
                ok = False
        return ok


@dataclass
class RecordingNotifier(Notifier):
    """For tests and dry runs: keeps every message."""

    sent: list[str] = field(default_factory=list)

    def send(self, text: str) -> bool:
        self.sent.append(text)
        return True


def from_env() -> Notifier:
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        return DiscordNotifier(os.environ["DISCORD_WEBHOOK_URL"])
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        return TelegramNotifier(os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"])
    return LogNotifier()


def once(session, key: str, now: datetime, notifier: Notifier, text: str) -> bool:
    """Send `text` only the first time `key` is seen; later sightings just count. Keys should
    carry their scope (e.g. a date) so a new day can alert again."""
    row = session.execute(
        insert(AlertLog)
        .values(key=key[:200], first_sent=now, last_seen=now, count=1)
        .on_conflict_do_nothing()
        .returning(AlertLog.key)
    ).scalar()
    if row is None:
        session.execute(
            AlertLog.__table__.update()
            .where(AlertLog.key == key[:200])
            .values(last_seen=now, count=AlertLog.count + 1)
        )
        return False
    notifier.send(text)
    return True


def with_link(text: str) -> str:
    link = run_link()
    return f"{text}\n{link}" if link else text
