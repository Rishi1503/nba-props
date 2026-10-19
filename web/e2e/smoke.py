"""Browser smoke test of the whole dashboard on a phone-sized screen (Stage 10).

    E2E_DATABASE_URL=postgresql+psycopg://.../props_e2e make e2e

Resets that (disposable!) database with migrations, seeds the demo, starts the API and the
built site, and walks every page in headless Chromium: log in, see staked picks, mark one
placed, price a milestone and a boost, open a player, Bets, Results (both charts), Models
(compare), Status, log out. Fails on any failed check or any browser console error.
Screenshots go to web/e2e/shots/. Needs `npx ng build` first and `pip install playwright`
(plus `playwright install chromium`).
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = str(Path(__file__).resolve().parents[2])
SHOTS = Path(__file__).resolve().parent / "shots"
SHOTS.mkdir(exist_ok=True)
DB = os.environ.get("E2E_DATABASE_URL", "")
if "e2e" not in DB and "demo" not in DB:
    sys.exit(
        "Set E2E_DATABASE_URL to a disposable database (its name must contain e2e or demo):"
        " this test wipes it."
    )
env = dict(
    os.environ,
    DATABASE_URL=DB,
    DASHBOARD_SECRET="e2e-secret-0123456789abcdef",
    DASHBOARD_ORIGINS="http://localhost:4200",
)
env["DASHBOARD_PASSWORD_HASH"] = subprocess.run(
    [
        "python3",
        "-c",
        "from nbaprops.api.auth import hash_password; print(hash_password('demo password 123'))",
    ],
    cwd=ROOT,
    capture_output=True,
    text=True,
    env=env,
).stdout.strip()
# a fresh demo every run: earlier runs' placed bets would otherwise leak into this one
for step in (["downgrade", "base"], ["upgrade", "head"]):
    subprocess.run(["alembic", *step], cwd=ROOT, env=env, check=True, capture_output=True)
subprocess.run(
    ["python3", "-m", "nbaprops.api.demo"], cwd=ROOT, env=env, check=True, capture_output=True
)
api = subprocess.Popen(
    ["python3", "-m", "uvicorn", "nbaprops.api.main:app", "--port", "8000"],
    cwd=ROOT,
    env=env,
    stdout=open(SHOTS / "api.log", "w"),
    stderr=subprocess.STDOUT,
)
site = subprocess.Popen(
    ["python3", "-m", "http.server", "4200", "--directory", f"{ROOT}/web/dist/web/browser"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    for _ in range(40):
        try:
            urllib.request.urlopen("http://localhost:8000/health")
            break
        except Exception:
            time.sleep(0.25)
    errors, results = [], []

    def check(name, ok):
        results.append((name, bool(ok)))

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://localhost:4200/")
        page.wait_for_url("**/login")
        check("unauthenticated visit redirects to login", page.url.endswith("/login"))
        page.fill("input[type=password]", "demo password 123")
        page.click("button[type=submit]")
        page.wait_for_selector("article.card.hc")
        page.wait_for_timeout(500)
        check("board shows picks", page.locator("article.card").count() >= 2)
        check(
            "picks carry player names",
            not page.locator("article a strong").first.inner_text().strip().isdigit(),
        )
        page.screenshot(path=f"{SHOTS}/1-board.png", full_page=True)
        # mark the first unplaced staked pick as placed
        page.locator("button", has_text="Placed").first.click()
        page.locator("input[name=stake]").fill("7.5")
        page.locator("button", has_text="Save").click()
        page.wait_for_selector("text=Placed: $7.50")
        check(
            "placing a bet shows it on the board", page.locator("text=Placed: $7.50").count() == 1
        )
        # manual milestone through the player picker
        picker = page.locator("select[name=mp]")
        check(
            "milestone picker lists the game's players by name",
            picker.locator("option").count() >= 2
            and not picker.locator("option").first.inner_text().isdigit(),
        )
        page.locator("button", has_text="Price").last.click()
        page.wait_for_selector("pre:has-text('EV')")
        check(
            "milestone price returns fair, EV and stake",
            page.locator("pre", has_text="EV").count() == 1,
        )
        page.locator("button", has_text="Price it").click()
        page.wait_for_function("document.querySelectorAll('pre').length >= 2")
        check("boost entry returns an answer", page.locator("pre").count() >= 2)
        page.screenshot(path=f"{SHOTS}/1b-board-tools.png", full_page=True)
        # player detail
        page.locator("article a strong").first.click()
        page.wait_for_selector("canvas")
        check("player page draws the distribution", page.locator("canvas").count() == 1)
        check(
            "player page lists recent games", page.locator("table").first.locator("tr").count() > 5
        )
        page.screenshot(path=f"{SHOTS}/2-player.png", full_page=True)
        for tab, sel, shot in (
            ("Bets", "text=Tax records", "3-bets"),
            ("Results", "canvas", "4-results"),
            ("Status", "text=Go-live", "6-status"),
        ):
            page.locator("nav a", has_text=tab).click()
            page.wait_for_selector(sel)
            page.wait_for_timeout(400)
            check(f"{tab} page renders", page.locator(sel).count() >= 1)
            if tab == "Results":
                check(
                    "results page draws CLV and calibration charts",
                    page.locator("canvas").count() == 2,
                )
            page.screenshot(path=f"{SHOTS}/{shot}.png", full_page=True)
        page.locator("nav a", has_text="Models").click()
        page.wait_for_selector("text=Compare")
        page.locator("button", has_text="Compare").click()
        page.wait_for_selector("table")
        check("models page compares runs", page.locator("table tr").count() > 5)
        page.screenshot(path=f"{SHOTS}/5-models.png", full_page=True)
        page.locator("button", has_text="Log out").click()
        page.wait_for_url("**/login")
        check("log out returns to login", page.url.endswith("/login"))
        b.close()
    for name, ok in results:
        print(("PASS " if ok else "FAIL ") + name)
    print("console errors:", errors or "none")
    sys.exit(0 if all(ok for _, ok in results) and not errors else 1)
finally:
    api.terminate()
    site.terminate()
