"""Third review of Stages 1-5: one test per fix."""

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from nbaprops.db.base import session_scope
from nbaprops.db.models import Player
from nbaprops.features import build as fb
from nbaprops.models import minutes as mm
from nbaprops.models import project as pj
from nbaprops.models.simulate import MARKETS, line_probs
from nbaprops.pipeline import stats
from tests import feature_fixtures as ff
from tests.test_revalidation import _reset
from tests.test_stats import FakeFetch

ROOT = Path(__file__).resolve().parents[1]

# --- Stage 1: daily job ------------------------------------------------------


@pytest.mark.db
def test_daily_job_retries_missing_box_scores_and_refreshes_positions():
    _reset(with_schedule=False)
    fetch = FakeFetch()
    stats.backfill_season(fetch, "2025-26")  # game 2's box score comes back empty
    with session_scope() as s:
        s.execute(
            Player.__table__.update().where(Player.player_id == 1642999).values(position=None)
        )  # a debut with no position yet
    fetch.calls.clear()
    stats.pull_day(fetch, date(2025, 10, 22))  # the day of game 2
    boxes = [k for e, k in fetch.calls if e == "boxscoretraditionalv3"]
    assert "0022500002" in boxes
    fetch.calls.clear()
    stats.pull_day(fetch, date(2025, 10, 23))  # a later day with no games
    assert ("boxscoretraditionalv3", "0022500002") in fetch.calls  # still retried
    fetch.calls.clear()
    stats.pull_day(fetch, date(2025, 10, 21))  # day of game 1: the debut's team plays
    index_calls = [k for e, k in fetch.calls if e == "playerindex"]
    assert index_calls == ["2025-26_20251021"]
    with session_scope() as s:
        assert s.scalar(select(Player.position).where(Player.player_id == 1630595)) == "G"


# --- Stage 3/4: playoffs -----------------------------------------------------


def _with_playoffs():
    inp = ff.build()
    cutoff = inp.games.loc[inp.games["season"] == "2025-26", "game_date"].sort_values().iloc[-7]
    for df in (inp.logs, inp.team_logs, inp.games):
        post = (
            (df["season"] == "2025-26")
            & (df["game_date"] >= cutoff)
            & (df["season_type"] == "regular")
        )
        df.loc[post, "season_type"] = "playoffs"
    return inp


def test_playoff_rows_are_marked_and_rest_signals_switched_off():
    df = fb.training_frame(_with_playoffs())
    post = df[df["season_type"] == "playoffs"]
    reg = df[df["season_type"] == "regular"]
    assert len(post) and (post["is_postseason"] == 1).all()
    assert (reg["is_postseason"] == 0).all()
    assert (post[["late_season", "clinched_top6", "eliminated"]] == 0).all().all()
    assert post["games_remaining"].isna().all()


def test_playoff_starters_are_not_flagged_low_confidence():
    df = fb.training_frame(_with_playoffs())
    model = mm.fit(df[df.season == "2024-25"])
    post = df[
        (df["season_type"] == "playoffs")
        & (df["started_share_l10"] >= 0.5)
        & (df["games_missed"] == 0)
        & (df["new_team"] == 0)
        & (df["career_games"] >= 5)
    ]
    assert len(post)
    _, s = mm.predict(model, post, df)
    assert not s["low_confidence"].any()


# --- Stage 5: whole-number lines ---------------------------------------------


def test_line_probabilities_for_half_and_whole_lines():
    grid = {"23.5": 0.70, "24.5": 0.55, "25.5": 0.40}
    assert line_probs(grid, 24.5) == pytest.approx((0.55, 0.45, 0.0))
    assert line_probs(grid, 25) == pytest.approx((0.40, 0.45, 0.15))  # push = 0.55 - 0.40
    assert line_probs(grid, 40) == (0.0, 1.0, 0.0) and line_probs(grid, 5) == (1.0, 0.0, 0.0)
    for line in (23, 24, 25):
        assert sum(line_probs(grid, line)) == pytest.approx(1.0)


# --- Stage 3/5: Questionable players -----------------------------------------


def test_mixing_uses_play_probability_for_teammates_only():
    n = 40_000
    frame = pd.DataFrame(
        {"game_id": ["G"] * 4, "player_id": [1, 2, 3, 4], "team_id": [10, 10, 10, 20]}
    )
    plays = {m: np.full(n, 10) for m in MARKETS}
    sits = {m: np.full(n, 20) for m in MARKETS}
    base = {("G", p): plays for p in (1, 2, 3, 4)}
    alt = {("G", p): sits for p in (2, 3, 4)}  # player 1 is the Questionable one
    keys = pd.DataFrame({"game_id": ["G"], "team_id": [10], "player_id": [1], "p_play": [0.3]})
    mixed = pj.mix_questionable(base, alt, keys, frame, "run", n)
    assert mixed[("G", 1)]["pts"].mean() == 10  # his own props assume he plays (else void)
    assert mixed[("G", 2)]["pts"].mean() == pytest.approx(0.3 * 10 + 0.7 * 20, abs=0.1)
    same = mixed[("G", 2)]["pts"] == mixed[("G", 3)]["pts"]
    assert same.all()  # one decision per run for the whole team
    assert mixed[("G", 4)]["pts"].mean() == 10  # the other team is unaffected


def test_questionable_scenarios_and_projection():
    inp = ff.build()
    day = ff.future_date()
    report = datetime(day.year, day.month, day.day, 17, 0, tzinfo=UTC)
    inp.injuries = pd.DataFrame(
        [
            {
                "report_ts": report,
                "game_id": ff.FUTURE_GAME,
                "team_id": ff.A,
                "player_id": ff.STAR,
                "status": "Questionable",
            },
            {
                "report_ts": report,
                "game_id": ff.FUTURE_GAME,
                "team_id": ff.D,
                "player_id": ff._pid(3, 8),
                "status": "Out",
            },
        ]
    )
    as_of = report + timedelta(minutes=10)
    base, alt, keys = fb.questionable_scenarios(inp, day, as_of, p_play=0.5)
    assert list(keys["player_id"]) == [ff.STAR]
    assert ff.STAR in set(base.player_id) and ff.STAR not in set(alt.player_id)
    mates_base = base[(base.team_id == ff.A) & (base.player_id != ff.STAR)]
    mates_alt = alt[alt.team_id == ff.A]
    assert (mates_base["absent_n"] == 0).all() and (mates_alt["absent_n"] == 1).all()
    assert ff._pid(3, 8) not in set(alt.player_id)  # the Out player stays out in both

    train = fb.training_frame(ff.build())
    bundle = pj.fit_bundle(
        train[train.season == "2024-25"],
        train[train.season == "2025-26"].head(2500),
        train,
        "q",
        n_sims=300,
    )
    proj = pj.project_frame(base, bundle, 2000, train, as_of, alt, keys)
    team_a = {int(p) for p in base.loc[base.team_id == ff.A, "player_id"]}
    mates = proj[proj.player_id.isin(team_a - {ff.STAR})]
    # teammates carry both scenarios; the Questionable player and the other team don't
    assert mates["scenarios"].notna().all()
    assert proj.loc[proj.player_id == ff.STAR, "scenarios"].isna().all()
    team_d = set(base.loc[base.team_id == ff.D, "player_id"])
    assert proj[proj.player_id.isin(team_d)]["scenarios"].isna().all()
    # the stored blend is exactly the p_play mix of the two calibrated scenarios (to rounding)
    for r in mates.itertuples():
        sc = r.scenarios
        assert sc["player_id"] == ff.STAR and sc["p_play"] == 0.5
        assert sc["in"].keys() == sc["out"].keys() == r.p_over.keys()
        for k, v in r.p_over.items():
            assert v == pytest.approx(0.5 * sc["in"][k] + 0.5 * sc["out"][k], abs=2e-4)
    # low confidence is the minutes model's own flag, no longer forced on the whole team
    assert not mates["low_confidence"].all()


def test_play_rates_from_logged_reports():
    tip = datetime(2026, 11, 1, 23, 0, tzinfo=UTC)
    games = pd.DataFrame({"game_id": ["g1", "g2"], "tip_time": [tip, tip], "status": "final"})
    inj = pd.DataFrame(
        [
            # player 1: Out in the morning, Questionable at the last report before tip, played
            {
                "report_ts": tip - timedelta(hours=8),
                "game_id": "g1",
                "player_id": 1,
                "status": "Out",
            },
            {
                "report_ts": tip - timedelta(hours=1),
                "game_id": "g1",
                "player_id": 1,
                "status": "Questionable",
            },
            {
                "report_ts": tip - timedelta(hours=1),
                "game_id": "g1",
                "player_id": 2,
                "status": "Questionable",
            },  # didn't play
            {
                "report_ts": tip - timedelta(hours=1),
                "game_id": "g2",
                "player_id": 3,
                "status": "Probable",
            },  # played
            {
                "report_ts": tip + timedelta(hours=1),
                "game_id": "g2",
                "player_id": 4,
                "status": "Questionable",
            },  # listed after tip: ignored
        ]
    )
    logs = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "player_id": [1, 2, 3],
            "minutes": [30, 0, 25],
            "dnp": [False, True, False],
        }
    )
    rates = fb.estimate_play_rates(inj, logs, games).set_index("status")
    assert rates.loc["Questionable", "listings"] == 2
    assert rates.loc["Questionable", "play_rate"] == pytest.approx(0.5)
    assert rates.loc["Probable", "play_rate"] == 1.0


# --- documentation can't drift from the code again ---------------------------


def test_every_command_is_documented_and_every_make_target_runs_a_real_command():
    run_src = (ROOT / "nbaprops/pipeline/run.py").read_text()
    commands = set(re.findall(r'add_parser\("([a-z\-]+)"', run_src))
    import nbaprops.pipeline.run as run

    documented = set(re.findall(r"^  ([a-z\-]+) ", run.__doc__, re.M))
    assert commands == documented
    makefile = (ROOT / "Makefile").read_text()
    used = set(re.findall(r"nbaprops\.pipeline\.run ([a-z\-]+)", makefile))
    assert used <= commands, used - commands
    phony = set(re.findall(r"^\.PHONY:(.*)$", makefile, re.M)[0].split())
    targets = set(re.findall(r"^([a-z0-9\-]+):", makefile, re.M))
    assert targets <= phony, targets - phony


def test_no_dead_code_in_tests():
    """Guard against half-edited assertions: a dead branch makes a test check nothing."""
    paths = [*ROOT.joinpath("tests").glob("*.py"), *ROOT.joinpath("web/e2e").glob("*.py")]
    for path in paths:
        text = path.read_text()
        assert "if " + "False" not in text, path.name
        assert "team_id_" + "dummy" not in text, path.name
        assert "or " + "True)" not in text, path.name  # a check that can't fail


def test_stage8_commands_are_documented():
    import nbaprops.ops.run as ops_run

    src = (ROOT / "nbaprops/ops/run.py").read_text()
    commands = set(re.findall(r'add_parser\("([a-z\-]+)"', src))
    # a command: a word at most 2 spaces in, then a run of spaces (formatters may dedent)
    documented = set(re.findall(r"^ {0,2}([a-z][a-z\-]+) {2,}\S", ops_run.__doc__, re.M))
    assert commands == documented
    makefile = (ROOT / "Makefile").read_text()
    assert set(re.findall(r"nbaprops\.ops\.run ([a-z\-]+)", makefile)) <= commands


def test_the_schedule_covers_every_pricing_run_and_closing_window():
    """The outside schedule (infra/schedule.yaml, New York time): the 10 AM, 3 PM and 6 PM
    pricing runs every day, and a run in the 35 minutes before every tip-off (closing lines):
    evenings every day, afternoons on weekends, when the NBA plays early games."""
    import yaml

    sched = yaml.safe_load((ROOT / "infra/schedule.yaml").read_text())
    assert sched["timezone"] == "America/New_York"
    wf = yaml.safe_load((ROOT / ".github/workflows/collect.yml").read_text())
    inputs = wf[True]["workflow_dispatch"]["inputs"]["command"]["description"]
    assert {j["command"] for j in sched["jobs"]} <= set(inputs.replace(" ", "").split("|"))

    def values(field: str, lo: int, hi: int) -> list[int]:
        if field == "*":
            return list(range(lo, hi + 1))
        out = []
        for part in field.split(","):
            a, _, b = part.partition("-")
            out += range(int(a), int(b or a) + 1)
        return out

    def runs_on(dow: int) -> set[int]:  # cron day of week: 0 Sunday ... 6 Saturday
        return {
            h * 60 + m
            for j in sched["jobs"]
            if j["command"] == "cycle" and dow in values(j["cron"].split()[4], 0, 6)
            for h in values(j["cron"].split()[1], 0, 23)
            for m in values(j["cron"].split()[0], 0, 59)
        }

    for dow in range(7):
        runs = runs_on(dow)
        for et in (10, 15, 18):
            assert et * 60 in runs, (dow, et)
        first_tip = 12 * 60 if dow in (0, 6) else 18 * 60
        for tip in range(first_tip, 23 * 60 + 1, 5):
            assert any(0 < tip - r <= 35 for r in runs), (dow, divmod(tip, 60))
        assert len(runs) <= 30  # minutes budget: every run bills at least one minute
