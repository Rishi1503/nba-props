"""Production after a holdout where some markets fail (the real 2025-26 result: PRA missed
one calibration decile). The verdict is read from the recorded clean evaluation, the holdout
is never evaluated again, and failed markets are switched off when the model is imported."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import delete

from nbaprops.backtest import run
from nbaprops.db.base import session_scope
from nbaprops.db.models import MarketControl
from nbaprops.ops import bundles

RUN = "bt-2025-26-202610212122"
SETTINGS = {"retrain_days": 7, "n_sims": 2000, "first_train_season": "2022-23"}


def recorded(tmp_path, markets: dict[str, bool]):
    reports = tmp_path / "reports"
    (reports / "backtest").mkdir(parents=True)
    (reports / "holdout_ledger.json").write_text(
        json.dumps(
            [
                {"season": "2025-26", "run_id": RUN, "clean": True, "passes": False},
                {"season": "2025-26", "run_id": "bt-rerun", "clean": False, "passes": True},
            ]
        )
    )
    report = {
        "run_id": RUN,
        "settings": SETTINGS,
        "layer_a": {
            "targets": {
                "passes": all(markets.values()),
                "markets": {m: {"passes": ok} for m, ok in markets.items()},
            }
        },
    }
    (reports / "backtest" / f"{RUN}.json").write_text(json.dumps(report))
    return reports


def test_the_verdict_is_the_recorded_clean_one(tmp_path):
    reports = recorded(tmp_path, {"pts": True, "pra": False})
    verdict, report = run.holdout_verdict(RUN, reports)
    assert verdict == {"pts": True, "pra": False} and report["settings"] == SETTINGS
    with pytest.raises(SystemExit, match="rerun"):
        run.holdout_verdict("bt-rerun", reports)
    with pytest.raises(SystemExit, match="isn't a recorded evaluation"):
        run.holdout_verdict("bt-other", reports)


class FakeBundle:
    def __init__(self, run_id):
        self.run_id, self.meta = run_id, {}

    def save(self, folder):
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps(self.meta))


def fake_training(monkeypatch, seen):
    lines = pd.DataFrame(
        {"game_date": ["2025-11-01"], "row_key": [1], "market": ["pts"], "p_raw": [0.5], "hit": [1]}
    )
    monkeypatch.setattr(run, "load_frame", lambda season, parquet: pd.DataFrame())

    def walk(df, season, settings):
        seen["settings"] = settings
        return SimpleNamespace(lines=lines), SimpleNamespace(lines=lines)

    monkeypatch.setattr(run.wf, "run", walk)

    def build(df, oos, run_id, first, features=None):
        seen["production_features"] = features
        return FakeBundle(run_id)

    monkeypatch.setattr(run, "production_bundle", build)


def test_production_from_the_holdout_never_evaluates_it_again(tmp_path, monkeypatch):
    reports = recorded(tmp_path, {"pts": True, "reb": True, "pra": False})
    monkeypatch.chdir(tmp_path)
    seen = {}
    fake_training(monkeypatch, seen)
    ledger_before = (reports / "holdout_ledger.json").read_text()
    out = run.main(["--season", "2025-26", "--from-holdout", RUN, "--reports", str(reports)])
    assert (reports / "holdout_ledger.json").read_text() == ledger_before  # nothing recorded
    assert sorted(p.name for p in (reports / "backtest").iterdir()) == [f"{RUN}.json"]
    s = seen["settings"]  # the recorded run's settings, so its cached models are reused
    assert (s.retrain_days, s.n_sims, s.first_train_season) == (7, 2000, "2022-23")
    meta = json.loads((tmp_path / out["bundle"] / "meta.json").read_text())
    assert meta["holdout"] == {
        "run_id": RUN,
        "passed": ["pts", "reb"],
        "failed": ["pra"],
        "clean": True,
        "accepted_rerun": None,
    }


def test_no_market_passing_means_no_production_model(tmp_path, monkeypatch):
    reports = recorded(tmp_path, {"pts": False, "pra": False})
    monkeypatch.chdir(tmp_path)
    fake_training(monkeypatch, {})
    with pytest.raises(SystemExit, match="No market passed"):
        run.main(["--season", "2025-26", "--from-holdout", RUN, "--reports", str(reports)])
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.db
def test_failed_markets_are_switched_off_at_import():
    with session_scope() as s:
        s.execute(delete(MarketControl))
    bundle = SimpleNamespace(
        meta={"holdout": {"run_id": RUN, "passed": ["pts"], "failed": ["pra"]}}
    )
    now = datetime(2026, 10, 22, tzinfo=UTC)
    with session_scope() as s:
        assert bundles.switch_off_failed_markets(s, bundle, now) == ["pra"]
    with session_scope() as s:
        rows = {m.market: (m.disabled, m.reason) for m in s.query(MarketControl).all()}
    assert set(rows) == {"pra"} and rows["pra"][0] is True
    assert RUN in rows["pra"][1] and "live evidence" in rows["pra"][1]
    with session_scope() as s:  # a bundle without holdout notes (a weekly retrain) changes nothing
        assert bundles.switch_off_failed_markets(s, SimpleNamespace(meta={}), now) == []


@pytest.mark.db
def test_bankroll_is_recorded_under_the_new_york_date(monkeypatch):
    """At 9:30 PM in New York it is already tomorrow in UTC (the database's clock): the
    bankroll must be dated today in New York, or that night's slate can't find it."""
    from datetime import date

    from nbaprops.db.models import BankrollEntry
    from nbaprops.ops import run as ops_run

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 22, 1, 30, tzinfo=UTC).astimezone(tz)

    with session_scope() as s:
        s.execute(delete(BankrollEntry))
    monkeypatch.setattr(ops_run, "datetime", Clock)
    ops_run.main(["bankroll", "1000"])
    with session_scope() as s:
        rows = [(r.entry_date, float(r.bankroll)) for r in s.query(BankrollEntry).all()]
    assert rows == [(date(2026, 10, 21), 1000.0)]


def test_a_rerun_needs_an_explicit_recorded_acceptance(tmp_path, monkeypatch):
    """After the absence-leakage fix the holdout had to be evaluated again: that rerun can
    only become production with --accept-rerun, and the reason travels with the model."""
    reports = recorded(tmp_path, {"pts": True, "pra": False})
    report = {
        "run_id": "bt-rerun",
        "settings": SETTINGS,
        "layer_a": {
            "targets": {
                "passes": True,
                "markets": {"pts": {"passes": True}, "reb": {"passes": True}},
            }
        },
    }
    (reports / "backtest" / "bt-rerun.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="--accept-rerun"):
        run.holdout_verdict("bt-rerun", reports)
    monkeypatch.chdir(tmp_path)
    fake_training(monkeypatch, {})
    why = "rerun after the absence-leakage fix"
    out = run.main(
        [
            "--season",
            "2025-26",
            "--from-holdout",
            "bt-rerun",
            "--accept-rerun",
            why,
            "--reports",
            str(reports),
        ]
    )
    meta = json.loads((tmp_path / out["bundle"] / "meta.json").read_text())
    assert meta["holdout"]["clean"] is False and meta["holdout"]["accepted_rerun"] == why
    assert meta["holdout"]["passed"] == ["pts", "reb"]


def test_model_inputs_without_a_group():
    from nbaprops.features.build import FEATURE_COLUMNS, FEATURE_GROUPS

    assert run.model_inputs(None) is None
    kept = run.model_inputs("roster")
    assert not set(FEATURE_GROUPS["roster"]) & set(kept)
    assert len(kept) == len(FEATURE_COLUMNS) - len(FEATURE_GROUPS["roster"])


def test_production_uses_the_inputs_the_rerun_evaluated(tmp_path, monkeypatch):
    """The 2025-26 rerun is evaluated without the roster inputs (no help once the leakage was
    gone): the walk-forward and the production model must both leave them out."""
    reports = recorded(tmp_path, {"pts": True})
    report = {
        "run_id": "bt-rerun",
        "settings": {**SETTINGS, "without": "roster"},
        "layer_a": {"targets": {"passes": True, "markets": {"pts": {"passes": True}}}},
    }
    (reports / "backtest" / "bt-rerun.json").write_text(json.dumps(report))
    monkeypatch.chdir(tmp_path)
    seen = {}
    fake_training(monkeypatch, seen)
    run.main(
        [
            "--season",
            "2025-26",
            "--from-holdout",
            "bt-rerun",
            "--accept-rerun",
            "fix",
            "--reports",
            str(reports),
        ]
    )
    expected = run.model_inputs("roster")
    assert seen["settings"].features == expected
    assert seen["production_features"] == expected
