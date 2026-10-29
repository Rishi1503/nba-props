"""The bankroll page: the paper record as if every recommended bet had been placed."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from nbaprops.db.base import session_scope
from nbaprops.site import build as site
from tests.test_stage9 import TODAY, history


def _now():
    return datetime.combine(TODAY, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20)


@pytest.mark.db
def test_the_page_adds_up_to_the_paper_record():
    history()
    with session_scope() as s:
        start = float(
            s.execute(
                text("SELECT bankroll FROM bankroll_ledger ORDER BY entry_date LIMIT 1")
            ).scalar()
            or 1000
        )
        pnl = float(
            s.execute(
                text("""SELECT coalesce(sum(pnl), 0) FROM picks
            WHERE mode = 'paper' AND stake > 0 AND tier <> 'boost_leg' AND result IS NOT NULL""")
            ).scalar()
        )
        par = float(
            s.execute(
                text("""SELECT coalesce(sum(pnl), 0) FROM parlays
            WHERE mode = 'paper' AND stake > 0 AND result IS NOT NULL""")
            ).scalar()
        )
        wins = s.execute(
            text("""SELECT count(*) FROM picks WHERE mode = 'paper' AND stake > 0
            AND tier <> 'boost_leg' AND result = 'win'""")
        ).scalar()
        data = site.snapshot(s, _now())
    assert data["start"]["bankroll"] == pytest.approx(start)
    assert data["bankroll"] == pytest.approx(start + pnl + par, abs=0.02)
    assert data["curve"][-1]["bankroll"] == pytest.approx(data["bankroll"], abs=0.02)
    assert data["singles"]["win"] == wins
    assert len(data["days"]) <= 21 and data["kill"]["verdict"]


@pytest.mark.db
def test_the_page_embeds_the_data_and_the_fingerprint_ignores_the_clock(tmp_path):
    history()
    with session_scope() as s:
        data = site.snapshot(s, _now())
    html = site.render(
        {
            **data,
            "days": [
                {
                    **data["days"][0],
                    "picks": [{**data["days"][0]["picks"][0], "player": "</script><b>x"}],
                }
            ],
        },
        _now(),
    )
    assert "/*__DATA__*/" not in html and "</script><b>" not in html  # escaped in the page
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(blob.replace("<\\/", "</"))["bankroll"] == data["bankroll"]
    assert site.fingerprint(data) == site.fingerprint(json.loads(json.dumps(data)))
    fp = site.build(tmp_path, _now())
    assert (tmp_path / "hash.txt").read_text().strip() == fp
    assert site.build(tmp_path, _now() + timedelta(hours=1)) == fp  # same data, same hash
