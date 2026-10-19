"""Loader smoke test: Stage 1 data in Postgres -> training features."""

import pytest
from sqlalchemy import func, select

from nbaprops.db.base import session_scope
from nbaprops.db.models import PlayerGameLog
from nbaprops.features import build as fb
from nbaprops.pipeline import stats
from tests.test_stats import FakeFetch, db  # noqa: F401  (db fixture)

pytestmark = pytest.mark.db


def test_training_set_from_database(db):  # noqa: F811
    stats.backfill_season(FakeFetch(), "2025-26")
    with session_scope() as s:
        played = s.scalar(
            select(func.count()).select_from(PlayerGameLog).where(PlayerGameLog.minutes > 0)
        )
        df = fb.build_training_set(s, ["2025-26"])
    assert len(df) == played == 32
    assert df[fb.FEATURE_COLUMNS].select_dtypes("number").shape[1] == len(fb.FEATURE_COLUMNS)
    assert (df["y_pra"] == df["y_pts"] + df["y_reb"] + df["y_ast"]).all()
    assert df["std_games"].eq(0).all()  # each player's first game of the season
