from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nbaprops.features import build as fb
from tests import feature_fixtures as ff


@pytest.fixture(scope="module")
def train():
    return fb.training_frame(ff.build())


def _row(df, player_id, nth_game_of_season, season="2025-26"):
    rows = df[(df.player_id == player_id) & (df.season == season)].sort_values("game_date")
    return rows.iloc[nth_game_of_season]


# --- leakage (the plan's required tests) -------------------------------------


def test_future_game_changes_no_existing_feature(train):
    inp = ff.build()
    last = inp.logs["game_date"].max()
    fake = inp.logs[inp.logs["game_date"] == last].copy()
    fake["game_id"] = fake["game_id"].str[:-2] + "99"
    fake["game_date"] = last + timedelta(days=3)
    fake[["pts", "reb", "ast", "minutes"]] = [999, 999, 999, 48]
    inp.logs = pd.concat([inp.logs, fake])
    tl = inp.team_logs[inp.team_logs["game_date"] == last].copy()
    tl["game_id"] = tl["game_id"].str[:-2] + "99"
    tl["game_date"] = last + timedelta(days=3)
    tl["pts"] = 999
    inp.team_logs = pd.concat([inp.team_logs, tl])
    g = inp.games[inp.games["game_date"] == last].copy()
    g["game_id"] = g["game_id"].str[:-2] + "99"
    g["game_date"] = last + timedelta(days=3)
    inp.games = pd.concat([inp.games, g])
    after = fb.training_frame(inp)
    after = after[after["game_date"] <= pd.Timestamp(last)].reset_index(drop=True)
    pd.testing.assert_frame_equal(train[fb.FEATURE_COLUMNS], after[fb.FEATURE_COLUMNS])


def test_a_games_own_result_never_reaches_its_features(train):
    inp = ff.build()
    target = _row(train, ff.STAR, 15)  # he played 20 of 24 (missed 4)
    mask = (inp.logs["game_id"] == target.game_id) & (inp.logs["player_id"] == ff.STAR)
    inp.logs.loc[mask, ["pts", "reb", "ast"]] = [999, 999, 999]
    after = fb.training_frame(inp)
    changed = after[(after.game_id == target.game_id) & (after.player_id == ff.STAR)].iloc[0]
    for col in fb.FEATURE_COLUMNS:
        a, b = target[col], changed[col]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), col
    assert changed["y_pts"] == 999


# --- rolling values ----------------------------------------------------------


def test_rolling_minutes_and_rates_match_hand_computation(train):
    inp = ff.build()
    pid = ff._pid(1, 2)
    r = _row(train, pid, 15)
    hist = inp.logs[
        (inp.logs.player_id == pid)
        & (inp.logs.season_type != "preseason")
        & (inp.logs.game_date < r.game_date.date())
    ].sort_values("game_date")
    last5 = hist.tail(5)
    assert r["min_avg_l5"] == pytest.approx(last5["minutes"].mean())
    assert r["pts_pm_l5"] == pytest.approx(last5["pts"].sum() / last5["minutes"].sum())
    season = hist[hist.season == "2025-26"]
    assert r["std_games"] == len(season)
    assert r["std_min_avg"] == pytest.approx(season["minutes"].mean())


def test_new_season_starts_from_prior_and_shrinks_toward_it(train):
    pid = ff._pid(1, 2)
    first = _row(train, pid, 0)
    assert first["std_games"] == 0 and first["prior_source"] == 2
    inp = ff.build()
    prev = inp.logs[
        (inp.logs.player_id == pid)
        & (inp.logs.season == "2024-25")
        & (inp.logs.season_type == "regular")
    ]
    prior = prev["pts"].sum() / prev["minutes"].sum()
    assert first["pts_pm_shrunk"] == pytest.approx(prior)  # no games yet -> exactly the prior
    later = _row(train, pid, 20)
    season = inp.logs[
        (inp.logs.player_id == pid)
        & (inp.logs.season == "2025-26")
        & (inp.logs.season_type == "regular")
        & (inp.logs.game_date < later.game_date.date())
    ]
    observed = season["pts"].sum() / season["minutes"].sum()
    expected = (season["pts"].sum() + fb.PRIOR_MINUTES * prior) / (
        season["minutes"].sum() + fb.PRIOR_MINUTES
    )
    assert later["pts_pm_shrunk"] == pytest.approx(expected)
    assert min(prior, observed) <= later["pts_pm_shrunk"] <= max(prior, observed)


def test_rookie_uses_position_prior(train):
    r = _row(train, ff.ROOKIE, 0)
    assert r["rookie"] == 1 and r["prior_source"] == 1 and r["career_games"] == 0


def test_first_season_has_no_prior(train):
    assert (train[train.season == "2024-25"]["prior_source"] == 0).all()


# --- teammates ---------------------------------------------------------------


def test_star_absence_shows_up_for_teammates_only_while_out(train):
    mate = ff._pid(0, 5)  # team A guard, same position as the star
    during = [_row(train, mate, i) for i in (10, 11, 12, 13)]
    before = _row(train, mate, 9)
    after = _row(train, mate, 14)
    for r in during:
        assert r["absent_n"] == 1 and r["absent_starters"] == 1
        assert r["absent_min_sum"] > 25 and r["same_pos_absent_min"] == r["absent_min_sum"]
    assert before["absent_n"] == 0 and after["absent_n"] == 0
    forward = _row(train, ff._pid(0, 2), 11)  # a forward: absence counted, not same position
    assert forward["absent_min_sum"] > 25 and forward["same_pos_absent_min"] == 0


def test_traded_player_is_not_absent_for_old_team_and_flags_new_team(train):
    old_mate = _row(train, ff._pid(1, 0), 16)  # team B after the trade
    assert old_mate["absent_n"] == 0
    with_c = train[(train.player_id == ff.TRADED) & (train.team_id == ff.C)].sort_values(
        "game_date"
    )
    assert with_c.iloc[0]["new_team"] == 1 and with_c.iloc[0]["games_with_team"] == 0
    assert with_c.iloc[3]["games_with_team"] == 3


# --- opponent, ratings, schedule, standings ------------------------------------


def test_strong_team_gets_positive_rating_margin(train):
    late = train[
        (train.season == "2025-26") & (train.team_id == ff.A) & (train.opp_team_id == ff.D)
    ]
    assert (late["rating_team_margin"].dropna() > 0).mean() > 0.8
    assert train["rating_total"].dropna().between(150, 300).all()


def test_schedule_rest_and_back_to_backs(train):
    a = train[(train.team_id == ff.A) & (train.season == "2025-26")].drop_duplicates("game_id")
    a = a.sort_values("game_date")
    gaps = a["game_date"].diff().dt.days
    assert ((gaps == 1) == (a["b2b"] == 1)).iloc[1:].all()
    assert (a["games_last_7"] <= 7).all() and (a["games_last_7"] >= 0).all()


def test_standings_count_only_earlier_regular_season_games(train):
    inp = ff.build()
    r = _row(train, ff._pid(0, 1), 10)
    g = inp.games[
        (inp.games.season == "2025-26")
        & (inp.games.season_type == "regular")
        & (inp.games.game_date < r.game_date.date())
    ]
    played = ((g.home_team_id == ff.A) | (g.away_team_id == ff.A)).sum()
    assert r["games_remaining"] == fb.REGULAR_SEASON_GAMES - played
    assert 1 <= r["conf_rank"] <= 2  # two teams per conference in the fixture


def test_opponent_allowed_by_position_is_per_minute(train):
    vals = train["opp_allow_pts_pm_l15"].dropna()
    assert vals.between(0.1, 1.5).all()


# --- live slate --------------------------------------------------------------


def _injuries(status: str, report_ts: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "report_ts": report_ts,
                "game_id": ff.FUTURE_GAME,
                "team_id": ff.A,
                "player_id": ff.STAR,
                "status": status,
            }
        ]
    )


def test_slate_excludes_players_ruled_out_and_counts_their_absence():
    day = ff.future_date()
    report = datetime(day.year, day.month, day.day, 17, 0, tzinfo=UTC)
    inp = ff.build()
    inp.injuries = _injuries("Out", report)
    slate = fb.slate_frame(inp, day, report + timedelta(minutes=30))
    assert ff.STAR not in set(slate.player_id)
    mates = slate[slate.team_id == ff.A]
    assert len(mates) >= 7 and (mates["absent_n"] == 1).all()
    assert set(slate.team_id) == {ff.A, ff.D}
    assert "y_minutes" not in slate.columns  # no outcomes for an unplayed game


def test_slate_ignores_injury_reports_after_as_of():
    day = ff.future_date()
    report = datetime(day.year, day.month, day.day, 17, 0, tzinfo=UTC)
    inp = ff.build()
    inp.injuries = _injuries("Out", report)
    slate = fb.slate_frame(inp, day, report - timedelta(hours=1))
    assert ff.STAR in set(slate.player_id)
    assert (slate[slate.team_id == ff.A]["absent_n"] == 0).all()


def test_slate_questionable_player_kept_and_flagged():
    day = ff.future_date()
    report = datetime(day.year, day.month, day.day, 17, 0, tzinfo=UTC)
    inp = ff.build()
    inp.injuries = _injuries("Questionable", report)
    slate = fb.slate_frame(inp, day, report + timedelta(minutes=5))
    star = slate[slate.player_id == ff.STAR].iloc[0]
    assert star["injury_status"] == "Questionable"


def test_slate_features_match_what_training_would_compute():
    """Same join for live and training: a finished game re-run as a slate gives identical
    form features for players who played."""
    inp = ff.build()
    train = fb.training_frame(ff.build())
    last_day = train["game_date"].max()
    slate = fb.slate_frame(inp, last_day.date(), datetime.now(UTC))
    cols = [
        c for c in fb.FEATURE_COLUMNS if not c.startswith("absent") and c != "same_pos_absent_min"
    ]
    t = train[train.game_date == last_day].set_index(["game_id", "player_id"])[cols]
    s = slate.set_index(["game_id", "player_id"])[cols]
    common = t.index.intersection(s.index)
    assert len(common) >= 20
    pd.testing.assert_frame_equal(
        t.loc[common].sort_index(), s.loc[common].sort_index(), check_dtype=False
    )


def test_market_lines_respect_as_of():
    inp = ff.build()
    day = ff.future_date()
    t0 = datetime(day.year, day.month, day.day, 14, 0, tzinfo=UTC)
    inp.game_lines = pd.DataFrame(
        [
            {
                "snapshot_ts": t0,
                "game_id": ff.FUTURE_GAME,
                "book": "pinnacle",
                "market": "spreads",
                "point": -6.5,
            },
            {
                "snapshot_ts": t0,
                "game_id": ff.FUTURE_GAME,
                "book": "pinnacle",
                "market": "totals",
                "point": 221.5,
            },
            {
                "snapshot_ts": t0 + timedelta(hours=4),
                "game_id": ff.FUTURE_GAME,
                "book": "pinnacle",
                "market": "spreads",
                "point": -9.0,
            },
        ]
    )
    early = fb.slate_frame(inp, day, t0 + timedelta(hours=1))
    home = early[early.team_id == ff.A].iloc[0]
    assert home["market_team_margin"] == 6.5 and home["market_total"] == 221.5
    away = early[early.team_id == ff.D].iloc[0]
    assert away["market_team_margin"] == -6.5
    late = fb.slate_frame(inp, day, t0 + timedelta(hours=5))
    assert late[late.team_id == ff.A].iloc[0]["market_team_margin"] == 9.0


def test_feature_frame_is_numeric_and_complete(train):
    x = train[fb.FEATURE_COLUMNS]
    assert all(np.issubdtype(dt, np.number) for dt in x.dtypes), x.dtypes[
        [not np.issubdtype(dt, np.number) for dt in x.dtypes]
    ]
    assert len(train) == len(train.drop_duplicates(["game_id", "player_id"]))
