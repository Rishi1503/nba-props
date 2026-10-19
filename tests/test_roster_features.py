"""Trades, roster changes, opponent absences and with/without splits.

Every expected value here is computed directly from the synthetic league's box scores, not by
calling the code under test. Storylines (see feature_fixtures): in 2025-26 TRADED moves B->C
after game 12 and STAR (A) misses games 10-13; in 2024-25 MOVER moves D->A after game 11 and
C_STAR misses games 8-11.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from nbaprops.features import build as fb
from nbaprops.models import minutes as mm
from nbaprops.models.gate import beats_anchor
from tests import feature_fixtures as ff

S0, S1 = "2024-25", "2025-26"


@pytest.fixture(scope="module")
def league():
    inp = ff.build()
    return inp, fb.training_frame(ff.build())


def reg_games(inp, team, season):
    tl = inp.team_logs
    g = tl[(tl.team_id == team) & (tl.season == season) & (tl.season_type == "regular")]
    return list(g.sort_values("game_date")["game_id"])


def box(inp, pid, game_ids, col="minutes"):
    lg = inp.logs.set_index(["game_id", "player_id"])
    return np.array([float(lg.loc[(g, pid), col]) for g in game_ids])


def row(df, game_id, pid):
    r = df[(df.game_id == game_id) & (df.player_id == pid)]
    assert len(r) == 1, (game_id, pid)
    return r.iloc[0]


def teammates(df, game_id, team, exclude):
    return df[(df.game_id == game_id) & (df.team_id == team) & (df.player_id != exclude)]


# --- 1. since joining the current team -----------------------------------------


def test_stint_counts_and_averages_follow_the_trade(league):
    inp, df = league
    b, c = reg_games(inp, ff.B, S1), reg_games(inp, ff.C, S1)
    for k in range(13):  # with B, continuing his 2024-25 stint (24 games)
        r = row(df, b[k], ff.TRADED)
        assert r.stint_games == min(24 + k, fb.STINT_GAMES_CAP) and r.stint_change == 0
    first = row(df, c[13], ff.TRADED)  # debut with C
    assert first.stint_games == 0 and first.stint_change == 1 and first.new_team == 1
    assert np.isnan(first.stint_min_avg_l10)
    m = box(inp, ff.TRADED, c[13:16])
    p = box(inp, ff.TRADED, c[13:16], "pts")
    third = row(df, c[15], ff.TRADED)
    assert third.stint_games == 2
    assert third.stint_min_avg_l10 == pytest.approx(m[:2].mean())  # C games only
    assert third.stint_pts_pm_l10 == pytest.approx(p[:2].sum() / m[:2].sum())
    assert third.stint_usage_l10 == pytest.approx(box(inp, ff.TRADED, c[13:15], "usage_pct").mean())
    # his overall last-10 still mixes in B games: that's why the stint inputs exist
    assert third.min_avg_l10 != pytest.approx(third.stint_min_avg_l10)
    assert row(df, c[22], ff.TRADED).new_team == 1  # 9 games in
    assert row(df, c[23], ff.TRADED).new_team == 0  # 10 games in


def test_stint_continues_across_seasons_and_rookies_are_not_new_team(league):
    inp, df = league
    a = reg_games(inp, ff.A, S1)
    r = row(df, a[0], ff.MOVER)  # traded to A after game 11 of 2024-25: 12 A games so far
    assert r.stint_change == 1 and r.stint_games == 12 and r.new_team == 0
    d = reg_games(inp, ff.D, S1)
    rookie = df[df.player_id == ff.ROOKIE]
    assert (rookie.stint_change == 0).all() and (rookie.new_team == 0).all()
    assert row(df, d[0], ff.ROOKIE).stint_games == 0


# --- 2. roster changes visible to teammates ------------------------------------


def test_departed_role_and_fade(league):
    inp, df = league
    b = reg_games(inp, ff.B, S1)
    for k in range(13):
        assert (teammates(df, b[k], ff.B, ff.TRADED).departed_min_sum == 0).all()
    for j in range(10):
        played = b[3 + j : 13]  # his games in B's previous-10 window
        apps = len(played)
        expect = (10 - j) / 10 * box(inp, ff.TRADED, played).mean() if apps >= 3 else 0.0
        expect_u = (
            (10 - j) / 10 * box(inp, ff.TRADED, played, "usage_pct").mean() if apps >= 3 else 0.0
        )
        mates = teammates(df, b[13 + j], ff.B, ff.TRADED)
        assert mates.departed_min_sum.to_numpy() == pytest.approx(expect), j
        assert mates.departed_usage_sum.to_numpy() == pytest.approx(expect_u), j
        assert (mates.absent_n == 0).all()  # traded away is not absent


def test_arrived_role_and_fade(league):
    inp, df = league
    c = reg_games(inp, ff.C, S1)
    b = reg_games(inp, ff.B, S1)
    role = box(inp, ff.TRADED, b[3:13]).mean()  # last 10 games before the trade
    role_u = box(inp, ff.TRADED, b[3:13], "usage_pct").mean()
    for k in range(13):
        assert (teammates(df, c[k], ff.C, ff.TRADED).arrived_min_sum == 0).all()
    for j in range(11):
        mates = teammates(df, c[13 + j], ff.C, ff.TRADED)
        w = max(0, 10 - j) / 10
        assert mates.arrived_min_sum.to_numpy() == pytest.approx(w * role), j
        assert mates.arrived_usage_sum.to_numpy() == pytest.approx(w * role_u), j
    assert row(df, c[13], ff.TRADED).arrived_min_sum == 0  # never counts himself


def test_arrival_in_the_earlier_season_too(league):
    inp, df = league
    a, d = reg_games(inp, ff.A, S0), reg_games(inp, ff.D, S0)
    role = box(inp, ff.MOVER, d[2:12]).mean()
    # A's game 12 is against D: his first game for A is against his old team. "Played in this
    # game" must not count as "played for D".
    assert d[12] == a[12]
    mates = teammates(df, a[12], ff.A, ff.MOVER)
    assert mates.arrived_min_sum.to_numpy() == pytest.approx(role)
    d_mates = teammates(df, d[12], ff.D, ff.MOVER)
    assert d_mates.departed_min_sum.to_numpy() == pytest.approx(role)


def test_returning_role_and_fade(league):
    inp, df = league
    a = reg_games(inp, ff.A, S1)
    role = box(inp, ff.STAR, a[0:10]).mean()  # his last 10 games before the absence
    role_u = box(inp, ff.STAR, a[0:10], "usage_pct").mean()
    assert row(df, a[14], ff.STAR).games_missed == 4
    for j, w in enumerate([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
        mates = teammates(df, a[14 + j], ff.A, ff.STAR)
        assert mates.returning_min_sum.to_numpy() == pytest.approx(w * role), j
        assert mates.returning_usage_sum.to_numpy() == pytest.approx(w * role_u), j
    assert row(df, a[14], ff.STAR).returning_min_sum == 0
    for k in range(14):
        assert (teammates(df, a[k], ff.A, ff.STAR).returning_min_sum == 0).all()


def test_return_in_the_earlier_season_uses_pre_absence_role(league):
    inp, df = league
    c = reg_games(inp, ff.C, S0)
    role = box(inp, ff.C_STAR, c[0:8]).mean()  # only 8 games before the absence
    mates = teammates(df, c[12], ff.C, ff.C_STAR)
    assert mates.returning_min_sum.to_numpy() == pytest.approx(role)


# --- 3. opponent absences -----------------------------------------------------


def test_opponent_absences_mirror_the_injured_team(league):
    inp, df = league
    pos = {
        int(p): fb.primary_position(x)
        for p, x in zip(inp.players.player_id, inp.players.position, strict=True)
    }
    a = reg_games(inp, ff.A, S1)
    for k in range(10, 14):
        game = df[df.game_id == a[k]]
        a_rows, opp = game[game.team_id == ff.A], game[game.team_id != ff.A]
        assert (a_rows.absent_n == 1).all() and (opp.opp_absent_n == 1).all()
        assert opp.opp_absent_min_sum.to_numpy() == pytest.approx(a_rows.absent_min_sum.iloc[0])
        star_min = a_rows.absent_min_sum.iloc[0]
        for r in opp.itertuples():
            same = pos.get(int(r.player_id), "U") == pos[ff.STAR]
            assert r.opp_same_pos_absent_min == pytest.approx(star_min if same else 0.0)
        assert (a_rows.opp_absent_n == 0).all()
    for k in (9, 14):
        assert (df[(df.game_id == a[k]) & (df.team_id != ff.A)].opp_absent_n == 0).all()


# --- 4. with vs without -------------------------------------------------------


def test_with_without_deltas_match_hand_calculation(league):
    inp, df = league
    a0, a1 = reg_games(inp, ff.A, S0), reg_games(inp, ff.A, S1)
    p = ff._pid(0, 1)
    with_games = a0 + a1[:10]  # STAR played all of these
    for j in range(4):  # games 10..13 of 2025-26, STAR out
        r = row(df, a1[10 + j], p)
        assert r.wo_games == j
        if j == 0:
            assert r.wo_min_delta == 0 and r.wo_pts_pm_delta == 0
            continue
        wo_games = a1[10 : 10 + j]
        shrink = j / (j + fb.WO_SHRINK_GAMES)
        mw, mi = box(inp, p, wo_games), box(inp, p, with_games)
        assert r.wo_min_delta == pytest.approx(shrink * (mw.mean() - mi.mean()))
        for s in ("pts", "reb", "ast", "fg3m"):
            sw, si = box(inp, p, wo_games, s), box(inp, p, with_games, s)
            assert getattr(r, f"wo_{s}_pm_delta") == pytest.approx(
                shrink * (sw.sum() / mw.sum() - si.sum() / mi.sum())
            )
        uw, ui = box(inp, p, wo_games, "usage_pct"), box(inp, p, with_games, "usage_pct")
        assert r.wo_usage_delta == pytest.approx(shrink * (uw.mean() - ui.mean()))
    back = row(df, a1[14], p)  # STAR back: nobody absent, so no split
    assert back.wo_games == 0 and back.wo_min_delta == 0


def test_with_without_needs_games_together():
    """Only two earlier games with the absent teammate: too few to compare against."""
    rotation = pd.DataFrame(
        {
            "game_id": ["g3"],
            "team_id": [1],
            "player_id": [9],
            "rotation_min": [30.0],
            "is_absent": [True],
        }
    )
    logs = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g1", "g2"],
            "player_id": [5, 5, 9, 9],
            "team_id": 1,
            "minutes": [20.0, 22.0, 30.0, 31.0],
            "usage_pct": 0.2,
            "pts": 10,
            "reb": 3,
            "ast": 2,
            "fg3m": 1,
            "dnp": False,
            "season_type": "regular",
            "game_date": pd.to_datetime(["2025-11-01", "2025-11-03"] * 2),
        }
    )
    t = pd.DataFrame(
        {
            "game_id": ["g3"],
            "player_id": [5],
            "team_id": [1],
            "game_date": pd.to_datetime(["2025-11-05"]),
        }
    )
    out = fb.with_without(logs, rotation, t)
    assert out.iloc[0].wo_games == 0 and out.iloc[0].wo_min_delta == 0  # 2 games with < 3


# --- departures: the same-season rule and today's roster -----------------------


def _mini(next_season: str):
    """Team 1: P plays g1-g4 then sits g5; teammate Q plays everything. P's next game is for
    team 2 in `next_season`."""
    dates = pd.to_datetime([f"2025-11-0{i}" for i in range(1, 6)])
    logs = []
    for i, d in enumerate(dates, start=1):
        logs.append(
            {
                "game_id": f"g{i}",
                "player_id": 99,
                "team_id": 1,
                "minutes": 32.0,
                "game_date": d,
                "season": "2025-26",
            }
        )
        if i <= 4:
            logs.append(
                {
                    "game_id": f"g{i}",
                    "player_id": 7,
                    "team_id": 1,
                    "minutes": 30.0,
                    "game_date": d,
                    "season": "2025-26",
                }
            )
    logs.append(
        {
            "game_id": "h1",
            "player_id": 7,
            "team_id": 2,
            "minutes": 25.0,
            "game_date": pd.Timestamp("2026-01-10" if next_season == "2025-26" else "2026-10-25"),
            "season": next_season,
        }
    )
    logs = pd.DataFrame(logs).assign(usage_pct=0.2, started=True, dnp=False, season_type="regular")
    team_logs = logs[logs.team_id == 1].drop_duplicates("game_id")[
        ["game_id", "team_id", "game_date", "season"]
    ]
    players = pd.DataFrame({"player_id": [7, 99], "pos": ["G", "F"]})
    tg = team_logs[team_logs.game_id == "g5"]
    return logs, team_logs, players, tg


def test_trade_in_season_is_a_departure_offseason_move_is_an_absence():
    logs, tl, players, tg = _mini("2025-26")
    summary, _ = fb.team_absences(logs, tl, players, tg)
    assert summary.iloc[0].departed_min_sum == pytest.approx(30.0)
    assert summary.iloc[0].absent_n == 0
    logs, tl, players, tg = _mini("2026-27")
    summary, _ = fb.team_absences(logs, tl, players, tg)
    assert summary.iloc[0].departed_min_sum == 0
    assert summary.iloc[0].absent_n == 1


def test_live_roster_places_a_traded_player_on_his_new_team():
    inp = ff.build()
    day = ff.future_date()
    as_of = datetime(day.year, day.month, day.day, 17, 0, tzinfo=UTC)
    x = ff._pid(3, 1)  # Team D guard, traded to A this morning (no game for A yet)
    plain = fb.slate_frame(ff.build(), day, as_of)
    assert set(plain.loc[plain.player_id == x, "team_id"]) == {ff.D}
    assert (plain.arrived_min_sum == 0).all() and (plain.departed_min_sum == 0).all()

    inp.players = inp.players.assign(current_team_id=np.nan)
    inp.players.loc[inp.players.player_id == x, "current_team_id"] = ff.A
    live = fb.slate_frame(inp, day, as_of)
    mine = live[live.player_id == x]
    assert list(mine.team_id) == [ff.A]
    assert mine.iloc[0].stint_games == 0 and mine.iloc[0].new_team == 1
    d_games = reg_games(inp, ff.D, S1)
    role = box(inp, x, d_games[-10:]).mean()
    a_mates = live[(live.team_id == ff.A) & (live.player_id != x)]
    d_mates = live[live.team_id == ff.D]
    assert a_mates.arrived_min_sum.to_numpy() == pytest.approx(
        box(inp, x, d_games[-10:]).mean()
    )  # his last 10 games, all for D
    assert d_mates.departed_min_sum.to_numpy() == pytest.approx(role)
    assert (d_mates.absent_n == 0).all()


# --- data hygiene ----------------------------------------------------------------


def test_unknown_starts_are_not_counted_as_bench():
    inp = ff.build()
    a = reg_games(inp, ff.A, S1)
    mask = (inp.logs.player_id == ff.STAR) & inp.logs.game_id.isin(a[2:10])
    inp.logs["started"] = inp.logs["started"].astype(object)
    inp.logs.loc[mask, "started"] = None  # 8 unknowns, 2 known starts in the window
    df = fb.training_frame(inp)
    assert (df[(df.game_id == a[10]) & (df.team_id == ff.A)].absent_starters == 1).all()


def test_later_attendance_changes_no_earlier_roster_feature():
    base = fb.training_frame(ff.build())
    inp = ff.build()
    a = reg_games(inp, ff.A, S1)
    last_day = pd.Timestamp(inp.logs.loc[inp.logs.game_id == a[-1], "game_date"].iloc[0])
    inp.logs = inp.logs[~((inp.logs.game_id == a[-1]) & (inp.logs.player_id == ff._pid(0, 1)))]
    changed = fb.training_frame(inp)
    cols = [
        c
        for c in fb.FEATURE_COLUMNS
        if c.startswith(
            (
                "stint_",
                "departed_",
                "arrived_",
                "returning_",
                "wo_",
                "opp_absent",
                "opp_same",
                "absent_",
                "new_team",
            )
        )
    ]
    key = ["game_id", "player_id"]
    before = base[base.game_date < last_day].set_index(key)[cols].sort_index()
    after = changed[changed.game_date < last_day].set_index(key)[cols].sort_index()
    pd.testing.assert_frame_equal(before, after)


# --- models use what they're given ----------------------------------------------


def test_minutes_anchor_blends_toward_new_team_minutes(league):
    inp, df = league
    c = reg_games(inp, ff.C, S1)
    r = df[(df.game_id == c[16]) & (df.player_id == ff.TRADED)]  # 3 games with C
    n = 3
    expect = (
        n / (n + mm.STINT_ANCHOR_GAMES) * r.stint_min_avg_l10.iloc[0]
        + (1 - n / (n + mm.STINT_ANCHOR_GAMES)) * r.min_ewm.iloc[0]
    )
    assert mm.anchor(r)[0] == pytest.approx(expect)
    settled = df[(df.player_id == ff._pid(0, 1)) & (df.season == S1)].head(3)
    assert mm.anchor(settled) == pytest.approx(settled.min_ewm.to_numpy())


def test_minutes_model_learns_roster_change_effects():
    """Departures add minutes, arrivals take them away: invisible to recent averages, learnable
    from the new inputs."""
    rng = np.random.default_rng(11)
    n = 8000
    base = rng.uniform(12, 36, n)
    dep = np.where(rng.random(n) < 0.25, rng.uniform(15, 32, n), 0.0)
    arr = np.where(rng.random(n) < 0.25, rng.uniform(15, 32, n), 0.0)
    df = pd.DataFrame({c: np.nan for c in mm.FEATURES}, index=range(n))
    df["min_ewm"] = df["min_avg_l10"] = df["min_avg_l5"] = df["min_avg_l3"] = base
    df["std_min_avg"] = base
    for c in (
        "games_missed",
        "new_team",
        "late_season",
        "clinched_top6",
        "b2b",
        "absent_starters",
        "absent_min_sum",
        "returning_min_sum",
        "wo_games",
    ):
        df[c] = 0
    df["career_games"] = 200
    df["started_share_l10"] = (base > 26).astype(float)
    df["departed_min_sum"], df["arrived_min_sum"] = dep, arr
    df["y_minutes"] = base + 0.25 * dep - 0.2 * arr + rng.normal(0, 2.0, n)
    df["player_id"] = rng.integers(0, 300, n)
    df["game_date"] = pd.Timestamp("2024-10-22") + pd.to_timedelta(rng.integers(0, 170, n), "D")
    model = mm.fit(df.iloc[:6000])
    assert model.best_rounds[0.5] > 0  # the gate keeps the trees: the effect is real
    probe = df.iloc[6000:6400].copy()
    probe["departed_min_sum"], probe["arrived_min_sum"] = 0.0, 0.0
    _, lo = mm.predict(model, probe, df)
    probe["departed_min_sum"] = 28.0
    _, hi = mm.predict(model, probe, df)
    gain = (hi["q50"] - lo["q50"]).to_numpy()
    assert 4.5 < np.median(gain) < 9.5  # true effect: 0.25 * 28 = 7 minutes


def test_gate_requires_a_significant_gain():
    rng = np.random.default_rng(0)
    la = rng.uniform(1, 3, 400)
    assert not beats_anchor(la, la + rng.normal(-0.01, 0.5, 400), 0.005)  # noise
    assert beats_anchor(la, la * 0.95, 0.005)  # consistent 5% gain
    assert not beats_anchor(la, la * 0.998, 0.005)  # consistent but below the minimum


def test_with_without_and_trade_inputs_reach_the_rate_models():
    from nbaprops.models import rates

    for c in ("wo_pts_pm_delta", "stint_pts_pm_l10", "arrived_usage_sum", "opp_absent_min_sum"):
        assert c in rates.FEATURE_COLUMNS


# --- daily roster refresh ----------------------------------------------------------


@pytest.mark.db
def test_daily_job_records_current_teams_but_backfill_of_old_seasons_does_not():
    from datetime import date

    from sqlalchemy import select

    from nbaprops.db.base import session_scope
    from nbaprops.db.models import Player, Team
    from nbaprops.pipeline import stats
    from tests.test_revalidation import _reset
    from tests.test_stats import FakeFetch

    _reset(with_schedule=False)
    with session_scope() as s:
        lal, bos = sorted(s.scalars(select(Team.team_id)).all())[:2]

    class RosterFetch(FakeFetch):
        def __call__(self, endpoint, cache_key, cache=True, **params):
            if endpoint == "playerindex":
                self.calls.append((endpoint, cache_key))
                return {
                    "PlayerIndex": [
                        {"PERSON_ID": 1630595, "POSITION": "G", "TEAM_ID": lal},
                        {"PERSON_ID": 1642999, "POSITION": "C", "TEAM_ID": 0},  # free agent
                    ]
                }
            return super().__call__(endpoint, cache_key, cache, **params)

    fetch = RosterFetch()
    stats.backfill_season(fetch, "2025-26")  # a past season: positions only
    with session_scope() as s:
        p = s.get(Player, 1630595)
        assert p.position == "G" and p.current_team_id is None
        s.get(Player, 1642999).current_team_id = bos  # stale team from an earlier refresh
    stats.pull_day(fetch, date(2025, 10, 23))
    with session_scope() as s:
        assert s.get(Player, 1630595).current_team_id == lal
        assert s.get(Player, 1642999).current_team_id is None  # released: no team
        assert s.get(Player, 1630595).roster_updated is not None
    assert [k for e, k in fetch.calls if e == "playerindex"][-1] == "2025-26_20251023"


def test_minutes_gate_is_one_decision_for_the_whole_distribution():
    """Keeping trees for some quantiles but not others distorts the distribution's shape (it
    showed up as miscalibrated points props), so the gate decides for all three together."""
    from tests import feature_fixtures as fx

    df = fb.training_frame(fx.build(n_regular=82))
    for seed in (1, 2, 3):
        kept = mm.fit(df[df.season == S0], params={"seed": seed}).best_rounds
        assert all(v > 0 for v in kept.values()) or all(v == 0 for v in kept.values()), kept


def test_clustered_calibration_check_counts_shared_outcomes_once():
    from nbaprops.models import calibrate

    rng = np.random.default_rng(5)
    n = 400
    p = np.full(n, 0.65)
    hit = (rng.random(n) < 0.72).astype(float)  # 7 points off
    plain = calibrate.calibration_failures(p, hit)
    same = calibrate.calibration_failures(p, hit, clusters=np.arange(n))  # all independent
    assert [f["bin"] for f in plain] == [f["bin"] for f in same]
    # the same 80 outcomes copied onto 5 lines each: 400 rows, but only 80 results
    p5, hit5 = np.repeat(p[:80], 5), np.repeat(hit[:80], 5)
    unclustered = calibration_z(calibrate.calibration_failures(p5, hit5, tolerance=0.0, z_crit=0.0))
    clustered = calibration_z(
        calibrate.calibration_failures(
            p5, hit5, tolerance=0.0, z_crit=0.0, clusters=np.repeat(np.arange(80), 5)
        )
    )
    assert clustered == pytest.approx(unclustered / np.sqrt(5), rel=0.15)


def calibration_z(failures):
    assert len(failures) == 1
    return failures[0]["z"]
