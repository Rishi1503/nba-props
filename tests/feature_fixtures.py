"""A small synthetic league for Stage 3 tests: 4 teams, 2 seasons, built-in storylines.

- Team A's star guard (STAR) misses regular-season games 10-13 of 2025-26.
- Team B's forward (TRADED) is traded to Team C after game 12 of 2025-26.
- Team D adds a rookie center (ROOKIE) in 2025-26.
- 2024-25 has its own storylines, so models can learn roster effects before being tested on
  2025-26: Team D's center (MOVER) is traded to Team A after game 11 and stays there, and
  Team C's top guard (C_STAR) misses games 8-11.
- Team A is strong, Team D weak; the schedule includes back-to-backs.
- One scheduled (unplayed) game, A vs D, follows the last 2025-26 date.
- Minutes vary about 8% game to game, and ~2% of games a regular starter exits early.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nbaprops.features.build import FeatureInputs
from nbaprops.pipeline.stats import possessions, usage_rate

TEAMS = [(1610612738, "BOS"), (1610612765, "DET"), (1610612760, "OKC"), (1610612759, "SAS")]
A, B, C, D = (t for t, _ in TEAMS)
STRENGTH = {A: 1.08, B: 1.0, C: 1.0, D: 0.92}
POSITIONS = ["G", "G", "F", "F", "C", "G", "F", "C", "F"]
BASE_MIN = [34, 33, 32, 30, 28, 24, 20, 14, 10]
BASE_PTS = [0.75, 0.6, 0.55, 0.5, 0.45, 0.45, 0.4, 0.35, 0.3]
STAR = 7000  # team A, player 0
TRADED = 7025  # team B, player 5
ROOKIE = 9999
MOVER = 7064  # team D, player 4 (center): traded to team A after game 11 of 2024-25
C_STAR = 7040  # team C, player 0: misses games 8-11 of 2024-25
SEASONS = ["2024-25", "2025-26"]
FUTURE_GAME = "0022500999"


def _pid(team_idx: int, i: int) -> int:
    return 7000 + team_idx * 20 + i


PAIRINGS = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
GAPS = [1, 2, 2, 3]  # day gaps between game dates; the 1s create back-to-backs


def build(n_regular: int = 24, seed: int = 7) -> FeatureInputs:
    rng = np.random.default_rng(seed)
    logs, team_logs, games = [], [], []
    for s_idx, season in enumerate(SEASONS):
        yr = int(season[:4])
        dates = [date(yr, 10, 5), date(yr, 10, 9)]  # preseason
        d = date(yr, 10, 22)
        for k in range(n_regular):
            dates.append(d)
            d += timedelta(days=GAPS[k % len(GAPS)])
        for k, day in enumerate(dates):
            pre = k < 2
            reg_idx = k - 2
            for pair_no, (h, a) in enumerate(PAIRINGS[k % 3]):
                if k % 2:
                    h, a = a, h
                prefix = "001" if pre else "002"
                gid = f"{prefix}{str(yr)[2:]}{k:03d}{pair_no:02d}"
                lines = {}
                for team_idx in (h, a):
                    team = TEAMS[team_idx][0]
                    roster = [(_pid(team_idx, i), i) for i in range(9)]
                    if s_idx == 1 and team == B and not pre and reg_idx > 12:
                        roster = [r for r in roster if r[0] != TRADED]
                    if s_idx == 1 and team == C and not pre and reg_idx > 12:
                        roster.append((TRADED, 5))
                    if s_idx == 1 and team == D:
                        roster.append((ROOKIE, 7))
                    # 2024-25: Team D's center (MOVER) is traded to Team A after game 11
                    moved = (s_idx == 0 and not pre and reg_idx > 11) or s_idx == 1
                    if team == D and moved:
                        roster = [r for r in roster if r[0] != MOVER]
                    if team == A and moved:
                        roster.append((MOVER, 4))
                    out = []
                    if s_idx == 1 and team == A and not pre and 10 <= reg_idx <= 13:
                        out = [STAR]
                    elif s_idx == 0 and team == C and not pre and 8 <= reg_idx <= 11:
                        out = [C_STAR]  # 2024-25: Team C's top guard misses four games
                    active = [(p, i) for p, i in roster if p not in out]
                    raw = np.array([BASE_MIN[i] for _, i in active], dtype=float)
                    raw *= rng.normal(1.0, 0.08, len(raw)).clip(0.6, 1.4)
                    # ~2% of games a regular starter exits early (injury, fouls)
                    exit_ = (raw >= 22) & (rng.random(len(raw)) < 0.02)
                    exit_min = rng.uniform(2, 10, len(raw)) * exit_
                    mins = np.where(
                        exit_,
                        exit_min,
                        raw / raw[~exit_].sum() * (240 - exit_min.sum()),
                    )
                    rows = []
                    for (pid, i), m in zip(active, mins, strict=True):
                        rate = BASE_PTS[i] * STRENGTH[team] * (1.06 if pid == STAR else 1)
                        pts = int(rng.poisson(rate * m))
                        fga = max(1, round(pts / 1.15))
                        fg3a = round(fga * 0.38)
                        fg3m = int(rng.binomial(fg3a, 0.36))
                        fta = round(pts * 0.25)
                        ftm = round(fta * 0.78)
                        fgm = max(0, min(fga, (pts - ftm - fg3m) // 2))
                        reb = int(rng.poisson((0.25 if POSITIONS[i] == "C" else 0.15) * m))
                        oreb = reb // 4
                        rows.append(
                            {
                                "game_id": gid,
                                "player_id": pid,
                                "team_id": team,
                                "started": False,
                                "dnp": False,
                                "dnp_reason": None,
                                "minutes": round(float(m), 2),
                                "pts": pts,
                                "reb": reb,
                                "oreb": oreb,
                                "dreb": reb - oreb,
                                "ast": int(
                                    rng.poisson((0.18 if POSITIONS[i] == "G" else 0.07) * m)
                                ),
                                "stl": 1,
                                "blk": 0,
                                "tov": int(rng.poisson(0.05 * m)),
                                "pf": 2,
                                "fgm": fgm,
                                "fga": fga,
                                "fg3m": fg3m,
                                "fg3a": fg3a,
                                "ftm": ftm,
                                "fta": fta,
                                "plus_minus": 0,
                            }
                        )
                    for r in sorted(rows, key=lambda r: -r["minutes"])[:5]:
                        r["started"] = True
                    for pid in out:
                        rows.append(
                            {
                                "game_id": gid,
                                "player_id": pid,
                                "team_id": team,
                                "started": False,
                                "dnp": True,
                                "dnp_reason": "DNP - Injury/Illness",
                                "minutes": 0,
                                **{
                                    c: 0
                                    for c in (
                                        "pts",
                                        "reb",
                                        "oreb",
                                        "dreb",
                                        "ast",
                                        "stl",
                                        "blk",
                                        "tov",
                                        "pf",
                                        "fgm",
                                        "fga",
                                        "fg3m",
                                        "fg3a",
                                        "ftm",
                                        "fta",
                                        "plus_minus",
                                    )
                                },
                            }
                        )
                    lines[team] = rows
                th, ta = TEAMS[h][0], TEAMS[a][0]
                tot = {
                    t: {
                        c: sum(r[c] for r in lines[t])
                        for c in (
                            "pts",
                            "fgm",
                            "fga",
                            "fg3m",
                            "fg3a",
                            "ftm",
                            "fta",
                            "oreb",
                            "dreb",
                            "reb",
                            "ast",
                            "stl",
                            "blk",
                            "tov",
                            "pf",
                        )
                    }
                    for t in (th, ta)
                }
                for t in (th, ta):
                    tot[t]["minutes"] = 240.0
                poss = possessions(tot[th], tot[ta])
                for t, o, is_home in ((th, ta, True), (ta, th, False)):
                    team_logs.append(
                        {
                            "game_id": gid,
                            "team_id": t,
                            "opp_team_id": o,
                            "is_home": is_home,
                            **tot[t],
                            "possessions": poss,
                            "pace": 48 * poss / 48,
                            "off_rtg": 100 * tot[t]["pts"] / poss,
                            "def_rtg": 100 * tot[o]["pts"] / poss,
                        }
                    )
                    for r in lines[t]:
                        r["usage_pct"] = usage_rate(r, tot[t]) if r["minutes"] else None
                        logs.append(
                            {
                                **r,
                                "game_date": day,
                                "season": season,
                                "season_type": "preseason" if pre else "regular",
                            }
                        )
                for row in team_logs[-2:]:
                    row.update(
                        game_date=day, season=season, season_type="preseason" if pre else "regular"
                    )
                games.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "season_type": "preseason" if pre else "regular",
                        "game_date": day,
                        "home_team_id": th,
                        "away_team_id": ta,
                        "home_score": tot[th]["pts"],
                        "away_score": tot[ta]["pts"],
                        "status": "final",
                    }
                )
    last = max(g["game_date"] for g in games)
    games.append(
        {
            "game_id": FUTURE_GAME,
            "season": "2025-26",
            "season_type": "regular",
            "game_date": last + timedelta(days=2),
            "home_team_id": A,
            "away_team_id": D,
            "home_score": None,
            "away_score": None,
            "status": "scheduled",
        }
    )
    players = pd.DataFrame(
        [{"player_id": _pid(t, i), "position": POSITIONS[i]} for t in range(4) for i in range(9)]
        + [{"player_id": ROOKIE, "position": "C"}]
    )
    teams = pd.DataFrame([{"team_id": t, "abbreviation": a} for t, a in TEAMS])
    return FeatureInputs(
        logs=pd.DataFrame(logs),
        team_logs=pd.DataFrame(team_logs),
        games=pd.DataFrame(games),
        players=players,
        teams=teams,
    )


def future_date() -> date:
    g = build().games
    return g.loc[g["game_id"] == FUTURE_GAME, "game_date"].iloc[0]
