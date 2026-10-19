"""Deterministic fixtures in nba_api's normalized-dict format (column names from nba_api's
expected_data). Player rows sum exactly to team rows, so validation should pass on good games.
"""

from datetime import date

BOS, DET, OKC, SAS = 1610612738, 1610612765, 1610612760, 1610612759

# (player_id, name) per team; real ids where the static list knows them, plus one rookie-like id
ROSTERS = {
    BOS: [
        (1627759, "Jaylen Brown"),
        (1630573, "Sam Hauser"),
        (1629057, "Robert Williams III"),
        (1628401, "Derrick White"),
        (1630202, "Payton Pritchard"),
        (1631199, "Ron Harper Jr."),
        (1641700, "Player Seven"),
        (1642999, "Rookie Unlisted"),
    ],
    DET: [
        (1630595, "Cade Cunningham"),
        (1631105, "Jalen Duren"),
        (1630180, "Saddiq Bey"),
        (1641709, "Ausar Thompson"),
        (1630587, "Isaiah Stewart"),
        (1641710, "Player Six"),
        (1641711, "Player Seven B"),
        (1641712, "Player Eight B"),
    ],
    OKC: [
        (1628983, "Shai Gilgeous-Alexander"),
        (1631096, "Chet Holmgren"),
        (1631114, "Jalen Williams"),
        (1627936, "Alex Caruso"),
        (1629652, "Luguentz Dort"),
        (1641717, "OKC Six"),
        (1641718, "OKC Seven"),
        (1641719, "OKC Eight"),
    ],
    SAS: [
        (1641705, "Victor Wembanyama"),
        (1641706, "Stephon Castle"),
        (1641707, "SAS Three"),
        (1641708, "SAS Four"),
        (1641720, "SAS Five"),
        (1641721, "SAS Six"),
        (1641722, "SAS Seven"),
        (1641723, "SAS Eight"),
    ],
}

STAT_COLS = [
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]


def _player_lines(team_id: int, total_minutes: float, seed: int) -> list[dict]:
    """Eight players; minutes sum to total_minutes; points = 2*FGM + FG3M + FTM."""
    shares = [0.15, 0.15, 0.14, 0.14, 0.13, 0.11, 0.10, 0.08]
    lines = []
    for i, (pid, name) in enumerate(ROSTERS[team_id]):
        k = (seed + i) % 5
        fgm, fga = 3 + k, 7 + 2 * k
        fg3m, fg3a = k % 3, 2 + k % 3
        ftm, fta = 1 + k % 2, 2 + k % 2
        oreb, dreb = 1 + k % 2, 3 + k
        line = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": name,
            "MIN": round(total_minutes * shares[i], 3),
            "FGM": fgm,
            "FGA": fga,
            "FG3M": fg3m,
            "FG3A": fg3a,
            "FTM": ftm,
            "FTA": fta,
            "OREB": oreb,
            "DREB": dreb,
            "REB": oreb + dreb,
            "AST": 2 + k,
            "STL": k % 2,
            "BLK": (k + 1) % 2,
            "TOV": 1 + k % 3,
            "PF": 2,
            "PTS": 2 * fgm + fg3m + ftm,
            "PLUS_MINUS": k - 2,
        }
        lines.append(line)
    # Fix rounding so minutes sum exactly
    lines[-1]["MIN"] = round(total_minutes - sum(x["MIN"] for x in lines[:-1]), 3)
    return lines


def _game(game_id, day: date, home, away, minutes, seed):
    home_lines = _player_lines(home, minutes, seed)
    away_lines = _player_lines(away, minutes, seed + 2)
    abbr = {BOS: "BOS", DET: "DET", OKC: "OKC", SAS: "SAS"}
    team_rows, player_rows = [], []
    for team, opp, lines, is_home in (
        (home, away, home_lines, True),
        (away, home, away_lines, False),
    ):
        matchup = f"{abbr[team]} vs. {abbr[opp]}" if is_home else f"{abbr[team]} @ {abbr[opp]}"
        totals = {c: sum(x[c] for x in lines) for c in STAT_COLS}
        team_rows.append(
            {
                "SEASON_ID": "22025",
                "TEAM_ID": team,
                "TEAM_ABBREVIATION": abbr[team],
                "TEAM_NAME": "",
                "GAME_ID": game_id,
                "GAME_DATE": day.isoformat(),
                "MATCHUP": matchup,
                "WL": "W",
                "MIN": minutes,
                **totals,
                "PLUS_MINUS": 0,
                "VIDEO_AVAILABLE": 1,
            }
        )
        for x in lines:
            player_rows.append(
                {
                    "SEASON_YEAR": "2025-26",
                    "TEAM_ID": team,
                    "TEAM_ABBREVIATION": abbr[team],
                    "TEAM_NAME": "",
                    "GAME_ID": game_id,
                    "GAME_DATE": f"{day.isoformat()}T00:00:00",
                    "MATCHUP": matchup,
                    "WL": "W",
                    **x,
                }
            )
    return team_rows, player_rows


def season_payload():
    """Regular season: one regulation game, one overtime game, one broken game."""
    t1, p1 = _game("0022500001", date(2025, 10, 21), DET, BOS, 240, 0)
    t2, p2 = _game("0022500002", date(2025, 10, 22), SAS, OKC, 265, 1)
    broken_team = [dict(t1[0], GAME_ID="0022500003")]  # only one team row
    return {"teams": t1 + t2 + broken_team, "players": p1 + p2}


def boxscore_rows(game_id="0022500001"):
    """BoxScoreTraditionalV3 PlayerStats rows for game 1: first five are starters, plus a DNP."""
    rows = []
    for team in (DET, BOS):
        for i, (pid, name) in enumerate(ROSTERS[team]):
            first, _, last = name.partition(" ")
            rows.append(
                {
                    "gameId": game_id,
                    "teamId": team,
                    "personId": pid,
                    "firstName": first,
                    "familyName": last,
                    "position": ["F", "F", "C", "G", "G"][i] if i < 5 else "",
                    "comment": "",
                    "minutes": "30:00",
                }
            )
        rows.append(
            {
                "gameId": game_id,
                "teamId": team,
                "personId": 1641999 if team == DET else 1641998,
                "firstName": "Injured",
                "familyName": "Player",
                "position": "",
                "comment": "DNP - Injury/Illness" if team == DET else "DNP - Coach's Decision",
                "minutes": "",
            }
        )
    return rows
