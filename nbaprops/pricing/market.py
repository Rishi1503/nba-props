"""The prop market at any moment, rebuilt from the change-only odds log (Stage 2).

`board` returns every prop quoted at time t: the latest row per prop key at or before t,
minus props withdrawn by then. A main line is one prop per (book, player, market) whatever
its value; alternates and milestones are one prop per line.

`consensus` is the plan's fair-price anchor: at each line DraftKings quotes, the median
de-vigged over probability across the consensus books quoting that same line with both
sides, and only when at least `min_books` do.
"""

import numpy as np
import pandas as pd

from nbaprops.pricing.odds_math import devig_two_way

TARGET_BOOK = "draftkings"
CONSENSUS_BOOKS = ("fanduel", "bovada", "unibet", "pinnacle")
PROP_COLS = ["game_id", "book", "player_id", "market", "line_type"]


def _line_key(odds: pd.DataFrame) -> pd.Series:
    """The line is part of a prop's identity except for main lines (which move)."""
    return odds["line"].astype(float).where(odds["line_type"] != "main", -1.0)


def board(odds: pd.DataFrame, t) -> pd.DataFrame:
    """Props live at time `t`: one timestamp, or a Series of per-row cutoffs aligned with
    `odds` by position (e.g. each row's game time). Withdrawn props are dropped. Alignment is
    positional, so a frame with a repeated index (after a concat) still works."""
    ts = pd.to_datetime(odds["snapshot_ts"], utc=True).dt.tz_convert(None).to_numpy()
    if isinstance(t, pd.Series):
        if len(t) != len(odds):
            raise ValueError("per-row cutoffs must align with the odds rows")
        cut = pd.to_datetime(t, utc=True).dt.tz_convert(None).to_numpy()
    else:
        cut = pd.Timestamp(t).tz_convert("UTC").tz_localize(None).to_datetime64()
    o = odds[(ts <= cut) & odds["player_id"].notna().to_numpy()]
    if o.empty:
        return o.assign(line=pd.Series(dtype=float))
    o = o.assign(_lk=_line_key(o)).sort_values(["snapshot_ts", "id"], kind="stable")
    last = o.drop_duplicates([*PROP_COLS, "_lk"], keep="last")
    withdrawn = last["is_withdrawn"].fillna(False).astype(bool)
    return last[~withdrawn].drop(columns="_lk").assign(line=lambda d: d["line"].astype(float))


def board_per_game(odds: pd.DataFrame, times: dict) -> pd.DataFrame:
    """Each game's board at its own time (e.g. flag time or close)."""
    o = odds[odds["game_id"].isin(times)]
    return board(o, o["game_id"].map(times))


def two_way(quotes: pd.DataFrame) -> pd.DataFrame:
    """Quotes with both sides priced, plus de-vigged probabilities (multiplicative)."""
    q = quotes.dropna(subset=["over_price", "under_price"]).copy()
    if q.empty:
        return q.assign(p_over_novig=pd.Series(dtype=float), p_under_novig=pd.Series(dtype=float))
    p = [
        devig_two_way(int(o), int(u))
        for o, u in zip(q["over_price"], q["under_price"], strict=True)
    ]
    q["p_over_novig"] = [a for a, _ in p]
    q["p_under_novig"] = [b for _, b in p]
    return q


def consensus(quotes: pd.DataFrame, min_books: int = 2) -> pd.DataFrame:
    """Median de-vigged P(over) across consensus books at each (game, player, market, line),
    kept only where at least `min_books` books quote that line on both sides."""
    q = two_way(quotes[quotes["book"].isin(CONSENSUS_BOOKS)])
    cols = ["game_id", "player_id", "market", "line"]
    if q.empty:
        return pd.DataFrame(columns=[*cols, "p_over_cons", "n_books"])
    # one quote per book per line (a book's main and alternate can coincide)
    q = q.drop_duplicates([*cols, "book"], keep="last")
    g = q.groupby(cols)["p_over_novig"].agg(p_over_cons="median", n_books="count").reset_index()
    return g[g["n_books"] >= min_books].reset_index(drop=True)


def fair_close(close_board: pd.DataFrame, picks: pd.DataFrame, min_books: int = 2) -> np.ndarray:
    """Fair closing probability of each pick's side (plan, Stage 9 CLV definition): the
    consensus no-vig close at the pick's line; if there's none, DraftKings' own two-way close
    with the vig removed; NaN when neither exists."""
    cols = ["game_id", "player_id", "market", "line"]
    cons = consensus(close_board, min_books).rename(columns={"p_over_cons": "_c"})
    dk = two_way(close_board[close_board["book"] == TARGET_BOOK])
    dk = dk.drop_duplicates(cols, keep="last")[[*cols, "p_over_novig"]].rename(
        columns={"p_over_novig": "_dk"}
    )
    m = (
        picks[cols + ["side"]]
        .merge(cons[[*cols, "_c"]], on=cols, how="left")
        .merge(dk, on=cols, how="left")
    )
    p_over = m["_c"].fillna(m["_dk"]).to_numpy(float)
    return np.where(m["side"].to_numpy() == "over", p_over, 1 - p_over)
