"""Weekly retrain (plan, Stage 8: Mondays, data through yesterday).

Refits the minutes, rate and scenario models on every training season through yesterday,
with the champion's own feature lists, and keeps the champion's calibrator (it was fit on
out-of-sample predictions; Stage 9 refits it on live results). The new bundle is stored as
a challenger: it projects in shadow and is promoted only by the champion-challenger check
(Stage 11), never automatically.
"""

from datetime import UTC, date, datetime

from nbaprops.features.build import build_training_set, prev_season
from nbaprops.models import minutes as mm
from nbaprops.models import rates as rm
from nbaprops.models import simulate
from nbaprops.models.project import Bundle, git_sha
from nbaprops.ops import bundles
from nbaprops.pipeline.stats import season_for_date

FIRST_TRAIN_SEASON = "2022-23"


def seasons_through(season: str) -> list[str]:
    out = [season]
    while out[-1] > "2021-22":
        out.append(prev_season(out[-1]))
    return sorted(out)


def retrain(session, today: date, first_train_season: str = FIRST_TRAIN_SEASON) -> str | None:
    """Returns the challenger's run_id, or None without a champion to start from."""
    champ_id = bundles.champion(session)
    if champ_id is None:
        return None
    champ = bundles.load(session, champ_id)
    df = build_training_set(session, seasons_through(season_for_date(today)))
    train = df[
        (df["season"] >= first_train_season)
        & (df["season_type"] != "preseason")
        & df["y_minutes"].notna()
        & (df["game_date"] < str(today))
    ]
    minutes_model = mm.fit(train, features=champ.minutes.features)
    rate_model = rm.fit(train, features=champ.rates.features)
    scenario = simulate.fit_scenario(train, minutes_model, df)
    run_id = f"wk-{today:%Y%m%d}"
    meta = {
        "trained_through": minutes_model.trained_through,
        "train_seasons": sorted(train["season"].unique().tolist()),
        "calibration": f"carried over from {champ_id}",
        "parent": champ_id,
        "created": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
    }
    bundles.save(
        session, Bundle(run_id, minutes_model, rate_model, scenario, champ.calibrator, meta)
    )
    return run_id
