"""Strategy config: every tunable threshold, loaded from config/strategy.yaml and validated."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Market = Literal["pts", "reb", "ast", "fg3m", "pra"]
Fraction = Field(ge=0.0, le=1.0)


class StrategyConfig(BaseModel):
    model_weight: float = Fraction
    min_consensus_books: int = Field(ge=1)
    ev_min: float = Fraction
    ev_min_model_only: float = Fraction
    model_only_shrink: float = Fraction
    hc_prob_min: float = Fraction
    max_edge: float = Field(gt=0.0)
    milestone_p_min: float = Fraction
    milestone_p_max: float = Fraction
    kelly_fraction: float = Field(gt=0.0, le=1.0)
    max_bet_pct: float = Fraction
    max_parlay_pct: float = Fraction
    max_satellites_per_slate: int = Field(ge=0)
    max_game_pct: float = Fraction
    max_daily_pct: float = Fraction
    sgp_ev_haircut: float = Fraction
    n_sims: int = Field(ge=1000)
    new_team_low_conf_games: int = Field(ge=0)
    lineup_poll_start_min: int = Field(ge=0)
    p_play_questionable: float = Fraction
    max_odds_age_min: int = Field(gt=0)
    stake_standard_flags: bool
    questionable_teammates: Literal["robust", "mixture", "block"]
    ramp_bets: int = Field(ge=0)
    ramp_factor: float = Fraction
    candidate_log_min_ev: float = Field(le=0)
    db_size_limit_mb: int = Field(gt=0)
    markets: list[Market]
    # Alt-line parlays (a paper experiment): 2-3 DraftKings "N+" rungs from different games
    # near +100, on the stats whose 55-95% probabilities passed the 2024-25 tail check.
    alt_combo_markets: list[Market] = ["reb", "fg3m", "ast"]
    alt_combo_shade: dict[Market, float] = {}  # extra shading for alt-parlay legs only
    # added to the model's chance of the over, per market, for every pick (see strategy.yaml)
    over_shift: dict[Market, float] = {"ast": -0.012}
    alt_combo_p_min: float = Field(0.65, ge=0.0, le=1.0)
    alt_combo_decimal_min: float = Field(1.8, gt=1.0)
    alt_combo_decimal_max: float = Field(2.6, gt=1.0)
    alt_combo_max_legs: int = Field(3, ge=2, le=4)
    max_alt_combos_per_slate: int = Field(2, ge=0)
    # alt-rung correction for points/PRA favored overs (make backtest-rungs writes it only
    # when its out-of-sample check passes); relative paths are from the repository root
    rung_calibration: str | None = "config/rung_calibration.json"

    @model_validator(mode="after")
    def _caps_are_ordered(self) -> "StrategyConfig":
        if not self.max_bet_pct <= self.max_game_pct <= self.max_daily_pct:
            raise ValueError("caps must satisfy max_bet_pct <= max_game_pct <= max_daily_pct")
        if self.ev_min_model_only < self.ev_min:
            raise ValueError("ev_min_model_only should not be lower than ev_min")
        if self.max_edge <= self.ev_min_model_only:
            raise ValueError("max_edge must be above the EV floors")
        if not self.milestone_p_min < self.milestone_p_max:
            raise ValueError("milestone_p_min must be below milestone_p_max")
        if not self.alt_combo_decimal_min < self.alt_combo_decimal_max:
            raise ValueError("alt_combo_decimal_min must be below alt_combo_decimal_max")
        return self


def load_strategy(path: str | Path = "config/strategy.yaml") -> StrategyConfig:
    with open(path) as f:
        return StrategyConfig.model_validate(yaml.safe_load(f))
