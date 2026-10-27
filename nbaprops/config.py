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
        return self


def load_strategy(path: str | Path = "config/strategy.yaml") -> StrategyConfig:
    with open(path) as f:
        return StrategyConfig.model_validate(yaml.safe_load(f))
