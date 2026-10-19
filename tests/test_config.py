import pytest
import yaml
from pydantic import ValidationError

from nbaprops.config import StrategyConfig, load_strategy


def test_repo_config_loads():
    cfg = load_strategy("config/strategy.yaml")
    assert cfg.kelly_fraction == 0.25
    assert cfg.max_bet_pct <= cfg.max_game_pct <= cfg.max_daily_pct
    assert "pra" in cfg.markets


def _raw():
    with open("config/strategy.yaml") as f:
        return yaml.safe_load(f)


def test_rejects_out_of_order_caps():
    raw = _raw() | {"max_game_pct": 0.01}
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(raw)


def test_rejects_unknown_market():
    raw = _raw() | {"markets": ["pts", "stl"]}
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(raw)


def test_rejects_fraction_above_one():
    raw = _raw() | {"model_weight": 1.5}
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(raw)
