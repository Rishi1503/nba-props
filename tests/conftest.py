"""Test-wide setup, loaded by pytest before any test module."""

import os
import tempfile
from pathlib import Path

import yaml

SHIPPED_STRATEGY = Path(__file__).resolve().parents[1] / "config" / "strategy.yaml"


def _uncapped_strategy() -> str:
    """The shipped strategy with the sanity cap (max_edge) off. The synthetic slates set prices
    far from the model on purpose, to produce picks, so their edges sit above the cap. The cap
    has its own tests: tests/test_pricing_engine.py, and on the shipped config in
    tests/test_slate.py::test_the_shipped_cap_holds_back_the_synthetic_slates_big_edges."""
    cfg = yaml.safe_load(SHIPPED_STRATEGY.read_text())
    cfg["max_edge"] = 10.0
    path = Path(tempfile.mkdtemp(prefix="strategy-")) / "strategy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


os.environ.setdefault("STRATEGY_CONFIG", _uncapped_strategy())
