.PHONY: setup db-up db-down migrate test lint fmt seed schedule snapshot injuries tick backfill stats-day features minutes stage5-eval stage5-train slate backtest-tune backtest-holdout backtest-layer-b price cycle settle retrain status import-champion production-from-holdout holdout-rerun bankroll review golive tune recalibrate api web demo e2e

setup:
	uv sync --all-extras --locked

db-up:
	docker compose up -d db

db-down:
	docker compose down

migrate:
	uv run alembic upgrade head

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff check --fix . && uv run ruff format .

# --- Stage 2 pipeline ---
seed:
	uv run python -m nbaprops.pipeline.run seed

schedule:
	uv run python -m nbaprops.pipeline.run schedule

snapshot:
	uv run python -m nbaprops.pipeline.run snapshot

injuries:
	uv run python -m nbaprops.pipeline.run injuries

tick:
	uv run python -m nbaprops.pipeline.run tick

# --- Stage 1 stats (run from your own machine: the stats site blocks many cloud IPs) ---
backfill:
	uv run python -m nbaprops.pipeline.run backfill

stats-day:
	uv run python -m nbaprops.pipeline.run stats-day

# --- Stage 3 ---
features:
	uv run python -m nbaprops.pipeline.run features

# --- Stage 4 (train on 2022-23 + 2023-24, judge on the 2024-25 tuning season) ---
minutes:
	uv run python -m nbaprops.models.minutes

# --- Stage 5 (train 2022-23 + 2023-24; judge or calibrate on the 2024-25 tuning season) ---
stage5-eval:
	uv run python -m nbaprops.models.project evaluate

stage5-train:
	uv run python -m nbaprops.models.project train

# Project today's slate with a saved bundle: make slate BUNDLE=artifacts/stage5/stage5-...
slate:
	uv run python -m nbaprops.models.project slate --bundle $(BUNDLE)

# Stage 6. Tune on 2024-25 (as often as you like), with the roster-input comparison.
backtest-tune:
	uv run python -m nbaprops.backtest.run --season 2024-25 --ablate roster

# The 2025-26 holdout: ONCE, after tuning is finished; a second run is refused (README).
# If it passes, the same run retrains on every season and saves the production bundle, so
# the holdout is looked at exactly once.
backtest-holdout:
	uv run python -m nbaprops.backtest.run --season 2025-26 --production

# Betting replay on a season with logged odds (from 2026-27 on).
backtest-layer-b:
	uv run python -m nbaprops.backtest.run --season $(SEASON) --layer-b

# Stage 7: price today's slate (paper picks). DATE=2026-10-20 [BOOST=0.30:25]
price:
	uv run python -m nbaprops.pricing.run --date $(DATE) $(if $(BOOST),--boost $(BOOST),)

# Stage 8: the production cycle and its pieces (the scheduler runs `cycle` every 15 minutes)
cycle:
	uv run python -m nbaprops.ops.run cycle

settle:
	uv run python -m nbaprops.ops.run settle

retrain:
	uv run python -m nbaprops.ops.run retrain

status:
	uv run python -m nbaprops.ops.run status

# Store the production bundle from `make backtest-holdout` as the champion: BUNDLE=artifacts/models/prod-...
# Record the bankroll as of today's New York date: make bankroll AMOUNT=1000
bankroll:
	uv run python -m nbaprops.ops.run bankroll $(AMOUNT)

import-champion:
	uv run python -m nbaprops.ops.run import-bundle $(BUNDLE) --champion

# Re-evaluate the holdout after a model fix (recorded as NOT clean) and build production
# from it: make holdout-rerun REASON="what changed" [WITHOUT=roster]
holdout-rerun:
	uv run python -m nbaprops.backtest.run --season 2025-26 --production \
		--rerun-holdout "$(REASON)" --accept-rerun "$(REASON)" $(if $(WITHOUT),--without $(WITHOUT),)

# Build the production bundle from the recorded holdout run, without re-evaluating it:
# make production-from-holdout RUN=bt-2025-26-...
production-from-holdout:
	uv run python -m nbaprops.backtest.run --season 2025-26 --from-holdout $(RUN)

# Stage 9: paper trading and go-live
review:
	uv run python -m nbaprops.ops.run review

golive:
	uv run python -m nbaprops.ops.run golive

tune:
	uv run python -m nbaprops.ops.run tune

recalibrate:
	uv run python -m nbaprops.ops.run recalibrate

# Stage 10: the dashboard
api:
	uv run --extra api uvicorn nbaprops.api.main:app --reload --port 8000

web:
	cd web && npm install && npx ng serve

demo:
	uv run python -m nbaprops.api.demo

# Browser smoke test of every page (disposable database!): E2E_DATABASE_URL=... make e2e
e2e:
	cd web && npx ng build && cd .. && uv run python web/e2e/smoke.py
