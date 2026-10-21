# nba-props

NBA player prop pricing model: data pipeline, game simulation, pricing engine, parlay builder,
backtests, and a single-user website. Built entirely on free resources.

The full design lives in the implementation plan doc. This repo follows its stage numbers.

## Status

| Stage | What | State |
| --- | --- | --- |
| 0 | Repo, schema, config, CI | Done |
| 1 | Historical stats backfill + daily stats | Built and tested; run the backfill from your own machine |
| 2 | Odds + injury logging | Built and tested; needs a PropLine key and a hosted DB to run |
| 2b | Starting lineups | No reliable free source; late changes come from injury reports and pulled props |
| 3 | Features | Built and tested; `make features` once the backfill has run |
| 4 | Minutes model | Built and tested; `make minutes` reports against the plan's targets |
| 5 | Rates, simulation, recalibration | Built and tested; `make stage5-eval`, `make stage5-train`, `make slate` |
| 6 | Backtesting (walk-forward, Layer A, Layer B) | Built and tested; `make backtest-tune`, then `make backtest-holdout` once |
| 7 | Pricing engine, parlays, boosts | Built and tested; `make price DATE=...` writes paper picks |
| 8 | Daily production pipeline | Built and tested; runs unattended once the secrets and a champion model are in place |
| 9 | Paper trading and go-live | Built and tested; needs weeks of real paper picks to say anything |
| 10 | Dashboard (FastAPI + Angular) | Built and tested, including a browser test of every page; `make demo` to click through |

## Setup (one time)

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop, git.

```bash
git clone <your-repo-url> nba-props && cd nba-props
cp .env.example .env          # fill in API keys as you get them
make setup                    # uv sync --extra dev
make db-up                    # Postgres 16 in Docker on localhost:5432
make migrate                  # alembic upgrade head: creates every table
make test                     # full suite (needs the database from make db-up)
```

Stage 0 is done when `make lint` and `make test` pass locally and CI is green on GitHub.

## Stage 1: box scores

`make backfill` loads 2021-22 through 2025-26 (preseason, regular season, play-in, playoffs).
2021-22 is a warm-up season: it only supplies last-season priors for 2022-23.


- Season-wide team and player game logs, plus the player index for positions: a handful of
  requests per season.
- One box-score request per game for starters and did-not-play reasons (about 1,300 per
  season). Expect roughly 30-40 minutes per season at the default one-second pace.
- Possessions, pace, offensive and defensive ratings, and usage are computed in Postgres
  with the Basketball-Reference formulas, so no second per-game request is needed.

Every response is cached in `data/cache/nba/`, so an interrupted backfill picks up where it
stopped; just run it again. `--refresh` re-downloads everything to pick up the NBA's stat
corrections. Daily pulls are never cached. The daily job also retries any box score still
missing from the last 30 days, and refreshes rosters from the NBA player index (positions,
and each player's current team, so live slates place a traded player on his new team). Validation issues (player points not summing to the team's,
minutes off by more than 1.5, games missing a team) are logged at the end of each season.

Run it from your own machine: the NBA stats site often blocks cloud servers. The daily
job (`make stats-day`, yesterday's games) is also attempted from GitHub each morning; if it
fails there, the workflow shows a warning and you run `make stats-day` locally instead.

Preseason games are loaded because preseason minutes feed the new-team priors, but models
must exclude `season_type = 'preseason'` from training targets.

## Stage 3: features

`nbaprops/features/build.py` turns box scores into 116 model inputs per player-game:
recent per-minute form (last 5/10/20 games), season-to-date rates shrunk toward a prior
(last season for returning players, the position average for rookies), minutes and role,
teammate absences, opponent defense overall and by position, rating-based margin and total,
rest and travel, and standings (including clinched and eliminated flags).

How leakage is prevented: every rolling number is stored as the state *after* each game,
and a target game joins only states from strictly earlier dates. Training (`training_frame`)
and live slates (`slate_frame`, `build_features`) use the same join. Tests prove it: adding
an extreme future game changes no earlier row, and changing a game's own result changes none
of that game's inputs.

- Absences: in finished games, a rotation player who didn't play; for upcoming games, a
  rotation player listed Out or Doubtful in the latest injury report before `as_of_ts`.
  Traded players stop counting for their old team.
- Live slates include the recent rotation, established players (for example a starter back
  from a long absence), anyone the injury report lists as able to play, and any extra
  players passed in, such as everyone with a DraftKings line.
- Spreads and totals: features use team-rating estimates, which exist for every season.
  Logged market lines (`market_*` columns) exist only from 2026-27, so they center the
  Stage 5 simulation instead of being training features, avoiding a train/live mismatch.

`make features` writes the training set to `data/features/train.parquet`.

## Stage 4: minutes model

`nbaprops/models/minutes.py` predicts minutes as a distribution for players expected to play
(whether they play comes from the injury report):

- **Normal games:** LightGBM predicts the 10th, 50th, and 90th percentiles; two half-normals
  joined at the median reproduce all three exactly.
- **Early exits** (injury, fouls, ejection): a per-player probability from earlier games only,
  shrunk toward the league rate, adds a lump between 0 and a low cap.
- The quantile models boost from a recency-weighted minutes average and must beat that
  anchor on held-out games by 0.5% and by more than 2 standard errors (paired, per game), or
  all three fall back to it together: tails and center never come from different models.
  Rate models use the same gate per stat. Calibration checks cluster by player, since one
  player's lines share outcomes and his estimation error.
- **Low-confidence rules** (plan, Stage 4) widen the range and flag the row: back from 3+
  missed games, new team, under 5 career games, or a starter on a clinched team late in the
  season. Stage 7 won't flag picks on low-confidence minutes.

`make minutes` trains on 2022-23 and 2023-24 and reports on 2024-25, the tuning season (the
2025-26 holdout stays untouched until Stage 6). The report gives MAE against the plan's
10-game-average baseline (target: at least 8% better, scored on the median, the point
prediction that minimizes absolute error) and 80% interval coverage (target: about 80%),
overall and for starters, bench, back-to-backs, key absences, returns, and new teams.

## Stage 5: rates, game simulation, recalibration

- **Rates** (`models/rates.py`): one LightGBM Poisson model per stat (points, rebounds,
  assists, threes) with a log-minutes offset, boosting from each player's shrunk season rate
  and gated on held-out games like the minutes model. Overdispersion is fitted per usage tier,
  along with two shared pieces of variance: teammates' common scoring environment and one
  player's linked points and assists.
- **Simulation** (`models/simulate.py`): each run draws a game total and margin (centered on
  the market when logged, else team ratings), ties starters' and bench minutes to how
  lopsided it is (a Gaussian copula that keeps each player's minutes distribution exact),
  scales scoring with the scenario, then draws points, rebounds and assists. Threes are drawn
  within points, so lines are always consistent; PRA is the sum. The shared pieces only add
  correlation: each player's total variance still matches the data. Seeded per run and game.
- **Market lines move projections**: the rate models' expectations rest on team ratings, so
  when the market expects a team to score more, its players shift up (capped at 15%).
- **Recalibration** (`models/calibrate.py`): isotonic per market, kept only if it beats raw
  probabilities on later data than it was fit on; otherwise that market passes through.

- **Questionable players**: their own props void if they sit, so they're projected as
  playing. Their teammates' props don't void, so teammates of each team's most important
  Questionable player are simulated both ways and mixed by `p_play_questionable` (a neutral
  0.5 until `python -m nbaprops.pipeline.run play-rates` measures it from logged reports),
  and flagged low-confidence.
- **Trades and roster changes**: a "stint" is a run of games for one team, resetting on any
  team change; minutes, usage and per-minute rates since joining (`stint_*`) sit beside the
  blended recent form, and the minutes anchor blends toward new-team minutes after a trade.
  Teammates see minutes and usage that left by trade (`departed_*`), arrived by trade
  (`arrived_*`, the new player's role before the trade) and returned from a 3+ game absence
  (`returning_*`, his role before it), each fading as recent form absorbs the change.
  Participation is checked per team, so a player traded to the opponent who faces his old
  team the same week is a departure, not a no-show.
- **Opponent absences**: the other team's missing players, minutes, usage and starters, and
  its absent minutes at the player's own position.
- **With/without splits**: for today's most important absent teammate, how this player's
  minutes, usage and per-minute rates differed in earlier games without him (same team),
  scaled by n / (n + 5) and requiring 3+ games together.
- **History has no trade dates**: a player who sits a game and never plays for that team
  again *this season* counts as departed for that game (the one deliberate look-ahead);
  live, today's roster says it directly.
- **Playoffs**: an `is_postseason` input, with the regular-season rest and tanking signals
  switched off in the playoffs and play-in, where they mean the opposite.
- **Whole-number lines**: `simulate.line_probs` gives over, under and push for any line from
  the stored half-point grid (over 25 is over 25.5; push is P(over 24.5) - P(over 25.5)).

Backtests know who actually played; live slates only have the injury report. Expect live
results to be slightly worse than backtests, which is why paper trading comes first.

`make stage5-eval` trains on 2022-23 and 2023-24, simulates 2024-25, fits recalibration on
its first half and judges the second half, against the plan's baselines: MAE of the simulated
median vs the season-average, and Brier and calibration vs a normal-distribution baseline. A
calibration decile fails only if it is off by more than 3 points and by more than 3 standard
errors (about 50 deciles are checked at once). `make stage5-train` saves a bundle;
`make slate BUNDLE=...` projects today's games into `projections` under the bundle's run id.

## Stage 6: backtesting

`python -m nbaprops.backtest.run --season 2025-26` (or the Make targets below) replays a season
the way production runs: every week, every model is retrained on games before that week and
the week's games are simulated. Nothing in a week can see that week or later: features are
leakage-proof, models train on earlier dates, early-exit history only counts earlier games,
and the calibrator is fit only on earlier out-of-sample predictions. A test rewrites every
outcome in the last week and requires every earlier prediction to be bit-for-bit identical.
The previous season runs first as a calibration warm-up. Trained models are cached under
`artifacts/backtest_cache/`, so reruns and comparisons don't retrain what hasn't changed.

The report (`reports/backtest/<run>.json` and `.md`) covers, per market:

- **Accuracy**: MAE and RMSE vs the baseline.
- **Probabilities**: Brier score, log loss (vs the baseline on the same lines), ECE.
- **Calibration**: deciles, with failures judged by player-clustered errors.
- **Distribution shape**: a PIT histogram with a verdict (flat, too narrow, too wide, or
  shifted, which call for different fixes).
- **Breakdowns**: by month, low-confidence flag, roster changes, and postseason.
- **Targets**: the Stage 5 targets, checked out of sample.

| Command | What |
| --- | --- |
| `make backtest-tune` | 2024-25, plus the same run without the roster inputs (paired comparison). Run as often as you like |
| `make backtest-holdout` | 2025-26, **once**, after tuning is finished. If it passes cleanly, the same run retrains on every season and saves a production bundle under `artifacts/models/`, calibrated on the walk-forward's out-of-sample predictions |
| `make backtest-layer-b SEASON=2026-27` | Betting replay on a season with logged odds |

**The holdout is used once.** Its first evaluation is recorded in `reports/holdout_ledger.json`
(commit it). A second run is refused unless you pass `--rerun-holdout "<reason>"`, and that
report is marked NOT CLEAN; production retraining needs a clean, passing holdout.

**Layer B** prices DraftKings' exact lines (pushes on whole numbers included) with the
walk-forward's probabilities. It applies the Stage 7 rules from `nbaprops/pricing/engine.py`,
which live pricing uses too, replaying the daily schedule (10:00 AM, 3:00 PM and 6:00 PM New
York) with first flag wins. Picks are graded from box
scores (players who didn't play are void), and CLV is measured against the fair close: the
consensus no-vig close, else DraftKings' own. Results are split by market, EV bucket,
consensus vs model-only, and high-confidence mode, with date-clustered standard errors and a
warning below 500 bets. The blend check chooses the model weight on the first half of the
dates and reports it on the second, so no number is tuned on the data it describes.

Expect each season to take a while on real data (weekly retraining of every model); measure
the first run before planning around it. `--retrain-days 14` halves the work.

## Stage 7: pricing

`make price DATE=2026-10-20` (or `python -m nbaprops.pricing.run --date ...`) prices the slate
with the active model run's projections and the latest odds, prints the ranked board, and
writes paper picks and parlays. `--dry-run` writes nothing; `--boost 0.30:25` adds a parlay
profit boost, `--boost-single 0.25:50` a boost on one straight bet.

| Piece | Rule |
| --- | --- |
| Fair price | 0.3 x model + 0.7 x consensus (median de-vigged FanDuel/Bovada/Unibet at DraftKings' line, 2+ books; if the books sit up to 1.5 points away, their line is read across with the model's shape); with no consensus, the model pulled halfway to DraftKings' no-vig price |
| Flag | EV >= 3% (5% model-only), model and consensus both see value on that side, minutes not low-confidence, line inside the model's simulated range |
| Questionable teammates | Projected as a blend of "he plays" / "he sits", with both scenarios stored. `questionable_teammates: robust` flags only bets that are +EV either way (`mixture`: the blend alone; `block`: never) |
| One per player | The prop with the highest expected growth at its Kelly stake (not the highest EV, which favors longshots across alternate lines) |
| Stake | Quarter Kelly (push-aware), 2% per bet, 4% per game, 10% per day across singles and parlays, counting earlier runs that day |
| Staked tiers | High-confidence picks (model >= 70%). Standard flags are logged at zero stake (`stake_standard_flags: false`) |
| Milestones | The model distribution is shifted to match the consensus main line's no-vig probability, then read at the milestone |
| Satellites | Up to 2 per slate: 2 flagged legs from different games, quarter Kelly capped at 1%, best expected growth |
| Boosts | Parlay boosts: best 3-4 flagged legs from different games, up to the boost's max stake. Single boosts: every prop repriced at the boosted odds (a boost can create the edge) |
| Same-game parlays | Joint probability from the simulation (dependence preserved), DraftKings' typed-in SGP price, 3 points of EV off |
| Pushes | Whole-number lines: a pushed leg drops out of a parlay and the rest pay; EV and Kelly count the push |

Safety rules for running three times a day plus after late news:

- **One pick per player per game per mode.** The first flag wins, so reruns never duplicate
  or overwrite a pick.
- **Tipped games are never priced.**
- **Stale boards are skipped.** A game is not priced when DraftKings hasn't answered a poll in
  `max_odds_age_min` (120). A consensus book that stopped answering is dropped from the
  consensus.
- **Injury news blocks stale projections.** A game whose injury statuses changed after its
  projections were made is not priced until it's re-projected.
- **Volume is visible.** The board reports how many props were blocked only by low
  confidence, and how many only by the Questionable-teammate rule, so the cost of each rule
  shows up in paper trading.

## Stage 8: the daily pipeline

GitHub Actions runs `python -m nbaprops.ops.run cycle` every 15 minutes. Each cycle decides
what to do from the time in New York:

| When | What |
| --- | --- |
| Every cycle | Odds and injuries first (closing lines before anything else), then settle whatever became final |
| 10:00 AM, 3:00 PM, 6:00 PM | Project the slate with the champion (challengers in shadow), price it, alert the new picks |
| Any cycle, on late news | Injury status changed since the projection: re-project and re-price that game. DraftKings pulled a player's props: re-price and alert a possible late scratch |
| 9:05 AM daily | Rosters, schedule, stats (try), settlement |
| Mondays 9:30 AM | Retrain through yesterday as a challenger (`retrain.yml`); it never prices picks until promoted |

Health checks alert once each: stale DraftKings odds on a game day, last night's box scores
missing after 9:00 AM, no champion model. A failing step alerts with a link to the run's log
and the rest of the cycle still runs; a crash before Python starts is caught by the
workflow's own alert step.

**Settlement** grades picks once the game is final and its box scores are loaded: result,
P&L at the price taken (your placed price for real bets), DraftKings' closing price, and CLV
(fair close minus the implied probability you took). Parlays settle when every leg has; a
pushed or void leg drops out. Real-money P&L updates the bankroll ledger; paper never does.

**Before it can run unattended:**

1. **Repository secrets:** `DATABASE_URL`, `PROPLINE_API_KEY`, and for alerts either
   `DISCORD_WEBHOOK_URL` or `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`.
2. **The production model:** after `make backtest-holdout` passes,
   `make import-champion BUNDLE=artifacts/models/prod-...` stores it in the database. Models
   live there because scheduled runners start empty; `promote RUN_ID` switches models, and
   rolling back is the same command.
3. **The bankroll:** a `bankroll_ledger` row with the season's bankroll.
4. **Box scores:** `make stats-day` each morning on your machine, if GitHub's attempt is
   blocked.

`make status` shows the champion, challengers and today's picks.

## Stage 9: paper trading and go-live

Every flagged pick is logged as a paper pick and settled with CLV (Stage 8). On top of that:

| Command | What |
| --- | --- |
| `make review` (Sunday night, `stage9.yml`) | CLV, ROI and hit rate vs breakeven, by stat, edge size, flag time, over/under, tier and consensus vs model-only; predicted EV vs realized CLV per edge bucket; parlays by tier; slippage on real bets. Switches off any market with negative season CLV after 100+ picks, and reports the go-live status |
| `make golive` | The checklist below, checked by code |
| `make tune` | Retunes the blend weight and EV floor on paper data only: replays every logged pricing run with the engine's own rules, chooses on the earlier 60% of dates and reports the choice and the current config on the later 40%. A report: it never changes the config |
| `make recalibrate` (1st of the month) | Fits a live calibration layer on recent results, per market, kept only if it beats no adjustment on later games |
| `python -m nbaprops.ops.run place PICK --price -110 --stake 20` | Records a real bet (the paper pick stays) |
| `... market reb --off "reason"` / `--on` | Switches a market off or back on |
| `... note-fix "what"` | Records a manual fix (restarts the clean-pipeline streak) |

Go-live checklist (real money only when every box is checked):

- [ ] **Volume:** 300+ settled paper picks, over 4+ weeks.
- [ ] **CLV:** average CLV positive, and positive in 3 of the last 4 weeks.
- [ ] **Calibration:** ECE under 3 points on live picks, with no bucket (20+ picks) off by more than 8.
- [ ] **Pipeline:** 14 straight days with every cycle succeeding and no manual fix recorded.
- [ ] **Bankroll:** recorded, and fixed since (no deposits or withdrawals).

Every pricing run also logs every prop it priced, flagged or not (`candidate_log`), which is
what makes tuning honest: a lower threshold or a different weight can only be judged on
props the current settings rejected. After go-live, real stakes are half the engine's for
the first 200 real bets, then full if real-money CLV holds (`ramp_bets`, `ramp_factor`); pick
alerts show the amount to bet.

## Stage 10: the dashboard

A phone-first website: an Angular 21 front end (`web/`) over a FastAPI back end
(`nbaprops/api/`) that reads Postgres directly.

| Page | What |
| --- | --- |
| Board | Picks by EV: DraftKings vs fair price, stake, badges for high-confidence, model-only, injury news since the flag, and line moves. One tap marks a pick placed. Parlay panel, boost entry, manual milestone and same-game parlay prices |
| Player | The simulated distribution with DraftKings' line marked, the last 10 games, the day's line moves |
| Record | Every day's picks and parlays with each leg marked hit or miss; success rates by week, month and year; and "if you'd tailed every pick": a flat stake on every pick (optionally parlays and tracked flags) since day one, with the running total |
| Bets | Placed bets with running P&L, real or paper; the yearly CSV for tax records |
| Results | CLV by day, the calibration plot, ROI and CLV by stat and edge size, filterable |
| Models | Two backtest reports side by side |
| Status | Go-live checklist, weekly CLV, market switches, bankroll, strategy settings; log out |

**Run it locally:**

```
python -m nbaprops.api.auth              # prompts for a password; prints the two secrets
export DASHBOARD_PASSWORD_HASH=... DASHBOARD_SECRET=... DASHBOARD_ORIGINS=http://localhost:4200
make demo                                # optional: fill an EMPTY database with demo data
make api                                 # http://localhost:8000 (docs at /docs)
make web                                 # http://localhost:4200
```

**Security.** One password (scrypt-hashed in an environment variable, never stored in
plain text) gets a 30-day signed token. An address is locked out for 15 minutes after 5
wrong passwords. CORS allows only `DASHBOARD_ORIGINS`. `/health` is the only open route
besides login.

**Where the API lives** is set at deploy time in `web/public/config.json` (`apiUrl`), so one
build works anywhere.

**Tests.**

- `pytest` covers every endpoint against the test database, plus a run of every migration
  down and up with data present.
- `cd web && npx ng test` runs the front end's unit tests.
- `E2E_DATABASE_URL=<disposable db> make e2e` walks every page in headless Chromium on a
  phone-sized screen and fails on any console error.

`web/.npmrc` sets `legacy-peer-deps`: npm 10.9 crashes resolving this tree's peers, and the
peer versions are pinned compatibly.

## Free-tier limits

Everything runs on free tiers, so two limits shape the design. Check the providers' current
terms when you set up; they change.

- **Database storage (about 0.5 GB on free Postgres tiers).** Odds are logged change-only,
  closing ticks included (writing every quote on every closing tick would be about 0.6 GB a
  season by itself), and the candidate log keeps only props near the flag threshold.
  `db_size_limit_mb` (500) drives a health alert at 80%, well before writes fail.
- **GitHub Actions minutes.** Private repositories get 2,000 free minutes a month, and every
  job bills at least one minute. The production schedule runs only when needed: 45 runs a
  day, about 1,400 minutes a month before the pricing runs' extra time, so expect roughly
  1,700-2,000 a month in season. A test checks that every pricing run and every closing
  window from noon to 11 PM is still covered. If minutes run short, make the repository
  public (unlimited minutes; secrets stay secret) or run the cycle from a free VM's cron.

## Stage 2: odds and injury logging

What runs, every 15 minutes during game hours (`tick`):

| When (New York time) | What |
| --- | --- |
| 10:00, 3:00 PM, 6:00 PM | Full odds snapshot for every game tipping within 36 hours, plus the injury report |
| Last 35 min before each tip | A closing snapshot on every tick (grading uses the latest), plus the injury report |
| Daily, ~9 AM | Refresh rosters (`seed`) and the schedule (`schedule`) |

Each odds call covers one game, all five books (DraftKings, FanDuel, Bovada, Unibet,
Pinnacle) and all tracked markets. A full slate costs roughly 100-150 of the free tier's
1,000 daily requests (three full snapshots, two or three closing snapshots per game, and the
event listings). The last 100 requests are reserved for closing snapshots.

Rows are written only when a line or price changes. When a book that's in a snapshot stops
quoting a prop it quoted before (pulled, suspended, or an alternate line removed), a
withdrawal row (`is_withdrawn`, prices NULL) records it, so a replay never treats a pulled
prop's last price as still available. A book missing from a whole response is not read as
withdrawing everything. `graded_props` takes one close per main line (the last before tip,
whatever its value) and drops props withdrawn at the close. Raw responses are saved to
`data/raw/` (uploaded as a 30-day artifact on GitHub) so anything can be re-parsed later.

### Turning it on

1. Get a free PropLine key at prop-line.com (email only, no card).
2. Create a free Postgres database (Neon or Supabase) and copy its connection string.
   Use the `postgresql+psycopg://` prefix.
3. In the GitHub repo: Settings, then Secrets and variables, then Actions. Add
   `DATABASE_URL` and `PROPLINE_API_KEY`.
4. Actions tab, "Collect odds and injuries", Run workflow with `seed`, then `schedule`,
   then `snapshot`. After that the schedule runs it automatically.

GitHub bills each scheduled run as at least one minute. This schedule uses about 1,700
of a private repo's 2,000 free monthly minutes; a public repo has no limit, or the same
`make tick` can run from cron on your own machine instead.

### Check on the first live run

These are built from published docs and tested against fixtures in the same shape.
Verify each once with real data:

- **Schedule feed fields** (`pipeline/schedule.py`): game ids, UTC tip times, team ids.
- **Milestone rungs** (`parse_props`): that DraftKings "20+" arrives as point 20 and is
  stored as line 19.5. If PropLine already sends 19.5, nothing changes.
- **Injury report file names** (`candidate_urls`): quarter-hour names are tried first,
  then the hourly name; check which one the NBA is using this season.
- **Unmatched players**: every run logs the match rate (target 98%). Names that matched no
  one go to a review queue: `python -m nbaprops.pipeline.run aliases` lists them with
  suggested players; `alias-add "Name" PLAYER_ID` fixes them, including past rows.
- **Pulled props**: when a book takes a player's props off the board (often a late scratch),
  the stale prices are skipped and the player is logged as a possible scratch.

There is no reliable free source of confirmed starting lineups (teams don't have to submit
them before tip), so late changes are caught by injury-report updates and pulled props.

Graded results: the `graded_props` view grades every prop's latest closing line against
the box score (over, under, push, or void if the player didn't play).

## Layout

```
config/strategy.yaml     every tunable threshold (one source of truth)
nbaprops/
  settings.py            env settings (.env): DB URL, API keys, timezone
  config.py              validated loader for strategy.yaml
  db/                    SQLAlchemy models + engine/session helpers
  pipeline/              Stages 1-2: stats, odds, injuries, lineups
  features/              Stage 3: build_features(game_date, as_of_ts)
  models/                Stages 4-5: minutes, rates, simulation, calibration
  pricing/               Stage 7: odds_math (done), engine, parlays
  backtest/              Stage 6
  api/                   Stage 10: FastAPI
migrations/              Alembic; 0001 = initial schema
web/                     Stage 10: Angular
infra/                   Stages 8, 11
```

## Conventions

- Thresholds change in `config/strategy.yaml`, never in code.
- Schema changes: edit `nbaprops/db/models.py`, then
  `uv run alembic revision --autogenerate -m "..."`, review the file, `make migrate`.
  CI runs `alembic check` and fails if models and migrations disagree.
- Prices are stored as American odds; all math converts to decimal via `pricing/odds_math.py`.
- Models declare no ORM relationships, so insert parent rows (teams, players, games)
  before children. The pipelines upsert in that order.
- Keep the GitHub repo private. Private repos get a monthly allowance of free Actions minutes,
  which is plenty for CI plus the Stage 8 schedules.
