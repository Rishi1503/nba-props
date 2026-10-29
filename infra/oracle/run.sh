#!/usr/bin/env bash
# One entry point for every scheduled job on the server (systemd runs it; see systemd/).
#   run.sh cycle    odds, settlement, projections and pricing when due (every 15 min evenings)
#   run.sh daily    the 9:05 AM routine: players, schedule, stats, settlement
#   run.sh deploy   pull new commits from GitHub and apply them (every 10 minutes)
#   run.sh alert "message"   send a message to Discord or Telegram (failed jobs use this)
# A lock keeps two jobs from ever running at once (GitHub's concurrency group, here).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
set +u; set -a; source .env; set +a; set -u  # .env values may contain "$"
CMD="${1:-cycle}"

alert() {
  local msg="nba-props server: $1"
  local body
  body=$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1]}))' "$msg")
  if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    curl -s -X POST -H "Content-Type: application/json" -d "$body" "$DISCORD_WEBHOOK_URL" >/dev/null
  elif [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
    curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
      -d chat_id="$TELEGRAM_CHAT_ID" --data-urlencode text="$msg" >/dev/null
  fi
  echo "$msg"
}

if [ "$CMD" = "alert" ]; then
  alert "${2:-a job failed}"
  exit 0
fi

exec 9>/tmp/nbaprops.lock
if ! flock -w 600 9; then
  echo "another job held the lock for 10 minutes; skipping $CMD"
  exit 1
fi

case "$CMD" in
  cycle)
    timeout 25m uv run python -m nbaprops.ops.run cycle
    ;;
  daily)
    status=0
    uv run python -m nbaprops.pipeline.run seed || status=1
    uv run python -m nbaprops.pipeline.run schedule || status=1
    # The NBA stats site often blocks cloud servers; the Mac job (infra/mac) covers it then.
    timeout 20m uv run python -m nbaprops.pipeline.run stats-day \
      || echo "stats-day failed on the server (the Mac job covers it)"
    uv run python -m nbaprops.ops.run settle || status=1
    (exit $status)  # the case's status, so the page still publishes below
    ;;
  deploy)
    git fetch -q origin main || exit 1
    [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && exit 0
    changed=$(git diff --name-only HEAD origin/main)
    if ! git merge --ff-only -q origin/main; then
      alert "deploy: can't fast-forward to GitHub's main (changes made on the server?)"
      exit 1
    fi
    uv sync -q --extra api && uv run alembic upgrade head || exit 1
    if echo "$changed" | grep -q '^web/'; then bash infra/oracle/build-web.sh || exit 1; fi
    sudo systemctl restart nbaprops-api
    echo "deployed $(git rev-parse --short HEAD)"
    ;;
  *)
    echo "unknown command: $CMD"; exit 2
    ;;
esac
rc=$?
# the bankroll page, after the jobs that change it (skipped without Cloudflare settings)
if [ "$CMD" != "deploy" ] && [ -n "${CLOUDFLARE_API_TOKEN:-}" ]; then
  bash infra/pages/publish.sh || echo "bankroll page not published"
fi
exit $rc
