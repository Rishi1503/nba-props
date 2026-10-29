#!/usr/bin/env bash
# One-time server setup (safe to rerun). On the server, from the repository folder:
#   bash infra/oracle/bootstrap.sh
# Needs the .env from your Mac first (see infra/oracle/README.md, step 4).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
[ -f .env ] || { echo "No .env: copy it from your Mac first (README step 4)."; exit 1; }
for k in DATABASE_URL PROPLINE_API_KEY DASHBOARD_PASSWORD_HASH DASHBOARD_SECRET; do
  grep -q "^$k=" .env || { echo ".env is missing $k (README step 4)"; exit 1; }
done

echo "== packages"
sudo apt-get update -q
sudo apt-get install -y -q git curl rsync build-essential gnupg debian-keyring \
  debian-archive-keyring apt-transport-https
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -q && sudo apt-get install -y -q caddy
fi
if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y -q nodejs
fi
command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== python environment and database"
uv sync --extra api
uv run alembic upgrade head

echo "== dashboard"
bash infra/oracle/build-web.sh
sudo cp infra/oracle/Caddyfile /etc/caddy/Caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy

echo "== scheduled jobs"
for f in infra/oracle/systemd/*; do
  sed "s#@REPO@#$REPO#g; s#@USER@#$USER#g; s#@HOME@#$HOME#g" "$f" \
    | sudo tee "/etc/systemd/system/$(basename "$f")" >/dev/null
done
chmod +x infra/oracle/*.sh
sudo systemctl daemon-reload
sudo systemctl enable --now nbaprops-api.service nbaprops-cycle.timer \
  nbaprops-daily.timer nbaprops-deploy.timer

echo "== private HTTPS for your devices (Tailscale)"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale status >/dev/null 2>&1 || sudo tailscale up
sudo tailscale serve --bg http://127.0.0.1:8080

echo
echo "Done. Next timers:"
systemctl list-timers 'nbaprops-*' --no-pager
echo
echo "Dashboard (on a device signed in to your Tailscale):"
sudo tailscale serve status | grep -o 'https://[^ ]*' | head -1
