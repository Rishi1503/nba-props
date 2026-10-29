#!/usr/bin/env bash
# Build the dashboard and publish it where Caddy serves it, pointed at /api on the same host.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO/web"
npm ci --no-audit --no-fund
npx ng build
DIST="$(dirname "$(find "$REPO/web/dist" -name index.html | head -1)")"
echo '{ "apiUrl": "/api" }' > "$DIST/config.json"
sudo mkdir -p /srv/nbaprops-web
sudo rsync -a --delete "$DIST/" /srv/nbaprops-web/
echo "dashboard published from $DIST"
