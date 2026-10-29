#!/usr/bin/env bash
# Build the bankroll page and publish it to Cloudflare Pages, only when its data changed
# (new picks, results or verdict), which keeps uploads well under the free plan's limit.
# Needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID; the project is created on first use.
set -euo pipefail
cd "$(dirname "$0")/../.."
P="${CF_PAGES_PROJECT:-nba-props-paper}"
API="https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/pages/projects/$P"
WRANGLER="npx --yes wrangler@3"

uv run python -m nbaprops.site.build --out site/dist
new="$(cat site/dist/hash.txt)"

subdomain() {
  curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" "$API" \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print((r.get("result") or {}).get("subdomain",""))'
}
SUB="$(subdomain)"
if [ -z "$SUB" ]; then
  echo "creating the Cloudflare Pages project $P"
  $WRANGLER pages project create "$P" --production-branch=main
  SUB="$(subdomain)"
fi
live="$(curl -fsS "https://$SUB/hash.txt" 2>/dev/null || true)"
if [ "$new" = "$live" ] && [ "${FORCE:-}" != "1" ]; then
  echo "bankroll page unchanged (https://$SUB)"
  exit 0
fi
$WRANGLER pages deploy site/dist --project-name="$P" --branch=main --commit-dirty=true
echo "bankroll page published: https://$SUB"
