#!/usr/bin/env bash
# The morning stats load on your Mac (the NBA stats site blocks most cloud servers).
# Runs at 8:30 AM; the Mac wakes itself at 8:25 (it needs to be plugged in, lid closed is fine).
#   bash infra/mac/install-statsday.sh          install
#   bash infra/mac/install-statsday.sh remove   uninstall
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.nbaprops.statsday.plist"
LOG="$HOME/Library/Logs/nbaprops-statsday.log"
if [ "${1:-}" = "remove" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  sudo pmset repeat cancel
  echo "removed"
  exit 0
fi
UV="$(command -v uv)" || { echo "uv not found on PATH"; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.nbaprops.statsday</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string>cd "$REPO" &amp;&amp; git pull --ff-only -q; set -a; source .env; set +a; echo "== \$(date)"; "$UV" run python -m nbaprops.pipeline.run stats-day</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PL
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
sudo pmset repeat wakeorpoweron MTWRFSU 08:25:00
echo "installed: stats load at 8:30 AM daily (log: $LOG); wake scheduled for 8:25 AM"
pmset -g sched | head -3
