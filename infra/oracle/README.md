# Running nba-props on a free Oracle Cloud server

One always-on server runs every job on time (no GitHub schedule, no Actions minutes), serves
the dashboard, and applies new commits from GitHub by itself. You reach the dashboard from
your phone through Tailscale: private to your devices, HTTPS, no domain needed.

| What | Where | When (New York time) |
|---|---|---|
| Morning routine: players, schedule, stats, settlement | server | 9:05 AM |
| Pricing runs | server | 10:00 AM, 3:00 PM (plus pre-tip passes) |
| Cycles: odds, settlement, injury changes, pre-tip pricing | server | :20 and :50, 11 AM-4 PM; every 15 min, 5-11:45 PM |
| Apply new commits from GitHub | server | every 10 min |
| Stats load (if the NBA site blocks the server) | your Mac | 8:30 AM, Mac wakes at 8:25 |

A failed job sends a Discord (or Telegram) message. Two jobs never run at once.

## 1. Oracle account

Sign up at oracle.com/cloud/free. It asks for a card to verify identity; Always Free resources
are never charged. Choose **US East (Ashburn)** as the home region (it can't be changed).

## 2. The server

Compute > Instances > Create instance:
- Image: **Canonical Ubuntu 24.04**
- Shape: **Ampere, VM.Standard.A1.Flex, 2 OCPUs, 12 GB** (Always Free allows up to 4 / 24)
- SSH keys: upload your Mac's public key (`cat ~/.ssh/id_ed25519.pub`; no key yet? run
  `ssh-keygen -t ed25519` and press Enter through the questions)

"Out of capacity" is common for free Ampere servers: try another availability domain, or
again later. No ports need opening: SSH is open by default, and the dashboard goes through
Tailscale.

## 3. The code on the server

From your Mac (the public IP is on the instance page):

```
ssh ubuntu@PUBLIC_IP
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 && cat ~/.ssh/id_ed25519.pub
```

On GitHub: nba-props > Settings > Deploy keys > Add deploy key, paste that line, read-only.
Then, still on the server:

```
git clone git@github.com:Rishi1503/nba-props.git ~/nba-props
```

## 4. Settings (.env)

On your Mac, add the dashboard login to `.env` (it asks twice for the password you'll use on
your phone, at least 12 characters, and appends two lines):

```
cd ~/Desktop/nba-props
uv run --extra api python -m nbaprops.api.auth >> .env
```

Make sure `.env` also has `DISCORD_WEBHOOK_URL` (or the Telegram pair) for alerts, then copy it:

```
scp .env ubuntu@PUBLIC_IP:~/nba-props/.env
```

## 5. Set up the server

```
ssh ubuntu@PUBLIC_IP
cd ~/nba-props && bash infra/oracle/bootstrap.sh
```

About 10 minutes. Near the end it prints a Tailscale login link: open it and sign up (free).
It finishes by listing the timers and printing the dashboard address
(`https://<name>.<tailnet>.ts.net`).

## 6. Your phone

Install the Tailscale app, sign in with the same account, open the dashboard address, and log
in with your password. In Safari: Share > Add to Home Screen.

## 7. Check it works

```
sudo systemctl start nbaprops@cycle                # one cycle now
journalctl -u nbaprops@cycle -n 40 --no-pager      # what it did
sudo systemctl start nbaprops@daily                # the morning routine: does stats-day work here?
journalctl -u nbaprops@daily -n 40 --no-pager
```

If the daily log shows the stats loading ("2026-27 regular: N games"), the server handles it
and you can skip step 8. If it shows "stats-day failed on the server", do step 8.

## 8. The Mac's morning stats load (only if step 7 needs it)

```
cd ~/Desktop/nba-props && bash infra/mac/install-statsday.sh
```

It asks for your Mac password once (to schedule the 8:25 AM wake). Keep the Mac plugged in
overnight; the lid can be closed. Log: `~/Library/Logs/nbaprops-statsday.log`.
Remove with `bash infra/mac/install-statsday.sh remove`.

## 9. Turn off GitHub's schedule

Once a day of server cycles looks right: GitHub > nba-props > Actions > Production cycle >
"..." > Disable workflow. (Re-enable it any time as a fallback.)

## Day to day

- Apply patches and push from your Mac as always; the server picks them up within 10 minutes
  (migrations included). Never edit files on the server: updates only fast-forward.
- Timers: `systemctl list-timers 'nbaprops-*'`
- Logs: `journalctl -u nbaprops@cycle --since today --no-pager`
- Run a job now: `sudo systemctl start nbaprops@cycle` (or `daily`, `deploy`)
- Dashboard backend: `sudo systemctl restart nbaprops-api`
