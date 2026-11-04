#!/usr/bin/env bash
# GitHub switches off scheduled workflows in a public repository after 60 days without repository
# activity. If the last commit on the checked-out (default) branch is older than $KEEPALIVE_DAYS,
# add an empty commit so the schedule keeps running. No effect on private repositories' behaviour.
# (Not verified: whether pushes to the `state` branch alone would also count as activity.)
set -euo pipefail

DAYS="${KEEPALIVE_DAYS:-45}"
last="$(git log -1 --format=%ct)"
age_days=$(( ( $(date +%s) - last ) / 86400 ))

if [ "$age_days" -lt "$DAYS" ]; then
  echo "keepalive: last commit is $age_days day(s) old; nothing to do"
  exit 0
fi
git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git commit --allow-empty --quiet -m "chore: keep scheduled workflows active"
git push --quiet
echo "keepalive: pushed an empty commit (last real commit was $age_days days old)"
