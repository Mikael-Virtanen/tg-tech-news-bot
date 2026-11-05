#!/usr/bin/env bash
# Save the bot's state: publish $STATE_DIR as the ONLY commit of the `state` branch.
#
# Each push replaces the branch with a fresh single commit (force-push), so the repository never
# grows: the state is a snapshot, not a history. Nothing is pushed if nothing changed.
# Safe because both workflows share one `concurrency:` group, so there is a single writer.
set -euo pipefail

STATE_DIR="${STATE_DIR:-.state}"
STATE_BRANCH="${STATE_BRANCH:-state}"
REMOTE="${STATE_REMOTE:-https://x-access-token:${GITHUB_TOKEN:?GITHUB_TOKEN is required}@github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}.git}"

if [ ! -d "$STATE_DIR" ] || [ -z "$(find "$STATE_DIR" -mindepth 1 -maxdepth 1 ! -name .git -print -quit)" ]; then
  echo "state: nothing to save"
  exit 0
fi

cd "$STATE_DIR"
if [ ! -d .git ]; then
  git init --quiet -b "$STATE_BRANCH"
  git remote add origin "$REMOTE"
else
  git remote set-url origin "$REMOTE"
fi
git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add -A
if git rev-parse -q --verify HEAD >/dev/null && git diff --cached --quiet; then
  echo "state: unchanged; not pushing"
  exit 0
fi

# Drop history: start a new root commit and force it onto the branch.
if git rev-parse -q --verify HEAD >/dev/null; then
  git checkout --quiet --orphan _next
fi
git commit --quiet -m "state: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Retry: a push that fails after the bot already posted is the one case where the next run would
# re-post everything this run sent. Two more attempts cover a flaky runner network.
pushed=0
for attempt in 1 2 3; do
  if git push --quiet --force origin "HEAD:refs/heads/$STATE_BRANCH"; then
    pushed=1
    break
  fi
  echo "state: push attempt $attempt of 3 failed" >&2
  if [ "$attempt" -lt 3 ]; then
    sleep $((attempt * 5))
  fi
done

if [ "$pushed" -ne 1 ]; then
  echo "::error title=State push failed::could not save the state after 3 attempts; the next run may re-post items sent in this run"
  exit 1
fi
echo "state: saved to branch '$STATE_BRANCH' at $(git rev-parse --short HEAD)"
