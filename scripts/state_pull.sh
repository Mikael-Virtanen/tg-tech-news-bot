#!/usr/bin/env bash
# Restore the bot's state: clone the `state` branch into $STATE_DIR (default .state).
# On the very first run the branch does not exist yet and an empty directory is created.
#
# SAFETY: "branch does not exist" (git ls-remote exit code 2) is the ONLY case treated as a
# first run. Any other failure (network, auth, outage) aborts the job. Mistaking an outage for a
# first run would let state_push.sh later overwrite the real state with an empty one, and the bot
# would then re-post everything it has ever posted.
set -euo pipefail

STATE_DIR="${STATE_DIR:-.state}"
STATE_BRANCH="${STATE_BRANCH:-state}"
REMOTE="${STATE_REMOTE:-https://x-access-token:${GITHUB_TOKEN:?GITHUB_TOKEN is required}@github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}.git}"

rm -rf "$STATE_DIR"

set +e
git ls-remote --exit-code --heads "$REMOTE" "$STATE_BRANCH" >/dev/null 2>&1
rc=$?
set -e

case "$rc" in
  0)
    git clone --quiet --depth 1 --branch "$STATE_BRANCH" "$REMOTE" "$STATE_DIR"
    echo "state: restored branch '$STATE_BRANCH' at $(git -C "$STATE_DIR" rev-parse --short HEAD)"
    ;;
  2)
    mkdir -p "$STATE_DIR"
    echo "state: branch '$STATE_BRANCH' does not exist yet; starting with an empty state"
    ;;
  *)
    echo "state: cannot reach the repository (git ls-remote exited with $rc); refusing to continue" >&2
    exit 1
    ;;
esac
