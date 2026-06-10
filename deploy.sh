#!/usr/bin/env bash
# deploy.sh — pull latest main, optionally reinstall deps, restart service, health-check.
# The PRD gate is enforced upstream: GitHub Actions CI must pass before a PR
# can merge to main.  By the time a commit lands here it has already cleared
# those checks.  This script is the delivery step, not the quality gate.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE="arabic-ocr"
HEALTH_URL="http://localhost:5000/status"

log() { echo "[deploy] $*"; }

# ── 1. Fetch & compare ────────────────────────────────────────────────────────

git -C "$REPO_DIR" fetch origin main --quiet

LOCAL=$(git -C "$REPO_DIR" rev-parse HEAD)
REMOTE=$(git -C "$REPO_DIR" rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already at $LOCAL — nothing to deploy."
    exit 0
fi

log "New commit detected: $LOCAL → $REMOTE"

# ── 2. Pull ───────────────────────────────────────────────────────────────────

git -C "$REPO_DIR" pull origin main --quiet

# ── 3. Reinstall deps if they changed ────────────────────────────────────────

DEPS_CHANGED=$(git -C "$REPO_DIR" diff "$LOCAL" HEAD -- requirements.txt setup.sh | wc -l)

if [ "$DEPS_CHANGED" -gt 0 ]; then
    log "Dependencies changed — reinstalling..."
    bash "$REPO_DIR/setup.sh"
fi

# ── 4. Restart service ────────────────────────────────────────────────────────

log "Restarting $SERVICE ..."
sudo systemctl restart "$SERVICE"

# ── 5. Health check (model takes ~60 s to load) ───────────────────────────────

log "Waiting for service to become ready (up to 10 min) ..."
for i in $(seq 1 60); do
    sleep 10
    SVC_STATUS=$(curl -sf "$HEALTH_URL" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null \
        || echo "")
    if [ "$SVC_STATUS" = "ready" ]; then
        log "Service is ready. Deployed $REMOTE successfully."
        exit 0
    fi
done

log "ERROR: Service did not become ready within 10 minutes after deploying $REMOTE."
exit 1
