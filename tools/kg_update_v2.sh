#!/bin/bash
# Knowledge Graph Update Script v2
# Refreshes graph from all sources, runs auto-linking, and updates MOC files
# Usage: bash tools/kg_update_v2.sh [--no-autolink] [--no-moc]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/../logs/kg_update_$(date '+%Y%m%d').log"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"

# Create logs dir if needed
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "=== Knowledge Graph Update v2 ==="
log "$TIMESTAMP"
log ""

# Step 1: Run the JSON graph builder (existing tool)
log "--- Step 1: Rebuilding JSON Knowledge Graph ---"
if [ -f "$SCRIPT_DIR/kg_builder.py" ]; then
    python3 "$SCRIPT_DIR/kg_builder.py" 2>&1 | tee -a "$LOG_FILE"
    log "[OK] kg_builder.py complete"
else
    log "[SKIP] kg_builder.py not found"
fi

log ""

# Step 2: Run auto-linker (new tool)
if [[ "${1:-}" != "--no-autolink" ]]; then
    log "--- Step 2: Auto-linking Entity Mentions ---"
    if [ -f "$SCRIPT_DIR/kg_auto_link.py" ]; then
        python3 "$SCRIPT_DIR/kg_auto_link.py" --auto --since 3 2>&1 | tee -a "$LOG_FILE"
        log "[OK] kg_auto_link.py complete"
    elif [ -f "$SCRIPT_DIR/kg_auto_link.py" ]; then
        python3 "$SCRIPT_DIR/kg_auto_link.py" --auto --since 3 2>&1 | tee -a "$LOG_FILE"
    else
        log "[SKIP] kg_auto_link.py not found"
    fi
    log ""
fi

# Step 3: Update MOC files (new tool)
if [[ "${1:-}" != "--no-moc" && "${2:-}" != "--no-moc" ]]; then
    log "--- Step 3: Updating MOC Files ---"
    if [ -f "$SCRIPT_DIR/kg_auto_link.py" ]; then
        python3 "$SCRIPT_DIR/kg_auto_link.py" --moc --auto 2>&1 | tee -a "$LOG_FILE"
        log "[OK] MOC update complete"
    else
        log "[SKIP] kg_auto_link.py not found"
    fi
    log ""
fi

# Step 4: Git commit changes
log "--- Step 4: Committing Changes ---"
VAULT_DIR=""
for vpath in "/home/clawdbot/obsidian-vault" "/workspace/obsidian-vault"; do
    if [ -d "$vpath/.git" ]; then
        VAULT_DIR="$vpath"
        break
    fi
done

if [ -n "$VAULT_DIR" ]; then
    cd "$VAULT_DIR"
    if git diff --quiet && git diff --staged --quiet; then
        log "[OK] No vault changes to commit"
    else
        git add -A
        git commit -m "kg-update: Auto-link and MOC update $(date '+%Y-%m-%d %H:%M')" 2>&1 | tee -a "$LOG_FILE"
        git push 2>&1 | tee -a "$LOG_FILE" || log "[WARN] git push failed (may need auth)"
        log "[OK] Committed vault changes"
    fi
else
    log "[SKIP] Vault git repo not found"
fi

log ""
log "--- Step 5: Vault Health Stats ---"
if [ -f "$SCRIPT_DIR/kg_stats.py" ]; then
    python3 "$SCRIPT_DIR/kg_stats.py" 2>/dev/null | head -30 | tee -a "$LOG_FILE"
else
    log "[SKIP] kg_stats.py not found"
fi

log ""
log "=== Update Complete ==="
log "Log saved to: $LOG_FILE"
log ""
log "Query examples:"
log "  python3 $SCRIPT_DIR/kg_query_vault.py projects --status active"
log "  python3 $SCRIPT_DIR/kg_query_vault.py links \"RAG Second Brain\""
log "  python3 $SCRIPT_DIR/kg_stats.py --full"
