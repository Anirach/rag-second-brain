#!/bin/bash
# Knowledge Graph Update Script
# Run via cron: */30 * * * * /home/clawdbot/clawd/tools/knowledge-graph/update_graph.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/home/clawdbot/clawd/logs/knowledge-graph.log"
VENV_PATH="/home/clawdbot/clawd/.venv"

mkdir -p "$(dirname "$LOG_FILE")"

{
    echo ""
    echo "=========================================="
    echo "Knowledge Graph Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # Set gog environment
    export GOG_ACCOUNT="anirach.m@fitm.kmutnb.ac.th"
    export GOG_KEYRING_PASSWORD="${GOG_KEYRING_PASSWORD:-}"

    # Activate venv if it exists
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    fi

    python3 "$SCRIPT_DIR/extract_knowledge.py"

    echo "Exit code: $?"
    echo "=========================================="
} >> "$LOG_FILE" 2>&1
