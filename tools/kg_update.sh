#!/bin/bash
# Knowledge Graph Update Script
# Refreshes graph from all sources and regenerates Obsidian notes
# Usage: bash tools/kg_update.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Knowledge Graph Update ==="
echo "$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Run builder
python3 "$SCRIPT_DIR/kg_builder.py"

echo ""
echo "=== Update Complete ==="
echo "Query examples:"
echo "  python3 $SCRIPT_DIR/kg_query.py stats"
echo "  python3 $SCRIPT_DIR/kg_query.py search \"Anirach\""
echo "  python3 $SCRIPT_DIR/kg_query.py projects --active"
echo "  python3 $SCRIPT_DIR/kg_query.py context \"NCD paper\""
