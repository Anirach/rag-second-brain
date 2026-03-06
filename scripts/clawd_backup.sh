#!/bin/bash
# Auto-backup clawd workspace to GitHub
# Runs daily at 03:00 Bangkok time

cd /home/clawdbot/clawd || exit 1

if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "Auto backup $(date '+%Y-%m-%d')" --quiet
    git push --quiet 2>&1
    echo "$(date): Backup pushed"
else
    echo "$(date): No changes"
fi
