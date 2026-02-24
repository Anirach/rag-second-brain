#!/bin/bash
# Auto-sync Obsidian vault with GitHub
# Runs via cron every 15 minutes

VAULT_DIR="/home/clawdbot/obsidian-vault"
cd "$VAULT_DIR" || exit 1

# Pull any remote changes first
git pull --rebase --quiet 2>/dev/null

# Check for local changes
if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M')" --quiet
    git push --quiet
    echo "$(date): Synced changes"
else
    echo "$(date): No changes"
fi
