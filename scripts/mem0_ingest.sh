#!/bin/bash
# Mem0 daily log ingestion cron job
# Runs daily at 03:00 UTC — extracts memories from yesterday's daily log
cd /home/clawdbot/clawd
source .venv/bin/activate

# Ingest yesterday's log (complete day)
YESTERDAY=$(date -u -d "yesterday" +%Y-%m-%d)
python3 tools/mem0_helper.py ingest "$YESTERDAY" >> /home/clawdbot/clawd/logs/mem0.log 2>&1
echo "[$(date -u)] Mem0 ingest complete for $YESTERDAY" >> /home/clawdbot/clawd/logs/mem0.log
