#!/bin/bash
# LightRAG indexing cron job
# Runs daily at 02:00 UTC — indexes new/changed vault files
cd /home/clawdbot/clawd
source .venv/bin/activate
python3 tools/lightrag_helper.py index >> /home/clawdbot/clawd/logs/lightrag.log 2>&1
echo "[$(date -u)] LightRAG index complete" >> /home/clawdbot/clawd/logs/lightrag.log
