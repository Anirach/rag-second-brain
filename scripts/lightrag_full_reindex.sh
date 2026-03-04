#!/bin/bash
# LightRAG full re-index (weekly, Sundays 01:00 UTC)
cd /home/clawdbot/clawd
source .venv/bin/activate
python3 tools/lightrag_helper.py index --full >> /home/clawdbot/clawd/logs/lightrag.log 2>&1
echo "[$(date -u)] LightRAG FULL re-index complete" >> /home/clawdbot/clawd/logs/lightrag.log
