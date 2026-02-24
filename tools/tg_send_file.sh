#!/bin/bash
# Usage: tg_send_file.sh /path/to/file.docx "Caption here"
# Sends a file to Anirach's Telegram chat
FILE="$1"
CAPTION="${2:-File delivered}"
TOKEN="8240497272:AAHLEhfZUcbSoheY6atwcRqAeD2ewUmq-_I"
CHAT_ID="7579913696"

if [ ! -f "$FILE" ]; then
  echo "ERROR: File not found: $FILE"
  exit 1
fi

FILENAME=$(basename "$FILE")
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendDocument"   -F "chat_id=${CHAT_ID}"   -F "document=@${FILE};filename=${FILENAME}"   -F "caption=${CAPTION}"
