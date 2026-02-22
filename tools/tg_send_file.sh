#!/bin/bash
# Send a file via Telegram. Accessible from sandbox.
# Usage: bash tools/tg_send_file.sh /path/to/file.docx "caption text"
TOKEN="8240497272:AAHLEhfZUcbSoheY6atwcRqAeD2ewUmq-_I"
CHAT_ID="7579913696"
FILE="$1"
CAPTION="${2:-📎 File delivery}"
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendDocument" \
  -F "chat_id=${CHAT_ID}" \
  -F "document=@${FILE}" \
  -F "caption=${CAPTION}" > /dev/null
echo "✅ Sent to Telegram: $(basename $FILE)"
