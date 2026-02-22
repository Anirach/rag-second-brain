#!/bin/bash
# Gmail Pub/Sub → OpenClaw Webhook Setup
# Requires: gcloud CLI installed and authenticated
#
# This script wires Gmail push notifications to OpenClaw so new emails
# trigger instant agent alerts instead of waiting for the cron digest.
#
# PREREQUISITES (one-time manual steps):
#   1. Install gcloud: https://cloud.google.com/sdk/docs/install
#   2. Run: gcloud auth login
#   3. Set your project: gcloud config set project YOUR_PROJECT_ID
#   4. Enable APIs: gcloud services enable pubsub.googleapis.com gmail.googleapis.com
#   5. Make sure tailscale is running for public HTTPS endpoint
#      OR use any other tunnel (ngrok, cloudflare) and set WEBHOOK_BASE_URL below

set -e

ACCOUNT="anirach.m@fitm.kmutnb.ac.th"
TOPIC_NAME="gmail-push-notifications"
SUBSCRIPTION_NAME="gmail-push-sub"
WEBHOOK_PORT=8788
WEBHOOK_PATH="/gmail-pubsub"

# Hook token from openclaw.json
HOOK_TOKEN=$(python3 -c "import json; print(json.load(open('/home/clawdbot/.openclaw/openclaw.json'))['hooks']['token'])")
OPENCLAW_HOOK_URL="http://localhost:18789/hooks/gmail"

# Public webhook URL — update this if using tailscale funnel or ngrok
# WEBHOOK_BASE_URL="https://your-tailscale-hostname.ts.net"
WEBHOOK_BASE_URL="${WEBHOOK_BASE_URL:-}"

echo "=== Gmail Pub/Sub Setup ==="
echo "Account: $ACCOUNT"
echo "Topic: $TOPIC_NAME"

if [ -z "$(which gcloud 2>/dev/null)" ]; then
    echo ""
    echo "❌ gcloud not installed. Install it first:"
    echo "   curl https://sdk.cloud.google.com | bash"
    echo "   Then re-run this script."
    exit 1
fi

PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "❌ No gcloud project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "GCloud project: $PROJECT"

# 1. Create Pub/Sub topic
echo "Creating Pub/Sub topic..."
gcloud pubsub topics create "$TOPIC_NAME" --project="$PROJECT" 2>/dev/null || echo "  (topic already exists)"

# 2. Grant Gmail permission to publish
echo "Granting Gmail publish permissions..."
gcloud pubsub topics add-iam-policy-binding "$TOPIC_NAME" \
    --member="serviceAccount:gmail-api-push@system.gserviceaccount.com" \
    --role="roles/pubsub.publisher" \
    --project="$PROJECT"

# 3. Start gog watch serve as background daemon
echo "Starting gog gmail watch serve..."
GOG_KEYRING_PASSWORD=openclaw GOG_ACCOUNT="$ACCOUNT" \
    gog gmail watch serve \
    --bind 127.0.0.1 \
    --port $WEBHOOK_PORT \
    --path "$WEBHOOK_PATH" \
    --hook-url "$OPENCLAW_HOOK_URL" \
    --hook-token "$HOOK_TOKEN" \
    --include-body \
    --save-hook &

GOG_PID=$!
echo "  gog watch serve PID: $GOG_PID"
sleep 2

# 4. Start Gmail watch (requires public push URL)
if [ -n "$WEBHOOK_BASE_URL" ]; then
    PUSH_URL="${WEBHOOK_BASE_URL}${WEBHOOK_PATH}"
    echo "Starting Gmail watch with push URL: $PUSH_URL"
    GOG_KEYRING_PASSWORD=openclaw GOG_ACCOUNT="$ACCOUNT" \
        gog gmail watch start \
        --topic "projects/$PROJECT/topics/$TOPIC_NAME" \
        --push-endpoint "$PUSH_URL"
    echo "✅ Gmail watch started!"
else
    echo ""
    echo "⚠️  WEBHOOK_BASE_URL not set. Gmail watch requires a public HTTPS URL."
    echo "   Options:"
    echo "   1. Tailscale Funnel: tailscale funnel 8788"
    echo "      Then: WEBHOOK_BASE_URL=https://hostname.ts.net bash tools/gmail_pubsub_setup.sh"
    echo "   2. ngrok: ngrok http 8788"
    echo "      Then: WEBHOOK_BASE_URL=https://xxxx.ngrok.io bash tools/gmail_pubsub_setup.sh"
    echo ""
    echo "   gog watch serve is running on localhost:$WEBHOOK_PORT (PID: $GOG_PID)"
    echo "   OpenClaw webhook is ready at: $OPENCLAW_HOOK_URL"
fi

echo ""
echo "=== Next Steps ==="
echo "1. Set up a public URL (tailscale funnel or ngrok)"
echo "2. Re-run with WEBHOOK_BASE_URL=https://... bash tools/gmail_pubsub_setup.sh"
echo "3. Gmail watch renews every 7 days — a cron job handles renewal automatically"
