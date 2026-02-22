# Gmail Real-Time Email Setup

## Status: BLOCKED — needs GCP project

### What's ready:
- cloudflared installed at /usr/local/bin/cloudflared
- ngrok installed at /usr/local/bin/ngrok  
- OpenClaw webhook endpoint enabled (/hooks/gmail)
- Hook token: stored in openclaw.json hooks.token
- gog gmail watch serve command confirmed working

### What's needed (one-time, ~10 min):
1. Create a GCP project at https://console.cloud.google.com
2. Enable APIs:
   - Cloud Pub/Sub API
   - Gmail API
3. Create a Pub/Sub topic: `gmail-push-notifications`
4. Grant publish permission to: gmail-api-push@system.gserviceaccount.com
5. Run:
   ```bash
   # Start tunnel
   cloudflared tunnel --url http://localhost:8788 --no-autoupdate &
   # Note the tunnel URL (e.g. https://xxxx.trycloudflare.com)
   
   # Start gog watch serve
   export GOG_KEYRING_PASSWORD=openclaw GOG_ACCOUNT=anirach.m@fitm.kmutnb.ac.th
   nohup gog gmail watch serve \
     --bind 127.0.0.1 --port 8788 --path /gmail-pubsub \
     --hook-url http://localhost:18789/hooks/gmail \
     --hook-token $(python3 -c "import json; print(json.load(open('/home/clawdbot/.openclaw/openclaw.json'))['hooks']['token'])") \
     --include-body --save-hook > ~/clawd/logs/gog-watch.log 2>&1 &
   
   # Start Gmail watch
   gog gmail watch start \
     --topic projects/YOUR_PROJECT_ID/topics/gmail-push-notifications \
     --hook-url https://xxxx.trycloudflare.com/gmail-pubsub \
     --include-body
   ```
6. Add PROJECT_ID to openclaw.json env: "GCP_PROJECT_ID": "your-project-id"

### Alternative (simpler): Poll-based near-realtime
If GCP setup is too much friction, can do 5-min email polling via cron instead.
