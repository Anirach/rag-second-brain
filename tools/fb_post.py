#!/usr/bin/env python3
"""Post to Facebook Page using Graph API."""
import json
import sys
import urllib.request
import urllib.parse

CONFIG_PATH = "/home/clawdbot/clawd/tools/fb_config.json"

def post_to_page(message: str) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    
    url = f"https://graph.facebook.com/v25.0/{cfg['page_id']}/feed"
    data = urllib.parse.urlencode({
        "message": message,
        "access_token": cfg["page_token"]
    }).encode()
    
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Read from stdin
        message = sys.stdin.read().strip()
    else:
        message = sys.argv[1]
    
    if not message:
        print("Error: No message provided", file=sys.stderr)
        sys.exit(1)
    
    result = post_to_page(message)
    print(json.dumps(result, indent=2))
    if "id" in result:
        post_id = result["id"]
        print(f"\n✅ Posted: https://www.facebook.com/{post_id}")
