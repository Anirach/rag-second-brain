#!/bin/bash
# Morning System Health Report — Arthur 🐕
# Checks real services on this VPS (srv1297304)
# Runs daily at 07:00 Bangkok time

TOKEN=$(python3 -c "import json; print(json.load(open('/home/clawdbot/.openclaw/openclaw.json'))['channels']['telegram']['botToken'])" 2>/dev/null)
TG_CHAT_ID="7579913696"

DAY=$(TZ="Asia/Bangkok" date "+%a %d %b %Y")
TIME=$(TZ="Asia/Bangkok" date "+%H:%M")

# ─── CHECK HELPER ──────────────────────────────────────────────────────────
check_http() {
    local url=$1
    local timeout=${2:-3}
    local code=$(curl -o /dev/null -s -w "%{http_code}" --max-time $timeout "$url" 2>/dev/null)
    echo $code
}

check_port() {
    nc -z -w3 127.0.0.1 $1 2>/dev/null && echo "open" || echo "closed"
}

# ─── OPENCLAW GATEWAY (18789) ──────────────────────────────────────────────
gw_ms=$(curl -o /dev/null -s -w "%{time_total}" --max-time 3 http://127.0.0.1:18789/ 2>/dev/null | python3 -c "import sys; print(round(float(sys.stdin.read())*1000))" 2>/dev/null || echo "?")
if [ "$gw_ms" != "?" ] && [ "$gw_ms" -lt 9999 ] 2>/dev/null; then
    gw_icon="✅"; gw_detail="Telegram OK (${gw_ms}ms)"
else
    gw_icon="❌"; gw_detail="Unreachable"
fi


# ─── YOUTUBESTUDY (3101) ───────────────────────────────────────────────────
yt_code=$(check_http "http://127.0.0.1:3101/" 3)
if [[ "$yt_code" == "200" ]]; then
    yt_icon="✅"; yt_detail="Online"
else
    yt_icon="❌"; yt_detail="Down (HTTP $yt_code)"
fi

# ─── LOBSTERBOARD (8080) ───────────────────────────────────────────────────
lb_code=$(check_http "http://127.0.0.1:8080/" 3)
if [[ "$lb_code" == "200" ]]; then
    lb_icon="✅"; lb_detail="Online"
else
    lb_icon="❌"; lb_detail="Down (HTTP $lb_code)"
fi

# ─── GATEWAY NODE (18792) ──────────────────────────────────────────────────
node_resp=$(curl -s --max-time 2 http://127.0.0.1:18792/ 2>/dev/null)
if [[ "$node_resp" == "OK" ]]; then
    node_icon="✅"; node_detail="OK"
else
    node_icon="❌"; node_detail="Unreachable"
fi

# ─── DOCKER ────────────────────────────────────────────────────────────────
running=$(docker ps -q 2>/dev/null | wc -l)
total=$(docker ps -aq 2>/dev/null | wc -l)
if [ "$running" -eq "$total" ] && [ "$total" -gt 0 ]; then
    docker_icon="✅"
else
    docker_icon="⚠️"
fi

# ─── DISK ──────────────────────────────────────────────────────────────────
disk_used=$(df -h / | awk 'NR==2 {print $3}')
disk_total=$(df -h / | awk 'NR==2 {print $2}')
disk_pct=$(df / | awk 'NR==2 {print $5}')
disk_pct_num=$(df / | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
if [ "$disk_pct_num" -lt 80 ]; then
    disk_icon="💾"
else
    disk_icon="⚠️"
fi

# ─── MEMORY ────────────────────────────────────────────────────────────────
mem_used=$(free -h | awk '/^Mem:/{print $3}')
mem_total=$(free -h | awk '/^Mem:/{print $2}')
mem_pct=$(free | awk '/^Mem:/{printf "%.0f", $3/$2*100}')

# ─── ISSUES ────────────────────────────────────────────────────────────────
issues=()
[[ "$gw_icon"   == "❌" ]] && issues+=("OpenClaw Gateway unreachable")
[[ "$yt_icon"   == "❌" ]] && issues+=("YouTubeStudy is down (port 3101)")
[[ "$lb_icon"   == "❌" ]] && issues+=("LobsterBoard is down (port 8080)")
[[ "$node_icon" == "❌" ]] && issues+=("Gateway Node unreachable (port 18792)")
[[ "$docker_icon" == "⚠️" ]] && issues+=("Docker: $running/$total containers running")
[[ "$disk_pct_num" -ge 80 ]] && issues+=("Disk usage high: $disk_pct")

if [ ${#issues[@]} -eq 0 ]; then
    issues_line="✅ All systems operational"
else
    issues_line=""
    for issue in "${issues[@]}"; do
        issues_line+="⚠️ $issue"$'\n'
    done
    issues_line="${issues_line%$'\n'}"
fi

# ─── FORMAT REPORT ─────────────────────────────────────────────────────────
REPORT="📊 *Daily System Health Report*
🕖 ${DAY}, ${TIME} ICT

*Core Services*
| Service | Status | Details |
|---|---|---|
| OpenClaw Gateway | ${gw_icon} | ${gw_detail} |
| YouTubeStudy | ${yt_icon} | ${yt_detail} |
| LobsterBoard | ${lb_icon} | ${lb_detail} |
| Gateway Node | ${node_icon} | ${node_detail} |

${docker_icon} Docker: ${running}/${total} containers running
${disk_icon} Disk: ${disk_used} / ${disk_total} used (${disk_pct})
🧠 Memory: ${mem_used} / ${mem_total} (${mem_pct}%)

${issues_line}"

# ─── SEND ──────────────────────────────────────────────────────────────────
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
    -H "Content-Type: application/json" \
    -d "{
        \"chat_id\": \"${TG_CHAT_ID}\",
        \"text\": $(echo "$REPORT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
        \"parse_mode\": \"Markdown\"
    }" > /dev/null 2>&1

echo "Health report sent at $(TZ='Asia/Bangkok' date)"
