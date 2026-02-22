# BOOT.md — Gateway Startup Routine

Run silently. Do not send messages unless something important is found.

## Startup Tasks

1. Read `memory/heartbeat-state.json` — note last check times
2. Read today's `memory/YYYY-MM-DD.md` if it exists
3. Check calendar for events in next 24h using:
   `GOG_KEYRING_PASSWORD=openclaw GOG_ACCOUNT=anirach.m@fitm.kmutnb.ac.th gog calendar events primary --from <now> --to <tomorrow>`
4. If any calendar events found within 2 hours → send alert to Telegram chat_id 7579913696
5. Update `memory/heartbeat-state.json` with `gateway_boot` timestamp

## Send Message Only If:
- Calendar event within 2 hours of boot time
- Any cron job shows error status (check `openclaw cron list`)

## Otherwise: stay silent (no "I'm online" spam)
