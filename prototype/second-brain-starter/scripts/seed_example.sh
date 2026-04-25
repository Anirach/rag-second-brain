#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/db/metadata.sqlite"
SEED="$ROOT/db/migrations/002_seed_example.sql"

if [[ ! -f "$DB" ]]; then
  echo "Database not initialized. Run bash scripts/init_db.sh first." >&2
  exit 1
fi

if command -v sqlite3 >/dev/null 2>&1; then
  sqlite3 "$DB" < "$SEED"
else
  python3 - "$DB" "$SEED" <<'PY'
import sqlite3
import sys
from pathlib import Path

db_path = Path(sys.argv[1])
seed_path = Path(sys.argv[2])
sql = seed_path.read_text()
conn = sqlite3.connect(db_path)
try:
    conn.executescript(sql)
    conn.commit()
finally:
    conn.close()
PY
fi

echo "Seeded example data into $DB"
