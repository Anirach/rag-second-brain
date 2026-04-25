#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/db/metadata.sqlite"
MIGRATION="$ROOT/db/migrations/001_init.sql"

if command -v sqlite3 >/dev/null 2>&1; then
  sqlite3 "$DB" < "$MIGRATION"
else
  python3 - "$DB" "$MIGRATION" <<'PY'
import sqlite3
import sys
from pathlib import Path

db_path = Path(sys.argv[1])
migration_path = Path(sys.argv[2])
sql = migration_path.read_text()
conn = sqlite3.connect(db_path)
try:
    conn.executescript(sql)
    conn.commit()
finally:
    conn.close()
PY
fi

echo "Initialized $DB"
