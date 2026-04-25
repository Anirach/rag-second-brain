# Second Brain Starter

A lean starter scaffold for a provenance-aware RAG second brain.

## Design goal

Compile raw information into durable, inspectable knowledge objects, then retrieve those objects before falling back to raw chunks.

## What this scaffold includes

- Filesystem layout for raw sources, extracted artifacts, and knowledge objects
- SQLite migration for metadata, provenance, reviews, and links
- Example source / chunk / entity / concept / synthesis files
- Basic config files (`purpose.md`, `schema.md`, `settings.json`)
- A tiny bootstrap script to initialize the SQLite database

## Suggested workflow

1. Drop a source into `raw/inbox/`
2. Normalize it into `raw/sources/{source_id}/`
3. Chunk and extract into `extract/`
4. Build or update markdown objects under `knowledge/`
5. Register evidence links and review items in SQLite
6. Query knowledge objects first, evidence second, raw chunks last

## Quick start

```bash
cd tmp/second-brain-starter
bash scripts/init_db.sh
bash scripts/seed_example.sh
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect('db/metadata.sqlite')
print(conn.execute("select name from sqlite_master where type='table' order by name").fetchall())
conn.close()
PY
PYTHONPATH=src python3 -m second_brain.cli --root . query transformer
```

## CLI

```bash
# Ingest plain text directly
PYTHONPATH=src python3 -m second_brain.cli --root . ingest-text \
  --title "Retrieval Augmented Generation" \
  --text "RAG combines parametric memory with external retrieval."

# Or ingest from a plain text file
PYTHONPATH=src python3 -m second_brain.cli --root . ingest-text \
  --title "My Note" \
  --file notes/my-note.txt

# Ingest a real file (pdf/doc/docx/txt/md/html)
PYTHONPATH=src python3 -m second_brain.cli --root . ingest-file papers/example.pdf

# Rebuild optional SQLite FTS indexes (falls back gracefully if FTS5 is unavailable)
PYTHONPATH=src python3 -m second_brain.cli --root . refresh-fts

# Query knowledge objects first, then chunks
PYTHONPATH=src python3 -m second_brain.cli --root . query "external retrieval"

# Assemble a grounded answer from retrieved objects and evidence
PYTHONPATH=src python3 -m second_brain.cli --root . answer "external retrieval"

# Optional OpenAI-compatible synthesis (requires config + API key env var)
PYTHONPATH=src python3 -m second_brain.cli --root . answer --llm "external retrieval"

# Detect and queue review/conflict items
PYTHONPATH=src python3 -m second_brain.cli --root . review-scan

# List open review items
PYTHONPATH=src python3 -m second_brain.cli --root . review-list --status open

# Resolve a review item
PYTHONPATH=src python3 -m second_brain.cli --root . review-resolve rev_20260425_001 accepted --notes "Checked and approved"

# Resolve and promote reviewed knowledge in one step
PYTHONPATH=src python3 -m second_brain.cli --root . review-resolve rev_20260425_001 accepted --promote --promote-target reviewed

# Inspect object trust state
PYTHONPATH=src python3 -m second_brain.cli --root . object-status --status reviewed

# Promote an object manually
PYTHONPATH=src python3 -m second_brain.cli --root . object-promote cpt_retrieval-augmented-generation --target trusted --reason "Cross-checked and approved"

# Scan for likely duplicate / redundant objects
PYTHONPATH=src python3 -m second_brain.cli --root . redundancy-scan

# Merge one object into another
PYTHONPATH=src python3 -m second_brain.cli --root . merge-objects ent_the-transformer ent_transformer --reason "duplicate label"

# Run the lightweight local UI
PYTHONPATH=src python3 -m second_brain.cli --root . web-ui --port 8765
# then open http://127.0.0.1:8765
```

## Storage model

- `raw/` — immutable source layer
- `extract/` — machine-generated intermediate layer
- `knowledge/` — durable human-editable artifacts
- `reviews/` — queued review and conflict records
- `db/` — metadata / provenance / review database
- `derived/` — embeddings, graph, cache, search sidecars

## Notes

- Keep raw sources immutable.
- Treat chat memory as separate from canonical knowledge.
- Derive graphs from the database; do not make graph storage the source of truth.
- The `answer` command works deterministically by default and only uses an external model when `--llm` is passed and config is valid.
- Answers now include cleaner human-readable citations inline, while still keeping raw ids in a separate citation map for auditability.
- `ingest-file` supports `.pdf`, `.doc`, `.docx`, `.txt`, `.md`, and `.html`; `.doc` is converted via headless LibreOffice.
- Image-only/scanned PDFs try an OCR fallback via `pdftoppm` + `tesseract`. If those binaries are missing, ingestion fails with a clear dependency message instead of silently pretending success.
- Optional LLM-assisted extraction can replace heuristic entity/summary extraction when `llm.enabled=true`, `llm.extractionEnabled=true`, and the configured API key is available.
- `review-scan` currently flags low-trust objects and evidence overlap; it is intentionally conservative and meant as a seed workflow.
- Status promotion updates both the SQLite row and the markdown frontmatter so trust state stays visible in the artifacts.
- `merge-objects` moves evidence links to the kept object, marks the source object deprecated, and records a `merged_into` link so redundancy handling is auditable.
- `web-ui` is intentionally thin: local-only, no auth, just enough to query, answer, inspect markdown artifacts, change object trust state, browse reviews, resolve review items, upload files into the ingest pipeline, run redundancy scans/merges, and show a project dashboard from the browser.
