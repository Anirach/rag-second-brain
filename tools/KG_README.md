# Knowledge Graph Memory System

Entity-relationship graph extracted from Arthur's memory files, emails, and calendar.

## Files

| File | Purpose |
|------|---------|
| `kg_builder.py` | Extracts entities & relationships, builds graph + Obsidian notes |
| `kg_query.py` | Query interface (search, connections, projects, context) |
| `kg_update.sh` | One-command refresh wrapper |
| `knowledge_graph.json` | The graph (nodes + edges + metadata) |

## Quick Start

```bash
# Build/refresh the graph
python3 tools/kg_builder.py

# Or use the wrapper
bash tools/kg_update.sh
```

## Queries

```bash
# Search anything
python3 tools/kg_query.py search "Naphatsara"

# Show connections for a node
python3 tools/kg_query.py connections "person:anirach"

# List projects (all or active only)
python3 tools/kg_query.py projects --active

# Timeline overview
python3 tools/kg_query.py timeline --days 7

# Pull all context for a topic
python3 tools/kg_query.py context "NCD paper"

# Graph statistics
python3 tools/kg_query.py stats
```

## Obsidian Integration

Notes generated at `/home/clawdbot/obsidian-vault/KnowledgeGraph/`:
- `People/` — Person profiles with wikilinks
- `Projects/` — Project notes with people & topics
- `Topics/` — Topic notes
- `Organizations/` — Org profiles
- `Events/` — Events & conferences
- `Documents/` — Referenced documents

Each note has YAML frontmatter and `[[wikilinks]]` for graph navigation.

## Data Sources

1. **Memory files** — `memory/*.md` + `MEMORY.md`
2. **Emails** — via `gog gmail search` (last 30 days)
3. **Calendar** — via `gog calendar events` (next 30 days)

## Entity Types

`Person`, `Project`, `Organization`, `Topic`, `Event`, `Document`

## Relationship Types

`works_on`, `collaborates_on`, `affiliated_with`, `related_to`, `submitted_to`, `developed_at`, `emailed`, `met_with`

## Automation

Add to heartbeat or cron for automatic updates:
```bash
bash /home/clawdbot/clawd/tools/kg_update.sh
```
