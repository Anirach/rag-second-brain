# Workshop DOCX Generator

Generates professional workshop documents matching the Vibe Coding Workshop style.

## Design System

| Element | Style |
|---------|-------|
| Cover title | 40pt bold, Purple (#7C3AED) |
| Cover subtitle | 26pt, Gray (#6B7280) |
| Part headers | 22pt white on purple background |
| Module headers | H1 18pt bold + colored duration badge |
| Section headers | H2 14pt bold |
| Body text | 12pt, justified |
| AI Prompt boxes | Green header → dark code block → light tip |
| Best Practice | Green background (#D4EDDA) |
| Vibe/Tip | Purple background (#FDF4FF) |
| Warning | Yellow background (#FFF3CD) |
| Tables | Purple header row, alternating shading |

## Usage

```bash
python3 workshop_docx_generator.py config.json output.docx
```

## Config Structure

See `example_config.json` for a complete example.

### Content Types

| Type | Fields | Description |
|------|--------|-------------|
| `text` | `value` | Body paragraph |
| `vibe` | `value` | Purple callout box |
| `best_practice` | `value` | Green callout box |
| `warning` | `value` | Yellow callout box |
| `bullets` | `items[]` | Bullet list |
| `code` | `value` | Dark code block |
| `ai_prompt` | `title`, `prompt`, `tip?` | AI prompt box (3-row) |
| `quick_ref` | `title`, `prompt` | Compact prompt box |
| `table` | `headers[]`, `rows[][]` | Data table |
| `numbered_table` | `headers[]`, `rows[][]` | Numbered step table |
| `heading` | `value` | H2 heading |
| `subheading` | `value` | Bold 12pt |
| `page_break` | — | Page break |

### Badge Colors

`purple` (default), `green`, `blue` — set per module via `badge_color`.
