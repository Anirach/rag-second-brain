# TOOLS.md - Local Notes

## Key Tools & Paths
- **Image gen:** `python3 /home/clawdbot/clawd/nano_banana_google.py "prompt" output.png` (Google AI, key in file)
- **DOCX template:** `/home/clawdbot/clawd/templates/reports/General_Report_Template.docx`
- **Research DOCX:** `python3 skills/professional-docx-generator/scripts/clean_table_docx_generator.py "Title" config.json out.docx`
- **GDrive upload:** `python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py /path/file --folder "FolderName" "name.ext"`
- **TG send file:** `bash /home/clawdbot/clawd/tools/tg_send_file.sh /path/file "caption"`
- **HA DB (ONLY):** `python3 tools/ha_query.py "SELECT ..."` — NEVER raw psql
- **PDF→PDF:** `libreoffice --headless --convert-to pdf input.docx`
- **Semantic Scholar:** `python3 tools/semantic_scholar.py "query"`
- **ChromaDB:** `source .venv/bin/activate && python3 tools/chroma_helper.py search "query"`
- **PPTX:** `NODE_PATH=/home/clawdbot/.npm-global/lib/node_modules node script.js`

## API Keys (quick ref)
- **Google Places/AI:** `AIzaSyAXtg3wHGZzIzh6Q0nUcaUvn12GlWsukEs`
- **GDrive:** creds at `gdrive/credentials.json`
- **OpenAI (skills):** in openclaw.json skills.entries

## Rules
- DOCX reports: always use template. Page X of Y, TOC, justified text, "View Source" links.
- Images: always upload to GDrive `Generated-Images` after gen.
- PPTX: post-process emoji (high Unicode → XML numeric refs). Layout: y=1.1–4.95.
- HA tables: `accreditationHistory`, `View_WH_Surveyor`, `hai_employee_gofive_empeo`, `bi_kpi*`, `pcu_*`
