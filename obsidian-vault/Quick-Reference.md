# Quick Reference — Arthur's Cheat Sheet

> Critical IDs, links, and facts. Always loaded at session start.
> Last updated: 2026-02-22

---

## 🔑 Key People
- **Anirach Mingkhwan** — Telegram chat_id `7579913696`, timezone Bangkok (UTC+7)

---

## 📁 Google Drive — ArthurBotData

### Course Materials
| Item | Drive ID / Link |
|------|----------------|
| DevOps VibeCoding Course (15-week DOCX) | `1C1cdWhNW2Xv_iJ8xe68DFqCdA3SLL1Pt` |
| Week 1 PPTX ✅ | `1JqeT-ffxswG8IvLg6-oyiY6MWFQq20KC` |
| Week 2 PPTX ✅ | `12dvquytK5Yv1psJBXAPtbZahbMUbPJhq` |
| Week 1 handout | `1uZeBvQjDU_QB9fSFU_Fh1IulDqYLSjsv` |
| Week 2 handout | `1-2kjiee4p7pSunMA0Y6SRrT7RRGlWjuG` |
| Week 3 handout | `1d5oNxlpot2c6cw6I9y2A1k1mQ6YOuTMI` |

### HA Analysis Reports
| Report | Drive ID |
|--------|---------|
| Strategic Analysis DOCX | `1pRz-nRbYRWbI9xoqarNPbdm85odHL_iu` |
| Strategic Analysis PPTX | `1Es4Qie-9KAOeEnkcriyJi_mG_hpMIi9W` |
| 6-Month Action Plan DOCX | `1TChJ0EoAw1vaVYciKvXF5PxUgVk4oeKM` |
| HCR Causality Analysis DOCX | `1wpWHU725JHpHapLSZms7eFluFbbfnkno` |

### Papers & Books
| Item | Drive Link |
|------|-----------|
| Three Old Men (EN) | [Link](https://docs.google.com/document/d/13pE5VHtxLosl2GC0tbSAA1R3RHczX8sr/edit) |
| Three Old Men (TH) | [Link](https://docs.google.com/document/d/1DBCrIPFbX9SyEu2ttF6l3mqwhFfwjDsh/edit) |

### Team Manuals v4
| Team | Link |
|------|------|
| Writing | [Link](https://docs.google.com/document/d/1HGcuywIWsBah4-djxW5fJwd-ioyUyLwb/edit) |
| Academic | [Link](https://docs.google.com/document/d/1q0OxJ8sy4CGQt9Ypyz_hXZafYRHpsK4K/edit) |
| Translation | [Link](https://docs.google.com/document/d/1Yw11G1Uw6IjuR7ZKDSAaLyyDHy8JXGIl/edit) |
| Course | [Link](https://docs.google.com/document/d/1uF1PwD-PSY_GGRqpRKntvWR8QIPPQz3u/edit) |
| Coding | [Link](https://docs.google.com/document/d/1CJgKyflyR3jdzu1waCUF5wnV2xub26kJ/edit) |

---

## 🐙 GitHub Repos
| Repo | URL |
|------|-----|
| Three Old Men | https://github.com/Anirach/three-old-men |
| RAG Second Brain | https://github.com/Anirach/rag-second-brain |
| ChartSense AI | https://github.com/Anirach/chartsense-ai |

---

## 🗄️ Key Local Paths
| Item | Path |
|------|------|
| Workspace | `/home/clawdbot/clawd` |
| SQLite memory DB | `/home/clawdbot/clawd/memory.db` |
| Memory CLI | `python3 /home/clawdbot/clawd/tools/memory_db.py` |
| HA query (safe) | `python3 /home/clawdbot/clawd/tools/ha_query.py` |
| HA de-ID tool | `/home/clawdbot/clawd/tools/ha_deid.py` |
| PPTX template | `/home/clawdbot/clawd/tmp/build_week2_pptx.js` |
| DOCX template | `/home/clawdbot/clawd/templates/reports/General_Report_Template.docx` |
| Sandbox Dockerfile | `/home/clawdbot/clawd/docker/Dockerfile.sandbox` |
| PPTX helper script | `NODE_PATH=/home/clawdbot/.npm-global/lib/node_modules node script.js` |

---

## ⚠️ Active Blockers / Pending Tasks

### HIGH PRIORITY
- **Week 6 PPTX REBUILD** — IaC/Terraform, failed 2026-02-22, 23min run
- **Weeks 7–15 PPTX** — 9 weeks remaining
- **Weeks 5–15 Handout DOCX** — 11 handouts remaining

### Infrastructure
- iPhone Obsidian → GitHub sync (method TBD)
- 22 unconfigured agents (no model set)
- `gog cal events` broken — needs `GOG_ACCOUNT` env var

---

## 📊 HA Database Quick Facts (Feb 2026)
- 1,584 hospitals | 591 surveyors | 120K+ manpower | 181 employees | 104 speakers
- 366 hospitals with expired accreditation (~25%)
- 250 at accreditation level 0
- Only 63 DHSA, 9 AHA (DHSA adoption: ~4%)
- 13 regions; Region 13 (Bangkok) largest at 252

---

## 🛠️ System Status
- **OpenClaw version:** 2026.2.19-2
- **Sandbox image:** `openclaw-sandbox:bookworm-slim` (uid fixed to 1001)
- **Model aliases:** `opus` = `anthropic/claude-opus-4-6`, `sonnet` = `anthropic/claude-sonnet-4-6`
- **Obsidian vault:** `/home/clawdbot/clawd/obsidian-vault/` (125→159 files, auto-sync 4x daily)
