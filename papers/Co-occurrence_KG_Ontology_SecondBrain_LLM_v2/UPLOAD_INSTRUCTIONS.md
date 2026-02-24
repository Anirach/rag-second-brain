# Upload Instructions for Google Drive

## Target Folder
**"Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"**

## Files to Upload

Run these commands from the host (not sandbox):

```bash
# Main paper files
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/main.tex --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/references.bib --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/paper_text.txt --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/README.md --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"

# Documentation
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/PROCESS_DOCUMENT.md --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/VERIFICATION_RESULTS.md --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"

# Code files
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/code/hybrid_memory.py --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/code/evaluation.py --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/code/visualizations.py --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/code/requirements.txt --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/code/README.md --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"

# Data files
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/datasets/DATASETS.md --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py ~/.openclaw/workspace-paper-architect/paper/tables/results_tables.csv --folder "Co-occurrence_KG_Ontology_SecondBrain_LLM_v2"
```

## PDF Generation (Required)

To generate paper.pdf, use Overleaf:

1. Go to https://www.overleaf.com/
2. Create account or login
3. New Project → Upload Project
4. Upload main.tex and references.bib
5. Click "Recompile"
6. Download PDF
7. Upload PDF to Google Drive folder

## DOCX Generation (Required)

Option 1: Use paper_text.txt
- Open paper_text.txt in Microsoft Word
- Save as .docx
- Upload to Google Drive folder

Option 2: Use Overleaf export
- After compiling in Overleaf
- Menu → Download → Source
- Convert using online tools

## File Checklist

| File | Status | Size |
|------|--------|------|
| main.tex | ✅ Ready | 46 KB |
| references.bib | ✅ Ready | 14 KB |
| paper_text.txt | ✅ Ready | 21 KB |
| README.md | ✅ Ready | 4 KB |
| PROCESS_DOCUMENT.md | ✅ Ready | 18 KB |
| VERIFICATION_RESULTS.md | ✅ Ready | 14 KB |
| code/hybrid_memory.py | ✅ Ready | 41 KB |
| code/evaluation.py | ✅ Ready | 22 KB |
| code/visualizations.py | ✅ Ready | 16 KB |
| code/requirements.txt | ✅ Ready | 152 B |
| code/README.md | ✅ Ready | 5 KB |
| paper.pdf | ⏳ Needs compilation | - |
| paper.docx | ⏳ Needs conversion | - |
