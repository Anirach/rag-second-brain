---
name: office-document-specialist-suite
description: Advanced suite for creating, editing, and analyzing Microsoft Office documents when multiple Office formats are involved (Word + Excel + PowerPoint together). Use when working across Office formats simultaneously, automated reporting with mixed formats, or batch document management. NOT for: Word-only tasks (use docx-generator-skill or word-docx), PowerPoint-only (use powerpoint-pptx), or simple single-format tasks.
metadata:
  {
    "openclaw": {
      "emoji": "📄",
      "requires": { 
        "bins": ["python3"], 
        "pip": ["python-docx", "openpyxl", "python-pptx"] 
      }
    }
  }
---

# Office Document Specialist Suite

A specialized toolset for professional document manipulation.

## Features

- **Word (.docx)**: Create and edit professional reports, manage styles, and insert tables/images.
- **Excel (.xlsx)**: Data analysis, automated spreadsheet generation, and complex formatting.
- **PowerPoint (.pptx)**: Automated slide deck creation from structured data.

## Usage

Each tool in the suite is designed to be called programmatically by the agent or via the provided CLI scripts.

## Installation

Run the included `setup.sh` to initialize the Python virtual environment and install dependencies.
