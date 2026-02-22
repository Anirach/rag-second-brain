# PDF Generation Skill

Generate PDFs from text, markdown, or structured content.

## Usage

Run the script with Node.js:

```bash
node skills/pdf-gen/generate.js --title "My Document" --content "Hello world" --output output.pdf
```

Or use programmatically from the workspace.

## Options

- `--title` - Document title (optional)
- `--content` - Text content or path to .md file
- `--output` - Output PDF path (default: output.pdf)

## Examples

**Simple text:**
```bash
node skills/pdf-gen/generate.js --content "This is my PDF content" --output report.pdf
```

**From markdown file:**
```bash
node skills/pdf-gen/generate.js --content ./notes.md --output notes.pdf
```

## Programmatic Use

```javascript
const { generatePDF } = require('./skills/pdf-gen/generate.js');
await generatePDF({ title: 'Report', content: 'Hello', output: 'report.pdf' });
```
