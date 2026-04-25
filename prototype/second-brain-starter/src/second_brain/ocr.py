from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class OcrUnavailableError(RuntimeError):
    pass


def ocr_pdf(path: str | Path, dpi: int = 300) -> str:
    pdf_path = Path(path).resolve()
    pdftoppm = shutil.which('pdftoppm')
    tesseract = shutil.which('tesseract')
    if not pdftoppm or not tesseract:
        missing = []
        if not pdftoppm:
            missing.append('pdftoppm')
        if not tesseract:
            missing.append('tesseract')
        raise OcrUnavailableError(f"OCR dependencies missing: {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prefix = tmp / 'page'
        render = subprocess.run(
            [pdftoppm, '-png', '-r', str(dpi), str(pdf_path), str(prefix)],
            check=False,
            capture_output=True,
            text=True,
        )
        if render.returncode != 0:
            raise RuntimeError(f"pdftoppm failed for {pdf_path}: {render.stderr.strip() or render.stdout.strip()}")

        images = sorted(tmp.glob('page-*.png'))
        if not images:
            raise RuntimeError(f"pdftoppm produced no images for {pdf_path}")

        parts: list[str] = []
        for image in images:
            out_base = image.with_suffix('')
            proc = subprocess.run(
                [tesseract, str(image), str(out_base), '-l', 'eng'],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"tesseract failed on {image.name}: {proc.stderr.strip() or proc.stdout.strip()}")
            txt_path = out_base.with_suffix('.txt')
            if txt_path.exists():
                text = txt_path.read_text(encoding='utf-8', errors='ignore').strip()
                if text:
                    parts.append(text)

        text = '\n\n'.join(parts).strip()
        if not text:
            raise RuntimeError(f"OCR produced no text for {pdf_path}")
        return text
