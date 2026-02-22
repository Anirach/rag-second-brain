#!/usr/bin/env python3
"""
voice_to_text.py — Transcribe audio files using OpenAI Whisper API.

Supported formats: .ogg, .mp3, .m4a, .wav, .webm, .flac
(Telegram sends voice notes as .ogg)

Usage:
  python3 voice_to_text.py /path/to/audio.ogg
  python3 voice_to_text.py /path/to/audio.ogg --save
  python3 voice_to_text.py /path/to/audio.ogg --language th
  python3 voice_to_text.py /path/to/audio.ogg --save --language en

Flags:
  --save             Append transcription to today's memory file
  --language <code>  Language hint (en, th, zh, ja, etc.) -- auto-detect if omitted
  --model <name>     Whisper model (default: whisper-1)
  --raw              Output raw text only (no labels) -- useful for piping

API key sources (in order):
  1. OPENAI_API_KEY environment variable
  2. /home/clawdbot/.openclaw/openclaw.json -> models.providers.openai.apiKey
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error

# Constants
WORKSPACE = Path(__file__).parent.parent          # /home/clawdbot/clawd
MEMORY_DIR = WORKSPACE / "memory"
OPENCLAW_CONFIG = Path("/home/clawdbot/.openclaw/openclaw.json")

SUPPORTED_FORMATS = {".ogg", ".mp3", ".m4a", ".wav", ".webm", ".flac", ".mp4", ".mpeg", ".mpga"}

MIME_TYPES = {
    ".ogg":  "audio/ogg",
    ".mp3":  "audio/mpeg",
    ".m4a":  "audio/mp4",
    ".wav":  "audio/wav",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".mp4":  "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
}


def get_api_key() -> str:
    """Resolve OpenAI API key from env or config file."""
    # 1. Environment variable
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    # 2. OpenClaw config file
    if OPENCLAW_CONFIG.exists():
        try:
            with open(OPENCLAW_CONFIG) as f:
                cfg = json.load(f)
            key = (cfg
                   .get("models", {})
                   .get("providers", {})
                   .get("openai", {})
                   .get("apiKey", ""))
            if key and not key.startswith("YOUR_") and key.startswith("sk-"):
                return key
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not read {OPENCLAW_CONFIG}: {e}", file=sys.stderr)

    raise SystemExit(
        "OpenAI API key not found.\n"
        "Set OPENAI_API_KEY environment variable, or configure in:\n"
        f"  {OPENCLAW_CONFIG}  ->  models.providers.openai.apiKey"
    )


def build_multipart(fields: dict, file_field: str, file_path: Path, mime_type: str):
    """
    Build multipart/form-data body using stdlib only (no requests).
    Returns (body_bytes, content_type_header_value).
    """
    boundary = "----OpenClawVoiceBoundary9f4a2b8c"

    parts = []
    for name, value in fields.items():
        if value is None:
            continue
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )

    filename = file_path.name
    file_data = file_path.read_bytes()
    file_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n"
        f"\r\n"
    )
    file_footer = f"\r\n--{boundary}--\r\n"

    body = b"".join(
        [p.encode("utf-8") for p in parts]
        + [file_header.encode("utf-8"), file_data, file_footer.encode("utf-8")]
    )

    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def transcribe(audio_path: Path, language: str = None, model: str = "whisper-1") -> str:
    """
    Transcribe audio via OpenAI Whisper API.

    Args:
        audio_path: Path to audio file
        language:   ISO-639-1 language code (None = auto-detect)
        model:      Whisper model (default: whisper-1)

    Returns:
        Transcribed text string
    """
    ext = audio_path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: '{ext}'\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    api_key = get_api_key()
    mime = MIME_TYPES.get(ext, "audio/mpeg")

    fields = {"model": model}
    if language:
        fields["language"] = language

    body, content_type = build_multipart(fields, "file", audio_path, mime)

    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("text", "").strip()

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(error_body)
            msg = err_json.get("error", {}).get("message", error_body)
        except Exception:
            msg = error_body
        raise SystemExit(f"OpenAI API error {e.code}: {msg}")

    except urllib.error.URLError as e:
        raise SystemExit(f"Network error: {e.reason}")


def save_to_memory(text: str, audio_path: Path) -> Path:
    """Append transcription to today's daily memory file."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    memory_file = MEMORY_DIR / f"{today}.md"

    timestamp = datetime.now().strftime("%H:%M")
    entry = (
        f"\n## Voice Note -- {timestamp}\n\n"
        f"**Source:** `{audio_path.name}`\n\n"
        f"{text}\n"
    )

    with open(memory_file, "a", encoding="utf-8") as f:
        f.write(entry)

    return memory_file


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper API",
    )
    parser.add_argument("audio_file", help="Path to audio file (.ogg, .mp3, .m4a, .wav, .webm, ...)")
    parser.add_argument("--save", action="store_true",
                        help="Append transcription to today's memory file (memory/YYYY-MM-DD.md)")
    parser.add_argument("--language", "-l", metavar="CODE",
                        help="Language hint: en, th, zh, ja, ko, ... (auto-detect if omitted)")
    parser.add_argument("--model", default="whisper-1",
                        help="Whisper model name (default: whisper-1)")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw text only -- useful for scripting/piping")
    args = parser.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()

    if not audio_path.exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if audio_path.stat().st_size == 0:
        print(f"File is empty: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not args.raw:
        lang_hint = f" [{args.language}]" if args.language else ""
        size_kb = audio_path.stat().st_size / 1024
        print(f"Transcribing: {audio_path.name} ({size_kb:.1f} KB){lang_hint} ...", file=sys.stderr)

    try:
        text = transcribe(audio_path, language=args.language, model=args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not text:
        print("No transcription returned (silent or unrecognisable audio?)", file=sys.stderr)
        sys.exit(1)

    if args.raw:
        print(text)
    else:
        print(f"\nTranscription:\n{text}\n")

    if args.save:
        saved = save_to_memory(text, audio_path)
        if not args.raw:
            print(f"Saved to: {saved}", file=sys.stderr)

    return text


if __name__ == "__main__":
    main()
