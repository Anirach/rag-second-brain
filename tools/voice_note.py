#!/usr/bin/env python3
"""
voice_note.py -- Higher-level voice note tool.

Transcribes an audio file and saves it to memory with a timestamp.
Optionally categorizes the note using quick_note.py categories.

Usage:
  python3 voice_note.py /path/to/audio.ogg
  python3 voice_note.py /path/to/audio.ogg --category remember
  python3 voice_note.py /path/to/audio.ogg --category todo
  python3 voice_note.py /path/to/audio.ogg --language th
  python3 voice_note.py /path/to/audio.ogg --category fact --language en

Categories (passed to quick_note.py):
  fact       -> KnowledgeGraph/Documents/Facts.md
  decision   -> KnowledgeGraph/Decisions/
  todo       -> KnowledgeGraph/Action-Items/
  remember   -> MEMORY.md (Lessons Learned)
  event      -> Daily memory note / KnowledgeGraph/Events/
  person     -> KnowledgeGraph/People/
  project    -> KnowledgeGraph/Projects/

If no --category given, saves to today's memory file only.
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
WORKSPACE = SCRIPT_DIR.parent                  # /home/clawdbot/clawd
MEMORY_DIR = WORKSPACE / "memory"
VOICE_TO_TEXT = SCRIPT_DIR / "voice_to_text.py"
QUICK_NOTE = SCRIPT_DIR / "quick_note.py"

VALID_CATEGORIES = {"fact", "decision", "todo", "remember", "event", "person", "project"}


def transcribe_audio(audio_path: Path, language: str = None) -> str:
    """
    Call voice_to_text.py to transcribe audio.
    Returns transcribed text, or raises RuntimeError on failure.
    """
    if not VOICE_TO_TEXT.exists():
        raise FileNotFoundError(
            f"voice_to_text.py not found at {VOICE_TO_TEXT}\n"
            "Please ensure both voice_to_text.py and voice_note.py are in the same directory."
        )

    cmd = [sys.executable, str(VOICE_TO_TEXT), "--raw", str(audio_path)]
    if language:
        cmd += ["--language", language]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error = result.stderr.strip() or "Unknown error"
        raise RuntimeError(f"Transcription failed: {error}")

    text = result.stdout.strip()
    if not text:
        raise RuntimeError("No transcription returned (silent or unrecognisable audio?)")

    return text


def save_to_memory(text: str, audio_path: Path) -> Path:
    """Append transcription to today's memory file with timestamp."""
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


def categorize_note(text: str, category: str) -> bool:
    """
    Route note to the right place using quick_note.py.
    Returns True on success, False on failure.
    """
    if not QUICK_NOTE.exists():
        print(f"Warning: quick_note.py not found at {QUICK_NOTE}", file=sys.stderr)
        return False

    cmd = [sys.executable, str(QUICK_NOTE), category, text]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Warning: quick_note.py failed: {result.stderr.strip()}", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio and save as a voice note",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 voice_note.py meeting.ogg
  python3 voice_note.py idea.m4a --category todo
  python3 voice_note.py note.wav --category remember --language en
  python3 voice_note.py msg.ogg --category fact --language th
        """,
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (.ogg, .mp3, .m4a, .wav, .webm, ...)",
    )
    parser.add_argument(
        "--category", "-c",
        metavar="CATEGORY",
        choices=sorted(VALID_CATEGORIES),
        help=f"Note category: {', '.join(sorted(VALID_CATEGORIES))}",
    )
    parser.add_argument(
        "--language", "-l",
        metavar="CODE",
        help="Language hint: en, th, zh, ja, ko, ... (auto-detect if omitted)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Skip saving to daily memory file (only save to category destination)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()

    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if audio_path.stat().st_size == 0:
        print(f"Error: File is empty: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Transcribe
    lang_hint = f" [{args.language}]" if args.language else ""
    size_kb = audio_path.stat().st_size / 1024
    print(f"Transcribing: {audio_path.name} ({size_kb:.1f} KB){lang_hint} ...", file=sys.stderr)

    try:
        text = transcribe_audio(audio_path, language=args.language)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTranscription:\n{text}\n")

    # 2. Save to daily memory file (unless --no-memory)
    if not args.no_memory:
        try:
            memory_file = save_to_memory(text, audio_path)
            print(f"Saved to memory: {memory_file}")
        except IOError as e:
            print(f"Warning: could not save to memory: {e}", file=sys.stderr)

    # 3. Categorize (if --category given)
    if args.category:
        if categorize_note(text, args.category):
            print(f"Categorized as: {args.category}")
        else:
            print(f"Warning: categorization failed, but memory was saved.", file=sys.stderr)

    return text


if __name__ == "__main__":
    main()
