from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .ingest import ingest_file, ingest_text
from .query import format_results, search
from .fts import refresh_fts
from .answer import answer_query, format_answer
from .review import format_review_scan, format_reviews, list_reviews, resolve_review, scan_reviews
from .status import format_object_statuses, get_object_status, list_object_statuses, promote_object_status, set_object_status
from .redundancy import format_redundancy_candidates, merge_objects, scan_redundancy
from .webapp import run_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="second-brain")
    parser.add_argument("--root", default=".", help="Path to second-brain-starter root")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest-text", help="Ingest a plain text source")
    ingest_parser.add_argument("--title", required=True)
    ingest_parser.add_argument("--text")
    ingest_parser.add_argument("--file")
    ingest_parser.add_argument("--source-type", default="note")
    ingest_parser.add_argument("--origin", default="cli")

    ingest_file_parser = sub.add_parser("ingest-file", help="Ingest a real file (pdf/doc/docx/txt/md/html)")
    ingest_file_parser.add_argument("path")
    ingest_file_parser.add_argument("--title")
    ingest_file_parser.add_argument("--origin", default="cli-file")

    query_parser = sub.add_parser("query", help="Query over knowledge objects and chunks")
    query_parser.add_argument("query")

    answer_parser = sub.add_parser("answer", help="Assemble an answer from retrieved objects and evidence")
    answer_parser.add_argument("query")
    answer_parser.add_argument("--llm", action="store_true", help="Use optional OpenAI-compatible LLM synthesis if configured")

    sub.add_parser("refresh-fts", help="Rebuild optional SQLite FTS indexes")

    review_scan_parser = sub.add_parser("review-scan", help="Detect draft/overlap review items and queue them")
    review_scan_parser.add_argument("--status", default="open", help="Unused placeholder for future filtering")

    review_list_parser = sub.add_parser("review-list", help="List review items")
    review_list_parser.add_argument("--status", default="open")

    review_resolve_parser = sub.add_parser("review-resolve", help="Resolve a review item")
    review_resolve_parser.add_argument("review_id")
    review_resolve_parser.add_argument("decision", choices=["accepted", "rejected", "deferred"])
    review_resolve_parser.add_argument("--notes", default="")
    review_resolve_parser.add_argument("--promote", action="store_true", help="Promote accepted object reviews")
    review_resolve_parser.add_argument("--promote-target", choices=["draft", "synthesized", "reviewed", "trusted", "deprecated"])

    object_status_parser = sub.add_parser("object-status", help="Show object statuses")
    object_status_parser.add_argument("object_id", nargs="?")
    object_status_parser.add_argument("--status")

    object_set_status_parser = sub.add_parser("object-set-status", help="Set object status explicitly")
    object_set_status_parser.add_argument("object_id")
    object_set_status_parser.add_argument("status", choices=["draft", "synthesized", "reviewed", "trusted", "deprecated"])
    object_set_status_parser.add_argument("--reason", default="")

    object_promote_parser = sub.add_parser("object-promote", help="Promote object status by one step or to a target")
    object_promote_parser.add_argument("object_id")
    object_promote_parser.add_argument("--target", choices=["draft", "synthesized", "reviewed", "trusted", "deprecated"])
    object_promote_parser.add_argument("--reason", default="")

    sub.add_parser("redundancy-scan", help="Find likely duplicate or overlapping objects")

    merge_parser = sub.add_parser("merge-objects", help="Merge one object into another and deprecate the source")
    merge_parser.add_argument("source_object_id")
    merge_parser.add_argument("target_object_id")
    merge_parser.add_argument("--reason", default="")

    web_parser = sub.add_parser("web-ui", help="Run the lightweight local web UI")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    settings = load_settings(Path(args.root))

    if args.command == "ingest-text":
        text = args.text
        if args.file:
            text = Path(args.file).read_text(encoding="utf-8")
        if not text:
            raise SystemExit("Provide --text or --file")
        result = ingest_text(settings, title=args.title, text=text, source_type=args.source_type, origin=args.origin)
        print(f"Ingested {result.source_id}")
        print(f"Chunks: {', '.join(result.chunk_ids)}")
        if result.entity_ids:
            print(f"Entities: {', '.join(result.entity_ids)}")
        print(f"Concept: {result.concept_id}")
        print(f"Synthesis: {result.synthesis_id}")
    elif args.command == "ingest-file":
        try:
            result = ingest_file(settings, args.path, title=args.title, origin=args.origin)
        except ValueError as exc:
            raise SystemExit(str(exc))
        print(f"Ingested {result.source_id}")
        print(f"Chunks: {', '.join(result.chunk_ids)}")
        if result.entity_ids:
            print(f"Entities: {', '.join(result.entity_ids)}")
        print(f"Concept: {result.concept_id}")
        print(f"Synthesis: {result.synthesis_id}")
    elif args.command == "query":
        print(format_results(search(settings, args.query)))
    elif args.command == "answer":
        print(format_answer(answer_query(settings, args.query, use_llm=args.llm)))
    elif args.command == "refresh-fts":
        ok = refresh_fts(settings)
        print("FTS index refreshed" if ok else "FTS unavailable; using scan fallback")
    elif args.command == "review-scan":
        print(format_review_scan(scan_reviews(settings)))
    elif args.command == "review-list":
        print(format_reviews(list_reviews(settings, status=args.status)))
    elif args.command == "review-resolve":
        item = resolve_review(
            settings,
            args.review_id,
            args.decision,
            notes=args.notes,
            promote=args.promote,
            promote_target=args.promote_target,
        )
        if not item:
            raise SystemExit(f"Review item not found: {args.review_id}")
        print(f"Resolved {item.review_id} as {item.status}")
    elif args.command == "object-status":
        if args.object_id:
            item = get_object_status(settings, args.object_id)
            if not item:
                raise SystemExit(f"Object not found: {args.object_id}")
            print(format_object_statuses([item]))
        else:
            print(format_object_statuses(list_object_statuses(settings, status=args.status)))
    elif args.command == "object-set-status":
        item = set_object_status(settings, args.object_id, args.status, reason=args.reason)
        if not item:
            raise SystemExit(f"Object not found: {args.object_id}")
        print(f"Updated {item.object_id} to {item.status}")
    elif args.command == "object-promote":
        item = promote_object_status(settings, args.object_id, target=args.target, reason=args.reason)
        if not item:
            raise SystemExit(f"Object not found: {args.object_id}")
        print(f"Promoted {item.object_id} to {item.status}")
    elif args.command == "redundancy-scan":
        print(format_redundancy_candidates(scan_redundancy(settings)))
    elif args.command == "merge-objects":
        result = merge_objects(settings, args.source_object_id, args.target_object_id, reason=args.reason)
        print(f"Merged {result.source_object_id} -> {result.target_object_id}")
    elif args.command == "web-ui":
        run_server(settings.root, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
