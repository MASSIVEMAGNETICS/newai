from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, List, Optional
from dataclasses import asdict

from .engine import get_engine

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _print_answer(payload: Any) -> None:
    print("\n=== ANSWER ===")
    print(payload.answer)
    print(f"\nConfidence: {payload.confidence}")
    print("\nSources:")
    if not payload.sources:
        print("  (none available)")
    for idx, src in enumerate(payload.sources, 1):
        print(f"  {idx}. {src.title} [{src.confidence}]")
        print(f"     {src.url}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Production-grade zero-pretraining retrieval engine."
    )
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
        help="Maximum number of sources to fetch and synthesize",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cached responses and re-run live retrieval",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    args = parser.parse_args(argv)

    question = args.question or "What is the latest on responsible AI regulation?"
    engine = get_engine()
    result = asyncio.run(engine.answer(question, max_sources=args.max_sources, force_refresh=args.force_refresh))

    if args.json:
        print(
            json.dumps(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "sources": [asdict(src) for src in result.sources],
                    "cached": result.cached,
                    "created_at": result.created_at,
                },
                indent=2,
            )
        )
    else:
        _print_answer(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
