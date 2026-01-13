# newai

Production-grade zero-pretraining retrieval + synthesis engine.

## Features
- Query decomposition into multiple search variants
- DuckDuckGo-based retrieval (no API keys)
- Lightweight HTML extraction with provenance scoring
- Confidence built from domain trust + recency
- SQLite-backed 24h cache for repeated questions
- CLI output in human-readable or JSON formats

## Quickstart
```bash
python -m newai "What are the latest trends in AI safety?"
```

Use JSON output:
```bash
python -m newai "How does retrieval augmented generation work?" --json
```

Force a live refresh instead of using cached memory:
```bash
python -m newai "Latest on EU AI regulation" --force-refresh
```

## Notes
- By default the cache is stored at `~/.cache/newai/memory.db` and reused for 24 hours.
- Network errors are handled gracefully; if no sources are reachable you will receive a helpful message instead of a crash.
