import asyncio
import atexit
import json
import logging
import ipaddress
import os
import re
import socket
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib import error, parse, request

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {"User-Agent": "newai-production/1.0 (+https://github.com/MASSIVEMAGNETICS/newai)"}
UTF8_MAX_BYTES_PER_CHAR = 4
MAX_CLAIM_LENGTH = 240


@dataclass
class SourceRecord:
    url: str
    title: str
    snippet: str
    domain_score: float
    recency_score: float
    confidence: float


@dataclass
class AnswerRecord:
    question: str
    answer: str
    confidence: float
    sources: List[SourceRecord]
    cached: bool
    created_at: float


class MemoryStore:
    """SQLite-backed cache for answered questions."""

    def __init__(self, path: Optional[str] = None) -> None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "newai")
        os.makedirs(cache_dir, exist_ok=True)
        self.path = path or os.path.join(cache_dir, "memory.db")
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries(
                question TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        atexit.register(self.close)

    def get(self, question: str, max_age_hours: float = 24.0) -> Optional[AnswerRecord]:
        cur = self._conn.execute(
            "SELECT response, created_at FROM queries WHERE question = ?", (question.strip(),)
        )
        row = cur.fetchone()
        if not row:
            return None
        response_json, created_at = row
        if time.time() - created_at > max_age_hours * 3600:
            return None
        payload = json.loads(response_json)
        return AnswerRecord(
            question=payload["question"],
            answer=payload["answer"],
            confidence=payload["confidence"],
            sources=[SourceRecord(**item) for item in payload.get("sources", [])],
            cached=True,
            created_at=created_at,
        )

    def save(self, answer: AnswerRecord) -> None:
        payload = {
            "question": answer.question,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "sources": [asdict(s) for s in answer.sources],
        }
        self._conn.execute(
            "INSERT OR REPLACE INTO queries(question, response, created_at) VALUES (?, ?, ?)",
            (answer.question.strip(), json.dumps(payload), answer.created_at),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class HTMLTextExtractor:
    """Lightweight HTML-to-text converter using regular expressions only."""

    script_re = re.compile(r"<script.*?>.*?</script>", re.I | re.S)
    style_re = re.compile(r"<style.*?>.*?</style>", re.I | re.S)
    tag_re = re.compile(r"<[^>]+>")
    whitespace_re = re.compile(r"\s+")

    @classmethod
    def extract(cls, html: str) -> str:
        text = cls.script_re.sub(" ", html)
        text = cls.style_re.sub(" ", text)
        text = cls.tag_re.sub(" ", text)
        text = unescape(text)
        text = cls.whitespace_re.sub(" ", text)
        return text.strip()


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._current_href: Optional[str] = None
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href:
            text = "".join(self._buffer)
            self.links.append((self._current_href, text))
            self._current_href = None
            self._buffer = []


def _is_safe_url(url: str) -> bool:
    try:
        parsed = parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_reserved:
            return False
    except ValueError:
        # Not an IP address, allow domain names
        pass
    return True


def _domain_score(url: str) -> float:
    hostname = parse.urlparse(url).hostname or ""
    if hostname.endswith(".gov"):
        return 0.95
    if hostname.endswith(".edu"):
        return 0.9
    if hostname.endswith(".org"):
        return 0.75
    return 0.6


def _recency_score(last_modified: Optional[str]) -> float:
    if not last_modified:
        return 0.5
    try:
        dt = parsedate_to_datetime(last_modified)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    except (ValueError, TypeError):
        return 0.5
    if age_days <= 1:
        return 1.0
    if age_days <= 7:
        return 0.9
    if age_days <= 30:
        return 0.8
    if age_days <= 180:
        return 0.65
    return 0.5


def _combine_confidence(domain_score: float, recency_score: float) -> float:
    return round(min(1.0, 0.7 * domain_score + 0.3 * recency_score), 3)


def _parse_links(html: str, limit: int) -> List[SourceRecord]:
    parser = _LinkCollector()
    parser.feed(html)
    results: List[SourceRecord] = []
    seen = set()
    for href, text in parser.links:
        if (
            href in seen
            or "duckduckgo.com" in href
            or "javascript:" in href
            or not _is_safe_url(href)
        ):
            continue
        seen.add(href)
        title = HTMLTextExtractor.extract(text)[:200] or "Untitled"
        if not title.strip():
            continue
        score = _combine_confidence(_domain_score(href), 0.5)
        results.append(
            SourceRecord(
                url=href,
                title=title,
                snippet="",
                domain_score=_domain_score(href),
                recency_score=0.5,
                confidence=score,
            )
        )
        if len(results) >= limit:
            break
    return results


class WebSearchClient:
    """Simple DuckDuckGo HTML scraper (no API keys required)."""

    def __init__(self, timeout: int = 8) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 3) -> List[SourceRecord]:
        encoded = parse.quote_plus(query.strip())
        url = f"https://duckduckgo.com/html/?q={encoded}"
        req = request.Request(url, headers=DEFAULT_HEADERS)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
        except (error.URLError, error.HTTPError, socket.timeout) as exc:  # pragma: no cover - defensive
            logger.warning("Search failed for %s: %s", query, exc)
            return []
        return _parse_links(html, limit)


class ContentFetcher:
    """Fetches pages and extracts plain text with provenance scoring."""

    def __init__(self, timeout: int = 8, max_chars: int = 5000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    def fetch(self, record: SourceRecord) -> SourceRecord:
        if not _is_safe_url(record.url):
            logger.info("Skipping unsafe URL: %s", record.url)
            return record
        req = request.Request(record.url, headers=DEFAULT_HEADERS)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                byte_limit = self.max_chars * UTF8_MAX_BYTES_PER_CHAR  # assume up to 4 bytes/char for UTF-8 safety
                raw = resp.read(byte_limit)  # limit bytes to keep memory bounded
                content = raw.decode("utf-8", errors="ignore")
                last_modified = resp.headers.get("Last-Modified")
        except error.HTTPError as exc:  # pragma: no cover - network defensive
            logger.info("HTTP error on %s: %s", record.url, exc)
            return record
        except (error.URLError, socket.timeout) as exc:  # pragma: no cover - network defensive
            logger.info("Fetch failed for %s: %s", record.url, exc)
            return record

        text = HTMLTextExtractor.extract(content)
        snippet = text[: self.max_chars]
        recency = _recency_score(last_modified)
        combined = _combine_confidence(record.domain_score, recency)
        return SourceRecord(
            url=record.url,
            title=record.title,
            snippet=snippet,
            domain_score=record.domain_score,
            recency_score=recency,
            confidence=combined,
        )


class Synthesizer:
    """Rule-based synthesizer with provenance-aware confidence."""

    @staticmethod
    def synthesize(question: str, sources: Sequence[SourceRecord]) -> AnswerRecord:
        if not sources:
            answer = (
                "No live sources were reachable. Please retry with network access or a different query."
            )
            return AnswerRecord(
                question=question,
                answer=answer,
                confidence=0.0,
                sources=[],
                cached=False,
                created_at=time.time(),
            )

        top_sources = [s for s in sources if s.snippet]
        if not top_sources:
            top_sources = list(sources)

        # Build a concise synthesis
        claims = []
        for src in top_sources[:5]:
            sentence = src.snippet.split(". ")
            head = (
                sentence[0].strip()
                if sentence and sentence[0].strip()
                else src.snippet[:MAX_CLAIM_LENGTH]
            )
            claims.append(f"- {head[:MAX_CLAIM_LENGTH]} (source: {src.url})")

        joined_claims = "\n".join(claims)
        aggregate_confidence = round(sum(s.confidence for s in top_sources) / len(top_sources), 3)
        summary = (
            f"Answer synthesized from {len(top_sources)} sources with confidence {aggregate_confidence}.\n"
            f"Key signals:\n{joined_claims}"
        )
        return AnswerRecord(
            question=question,
            answer=summary,
            confidence=aggregate_confidence,
            sources=list(top_sources),
            cached=False,
            created_at=time.time(),
        )


class NewAIEngine:
    """Production-grade zero-pretraining retrieval+synthesis engine."""

    def __init__(self, memory: Optional[MemoryStore] = None) -> None:
        self.memory = memory or MemoryStore()
        self.search_client = WebSearchClient()
        self.fetcher = ContentFetcher()
        self.synthesizer = Synthesizer()

    @staticmethod
    def _expand_queries(question: str) -> List[str]:
        base = question.strip()
        return [
            base,
            f"{base} latest insights",
            f"{base} evidence and sources",
        ]

    async def _search_variation(self, query: str, limit: int) -> List[SourceRecord]:
        return await asyncio.to_thread(self.search_client.search, query, limit)

    async def _fetch_sources(self, sources: Iterable[SourceRecord]) -> List[SourceRecord]:
        tasks = [asyncio.to_thread(self.fetcher.fetch, src) for src in sources]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        return list(results)

    async def answer(
        self, question: str, *, max_sources: int = 5, force_refresh: bool = False
    ) -> AnswerRecord:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")

        if not force_refresh:
            cached = self.memory.get(question)
            if cached:
                logger.info("Cache hit for question: %s", question)
                return cached

        logger.info("Starting retrieval for: %s", question)
        variations = self._expand_queries(question)
        search_tasks = [self._search_variation(q, max_sources) for q in variations]
        search_results_nested = await asyncio.gather(*search_tasks)
        merged: List[SourceRecord] = []
        seen_urls = set()
        for results in search_results_nested:
            for item in results:
                if item.url in seen_urls:
                    continue
                seen_urls.add(item.url)
                merged.append(item)
        merged = merged[: max_sources * 2]  # keep top across variations

        fetched = await self._fetch_sources(merged[:max_sources])
        answer = self.synthesizer.synthesize(question, fetched)
        self.memory.save(answer)
        return answer


_ENGINE: Optional[NewAIEngine] = None


def get_engine() -> NewAIEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = NewAIEngine()
    return _ENGINE


async def ask(
    question: str,
    max_sources: int = 5,
    force_refresh: bool = False,
    engine: Optional[NewAIEngine] = None,
) -> AnswerRecord:
    active_engine = engine or get_engine()
    return await active_engine.answer(question, max_sources=max_sources, force_refresh=force_refresh)
