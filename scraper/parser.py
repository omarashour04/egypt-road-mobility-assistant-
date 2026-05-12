"""
scraper/parser.py
=================
Cleans raw HTML and splits text into overlapping chunks.
Preserves source metadata (URL, title, section heading, language).
"""

import re
import json
import unicodedata
from typing import Optional
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from loguru import logger

import sys
sys.path.insert(0, str(__file__).split("/scraper")[0])
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN


# ── HTML tags that carry meaningful content ───────────────────────────────────
CONTENT_TAGS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "td", "th", "article", "section",
    "blockquote", "figcaption", "span", "div",
]

# ── Tags to strip entirely (scripts, styles, nav, boilerplate) ────────────────
NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "input", "select",
    "iframe", "noscript", "meta", "link", "svg",
    "advertisement", "ads",
]

# ── Patterns to remove from cleaned text ──────────────────────────────────────
NOISE_PATTERNS = [
    r"http[s]?://\S+",                    # URLs
    r"\S+@\S+\.\S+",                      # emails
    r"[\U00010000-\U0010ffff]",           # emoji / surrogate pairs
    r"\s{3,}",                            # 3+ consecutive spaces → single space
    r"\n{3,}",                            # 3+ consecutive newlines → double
]


class HTMLParser:
    """Cleans a raw HTML page and extracts structured text blocks."""

    def __init__(self):
        self._noise_re = [re.compile(p, re.UNICODE) for p in NOISE_PATTERNS]

    def parse(self, html: str, url: str, source_name: str, group: str) -> list[dict]:
        """
        Parse a single HTML page into a list of chunk dicts.

        Each chunk dict has:
            text        : cleaned text content
            url         : source URL
            source_name : config source identifier
            group       : human-readable category label
            title       : page <title>
            section     : nearest heading above this block
            language    : detected language ("ar" / "en" / "unknown")
            chunk_index : position within this page
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove all noise tags entirely
        for tag in NOISE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Extract page title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else url

        # Extract text blocks with their nearest heading
        blocks = self._extract_blocks(soup)

        if not blocks:
            logger.warning(f"No content extracted from {url}")
            return []

        # Join blocks into full text, then chunk
        full_text = "\n\n".join(blocks)
        full_text = self._clean_text(full_text)

        chunks = self._chunk_text(full_text)

        results = []
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) < MIN_CHUNK_LEN // 5:  # rough word count guard
                continue
            lang = self._detect_language(chunk)
            results.append({
                "text":         chunk,
                "url":          url,
                "source_name":  source_name,
                "group":        group,
                "title":        title,
                "language":     lang,
                "chunk_index":  i,
            })

        logger.debug(f"Parsed {len(results)} chunks from {url}")
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_blocks(self, soup: BeautifulSoup) -> list[str]:
        """Walk the DOM and collect non-empty text blocks from content tags."""
        blocks = []
        current_heading = ""

        for el in soup.find_all(CONTENT_TAGS):
            text = el.get_text(separator=" ", strip=True)
            if not text or len(text) < 10:
                continue

            # Track headings so they prefix following paragraphs
            if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                current_heading = text
                blocks.append(text)
            else:
                # Prepend heading context if available
                if current_heading:
                    blocks.append(f"{current_heading}: {text}")
                else:
                    blocks.append(text)

        return blocks

    def _clean_text(self, text: str) -> str:
        """Normalize unicode, remove noise patterns, collapse whitespace."""
        # Normalize Arabic text (NFC)
        text = unicodedata.normalize("NFC", text)

        # Apply noise regex patterns
        for pattern in self._noise_re:
            replacement = " " if r"\s" in pattern.pattern else ""
            text = pattern.sub(replacement, text)

        # Collapse whitespace
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks by word count.
        Uses a sliding window: step = CHUNK_SIZE - CHUNK_OVERLAP
        """
        words = text.split()
        if not words:
            return []

        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        chunks = []

        for start in range(0, len(words), step):
            end = start + CHUNK_SIZE
            chunk = " ".join(words[start:end])
            if len(chunk) >= MIN_CHUNK_LEN:
                chunks.append(chunk)
            if end >= len(words):
                break

        return chunks

    def _detect_language(self, text: str) -> str:
        """Detect language using langdetect. Returns 'ar', 'en', or 'unknown'."""
        try:
            lang = detect(text)
            if lang == "ar":
                return "ar"
            elif lang == "en":
                return "en"
            else:
                return lang
        except LangDetectException:
            return "unknown"


# ── Convenience function ──────────────────────────────────────────────────────

_parser_instance: Optional[HTMLParser] = None

def get_parser() -> HTMLParser:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = HTMLParser()
    return _parser_instance
