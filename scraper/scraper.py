"""
scraper/scraper.py
==================
Web scraper for Egyptian traffic law sources.

Key design decisions:
- Per-site CSS selectors to extract ONLY the article body, not navigation,
  sidebars, footers, or related-articles sections. This is critical — scraping
  full page text produces garbage chunks that confuse the retriever.
- Automatic fallback selector chain if the primary selector yields too little.
- Min content length filter to discard pages that returned nav-only content.
- Respects SCRAPE_DELAY_SEC between requests.
"""

import logging
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import (
    MIN_CHUNK_LEN,
    SCRAPE_DELAY_SEC,
    SCRAPE_HEADERS,
    SCRAPE_MAX_DEPTH,
    SCRAPE_MAX_RETRIES,
    SCRAPE_TIMEOUT_SEC,
    SCRAPED_DIR,
    SOURCES,
)

logger = logging.getLogger(__name__)

# ── Per-domain article body selectors ────────────────────────────────────────
# Order matters: first match with sufficient text wins.
# Each entry is (css_selector, min_words_to_accept).
# The final fallback ("body", 50) catches everything else.

SITE_SELECTORS: dict[str, list[tuple[str, int]]] = {
    "youm7.com": [
        ("div.content-body",          80),
        ("div.story-body",            80),
        ("article.content",           80),
        ("div#content-body",          80),
    ],
    "masrawy.com": [
        ("div.story-content",         80),
        ("div.article-content",       80),
        ("div.content-text",          80),
        ("div#article-body",          80),
    ],
    "elbalad.news": [
        ("div.article-content",       80),
        ("div.story-body",            80),
        ("div.post-content",          80),
    ],
    "dostor.org": [
        ("div.article-content",       80),
        ("div.news-body",             80),
        ("div.content-area",          80),
    ],
    "albawabhnews.com": [
        ("div.article-body",          80),
        ("div.story-content",         80),
    ],
    "mohamah.net": [
        ("div.entry-content",         80),
        ("div.post-content",          80),
        ("article",                   80),
    ],
    "tahiamasr.com": [
        ("div.article-content",       80),
        ("div.news-content",          80),
        ("div.post-content",          80),
        ("div.content-body",          80),
    ],
    "elwatannews.com": [
        ("div.article-body",          80),
        ("div.content-text",          80),
        ("div.news-details",          80),
    ],
    "almasryalyoum.com": [
        ("div.article-content",       80),
        ("div.story-body",            80),
        ("div.news-text",             80),
    ],
    "zyadda.com": [
        ("div.entry-content",         80),
        ("article",                   80),
        ("div.post-content",          80),
    ],
    "adcidl.com": [
        ("div.content",               50),
        ("main",                      50),
        ("article",                   50),
    ],
    "expatfocus.com": [
        ("div.article-content",       50),
        ("div.guide-content",         50),
        ("main article",              50),
        ("div.content",               50),
    ],
    "internationaldrivingpermit.org": [
        ("div.entry-content",         50),
        ("article",                   50),
        ("main",                      50),
    ],
    "wikiwand.com": [
        ("div.article-content",       50),
        ("div#article-content",       50),
        ("section.article__body",     50),
        ("div.page-content",          50),
    ],
}

# Generic fallback selector chain tried for any domain not in SITE_SELECTORS
GENERIC_SELECTORS: list[tuple[str, int]] = [
    ("article",                       80),
    ("div.article-content",           80),
    ("div.entry-content",             80),
    ("div.post-content",              80),
    ("div.content-body",              80),
    ("div.story-body",                80),
    ("main",                          60),
    ("div#main-content",              60),
]

# Tags whose text is always removed regardless of selector (noise sources)
NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", ".related-articles", ".social-share",
    ".tags", ".breadcrumb", ".pagination", ".comments",
    ".advertisement", ".ads", "noscript", "iframe",
]


def _get_domain(url: str) -> str:
    """Extract base domain (e.g. 'youm7.com') from a URL."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    # strip www. prefix
    return hostname.removeprefix("www.")


def _clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove noise tags from soup in-place."""
    for selector in NOISE_TAGS:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def _extract_text(soup: BeautifulSoup, url: str) -> str:
    """
    Extract article body text using per-site selectors.
    Falls back through generic selectors, then full body.
    """
    domain = _get_domain(url)
    selectors = SITE_SELECTORS.get(domain, []) + GENERIC_SELECTORS

    # Clean noise tags first
    _clean_soup(soup)

    for css_selector, min_words in selectors:
        try:
            el = soup.select_one(css_selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                words = len(text.split())
                if words >= min_words:
                    logger.debug(f"  Selector '{css_selector}' matched: {words} words")
                    return text
        except Exception:
            continue

    # Last resort: full body text (likely to include sidebar garbage)
    logger.warning(f"  No selector matched for {url} — using full body (may include noise)")
    body_text = soup.get_text(separator="\n", strip=True)
    return body_text


def _fetch(url: str, session: requests.Session) -> Optional[str]:
    """Fetch a URL and return article text, or None on failure."""
    for attempt in range(1, SCRAPE_MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=SCRAPE_HEADERS, timeout=SCRAPE_TIMEOUT_SEC)
            if resp.status_code == 403:
                logger.warning(f"403 Forbidden: {url} — skipping")
                return None
            if resp.status_code == 404:
                logger.warning(f"404 Not Found: {url} — skipping")
                return None
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            text = _extract_text(soup, url)

            # Reject if text is too short to be real content
            if len(text.strip()) < MIN_CHUNK_LEN * 2:
                logger.warning(f"  Content too short ({len(text)} chars) for {url} — skipping")
                return None

            return text

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt}/{SCRAPE_MAX_RETRIES}: {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt}/{SCRAPE_MAX_RETRIES}: {url} — {e}")

        if attempt < SCRAPE_MAX_RETRIES:
            time.sleep(SCRAPE_DELAY_SEC * attempt)  # exponential-ish back-off

    return None


def _get_links(soup: BeautifulSoup, base_url: str, domain_filter: str) -> list[str]:
    """Extract same-domain links from a page for follow_links=True sources."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if _get_domain(full) == domain_filter:
            links.append(full)
    return list(set(links))


def scrape_all() -> list[dict]:
    """
    Scrape all sources defined in config.SOURCES.

    Returns:
        List of dicts: {
            "text":        str,
            "source_url":  str,
            "source_name": str,
            "group":       str,
            "language":    str,
        }
    """
    session = requests.Session()
    session.headers.update(SCRAPE_HEADERS)

    all_docs: list[dict] = []
    seen_urls: set[str] = set()

    for source in SOURCES:
        name        = source["name"]
        urls        = source["urls"]
        group       = source["group"]
        language    = source["language"]
        follow      = source.get("follow_links", False)
        max_depth   = source.get("max_depth", SCRAPE_MAX_DEPTH)

        logger.info(f"Scraping source: {name} ({len(urls)} seed URLs)")

        to_visit: list[tuple[str, int]] = [(u, 0) for u in urls]

        while to_visit:
            url, depth = to_visit.pop(0)
            if url in seen_urls:
                continue
            seen_urls.add(url)

            logger.info(f"  [{depth}] {url}")
            time.sleep(SCRAPE_DELAY_SEC)

            text = _fetch(url, session)
            if not text:
                continue

            all_docs.append({
                "text":        text,
                "source_url":  url,
                "source_name": name,
                "group":       group,
                "language":    language,
            })
            logger.info(f"  ✓ {len(text.split())} words extracted")

            # Follow links if configured and within depth limit
            if follow and depth < max_depth:
                try:
                    resp = session.get(url, headers=SCRAPE_HEADERS, timeout=SCRAPE_TIMEOUT_SEC)
                    soup = BeautifulSoup(resp.text, "html.parser")
                    domain = _get_domain(url)
                    links = _get_links(soup, url, domain)
                    for link in links:
                        if link not in seen_urls:
                            to_visit.append((link, depth + 1))
                    logger.debug(f"  Found {len(links)} follow links at depth {depth}")
                except Exception as e:
                    logger.warning(f"  Could not extract links from {url}: {e}")

    logger.info(f"Scraping complete: {len(all_docs)} documents from {len(seen_urls)} URLs")
    return all_docs