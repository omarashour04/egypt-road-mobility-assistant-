"""
rag/retriever.py
================
Retrieval layer — the upgraded version that wires together:
  1. QueryEnhancer  → HyDE hypothesis + bilingual translation
  2. FAISSStore     → vector search (raw query + hypothesis + translation)
  3. Reranker       → cross-encoder re-scoring of merged candidates

The retriever is the single entry point for the pipeline; it hides all this
complexity from the generator and the API layer.

Retrieval flow:
    query
      ├─► HyDE hypothesis ──────────► FAISS search (fetch_k results)
      ├─► bilingual translation ────► FAISS search (fetch_k results)
      └─► raw query ────────────────► FAISS search (fetch_k results)
                                              │
                                       merge + deduplicate
                                              │
                                       cross-encoder rerank
                                              │
                                        top_k chunks ──► generator
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TOP_K,
    MIN_SCORE,
    RERANKER_ENABLED,
    RERANKER_FETCH_K,
    HYDE_ENABLED,
)
from indexer.faiss_store import get_store
from indexer.embedder import get_embedder


class Retriever:
    """
    Full retrieval pipeline: enhance → search → rerank → return top_k chunks.
    """

    def __init__(self):
        self.store    = get_store()
        self.embedder = get_embedder()

        # Lazy-load optional components
        self._enhancer = None
        self._reranker = None

        logger.info("Retriever initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        use_hyde: bool = True,
        use_translation: bool = True,
        use_reranker: bool = True,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:           Raw user question (Arabic or English)
            top_k:           Final number of chunks to return
            use_hyde:        Whether to use HyDE (requires HYDE_ENABLED in config)
            use_translation: Whether to expand with bilingual translation
            use_reranker:    Whether to apply cross-encoder reranking

        Returns:
            List of chunk dicts sorted by relevance, length = top_k (or fewer
            if the index doesn't have enough candidates above MIN_SCORE).
        """
        normalized_query = self._normalize(query)
        fetch_k = RERANKER_FETCH_K if (use_reranker and RERANKER_ENABLED) else top_k

        # ── Step 1: build a set of queries to search ──────────────────────────
        search_queries: list[tuple[str, str]] = [
            (normalized_query, "raw"),
        ]

        if use_hyde and HYDE_ENABLED:
            enhancer  = self._get_enhancer()
            hypothesis = enhancer.get_hyde_query(query)
            if hypothesis:
                search_queries.append((self._normalize(hypothesis), "hyde"))

        if use_translation:
            if self._get_enhancer() is None:
                pass  # enhancer already loaded above or will be loaded now
            enhancer    = self._get_enhancer()
            translation = enhancer.get_translation(query)
            if translation:
                search_queries.append((self._normalize(translation), "translation"))

        # ── Step 2: search FAISS with each query, merge results ───────────────
        seen_ids: set[str] = set()
        all_candidates: list[dict] = []

        for search_text, search_type in search_queries:
            results = self.store.search(
                query=search_text,
                top_k=fetch_k,
                min_score=MIN_SCORE,
            )
            for chunk in results:
                chunk_id = chunk.get("chunk_id") or f"{chunk.get('url','')}_{chunk.get('chunk_index','')}"
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    chunk["search_type"] = search_type   # for debugging
                    all_candidates.append(chunk)

        logger.debug(
            f"Retriever: {len(search_queries)} search(es) → "
            f"{len(all_candidates)} unique candidates"
        )

        if not all_candidates:
            logger.warning("Retriever: no candidates found above MIN_SCORE")
            return []

        # ── Step 3: rerank ────────────────────────────────────────────────────
        if use_reranker and RERANKER_ENABLED and len(all_candidates) > top_k:
            reranker = self._get_reranker()
            final    = reranker.rerank(query, all_candidates, top_k=top_k)
        else:
            # Sort by original FAISS score and truncate
            final = sorted(all_candidates, key=lambda c: c.get("score", 0), reverse=True)[:top_k]

        logger.debug(
            f"Retriever: returning {len(final)} chunks | "
            f"top source: {final[0].get('source_name','?') if final else 'none'}"
        )
        return final

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_enhancer(self):
        if self._enhancer is None:
            from rag.query_enhancer import get_query_enhancer
            self._enhancer = get_query_enhancer()
        return self._enhancer

    def _get_reranker(self):
        if self._reranker is None:
            from rag.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Light normalization for Arabic/English text before embedding.
        - NFC unicode normalization
        - Strip excessive whitespace
        - Normalize Arabic letter forms (alef variants → bare alef)
        """
        text = unicodedata.normalize("NFC", text)
        text = " ".join(text.split())

        # Arabic letter normalization — alef variants
        ar_norm = {
            "\u0622": "\u0627",  # آ → ا
            "\u0623": "\u0627",  # أ → ا
            "\u0625": "\u0627",  # إ → ا
            "\u0671": "\u0627",  # ٱ → ا
            "\u0649": "\u064a",  # ى → ي
            "\u0629": "\u0647",  # ة → ه
        }
        for src, dst in ar_norm.items():
            text = text.replace(src, dst)

        return text


# ── Singleton ─────────────────────────────────────────────────────────────────
_retriever_instance: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
