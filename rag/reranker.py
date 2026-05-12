"""
rag/reranker.py
===============
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 via sentence-transformers.
Reranks retrieved candidates so the most relevant chunks surface to the top.

No Ollama dependency here — runs entirely via sentence-transformers on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config import RERANKER_MODEL_NAME, RERANKER_TOP_K


class Reranker:
    """
    Wraps a CrossEncoder model for reranking retrieved chunks.
    The model is loaded from HuggingFace / local cache on first init.
    """

    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading cross-encoder reranker: {RERANKER_MODEL_NAME}")
        self.model      = CrossEncoder(RERANKER_MODEL_NAME)
        self.top_k      = RERANKER_TOP_K
        self.model_name = RERANKER_MODEL_NAME
        logger.success(f"Reranker ready: {RERANKER_MODEL_NAME}")

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
        """
        Score each chunk against the query using the cross-encoder,
        then sort descending by rerank score and return top_k.

        Each chunk dict must have a 'text' key.
        The rerank_score is added in-place to each chunk dict.
        """
        if not chunks:
            return chunks

        k      = top_k if top_k is not None else self.top_k
        pairs  = [(query, c.get("text", "")) for c in chunks]
        scores = self.model.predict(pairs).tolist()

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:k]


# ── Singleton ─────────────────────────────────────────────────────────────────
_reranker_instance: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance