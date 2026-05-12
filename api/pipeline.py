"""
api/pipeline.py
===============
Singleton RAG pipeline — loads once at startup, shared across all requests.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config import RERANKER_ENABLED


class RAGPipeline:

    def __init__(self):
        logger.info("Initializing RAG pipeline...")
        from indexer.embedder import Embedder
        from indexer.faiss_store import get_store
        from rag.retriever import Retriever
        from rag.generator import Generator

        self.embedder  = Embedder()
        self.store     = get_store()
        self.retriever = Retriever()

        self.reranker = None
        if RERANKER_ENABLED:
            try:
                from rag.reranker import Reranker
                self.reranker = Reranker()
                logger.success("Reranker loaded")
            except Exception as e:
                logger.warning(f"Reranker failed to load (will run without it): {e}")

        self.generator = Generator()
        logger.success("RAG pipeline ready")

    def run(
        self,
        question: str,
        top_k: int = 5,
        session_history: str = "",
        use_hyde: bool = True,
        use_translation: bool = True,
        use_reranker: bool = True,
        thinking_mode: bool = False,
    ) -> dict:
        t0 = time.time()

        # ── Retrieve ──────────────────────────────────────────────
        chunks = self.retriever.retrieve(question, top_k=top_k)
        hyde_used = False  # set True below if HyDE ran

        # ── Rerank ────────────────────────────────────────────────
        reranker_used = False
        if self.reranker and use_reranker and chunks:
            chunks = self.reranker.rerank(question, chunks)
            reranker_used = True

        t1 = time.time()

        # ── Generate ──────────────────────────────────────────────
        result = self.generator.generate(
            question=question,
            context_chunks=chunks,
            history_block=session_history,
            thinking=thinking_mode,
        )
        t2 = time.time()

        # ── Build source dicts ────────────────────────────────────
        sources = [
            {
                "url":          c.get("url", ""),
                "source_name":  c.get("source_name", ""),
                "group":        c.get("group", ""),
                "language":     c.get("language", ""),
                "score":        float(c.get("score", 0)),
                "rerank_score": float(c.get("rerank_score", 0)),
                "search_type":  c.get("search_type", "raw"),
                "text_preview": c.get("text", "")[:150],
            }
            for c in chunks
        ]

        return {
            "answer":        result["answer"],
            "model_used":    result["model"],
            "sources":       sources,
            "chunks_used":   len(chunks),
            "hyde_used":     hyde_used,
            "reranker_used": reranker_used,
            "has_history":   bool(session_history),
            "latency_ms": {
                "retrieval":  int((t1 - t0) * 1000),
                "generation": int((t2 - t1) * 1000),
                "total":      int((t2 - t0) * 1000),
            },
        }


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline