"""
indexer/faiss_store.py
======================
Builds a FAISS flat inner-product index, saves/loads it from disk,
and provides a search method returning top-K chunks with scores.

Using faiss-cpu — GPU is handled by Ollama for the models themselves.
FAISS index operations are fast enough on CPU for our corpus size (~600 chunks).

Run directly to build the index from scraped chunks:
    python -m indexer.faiss_store
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH, FAISS_METADATA_PATH,
    FAISS_DIM, EMBED_BATCH_SIZE, SCRAPED_DIR, TOP_K, MIN_SCORE,
)
from indexer.embedder import get_embedder


class FAISSStore:
    """
    Manages the FAISS vector index and its associated chunk metadata.

    Index type: IndexFlatIP (flat inner product)
    - Exact search — correct for our corpus size (< 100K vectors)
    - With L2-normalized vectors: inner product = cosine similarity
    - Scores in [-1, 1]; higher = more similar
    """

    def __init__(self):
        self.index: faiss.Index = None
        self.metadata: list[dict] = []
        self.embedder = get_embedder()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """
        Encode all chunks and build the FAISS index.
        chunks must each have a 'text' key.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        logger.info(f"Building FAISS index from {len(chunks)} chunks...")

        texts = [c["text"] for c in chunks]

        # Encode in batches — show progress bar
        all_embeddings = []
        for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE),
                      desc="Encoding chunks via Ollama"):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            embs  = self.embedder.encode(batch)
            all_embeddings.append(embs)

        matrix = np.vstack(all_embeddings).astype(np.float32)

        # Rebuild dim from actual embedding output in case it differs from config
        actual_dim = matrix.shape[1]
        if actual_dim != FAISS_DIM:
            logger.warning(
                f"Using actual embedding dim {actual_dim} instead of "
                f"config FAISS_DIM={FAISS_DIM}"
            )

        self.index    = faiss.IndexFlatIP(actual_dim)
        self.metadata = chunks
        self.index.add(matrix)

        logger.success(
            f"Index built: {self.index.ntotal} vectors, dim={actual_dim}"
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save index and metadata to disk."""
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
            for chunk in self.metadata:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        logger.success(f"Index  → {FAISS_INDEX_PATH}")
        logger.success(f"Metadata → {FAISS_METADATA_PATH}")

    def load(self) -> None:
        """Load index and metadata from disk."""
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}.\n"
                "Run the indexer first:  python -m indexer.faiss_store"
            )

        logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        self.metadata = []
        with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        logger.success(
            f"Index loaded: {self.index.ntotal} vectors, "
            f"{len(self.metadata)} metadata records"
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> list[dict]:
        """
        Search for the most relevant chunks for a query.

        Returns list of chunk dicts (text, url, group, title, language,
        chunk_index, score) sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        query_vec = self.embedder.encode_query(query)

        # Over-fetch then filter by min_score
        scores, indices = self.index.search(query_vec, top_k * 2)
        scores  = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            if float(score) < min_score:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)
            if len(results) == top_k:
                break

        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if self.index is None:
            return {"status": "not loaded"}

        source_counts, lang_counts = {}, {}
        for m in self.metadata:
            src  = m.get("source_name", "unknown")
            lang = m.get("language", "unknown")
            source_counts[src]  = source_counts.get(src, 0) + 1
            lang_counts[lang]   = lang_counts.get(lang, 0) + 1

        return {
            "total_vectors": self.index.ntotal,
            "dimension":     self.index.d,
            "sources":       source_counts,
            "languages":     lang_counts,
        }


# ── Singleton accessor ────────────────────────────────────────────────────────
_store_instance: FAISSStore = None

def get_store() -> FAISSStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = FAISSStore()
        _store_instance.load()
    return _store_instance


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks_path = SCRAPED_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        logger.error(
            f"Chunks file not found: {chunks_path}\n"
            "Run the scraper first:  python -m scraper.scraper"
        )
        sys.exit(1)

    logger.info(f"Loading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks")

    store = FAISSStore()
    store.build(chunks)
    store.save()

    logger.info(json.dumps(store.stats(), ensure_ascii=False, indent=2))
