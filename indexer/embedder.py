"""
indexer/embedder.py
===================
Encodes text chunks into dense vectors using Ollama's embedding endpoint.
Handles both old (dict) and new (object) Ollama Python client response formats.
"""

import numpy as np
from pathlib import Path
from typing import Union

import ollama
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBED_MODEL_NAME, EMBED_BATCH_SIZE, FAISS_DIM, OLLAMA_BASE_URL


def _extract_embeddings(resp) -> list:
    """
    Extract embedding list from Ollama response.
    Handles both response formats:
      - New client: resp.embeddings  (object with attribute)
      - Old client: resp['embeddings'] (plain dict)
    """
    if hasattr(resp, "embeddings"):
        return resp.embeddings
    elif isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"]
    elif isinstance(resp, dict) and "embedding" in resp:
        # Some versions return single 'embedding' key
        return [resp["embedding"]]
    else:
        raise ValueError(f"Cannot extract embeddings from response: {type(resp)} — {resp}")


class Embedder:

    def __init__(self):
        logger.info(f"Initializing Ollama embedder: {EMBED_MODEL_NAME}")
        self.model  = EMBED_MODEL_NAME
        self.dim    = FAISS_DIM
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            resp = self.client.embed(model=self.model, input=["test"])
            embeddings = _extract_embeddings(resp)
            actual_dim = len(embeddings[0])
            if actual_dim != self.dim:
                logger.warning(
                    f"Embedding dim mismatch: model={actual_dim}, config={self.dim}. "
                    f"Auto-correcting FAISS_DIM to {actual_dim}."
                )
                import config as cfg
                cfg.FAISS_DIM = actual_dim
                self.dim = actual_dim
            logger.success(f"Ollama embedder ready: {self.model} (dim={self.dim})")
        except ollama.ResponseError as e:
            if "not found" in str(e).lower():
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Run: ollama pull {self.model}"
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
                f"Make sure Ollama is running. Error: {e}"
            ) from e

    def encode(self, texts: Union[str, list[str]], show_progress: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        batches = range(0, len(texts), EMBED_BATCH_SIZE)
        if show_progress:
            batches = tqdm(batches, desc="Encoding chunks")

        for start in batches:
            batch = texts[start : start + EMBED_BATCH_SIZE]
            resp  = self.client.embed(model=self.model, input=batch)
            batch_embs = np.array(_extract_embeddings(resp), dtype=np.float32)
            all_embeddings.append(batch_embs)

        matrix = np.vstack(all_embeddings)
        norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms  = np.where(norms == 0, 1, norms)
        return (matrix / norms).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode(query).reshape(1, -1)


_embedder_instance: Embedder = None

def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
