"""
api/schemas.py
==============
Pydantic v2 request/response models. These drive the Swagger UI docs.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# Suppress Pydantic v2 protected-namespace warnings for model_* fields
class _Base(BaseModel):
    model_config = {"protected_namespaces": ()}


# ── /ask endpoint ─────────────────────────────────────────────────────────────

class AskRequest(_Base):
    question: str = Field(
        description="Your question in Arabic or English",
        examples=[
            "ما هي غرامة استخدام الموبايل أثناء القيادة؟",
            "What are the steps to renew my driving license?",
            "How do I transfer car ownership in Egypt?",
            "ما هي إجراءات نقل ملكية السيارة؟",
        ],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID from a previous /ask response. "
            "Pass this to maintain conversation context across multiple questions. "
            "Omit to start a new session."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of source chunks to retrieve (default 5)",
    )
    use_hyde: bool = Field(
        default=True,
        description=(
            "Enable HyDE (Hypothetical Document Embeddings). "
            "Generates a hypothetical answer first, then searches with it. "
            "Significantly improves retrieval for Arabic colloquial queries."
        ),
    )
    use_reranker: bool = Field(
        default=True,
        description=(
            "Enable cross-encoder reranking of retrieved chunks. "
            "More accurate than bi-encoder similarity alone."
        ),
    )
    thinking_mode: bool = Field(
        default=False,
        description=(
            "Enable Qwen3 chain-of-thought reasoning. "
            "Slower but better for complex multi-part questions."
        ),
    )


class SourceCitation(_Base):
    url:          str   = Field(description="Source URL")
    source_name:  str   = Field(description="Internal source identifier")
    group:        str   = Field(description="Domain group (e.g. 'driving_license')")
    language:     str   = Field(description="'ar' or 'en'")
    score:        float = Field(description="FAISS cosine similarity score")
    rerank_score: float = Field(description="Cross-encoder rerank score (0 if reranker disabled)")
    search_type:  str   = Field(description="How this chunk was found: raw / hyde / translation")
    text_preview: str   = Field(description="First 150 chars of the chunk text")


class LatencyInfo(_Base):
    retrieval:  int = Field(description="Retrieval time in milliseconds")
    generation: int = Field(description="Generation time in milliseconds")
    total:      int = Field(description="Total end-to-end time in milliseconds")


class AskResponse(_Base):
    answer:        str              = Field(description="Generated answer")
    session_id:    str              = Field(description="Session ID — pass back in next request to continue conversation")
    sources:       list[SourceCitation] = Field(description="Source chunks used for generation")
    chunks_used:   int              = Field(description="Number of chunks passed to the generator")
    model_used:    str              = Field(description="Generator model name")
    hyde_used:     bool             = Field(description="Whether HyDE was applied")
    reranker_used: bool             = Field(description="Whether the cross-encoder reranker was applied")
    has_history:   bool             = Field(description="Whether conversation history was included")
    latency_ms:    LatencyInfo      = Field(description="Latency breakdown in milliseconds")


# ── /session endpoints ────────────────────────────────────────────────────────

class SessionTurn(_Base):
    question:  str = Field(description="User question")
    answer:    str = Field(description="Assistant answer")
    timestamp: str = Field(description="ISO timestamp of this turn")


class SessionHistoryResponse(_Base):
    session_id:  str              = Field(description="Session ID")
    turn_count:  int              = Field(description="Number of turns in this session")
    turns:       list[SessionTurn] = Field(description="All turns in this session")


class SessionDeleteResponse(_Base):
    session_id: str  = Field(description="Session ID")
    deleted:    bool = Field(description="True if the session existed and was deleted")


# ── /scrape endpoint ──────────────────────────────────────────────────────────

class ScrapeRequest(_Base):
    group: Optional[str] = Field(
        default=None,
        description=(
            "Scrape only sources in this domain group. "
            "Available: traffic_law, driving_license, vehicle_registration, "
            "accident_liability, commercial_vehicles, driver_fitness, "
            "international_driving, road_infrastructure. "
            "Omit to scrape all domains."
        ),
    )
    rebuild_index: bool = Field(
        default=True,
        description="Rebuild the FAISS index after scraping",
    )


class ScrapeResponse(_Base):
    status:        str  = Field(description="'success' or 'error'")
    chunks_scraped: int = Field(description="Number of new chunks added")
    index_rebuilt:  bool = Field(description="Whether the FAISS index was rebuilt")
    message:        str  = Field(description="Human-readable status message")


# ── /health endpoint ──────────────────────────────────────────────────────────

class HealthResponse(_Base):
    status:          str  = Field(description="'ok' or 'degraded'")
    index_loaded:    bool = Field(description="Is the FAISS index loaded?")
    index_size:      int  = Field(description="Number of vectors in the index")
    model_loaded:    bool = Field(description="Is the generator model loaded?")
    model_name:      str  = Field(description="Generator model name")
    reranker_loaded: bool = Field(description="Is the cross-encoder reranker loaded?")
    active_sessions: int  = Field(description="Number of active conversation sessions")


# ── /stats endpoint ───────────────────────────────────────────────────────────

class StatsResponse(_Base):
    total_vectors:   int  = Field(description="Total vectors in FAISS index")
    dimension:       int  = Field(description="Embedding dimension")
    sources:         dict = Field(description="Chunk count per source name")
    groups:          dict = Field(description="Chunk count per domain group")
    languages:       dict = Field(description="Chunk count per language")
    active_sessions: int  = Field(description="Number of live conversation sessions")
