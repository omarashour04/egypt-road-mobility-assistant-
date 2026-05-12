"""
api/main.py
===========
FastAPI application for the Egyptian Road & Mobility Assistant.

Endpoints:
    POST /ask                       — RAG question answering (with sessions)
    GET  /session/{id}/history      — View conversation history
    DELETE /session/{id}            — Delete a session
    POST /scrape                    — Trigger scraping and index rebuild
    GET  /health                    — Service health check
    GET  /stats                     — Index and session statistics
    GET  /docs                      — Swagger UI (auto-generated)
    GET  /redoc                     — ReDoc UI (auto-generated)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    API_TITLE, API_VERSION, API_DESCRIPTION,
    RERANKER_ENABLED,
)
from api.schemas import (
    AskRequest, AskResponse, LatencyInfo, SourceCitation,
    SessionHistoryResponse, SessionTurn, SessionDeleteResponse,
    ScrapeRequest, ScrapeResponse,
    HealthResponse, StatsResponse,
)
from api.pipeline import get_pipeline
from api.session import get_session_store
from indexer.faiss_store import get_store


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Egyptian Road & Mobility Assistant API...")
    try:
        get_pipeline()
        logger.success("Pipeline loaded")
    except Exception as e:
        logger.error(f"Pipeline startup failed: {e}")

    # Start session cleanup background task
    session_store = get_session_store()
    session_store.start_cleanup()
    logger.info("Session store started")

    yield

    session_store.stop_cleanup()
    logger.info("API shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# After app definition, before your endpoints:
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("demo_frontend.html")
# ── /ask ──────────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about Egyptian road law",
    description="""
Submit a question in **Arabic or English**. The system retrieves relevant
legal text using HyDE + cross-encoder reranking, then generates a grounded answer.

Pass the returned `session_id` in follow-up questions to maintain context.

**Covered domains:**
- Traffic laws and fines
- Driving licenses (new, renewal, replacement)
- Vehicle registration and ownership transfer
- Accident liability and insurance
- Commercial vehicles and taxis
- Driver fitness and medical requirements
- International driving permits
- Road infrastructure and speed cameras
""",
    tags=["QA"],
)
async def ask(request: AskRequest) -> AskResponse:
    pipeline      = get_pipeline()
    session_store = get_session_store()

    # Get or create session
    session = session_store.get_or_create(request.session_id)
    history_block = session.build_history_block()

    try:
        result = pipeline.run(
            question=request.question,
            top_k=request.top_k,
            session_history=history_block,
            use_hyde=request.use_hyde,
            use_translation=True,
            use_reranker=request.use_reranker,
            thinking_mode=request.thinking_mode,
        )
    except Exception as e:
        logger.error(f"/ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Record this turn in the session
    session_store.add_turn(
        session_id=session.session_id,
        question=request.question,
        answer=result["answer"],
    )

    return AskResponse(
        answer=result["answer"],
        session_id=session.session_id,
        sources=[SourceCitation(**s) for s in result["sources"]],
        chunks_used=result["chunks_used"],
        model_used=result["model_used"],
        hyde_used=result["hyde_used"],
        reranker_used=result["reranker_used"],
        has_history=result["has_history"],
        latency_ms=LatencyInfo(**result["latency_ms"]),
    )


# ── /session endpoints ────────────────────────────────────────────────────────

@app.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
    tags=["Sessions"],
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    session_store = get_session_store()
    turns = session_store.get_history(session_id)

    session = session_store.get_session(session_id)
    turn_count = session.turn_count if session else 0

    return SessionHistoryResponse(
        session_id=session_id,
        turn_count=turn_count,
        turns=[SessionTurn(**t) for t in turns],
    )


@app.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    summary="Delete a conversation session",
    tags=["Sessions"],
)
async def delete_session(session_id: str) -> SessionDeleteResponse:
    session_store = get_session_store()
    deleted = session_store.delete_session(session_id)
    return SessionDeleteResponse(session_id=session_id, deleted=deleted)


# ── /scrape ───────────────────────────────────────────────────────────────────

@app.post(
    "/scrape",
    response_model=ScrapeResponse,
    summary="Trigger web scraping and index rebuild",
    tags=["Admin"],
)
async def scrape(request: ScrapeRequest, background_tasks: BackgroundTasks) -> ScrapeResponse:
    """
    Re-scrapes the configured sources (optionally filtered by domain group)
    and rebuilds the FAISS index.

    This runs synchronously and may take several minutes.
    """
    try:
        from scraper.scraper import Scraper, save_chunks
        from indexer.faiss_store import FAISSStore
        from config import SOURCES

        scraper = Scraper()

        if request.group:
            sources_to_scrape = [s for s in SOURCES if s["group"] == request.group]
            if not sources_to_scrape:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown group '{request.group}'. "
                           f"Available: {sorted(set(s['group'] for s in SOURCES))}"
                )
            chunks = []
            for source in sources_to_scrape:
                chunks.extend(scraper.scrape_source(source))
        else:
            chunks = scraper.scrape_all()

        save_chunks(chunks)

        index_rebuilt = False
        if request.rebuild_index and chunks:
            store = FAISSStore()
            store.build(chunks)
            store.save()
            index_rebuilt = True

        return ScrapeResponse(
            status="success",
            chunks_scraped=len(chunks),
            index_rebuilt=index_rebuilt,
            message=f"Scraped {len(chunks)} chunks. Index rebuilt: {index_rebuilt}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/scrape failed: {e}")
        return ScrapeResponse(
            status="error",
            chunks_scraped=0,
            index_rebuilt=False,
            message=str(e),
        )


# ── /health ───────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    session_store = get_session_store()
    index_size = 0
    index_loaded = False

    try:
        store = get_store()
        index_size   = store.index.ntotal if store.index else 0
        index_loaded = store.index is not None
    except Exception:
        pass

    pipeline_ok = False
    model_name  = ""
    try:
        pipeline   = get_pipeline()
        pipeline_ok = True
        model_name  = pipeline.generator.model
    except Exception:
        pass

    reranker_loaded = False
    if RERANKER_ENABLED:
        try:
            from rag.reranker import _reranker_instance
            reranker_loaded = _reranker_instance is not None
        except Exception:
            pass

    status = "ok" if (index_loaded and pipeline_ok) else "degraded"

    return HealthResponse(
        status=status,
        index_loaded=index_loaded,
        index_size=index_size,
        model_loaded=pipeline_ok,
        model_name=model_name,
        reranker_loaded=reranker_loaded,
        active_sessions=session_store.active_session_count,
    )


# ── /stats ────────────────────────────────────────────────────────────────────

@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Index and session statistics",
    tags=["System"],
)
async def stats() -> StatsResponse:
    session_store = get_session_store()

    try:
        store      = get_store()
        index_stats = store.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Compute per-group counts from metadata
    group_counts: dict[str, int] = {}
    for meta in store.metadata:
        g = meta.get("group", "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1

    return StatsResponse(
        total_vectors=index_stats.get("total_vectors", 0),
        dimension=index_stats.get("dimension", 0),
        sources=index_stats.get("sources", {}),
        groups=group_counts,
        languages=index_stats.get("languages", {}),
        active_sessions=session_store.active_session_count,
    )
