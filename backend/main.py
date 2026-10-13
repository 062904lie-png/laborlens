"""
LaborLens – FastAPI Entry Point
Run with:  python run.py   (from backend/ folder)
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from threading import Lock

from routers import auth, chat, admin, knowledge_base, faqs
from routers import live_voice
from routers.services.database import execute_one, init_db
from routers.services.chroma_service import get_collection_stats, init_chroma
from routers.services.performance import begin_request, end_request
from routers.services.provider_runtime import (
    configured_provider_order,
    health_for,
    provider_health_snapshot,
)
from routers.services import (
    chroma_service,
    gemini_service,
    openai_service,
    ingestion_service,
    legal_answer_validator,
    rag_policy,
)


logger = logging.getLogger(__name__)
READY_CHECK_TIMEOUT_SECONDS = 3
_READINESS_SEARCH_INDEX_PROBE_LOCK = Lock()


def _readiness_search_index_probe():
    """Keep timed-out Chroma probes from stacking up in the shared thread pool."""
    if not _READINESS_SEARCH_INDEX_PROBE_LOCK.acquire(blocking=False):
        raise RuntimeError("a previous search-index readiness probe is still running")
    try:
        return get_collection_stats()
    finally:
        # asyncio cannot stop a synchronous Chroma call after its awaiter times out.
        # Keep the lock until that worker actually finishes so later polls fail fast.
        _READINESS_SEARCH_INDEX_PROBE_LOCK.release()


def _module_path(module) -> str:
    file_name = getattr(module, "__file__", "")
    return str(Path(file_name).resolve()) if file_name else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from routers.services.backend_diagnostics import install_log_handler
    from routers.services.admin_runtime_settings import load_saved_settings
    install_log_handler()
    loaded_settings = load_saved_settings()
    from config import settings
    app.title = settings.APP_NAME
    if loaded_settings:
        logger.info('Loaded %d saved admin runtime settings', len(loaded_settings))
    print("Starting LaborLens...")
    logger.info(
        "LaborLens runtime files | main=%s | auth=%s | chat=%s | admin=%s | kb=%s | "
        "chroma=%s | rag_policy=%s | openai=%s | ingestion=%s | validator=%s",
        str(Path(__file__).resolve()),
        _module_path(auth),
        _module_path(chat),
        _module_path(admin),
        _module_path(knowledge_base),
        _module_path(chroma_service),
        _module_path(rag_policy),
        _module_path(openai_service),
        _module_path(ingestion_service),
        _module_path(legal_answer_validator),
    )
    await init_db()
    await init_chroma()
    await knowledge_base.recover_interrupted_replacements()
    print("Ready!")
    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LaborLens API",
        description="AI-Powered Labour and Employment Law Advisor",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(chroma_service.EmbeddingMismatchError)
    async def embedding_mismatch_response(request: Request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://laborlens-lac.vercel.app", "https://3.104.175.253"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        timing = token = None
        if request.method == "POST" and request.url.path in {
            "/api/chat/message",
            "/api/chat/guest-message",
        }:
            timing, token = begin_request()
        try:
            response = await call_next(request)
        finally:
            if timing is not None and token is not None:
                end_request(timing, token)
        if request.url.path.startswith("/api/chat/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.include_router(auth.router,           prefix="/api/auth",  tags=["Auth"])
    app.include_router(faqs.router,           prefix="/api/faqs",  tags=["FAQs"])
    app.include_router(chat.router,           prefix="/api/chat",  tags=["Chat"])
    app.include_router(admin.router,          prefix="/api/admin", tags=["Admin"])
    app.include_router(knowledge_base.router, prefix="/api/kb",    tags=["KB"])
    app.include_router(live_voice.router, prefix="/api/voice", tags=["Voice"])

    @app.get("/")
    async def root():
        return {"status": "running"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/ready")
    async def readiness():
        checks = {"database": "ok", "search_index": "ok"}

        try:
            result = await asyncio.wait_for(
                execute_one("SELECT 1 AS ready"),
                timeout=READY_CHECK_TIMEOUT_SECONDS,
            )
            if result is None:
                raise RuntimeError("database readiness query returned no row")
        except Exception as exc:
            checks["database"] = "unavailable"
            logger.warning(
                "Readiness check failed | dependency=database | error=%s",
                type(exc).__name__,
            )

        try:
            stats = await asyncio.wait_for(
                asyncio.to_thread(_readiness_search_index_probe),
                timeout=READY_CHECK_TIMEOUT_SECONDS,
            )
            if stats.get("total_chunks") is None:
                raise RuntimeError("search index readiness query returned no count")
        except Exception as exc:
            checks["search_index"] = "unavailable"
            logger.warning(
                "Readiness check failed | dependency=search_index | error=%s",
                type(exc).__name__,
            )

        is_ready = all(status == "ok" for status in checks.values())
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "not_ready", "checks": checks},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/status")
    async def public_status():
        for model_name in gemini_service.configured_gemini_models():
            health_for(gemini_service.gemini_health_key(model_name))
        doc_stats = await execute_one(
            """
            SELECT
                COUNT(*) AS total_documents,
                SUM(CASE WHEN status='indexed' THEN 1 ELSE 0 END) AS indexed_documents,
                SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) AS processing_documents,
                MAX(last_indexed_at) AS last_document_update
            FROM uploaded_documents
            """
        )
        chroma = get_collection_stats()
        return {
            "api": "online",
            # This endpoint does not make a billable/side-effecting model call,
            # so provider availability cannot be confirmed by a status read.
            "ai_service": "unverified",
            "providers": {
                "order": configured_provider_order(),
                "health": provider_health_snapshot(),
            },
            "knowledge_base": {
                "status": "available" if (chroma.get("total_chunks") or 0) > 0 else "empty",
                "total_chunks": chroma.get("total_chunks", 0),
                "embedding_model": chroma.get("embedding_model"),
                "last_reset_reason": chroma.get("last_reset_reason"),
                **(doc_stats or {}),
            },
        }

    return app


app = create_app()
