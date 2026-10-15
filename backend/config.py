"""
LaborLens – Configuration
Reads all settings from .env file.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os


BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # App
    APP_NAME: str = "LaborLens"
    DEBUG: bool = False
    LABORLENS_DIAGNOSTICS_ENABLED: bool = True
    LABORLENS_DIAGNOSTICS_PATH: str = ''
    LABORLENS_DIAGNOSTICS_MAX_EVENTS: int = 10000
    LABORLENS_DIAGNOSTICS_RETENTION_DAYS: int = 30
    SECRET_KEY: str = "change-me"

    # Groq
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-6-luna"
    LABORLENS_OPENAI_ENABLED: bool = True
    LABORLENS_MODEL_POOLS_ENABLED: bool = True
    LABORLENS_LLM_RERANK_ENABLED: bool = True
    LABORLENS_CONDITIONAL_RERANK_ENABLED: bool = True
    # Local relevance signals, never evidence support. Margin corresponds to
    # the existing full term-overlap bonus; confidence reuses the source floor.
    LABORLENS_RERANK_LOCAL_MARGIN: float = 0.10
    LABORLENS_RERANK_LEXICAL_AGREEMENT: float = 0.60
    # All three conditions are required before bypassing the model reranker.
    LABORLENS_RERANK_SKIP_MIN_VECTOR_SCORE: float = 0.72
    LABORLENS_RERANK_SKIP_MIN_VECTOR_MARGIN: float = 0.10
    LABORLENS_RERANK_SKIP_MIN_CONTENT_OVERLAP: int = 3
    LABORLENS_LIGHT_PROMPT_ENABLED: bool = True
    LABORLENS_LIGHT_QUESTION_CHAR_LIMIT: int = 160
    # Explicitly authorized extra provider call: current question only, no
    # documents/history; optional and bounded, with original search retained.
    LABORLENS_QUERY_REFORMULATION_ENABLED: bool = True
    LABORLENS_RERANKER_POOL: str = (
        "openai/gpt-6-luna,gemini/gemini-3.5-flash-lite,gemini/gemini-3.1-flash-lite,"
        "groq/openai/gpt-oss-120b,groq/openai/gpt-oss-20b"
    )
    # Five OpenAI generators share load; Gemini is fallback-only.
    LABORLENS_GENERATOR_POOL: str = (
        "openai/gpt-6-luna,openai/gpt-5.6-luna,"
        "openai/gpt-5-nano,openai/gpt-4.1-nano,openai/gpt-5.4-nano,"
        "gemini/gemini-3.5-flash-lite,gemini/gemini-3.1-flash-lite"
    )
    # Process-local scheduling: fill idle primary models before reusing busy ones.
    # Optional finite caps can wait within the deadline; provider quotas apply.
    # Zero means no app-local generation concurrency cap.
    LABORLENS_GENERATION_CONCURRENCY_PER_MODEL: int = 0
    LABORLENS_GENERATION_QUEUE_WAIT_SECONDS: float = 2.0
    LABORLENS_GENERATION_QUEUE_MAX_WAITERS: int = 64
    # Long-query reformulation is a separate optional task; keep it on Gemini/Groq.
    LABORLENS_QUERY_POOL: str = (
        "gemini/gemini-3.5-flash-lite,gemini/gemini-3.1-flash-lite,"
        "groq/openai/gpt-oss-120b,groq/openai/gpt-oss-20b"
    )
    LABORLENS_DEFAULT_MODEL_RPM: int = 5
    # The five requested OpenAI generators bypass app-local minute caps;
    # these settings apply to the other models. Provider quotas still apply.
    LABORLENS_MODEL_RPM_LIMITS: str = "{}"
    # Provider quotas are authoritative. Local token guards stay off unless a
    # deployment intentionally configures its own conservative limits.
    LABORLENS_MODEL_TPM_LIMITS: str = "{}"
    LABORLENS_RERANK_TIMEOUT_SECONDS: float = 24.0
    LABORLENS_RERANK_ATTEMPT_SECONDS: float = 10.0
    LABORLENS_RERANK_REQUEST_TOKEN_BUDGET: int = 6500
    LABORLENS_QUERY_REQUEST_TOKEN_BUDGET: int = 16000
    LABORLENS_RERANK_PASSAGE_CHAR_LIMIT: int = 900
    # Optional ranking (including batches/backups) must not monopolize quota.
    LABORLENS_RERANK_MAX_MODEL_CALLS: int = 5
    LABORLENS_SECTION_CONTEXT_CHAR_LIMIT: int = 3500
    LABORLENS_SECTION_CONTEXT_TOKEN_LIMIT: int = 1200
    LABORLENS_SECTION_CONTEXT_MAX_PARTS: int = 32
    # Checking an answer is independent of optional ranking. A failed check
    # must never publish unchecked legal prose.
    LABORLENS_ANSWER_VERIFY_ENABLED: bool = True
    # Normal chat now uses source-based generation without the legacy audit.
    LABORLENS_CHAT_REQUEST_SECONDS: float = 25.0
    # Explicit temporary experiment. Disabled by default; never implies checked support.
    LABORLENS_ANSWER_VERIFY_TEST_BYPASS: bool = False
    LABORLENS_VERIFY_TIMEOUT_SECONDS: float = 24.0
    LABORLENS_VERIFY_ATTEMPT_SECONDS: float = 12.0
    # One bounded recovery/repair allowance and one final check, shared across
    # retries. They never reset per model or authorize an unchecked answer.
    LABORLENS_ANSWER_RECOVERY_SECONDS: float = 20.0
    # Same ordered backup chain as reranking; quotas remain shared by model.
    LABORLENS_VERIFIER_POOL: str = (
        "openai/gpt-6-luna,gemini/gemini-3.5-flash-lite,gemini/gemini-3.1-flash-lite,"
        "groq/openai/gpt-oss-120b,groq/openai/gpt-oss-20b"
    )
    LABORLENS_VERIFY_REQUEST_TOKEN_BUDGET: int = 12000
    LABORLENS_VERIFY_SIMPLE_OUTPUT_TOKENS: int = 800
    LABORLENS_VERIFY_NORMAL_OUTPUT_TOKENS: int = 1200
    LABORLENS_VERIFY_COMPLEX_OUTPUT_TOKENS: int = 4000
    LABORLENS_VERIFY_SIMPLE_MAX_CLAIMS: int = 3
    LABORLENS_VERIFY_NORMAL_MAX_CLAIMS: int = 8
    LABORLENS_SIMPLE_GENERATION_MAX_CALLS: int = 7
    LABORLENS_SIMPLE_VERIFY_MAX_CALLS: int = 5
    # Useful time per generation attempt, not the total divided by pool size.
    LABORLENS_GENERATION_ATTEMPT_SECONDS: float = 12.0
    LABORLENS_MODEL_BACKUP_RESERVE_SECONDS: float = 4.0
    LABORLENS_LIVE_MODEL: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    LABORLENS_LIVE_ENABLED: bool = False
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_MAX_TOKENS: int = 2048
    # Combined estimated input/output budget with headroom below the account TPM cap.
    GROQ_REQUEST_TOKEN_BUDGET: int = 10000
    GROQ_TEMPERATURE: float = 0.7
    GROQ_TOP_P: float = 0.9
    # Medium is a balanced baseline while validating completion token budgets.
    GROQ_REASONING_EFFORT: str = "medium"
    # Provider completion includes reasoning; visible answers have a separate ceiling.
    LABORLENS_REASONING_COMPLETION_TOKENS: int = 4096

    # Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LABORLENS_GEMINI_ENABLED: bool = True
    LABORLENS_GEMINI_MODELS: str = (
        "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    )

    # The per-role model pools determine model order. These settings are retained
    # for legacy provider paths such as answer translation.
    LABORLENS_PRIMARY_PROVIDER: str = "gemini"
    LABORLENS_FALLBACK_PROVIDER: str = "groq"
    LABORLENS_PROVIDER_TIMEOUT_SECONDS: float = 30.0
    # Retrieval is bounded within the original request deadline. Optional
    # discovery/ranking leaves time to finalize evidence, draft and verify it.
    LABORLENS_RETRIEVAL_NORMAL_SECONDS: float = 60.0
    LABORLENS_RETRIEVAL_COMPLEX_SECONDS: float = 75.0
    LABORLENS_RETRIEVAL_GENERATION_RESERVE_SECONDS: float = 15.0
    LABORLENS_RETRIEVAL_FINALIZATION_RESERVE_SECONDS: float = 8.0
    LABORLENS_RETRIEVAL_RANKING_SECONDS: float = 12.0
    LABORLENS_NORMAL_REQUEST_BUDGET_SECONDS: float = 45.0
    LABORLENS_COMPLEX_REQUEST_BUDGET_SECONDS: float = 60.0
    LABORLENS_GROQ_TIMEOUT_SECONDS: float = 30.0
    LABORLENS_OPENAI_TIMEOUT_SECONDS: float = 30.0
    LABORLENS_GEMINI_TIMEOUT_SECONDS: float = 30.0
    LABORLENS_GEMINI_TOTAL_BUDGET_SECONDS: float = 30.0
    LABORLENS_GEMINI_MAX_MODELS_PER_REQUEST: int = 2
    LABORLENS_PROVIDER_FAILURE_THRESHOLD: int = 3
    LABORLENS_PROVIDER_COOLDOWN_SECONDS: float = 300.0
    LABORLENS_MAX_OUTPUT_TOKENS: int = 1200
    LABORLENS_MAX_OUTPUT_TOKENS_COMPLEX: int = 2000
    # Visible-answer budgets. Keep the hard ceiling below provider defaults so
    # a short request cannot accidentally consume the complex budget.
    LABORLENS_SIMPLE_OUTPUT_TOKENS: int = 500
    LABORLENS_HARD_OUTPUT_LIMIT: int = 2048
    LABORLENS_CONTINUATION_TOKENS: int = 200
    LABORLENS_STREAMING_ENABLED: bool = True

    # Safe retrieval caches contain only normalized queries, embeddings, and
    # document results. They never contain final answers or conversation data.
    LABORLENS_RETRIEVAL_CACHE_SIZE: int = 256
    LABORLENS_RETRIEVAL_CACHE_TTL_SECONDS: float = 300.0

    # MySQL
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "laborlens_db"
    DB_USER: str = "root"
    DB_PASSWORD: str = ""

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "labor_code_docs"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"

    # Semantic boundary detection refines text chunks inside existing legal
    # article/section boundaries. Parent passages keep the complete context.
    SEMANTIC_CHUNKING_ENABLED: bool = True
    SEMANTIC_CHUNK_MIN_CHARS: int = 320
    SEMANTIC_CHUNK_MAX_UNITS: int = 160
    SEMANTIC_CHUNK_DISTANCE_PERCENTILE: float = 90.0
    SEMANTIC_CHUNK_MIN_DISTANCE: float = 0.16

    # Targeted RAG retrieval. These limits keep the normal path small while
    # allowing one bounded expansion when the first evidence set is incomplete.
    # Broader search candidates; legal selection narrows the generation context.
    RAG_INITIAL_TOP_K: int = 30
    RAG_EXPANDED_TOP_K: int = 50
    RAG_CANDIDATE_POOL_SIZE: int = 50
    RAG_MAX_CONTEXT_CHUNKS: int = 12
    # Document-driven selection uses useful coverage within this input budget.
    # RAG_MAX_CONTEXT_CHUNKS remains only for legacy topic-based callers.
    RAG_CONTEXT_TOKEN_BUDGET: int = 8000
    RAG_MAX_RETRIEVAL_STAGES: int = 3
    RAG_MAX_CATEGORIES: int = 3
    RAG_VECTOR_SCORE_FLOOR: float = 0.20
    RAG_NEAR_DUPLICATE_THRESHOLD: float = 0.90
    RAG_MIN_RELEVANCE_SCORE: float = 0.30
    RAG_RELIABLE_SOURCE_THRESHOLD: float = 0.65
    # Optional learned topic hints supplement global discovery; they never
    # filter documents, supply rules, or establish answer support.
    LDA_RERANK_ENABLED: bool = True
    LDA_SEARCH_HINTS_ENABLED: bool = True
    # 0 chooses a data-dependent count; positive values request a manual count.
    # This is a CPU/memory safety ceiling, not a ten-topic semantic restriction.
    LDA_RERANK_TOPIC_COUNT: int = 0
    LDA_RERANK_MAX_TOPICS: int = 128
    LDA_CHUNK_ASSIGNMENT_BATCH_SIZE: int = 128
    # 0 includes every usable approved chunk; positive values opt into a cap.
    LDA_RERANK_MAX_TRAINING_CHUNKS: int = 0
    LDA_TRAINING_BATCH_SIZE: int = 128
    LDA_RERANK_MIN_CORPUS_CHUNKS: int = 80
    LDA_RERANK_MAX_FEATURES: int = 12000
    LDA_RERANK_MAX_ITER: int = 8
    LDA_RERANK_MIN_QUERY_CONFIDENCE: float = 0.42
    LDA_RERANK_MIN_BASE_SCORE: float = 0.45
    LDA_RERANK_MIN_SIMILARITY: float = 0.15
    LDA_RERANK_MAX_BOOST: float = 0.04

    # JWT
    JWT_SECRET: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # Frontend / OAuth
    FRONTEND_URL: str = "http://127.0.0.1:5500/index.html"
    OAUTH_STATE_DB_PATH: str = str(BACKEND_DIR / "data" / "oauth_state.sqlite3")
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = ""
    # Firebase Auth project used to verify client-side Google sign-in tokens.
    FIREBASE_PROJECT_ID: str = "laborlens-31a9c"

    # Email / Password reset
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    SMTP_FROM_NAME: str = "LaborLens"

    # Uploads
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    # Shared by API workers on this host; keep on persistent local storage.
    GUEST_USAGE_DB_PATH: str = str(BACKEND_DIR / "data" / "guest_usage.sqlite3")

    # OCR
    TESSERACT_CMD: str = ""

    class Config:
        env_file = str(BACKEND_DIR / ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
