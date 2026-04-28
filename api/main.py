"""
FastAPI Application — Enterprise GenAI Knowledge Platform
Main application entry point with CORS, startup, and route registration.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from api.routes import chat, documents, models, governance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────── Application Components ────────────
# Shared instances initialized at startup
_components: dict = {}


def get_components() -> dict:
    """Access shared application components."""
    return _components


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize components on startup."""
    settings = get_settings()
    logger.info(f"🚀 Starting {settings.app_name}")

    # Initialize core components
    from core.document_processor import DocumentProcessor
    from core.vector_store import VectorStoreManager
    from core.rag_engine import RAGEngine
    from core.model_garden import ModelGarden
    from governance.governance_engine import GovernanceEngine

    vector_store = VectorStoreManager()
    rag_engine = RAGEngine(vector_store=vector_store)

    _components["document_processor"] = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    _components["vector_store"] = vector_store
    _components["rag_engine"] = rag_engine
    _components["model_garden"] = ModelGarden()
    _components["governance_engine"] = GovernanceEngine(
        enable_toxicity=settings.enable_toxicity_filter,
        enable_hallucination=settings.enable_hallucination_detection,
        enable_prompt_guard=settings.enable_prompt_guard,
        toxicity_threshold=settings.toxicity_threshold,
        hallucination_threshold=settings.hallucination_threshold,
    )

    # Ensure upload directory exists
    Path(settings.document_upload_dir).mkdir(parents=True, exist_ok=True)

    logger.info("✅ All components initialized")
    yield
    logger.info("👋 Shutting down")


# ──────────── FastAPI App ────────────

app = FastAPI(
    title="Enterprise GenAI Knowledge Platform",
    description=(
        "Enterprise-grade Generative AI platform with RAG pipelines, "
        "Model Garden, AI Governance, and evaluation metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(models.router, prefix="/api", tags=["Model Garden"])
app.include_router(governance.router, prefix="/api", tags=["Governance"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Enterprise GenAI Knowledge Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    components = get_components()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "rag_engine": "ready" if "rag_engine" in components else "not initialized",
            "vector_store": "ready" if "vector_store" in components else "not initialized",
            "model_garden": "ready" if "model_garden" in components else "not initialized",
            "governance": "ready" if "governance_engine" in components else "not initialized",
        },
    }
