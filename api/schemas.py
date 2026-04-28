"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ──────────── Chat ────────────

class ChatRequest(BaseModel):
    """Chat query request."""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    model_name: Optional[str] = Field(None, description="Model to use (default: gpt-4o-mini)")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    enable_governance: bool = Field(True, description="Run governance checks")


class SourceInfo(BaseModel):
    """Document source reference."""
    document: str
    file_type: str
    chunk_index: int = -1


class GovernanceCheckResult(BaseModel):
    """Individual governance check result."""
    is_toxic: Optional[bool] = None
    is_safe: Optional[bool] = None
    is_hallucinated: Optional[bool] = None
    score: Optional[float] = None
    risk_level: Optional[str] = None
    details: Optional[str] = None


class GovernanceReport(BaseModel):
    """Unified governance report."""
    pre_check_passed: bool = True
    post_check_passed: bool = True
    confidence_score: float = 1.0
    warnings: list[str] = []
    details: dict = {}


class ChatResponse(BaseModel):
    """Chat query response."""
    answer: str
    sources: list[SourceInfo] = []
    model_used: str
    latency_ms: int
    retrieved_chunks: int
    governance: Optional[GovernanceReport] = None


# ──────────── Documents ────────────

class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    filename: str
    chunks_created: int
    status: str = "processed"
    message: str = ""


class DocumentStats(BaseModel):
    """Vector store statistics."""
    total_documents: int
    unique_sources: int
    source_files: list[str]
    collection_name: str


class DocumentListItem(BaseModel):
    """Item in the document list."""
    name: str
    file_type: str
    chunks: int = 0


# ──────────── Models ────────────

class ModelInfo(BaseModel):
    """Model metadata."""
    name: str
    provider: str
    version: str = ""
    description: str = ""
    cost_tier: str = "medium"
    max_tokens: int = 4096
    capabilities: list[str] = []
    status: str = "active"


class ModelCompareRequest(BaseModel):
    """Request to compare multiple models."""
    question: str = Field(..., min_length=1, max_length=2000)
    model_names: list[str] = Field(..., min_length=2, max_length=5)
    context: str = Field("", description="Optional context for grounded comparison")


class ModelCompareResult(BaseModel):
    """Single model's comparison result."""
    model: str
    answer: str
    latency_ms: int
    status: str
    cost_tier: str = "unknown"


class UsageMetrics(BaseModel):
    """Model usage metrics."""
    total_requests: int = 0
    avg_latency_ms: int = 0
    min_latency_ms: int = 0
    max_latency_ms: int = 0
    total_response_chars: int = 0
    models_used: list[str] = []


# ──────────── Governance ────────────

class GovernanceCheckRequest(BaseModel):
    """Request to run governance checks on text."""
    text: str = Field(..., min_length=1, max_length=5000)
    check_type: str = Field("all", description="Type: all, toxicity, prompt_safety")


# ──────────── Evaluation ────────────

class EvaluationRequest(BaseModel):
    """Request to evaluate a RAG response."""
    question: str
    answer: str
    context: str


class EvaluationMetric(BaseModel):
    """Single evaluation metric result."""
    metric: str
    score: float
    interpretation: str


class EvaluationResponse(BaseModel):
    """Full evaluation report."""
    faithfulness: EvaluationMetric
    answer_relevance: EvaluationMetric
    overall_score: float
    overall_interpretation: str


# ──────────── Health ────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    components: dict = {}
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
