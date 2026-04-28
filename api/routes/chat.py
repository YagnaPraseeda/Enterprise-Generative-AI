"""
Chat API Routes
Handles conversational RAG queries with optional governance checks.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, SourceInfo, GovernanceReport

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory conversation history
_conversation_history: list[dict] = []


def _get_components():
    from api.main import get_components
    return get_components()


@router.post("/chat", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Submit a question to the RAG pipeline.
    Returns an answer with sources, model info, and governance report.
    """
    components = _get_components()
    rag_engine = components.get("rag_engine")
    governance_engine = components.get("governance_engine")

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    governance_report = None

    # Pre-check (input governance)
    if request.enable_governance and governance_engine:
        pre_check = governance_engine.pre_check(request.question)
        if not pre_check["is_allowed"]:
            return ChatResponse(
                answer=f"⚠️ Your question was blocked by governance controls: {pre_check['blocked_reason']}",
                sources=[],
                model_used=request.model_name or "N/A",
                latency_ms=0,
                retrieved_chunks=0,
                governance=GovernanceReport(
                    pre_check_passed=False,
                    warnings=[pre_check["blocked_reason"]],
                    details=pre_check.get("checks", {}),
                ),
            )

    # Run RAG pipeline
    try:
        result = rag_engine.query(
            question=request.question,
            model_name=request.model_name,
            k=request.top_k,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Post-check (output governance)
    if request.enable_governance and governance_engine and result.get("context"):
        post_check = governance_engine.post_check(
            answer=result["answer"],
            context=result["context"],
        )
        governance_report = GovernanceReport(
            pre_check_passed=True,
            post_check_passed=post_check["is_safe"],
            confidence_score=post_check["confidence_score"],
            warnings=post_check["warnings"],
            details=post_check.get("checks", {}),
        )

    # Build source references
    sources = [
        SourceInfo(**src) for src in result.get("sources", [])
    ]

    # Store in history
    _conversation_history.append({
        "timestamp": datetime.now().isoformat(),
        "question": request.question,
        "answer": result["answer"],
        "model": result["model_used"],
        "sources": [s.document for s in sources],
    })

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        model_used=result["model_used"],
        latency_ms=result["latency_ms"],
        retrieved_chunks=result["retrieved_chunks"],
        governance=governance_report,
    )


@router.get("/chat/history")
async def get_chat_history(limit: int = 20):
    """Get recent conversation history."""
    return {
        "history": _conversation_history[-limit:],
        "total": len(_conversation_history),
    }


@router.delete("/chat/history")
async def clear_chat_history():
    """Clear conversation history."""
    _conversation_history.clear()
    return {"message": "Conversation history cleared"}
