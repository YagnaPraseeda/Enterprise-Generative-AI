"""
Governance API Routes
Run safety checks and retrieve governance reports.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import GovernanceCheckRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_components():
    from api.main import get_components
    return get_components()


@router.post("/governance/check")
async def run_governance_check(request: GovernanceCheckRequest):
    """
    Run governance checks on arbitrary text.
    Useful for testing governance controls independently.
    """
    components = _get_components()
    governance_engine = components.get("governance_engine")

    if not governance_engine:
        raise HTTPException(status_code=503, detail="Governance engine not initialized")

    result = {}

    if request.check_type in ("all", "toxicity"):
        result["toxicity"] = governance_engine.toxicity_filter.check(request.text)

    if request.check_type in ("all", "prompt_safety"):
        result["prompt_safety"] = governance_engine.prompt_guard.check(request.text)

    return {
        "text_length": len(request.text),
        "check_type": request.check_type,
        "results": result,
    }


@router.get("/governance/config")
async def get_governance_config():
    """Get current governance configuration."""
    components = _get_components()
    governance_engine = components.get("governance_engine")

    if not governance_engine:
        raise HTTPException(status_code=503, detail="Governance engine not initialized")

    return {
        "toxicity_enabled": governance_engine.enable_toxicity,
        "hallucination_enabled": governance_engine.enable_hallucination,
        "prompt_guard_enabled": governance_engine.enable_prompt_guard,
        "status": "active",
    }
