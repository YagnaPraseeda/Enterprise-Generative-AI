"""
Model Garden API Routes
Manage models, compare responses, and track usage metrics.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import ModelInfo, ModelCompareRequest, ModelCompareResult, UsageMetrics

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_components():
    from api.main import get_components
    return get_components()


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List all registered models in the Model Garden."""
    components = _get_components()
    model_garden = components.get("model_garden")

    if not model_garden:
        raise HTTPException(status_code=503, detail="Model Garden not initialized")

    models = model_garden.list_models()
    return [ModelInfo(**m) for m in models]


@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed info for a specific model."""
    components = _get_components()
    model_garden = components.get("model_garden")

    if not model_garden:
        raise HTTPException(status_code=503, detail="Model Garden not initialized")

    info = model_garden.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return info


@router.post("/models/compare", response_model=list[ModelCompareResult])
async def compare_models(request: ModelCompareRequest):
    """
    Compare responses from multiple models for the same question.
    Returns side-by-side results with latency and cost information.
    """
    components = _get_components()
    model_garden = components.get("model_garden")

    if not model_garden:
        raise HTTPException(status_code=503, detail="Model Garden not initialized")

    # Validate models exist
    available = {m["name"] for m in model_garden.list_models()}
    invalid = [m for m in request.model_names if m not in available]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown models: {invalid}. Available: {sorted(available)}",
        )

    try:
        results = model_garden.compare_models(
            question=request.question,
            model_names=request.model_names,
            context=request.context,
        )
        return [ModelCompareResult(**r) for r in results]

    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/metrics", response_model=UsageMetrics)
async def get_model_metrics(model_name: str):
    """Get usage metrics for a specific model."""
    components = _get_components()
    model_garden = components.get("model_garden")

    if not model_garden:
        raise HTTPException(status_code=503, detail="Model Garden not initialized")

    metrics = model_garden.get_usage_metrics(model_name)
    return UsageMetrics(**metrics)


@router.get("/models-metrics/all", response_model=UsageMetrics)
async def get_all_metrics():
    """Get aggregated usage metrics across all models."""
    components = _get_components()
    model_garden = components.get("model_garden")

    if not model_garden:
        raise HTTPException(status_code=503, detail="Model Garden not initialized")

    metrics = model_garden.get_usage_metrics()
    return UsageMetrics(**metrics)
