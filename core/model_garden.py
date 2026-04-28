"""
Model Garden
Manages multiple LLMs — registration, selection, comparison, and usage tracking.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Path to the default model registry
REGISTRY_PATH = Path(__file__).parent / "model_registry.json"


class ModelGarden:
    """
    Enterprise Model Garden for managing multiple LLM providers.

    Features:
    - Model registration and metadata management
    - Multi-model comparison
    - Usage tracking and metrics
    """

    def __init__(self):
        self.settings = get_settings()
        self.models: dict[str, dict] = {}
        self.usage_log: list[dict] = []

        # Load default models from registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load models from the JSON registry file."""
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH, "r") as f:
                registry = json.load(f)
                for model_config in registry:
                    self.models[model_config["name"]] = model_config
            logger.info(f"Loaded {len(self.models)} models from registry")
        else:
            logger.warning("Model registry not found, starting with empty garden")

    def register_model(
        self,
        name: str,
        provider: str,
        version: str = "1.0",
        description: str = "",
        cost_tier: str = "medium",
        max_tokens: int = 4096,
        capabilities: Optional[list[str]] = None,
    ) -> dict:
        """Register a new model in the garden."""
        model_config = {
            "name": name,
            "provider": provider,
            "version": version,
            "description": description,
            "cost_tier": cost_tier,
            "max_tokens": max_tokens,
            "supports_streaming": True,
            "capabilities": capabilities or ["text-generation", "chat"],
            "status": "active",
            "registered_at": datetime.now().isoformat(),
        }
        self.models[name] = model_config
        logger.info(f"Registered model: {name} ({provider})")
        return model_config

    def list_models(self) -> list[dict]:
        """Return all registered models with metadata."""
        return list(self.models.values())

    def get_model_info(self, name: str) -> Optional[dict]:
        """Get metadata for a specific model."""
        return self.models.get(name)

    def get_llm(self, name: str) -> ChatGroq:
        """Get an LLM instance for a registered model."""
        if name not in self.models:
            raise ValueError(
                f"Model '{name}' not found. Available: {list(self.models.keys())}"
            )

        model_config = self.models[name]
        return ChatGroq(
            model=name,
            groq_api_key=self.settings.groq_api_key,
            temperature=0.1,
            max_tokens=min(model_config.get("max_tokens", 4096), 1024),
        )

    def compare_models(
        self,
        question: str,
        model_names: list[str],
        context: str = "",
    ) -> list[dict]:
        """Run the same query across multiple models and return comparison."""
        results = []

        for model_name in model_names:
            try:
                llm = self.get_llm(model_name)
                start_time = time.time()

                if context:
                    prompt = (
                        f"Based on this context:\n{context}\n\n"
                        f"Answer this question: {question}"
                    )
                else:
                    prompt = question

                response = llm.invoke(prompt)
                latency_ms = round((time.time() - start_time) * 1000)

                result = {
                    "model": model_name,
                    "answer": response.content,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "cost_tier": self.models[model_name].get("cost_tier", "unknown"),
                }

                # Log usage
                self.log_usage(model_name, latency_ms, len(response.content))

            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                result = {
                    "model": model_name,
                    "answer": f"Error: {str(e)}",
                    "latency_ms": 0,
                    "status": "error",
                    "cost_tier": self.models.get(model_name, {}).get(
                        "cost_tier", "unknown"
                    ),
                }

            results.append(result)

        return results

    def log_usage(
        self,
        model_name: str,
        latency_ms: int,
        response_length: int,
    ) -> None:
        """Track model usage for metrics and reporting."""
        entry = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "response_length": response_length,
        }
        self.usage_log.append(entry)

    def get_usage_metrics(self, model_name: Optional[str] = None) -> dict:
        """Get aggregated usage metrics for one or all models."""
        logs = self.usage_log
        if model_name:
            logs = [l for l in logs if l["model"] == model_name]

        if not logs:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0,
                "total_response_chars": 0,
                "models_used": [],
            }

        latencies = [l["latency_ms"] for l in logs]
        models_used = list(set(l["model"] for l in logs))

        return {
            "total_requests": len(logs),
            "avg_latency_ms": round(sum(latencies) / len(latencies)),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "total_response_chars": sum(l["response_length"] for l in logs),
            "models_used": models_used,
        }
