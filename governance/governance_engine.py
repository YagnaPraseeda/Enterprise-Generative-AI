"""
Governance Engine
Orchestrates all governance checks (toxicity, prompt safety, hallucination)
into a unified pipeline.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from governance.toxicity_filter import ToxicityFilter
from governance.hallucination_detector import HallucinationDetector
from governance.prompt_guard import PromptGuard

logger = logging.getLogger(__name__)


class GovernanceEngine:
    """
    Enterprise AI Governance Engine.

    Provides a unified interface for running all safety and compliance
    checks on both user inputs and LLM outputs.
    """

    def __init__(
        self,
        enable_toxicity: bool = True,
        enable_hallucination: bool = True,
        enable_prompt_guard: bool = True,
        toxicity_threshold: float = 0.7,
        hallucination_threshold: float = 0.3,
    ):
        self.enable_toxicity = enable_toxicity
        self.enable_hallucination = enable_hallucination
        self.enable_prompt_guard = enable_prompt_guard

        # Initialize filters
        if enable_toxicity:
            self.toxicity_filter = ToxicityFilter(threshold=toxicity_threshold)
        if enable_hallucination:
            self.hallucination_detector = HallucinationDetector(
                threshold=hallucination_threshold
            )
        if enable_prompt_guard:
            self.prompt_guard = PromptGuard()

    def pre_check(self, user_input: str) -> dict:
        """
        Run pre-generation checks on user input.

        Checks:
        1. Toxicity of user message
        2. Prompt injection attempts

        Returns:
            dict with keys:
                - is_allowed (bool): whether the input should be processed
                - checks (dict): results from each check
                - blocked_reason (str): reason if blocked
                - risk_summary (str): overall risk summary
        """
        checks = {}
        blocked_reasons = []

        # Toxicity check
        if self.enable_toxicity:
            toxicity_result = self.toxicity_filter.check(user_input)
            checks["toxicity"] = toxicity_result
            if toxicity_result["is_toxic"]:
                blocked_reasons.append(
                    f"Toxic content: {', '.join(toxicity_result['flagged_categories'])}"
                )

        # Prompt injection check
        if self.enable_prompt_guard:
            prompt_result = self.prompt_guard.check(user_input)
            checks["prompt_safety"] = prompt_result
            if not prompt_result["is_safe"]:
                blocked_reasons.append(
                    f"Prompt injection: {prompt_result['risk_type']}"
                )

        is_allowed = len(blocked_reasons) == 0

        if is_allowed:
            risk_summary = "✅ Input passed all governance checks"
        else:
            risk_summary = (
                f"🚫 Input blocked: {'; '.join(blocked_reasons)}"
            )

        return {
            "is_allowed": is_allowed,
            "checks": checks,
            "blocked_reason": "; ".join(blocked_reasons) if blocked_reasons else None,
            "risk_summary": risk_summary,
        }

    def post_check(
        self,
        answer: str,
        source_documents: Optional[list[Document]] = None,
        context: Optional[str] = None,
    ) -> dict:
        """
        Run post-generation checks on LLM output.

        Checks:
        1. Toxicity of generated response
        2. Hallucination detection (if source documents provided)

        Returns:
            dict with keys:
                - is_safe (bool): whether the response is safe to display
                - checks (dict): results from each check
                - warnings (list): non-blocking warnings
                - confidence_score (float): overall confidence in response safety
        """
        checks = {}
        warnings = []
        confidence_factors = []

        # Toxicity check on output
        if self.enable_toxicity:
            toxicity_result = self.toxicity_filter.check(answer)
            checks["toxicity"] = toxicity_result
            if toxicity_result["is_toxic"]:
                warnings.append("Response contains potentially toxic content")
            confidence_factors.append(1.0 - toxicity_result.get("max_score", 0))

        # Hallucination check
        if self.enable_hallucination and (source_documents or context):
            if source_documents:
                hall_result = self.hallucination_detector.score(
                    answer, source_documents
                )
            else:
                hall_result = self.hallucination_detector.score_with_context(
                    answer, context
                )
            checks["hallucination"] = hall_result
            if hall_result["is_hallucinated"]:
                warnings.append(
                    f"High hallucination risk (score: {hall_result['score']:.2%})"
                )
            confidence_factors.append(hall_result.get("score", 0.5))

        # Calculate overall confidence
        confidence_score = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.5
        )

        is_safe = not any(
            checks.get("toxicity", {}).get("is_toxic", False)
            for _ in [1]  # Check toxicity only
        )

        return {
            "is_safe": is_safe,
            "checks": checks,
            "warnings": warnings,
            "confidence_score": round(confidence_score, 4),
        }

    def full_check(
        self,
        user_input: str,
        answer: str,
        source_documents: Optional[list[Document]] = None,
        context: Optional[str] = None,
    ) -> dict:
        """Run both pre-check and post-check and return unified report."""
        pre = self.pre_check(user_input)
        post = self.post_check(answer, source_documents, context)

        return {
            "pre_check": pre,
            "post_check": post,
            "overall_safe": pre["is_allowed"] and post["is_safe"],
            "confidence_score": post["confidence_score"],
        }
