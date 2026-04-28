"""
Hallucination Detection
Scores how grounded an LLM response is relative to its source documents.
Uses embedding cosine similarity to detect potential hallucinations.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import get_settings

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects hallucinated (ungrounded) content by comparing the LLM response
    against the source documents used to generate it.

    Approach:
    1. Embed the generated answer
    2. Embed the source context
    3. Compute cosine similarity
    4. Low similarity = high hallucination risk
    """

    RISK_LEVELS = {
        "low": (0.7, 1.0),      # Well-grounded
        "medium": (0.4, 0.7),   # Partially grounded
        "high": (0.0, 0.4),     # Likely hallucinated
    }

    def __init__(self, threshold: float = 0.3):
        settings = get_settings()
        self.threshold = threshold
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def score(
        self,
        answer: str,
        source_documents: list[Document],
    ) -> dict:
        """
        Score hallucination risk of an answer given its source documents.

        Returns:
            dict with keys:
                - score (float): similarity score (0-1, higher = more grounded)
                - risk_level (str): low/medium/high
                - explanation (str): human-readable explanation
                - is_hallucinated (bool): whether score is below threshold
        """
        if not source_documents:
            return {
                "score": 0.0,
                "risk_level": "high",
                "explanation": "No source documents provided — cannot verify",
                "is_hallucinated": True,
            }

        try:
            # Combine source documents into one context string
            source_text = " ".join(
                doc.page_content for doc in source_documents
            )

            # Embed both answer and source
            answer_embedding = self.embeddings.embed_query(answer)
            source_embedding = self.embeddings.embed_query(source_text[:8000])

            # Compute cosine similarity
            sim = cosine_similarity(
                [answer_embedding], [source_embedding]
            )[0][0]

            sim = float(np.clip(sim, 0.0, 1.0))

            # Determine risk level
            risk_level = "high"
            for level, (low, high) in self.RISK_LEVELS.items():
                if low <= sim < high:
                    risk_level = level
                    break

            is_hallucinated = sim < self.threshold

            if risk_level == "low":
                explanation = (
                    f"✅ Response is well-grounded in source documents. "
                    f"Similarity: {sim:.2%}"
                )
            elif risk_level == "medium":
                explanation = (
                    f"⚠️ Response is partially grounded. Some claims may not "
                    f"be directly supported. Similarity: {sim:.2%}"
                )
            else:
                explanation = (
                    f"🚨 High hallucination risk! Response may contain "
                    f"ungrounded claims. Similarity: {sim:.2%}"
                )

            return {
                "score": round(sim, 4),
                "risk_level": risk_level,
                "explanation": explanation,
                "is_hallucinated": is_hallucinated,
            }

        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return {
                "score": 0.0,
                "risk_level": "unknown",
                "explanation": f"Detection error: {str(e)}",
                "is_hallucinated": False,
            }

    def score_with_context(
        self,
        answer: str,
        context: str,
    ) -> dict:
        """
        Simplified scoring using a pre-formatted context string
        instead of Document objects.
        """
        doc = Document(page_content=context)
        return self.score(answer, [doc])
