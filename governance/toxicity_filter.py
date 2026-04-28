"""
Toxicity Detection Filter
Detects harmful, toxic, or inappropriate content using the Detoxify model.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-load the model to avoid startup delay
_model = None


def _get_model():
    """Lazy-load the Detoxify model."""
    global _model
    if _model is None:
        try:
            from detoxify import Detoxify
            _model = Detoxify("original")
            logger.info("Toxicity detection model loaded")
        except Exception as e:
            logger.error(f"Failed to load toxicity model: {e}")
            _model = "unavailable"
    return _model


class ToxicityFilter:
    """
    Detects toxic, harmful, or inappropriate content.

    Uses the Detoxify library (based on BERT) to score text across
    multiple toxicity categories.
    """

    # Categories checked by Detoxify
    CATEGORIES = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
    ]

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def check(self, text: str) -> dict:
        """
        Check text for toxicity.

        Returns:
            dict with keys:
                - is_toxic (bool): whether any category exceeds threshold
                - scores (dict): per-category scores
                - max_score (float): highest toxicity score
                - flagged_categories (list): categories exceeding threshold
                - details (str): human-readable summary
        """
        model = _get_model()

        # Fallback if model not available
        if model == "unavailable":
            return {
                "is_toxic": False,
                "scores": {},
                "max_score": 0.0,
                "flagged_categories": [],
                "details": "Toxicity model unavailable — skipping check",
            }

        try:
            results = model.predict(text)

            # Round scores for readability
            scores = {k: round(v, 4) for k, v in results.items()}
            max_score = max(scores.values()) if scores else 0.0

            flagged = [
                cat for cat, score in scores.items()
                if score >= self.threshold
            ]

            is_toxic = len(flagged) > 0

            if is_toxic:
                details = (
                    f"⚠️ Toxic content detected! "
                    f"Flagged categories: {', '.join(flagged)}. "
                    f"Max score: {max_score:.2%}"
                )
            else:
                details = f"✅ Content is safe. Max toxicity score: {max_score:.2%}"

            return {
                "is_toxic": is_toxic,
                "scores": scores,
                "max_score": max_score,
                "flagged_categories": flagged,
                "details": details,
            }

        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return {
                "is_toxic": False,
                "scores": {},
                "max_score": 0.0,
                "flagged_categories": [],
                "details": f"Toxicity check error: {str(e)}",
            }
