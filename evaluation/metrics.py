"""
RAG Evaluation Metrics
Evaluates RAG pipeline quality: faithfulness, relevance, context precision.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from config.settings import get_settings

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline responses using multiple quality metrics.

    Metrics:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevance: Does the answer address the question?
    - Context Precision: How relevant are the retrieved documents?
    - Overall Score: Weighted combination of all metrics
    """

    def __init__(self):
        settings = get_settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatGroq(
            model=settings.default_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.0,
            max_tokens=256,
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        return self.embeddings.embed_documents(texts)

    def _cosine_sim(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = self._embed_texts([text_a, text_b])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def evaluate_faithfulness(self, answer: str, context: str) -> dict:
        """
        Measure how faithful the answer is to the provided context.
        Uses embedding similarity + LLM-as-judge.
        """
        sim_score = self._cosine_sim(answer, context)

        try:
            eval_prompt = (
                "You are an evaluation judge. Score how faithfully this answer "
                "reflects the given context on a scale of 0.0 to 1.0.\n\n"
                f"Context: {context[:3000]}\n\n"
                f"Answer: {answer}\n\n"
                "Respond with ONLY a number between 0.0 and 1.0:"
            )
            response = self.llm.invoke(eval_prompt)
            llm_score = float(response.content.strip())
            llm_score = np.clip(llm_score, 0.0, 1.0)
        except (ValueError, Exception):
            llm_score = sim_score

        combined = float(0.4 * sim_score + 0.6 * llm_score)

        return {
            "metric": "faithfulness",
            "score": round(combined, 4),
            "embedding_score": round(sim_score, 4),
            "llm_score": round(float(llm_score), 4),
            "interpretation": self._interpret_score(combined),
        }

    def evaluate_relevance(self, answer: str, question: str) -> dict:
        """
        Measure how relevant the answer is to the question asked.
        """
        sim_score = self._cosine_sim(answer, question)

        try:
            eval_prompt = (
                "You are an evaluation judge. Score how relevant this answer is "
                "to the question on a scale of 0.0 to 1.0.\n\n"
                f"Question: {question}\n\n"
                f"Answer: {answer}\n\n"
                "Respond with ONLY a number between 0.0 and 1.0:"
            )
            response = self.llm.invoke(eval_prompt)
            llm_score = float(response.content.strip())
            llm_score = np.clip(llm_score, 0.0, 1.0)
        except (ValueError, Exception):
            llm_score = sim_score

        combined = float(0.4 * sim_score + 0.6 * llm_score)

        return {
            "metric": "answer_relevance",
            "score": round(combined, 4),
            "embedding_score": round(sim_score, 4),
            "llm_score": round(float(llm_score), 4),
            "interpretation": self._interpret_score(combined),
        }

    def evaluate_context_precision(
        self,
        retrieved_docs: list[Document],
        question: str,
    ) -> dict:
        """
        Measure how relevant the retrieved context documents are to the question.
        """
        if not retrieved_docs:
            return {
                "metric": "context_precision",
                "score": 0.0,
                "per_document_scores": [],
                "interpretation": "No documents retrieved",
            }

        question_emb = self.embeddings.embed_query(question)
        doc_scores = []

        for doc in retrieved_docs:
            doc_emb = self.embeddings.embed_query(doc.page_content[:2000])
            sim = cosine_similarity([question_emb], [doc_emb])[0][0]
            doc_scores.append({
                "source": doc.metadata.get("source", "Unknown"),
                "score": round(float(sim), 4),
            })

        avg_score = float(np.mean([d["score"] for d in doc_scores]))

        return {
            "metric": "context_precision",
            "score": round(avg_score, 4),
            "per_document_scores": doc_scores,
            "interpretation": self._interpret_score(avg_score),
        }

    def run_evaluation(
        self,
        question: str,
        answer: str,
        context: str,
        retrieved_docs: Optional[list[Document]] = None,
    ) -> dict:
        """Run all evaluation metrics and return a comprehensive report."""
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)

        results = {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
        }

        if retrieved_docs:
            context_precision = self.evaluate_context_precision(
                retrieved_docs, question
            )
            results["context_precision"] = context_precision

        scores = [r["score"] for r in results.values()]
        overall = float(np.mean(scores))

        results["overall"] = {
            "score": round(overall, 4),
            "interpretation": self._interpret_score(overall),
            "metrics_evaluated": len(scores),
        }

        return results

    @staticmethod
    def _interpret_score(score: float) -> str:
        """Convert a numeric score to a human-readable interpretation."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
