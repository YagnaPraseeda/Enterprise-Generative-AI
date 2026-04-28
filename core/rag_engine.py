"""
RAG (Retrieval Augmented Generation) Engine
Orchestrates the full RAG pipeline: query → retrieve → generate.
"""

import logging
import time
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings
from core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Enterprise-tuned system prompt (with document context)
SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant. You help users with both
enterprise document questions AND general knowledge queries.

RULES:
1. When document context is provided, prioritize answering from those documents and cite sources.
2. If the context does not contain the answer, use your general knowledge to help the user.
3. Always be professional, concise, and accurate.
4. If the question is ambiguous, ask for clarification.
5. Structure your response clearly with bullet points or numbered lists when appropriate.
6. When using general knowledge (not from documents), mention that the answer comes from
   your general training rather than enterprise documents."""

RAG_PROMPT_TEMPLATE = """Use the following context from enterprise documents to answer the question.
If the context is relevant, base your answer on it and cite the sources.
If the context is not relevant to the question, use your general knowledge to provide a helpful answer.

CONTEXT:
{context}

---

QUESTION: {question}

Provide a comprehensive, helpful answer."""

# Prompt for when no documents are retrieved
GENERAL_PROMPT_TEMPLATE = """Answer the following question using your general knowledge.
Be professional, concise, and helpful.

QUESTION: {question}"""


class RAGEngine:
    """Orchestrates the Retrieval Augmented Generation pipeline."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.settings = get_settings()
        self.vector_store = vector_store or VectorStoreManager()

        # Build prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RAG_PROMPT_TEMPLATE),
        ])

    def _get_llm(self, model_name: Optional[str] = None) -> ChatGroq:
        """Get an LLM instance for the specified model."""
        model = model_name or self.settings.default_model
        return ChatGroq(
            model=model,
            groq_api_key=self.settings.groq_api_key,
            temperature=0.1,
            max_tokens=1024,
        )

    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, documents: list[Document]) -> list[dict]:
        """Extract source metadata from retrieved documents."""
        sources = []
        seen = set()
        for doc in documents:
            source_name = doc.metadata.get("source", "Unknown")
            if source_name not in seen:
                seen.add(source_name)
                sources.append({
                    "document": source_name,
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                })
        return sources

    def query(
        self,
        question: str,
        model_name: Optional[str] = None,
        k: int = 5,
    ) -> dict:
        """
        Execute the full RAG pipeline.

        Returns:
            dict with keys: answer, sources, model_used, latency_ms,
                           retrieved_chunks, context
        """
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(question, k=k)

        if not retrieved_docs:
            # No documents found — answer using general knowledge
            llm = self._get_llm(model_name)
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", GENERAL_PROMPT_TEMPLATE),
            ])
            chain = general_prompt | llm
            response = chain.invoke({"question": question})
            latency_ms = round((time.time() - start_time) * 1000)

            return {
                "answer": response.content,
                "sources": [],
                "model_used": model_name or self.settings.default_model,
                "latency_ms": latency_ms,
                "retrieved_chunks": 0,
                "context": "",
            }

        # Step 2: Format context
        context = self._format_context(retrieved_docs)

        # Step 3: Generate response
        llm = self._get_llm(model_name)
        chain = self.prompt | llm

        response = chain.invoke({
            "context": context,
            "question": question,
        })

        latency_ms = round((time.time() - start_time) * 1000)

        # Step 4: Build structured response
        result = {
            "answer": response.content,
            "sources": self._extract_sources(retrieved_docs),
            "model_used": model_name or self.settings.default_model,
            "latency_ms": latency_ms,
            "retrieved_chunks": len(retrieved_docs),
            "context": context,
        }

        logger.info(
            f"RAG query completed in {latency_ms}ms using {result['model_used']}"
        )
        return result

    def query_with_scores(
        self,
        question: str,
        model_name: Optional[str] = None,
        k: int = 5,
    ) -> dict:
        """Execute RAG pipeline with relevance scores for retrieved documents."""
        start_time = time.time()

        results_with_scores = self.vector_store.similarity_search_with_scores(
            question, k=k
        )

        if not results_with_scores:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "model_used": model_name or self.settings.default_model,
                "latency_ms": round((time.time() - start_time) * 1000),
                "retrieved_chunks": 0,
                "context": "",
                "relevance_scores": [],
            }

        docs = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]

        context = self._format_context(docs)

        llm = self._get_llm(model_name)
        chain = self.prompt | llm
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        latency_ms = round((time.time() - start_time) * 1000)

        return {
            "answer": response.content,
            "sources": self._extract_sources(docs),
            "model_used": model_name or self.settings.default_model,
            "latency_ms": latency_ms,
            "retrieved_chunks": len(docs),
            "context": context,
            "relevance_scores": scores,
        }
