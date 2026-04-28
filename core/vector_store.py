"""
Vector Store Manager
Manages ChromaDB vector database for document storage and similarity search.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the ChromaDB vector store for document embeddings."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "enterprise_docs",
    ):
        settings = get_settings()
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.collection_name = collection_name

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Local embeddings — no API key needed, runs entirely on your machine
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Lazy-initialized vector store
        self._store: Optional[Chroma] = None

    @property
    def store(self) -> Chroma:
        """Lazy-load the vector store."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._store

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store")
        ids = self.store.add_documents(documents)
        logger.info(f"Successfully added {len(ids)} documents")
        return ids

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[Document]:
        """Search for similar documents using a query string."""
        logger.info(f"Searching for: '{query[:80]}...' (top {k})")
        results = self.store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} results")
        return results

    def similarity_search_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Search similar documents and return with relevance scores."""
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        return results

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        collection = self.store._collection
        count = collection.count()

        # Get unique sources
        sources = set()
        if count > 0:
            result = collection.get(include=["metadatas"])
            if result and result.get("metadatas"):
                for meta in result["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])

        return {
            "total_documents": count,
            "unique_sources": len(sources),
            "source_files": sorted(list(sources)),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.store.delete_collection()
        self._store = None

    def reset(self) -> None:
        """Reset the vector store by deleting and recreating the collection."""
        self.delete_collection()
        self._store = None
        logger.info("Vector store reset complete")
