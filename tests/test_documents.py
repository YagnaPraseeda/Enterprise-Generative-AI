"""
Tests for the document processor.
"""

import pytest
from pathlib import Path
from core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document loading and chunking."""

    def setup_method(self):
        self.processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

    def test_supported_extensions(self):
        exts = self.processor.get_supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts
        assert ".md" in exts

    def test_load_text_file(self):
        sample_dir = Path(__file__).parent.parent / "data" / "sample_docs"
        txt_files = list(sample_dir.glob("*.txt"))

        if txt_files:
            docs = self.processor.load_document(str(txt_files[0]))
            assert len(docs) > 0
            assert docs[0].page_content
            assert docs[0].metadata["source"] == txt_files[0].name

    def test_chunk_documents(self):
        from langchain_core.documents import Document

        docs = [Document(page_content="A " * 500, metadata={"source": "test.txt"})]
        chunks = self.processor.chunk_documents(docs)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_size" in chunk.metadata

    def test_process_directory(self):
        sample_dir = Path(__file__).parent.parent / "data" / "sample_docs"
        if sample_dir.exists():
            chunks = self.processor.process_directory(str(sample_dir))
            assert len(chunks) > 0

    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.processor.load_document("test.xyz")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.processor.load_document("nonexistent_file.txt")
