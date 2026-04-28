"""
Document Management API Routes
Handles document upload, processing, and vector store management.
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from api.schemas import DocumentUploadResponse, DocumentStats
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_components():
    from api.main import get_components
    return get_components()


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    Supported formats: PDF, DOCX, TXT, MD
    """
    components = _get_components()
    doc_processor = components.get("document_processor")
    vector_store = components.get("vector_store")
    settings = get_settings()

    if not doc_processor or not vector_store:
        raise HTTPException(status_code=503, detail="Components not initialized")

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in doc_processor.get_supported_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {doc_processor.get_supported_extensions()}"
        )

    # Save uploaded file
    upload_dir = Path(settings.document_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process document
        chunks = doc_processor.process_file(str(file_path))

        # Store in vector database
        vector_store.add_documents(chunks)

        return DocumentUploadResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            status="processed",
            message=f"Successfully processed {file.filename} into {len(chunks)} chunks",
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/documents/ingest-samples")
async def ingest_sample_documents():
    """
    Ingest the built-in sample enterprise documents.
    This is a convenience endpoint for initial setup.
    """
    components = _get_components()
    doc_processor = components.get("document_processor")
    vector_store = components.get("vector_store")

    if not doc_processor or not vector_store:
        raise HTTPException(status_code=503, detail="Components not initialized")

    sample_dir = Path(__file__).parent.parent.parent / "data" / "sample_docs"

    if not sample_dir.exists():
        raise HTTPException(status_code=404, detail="Sample docs directory not found")

    try:
        chunks = doc_processor.process_directory(str(sample_dir))
        if chunks:
            vector_store.add_documents(chunks)

        return {
            "status": "success",
            "chunks_created": len(chunks),
            "message": f"Ingested {len(chunks)} chunks from sample documents",
        }

    except Exception as e:
        logger.error(f"Sample ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/stats", response_model=DocumentStats)
async def get_document_stats():
    """Get vector store statistics."""
    components = _get_components()
    vector_store = components.get("vector_store")

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    stats = vector_store.get_stats()
    return DocumentStats(**stats)


@router.get("/documents")
async def list_documents():
    """List all indexed document sources."""
    components = _get_components()
    vector_store = components.get("vector_store")

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    stats = vector_store.get_stats()
    return {
        "documents": [
            {"name": name, "status": "indexed"}
            for name in stats.get("source_files", [])
        ],
        "total": stats.get("unique_sources", 0),
    }


@router.delete("/documents/reset")
async def reset_vector_store():
    """Reset the vector store (delete all documents)."""
    components = _get_components()
    vector_store = components.get("vector_store")

    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        vector_store.reset()
        return {"status": "success", "message": "Vector store reset complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
