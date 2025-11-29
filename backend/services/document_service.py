from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from backend.models.database import Document
from backend.models.schemas import DocumentCreate, DocumentResponse
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing documents"""
    
    @staticmethod
    async def create_document(db: AsyncSession, doc_data: DocumentCreate, file_path: str) -> Document:
        """Create a new document record"""
        doc = Document(
            filename=doc_data.filename,
            file_path=file_path,
            processing_status="pending",
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)
        return doc
    
    @staticmethod
    async def get_document(db: AsyncSession, doc_id: int) -> Optional[Document]:
        """Get document by ID"""
        result = await db.execute(select(Document).where(Document.id == doc_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all_documents(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Document]:
        """Get all documents with pagination"""
        result = await db.execute(
            select(Document)
            .order_by(Document.upload_time.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def update_document_status(
        db: AsyncSession,
        doc_id: int,
        status: str,
        num_pages: Optional[int] = None,
        num_chunks: Optional[int] = None,
        doc_metadata: Optional[dict] = None
    ) -> Optional[Document]:
        """Update document processing status and metadata"""
        doc = await DocumentService.get_document(db, doc_id)
        if doc:
            doc.processing_status = status
            if num_pages is not None:
                doc.num_pages = num_pages
            if num_chunks is not None:
                doc.num_chunks = num_chunks
            if doc_metadata is not None:
                doc.doc_metadata = doc_metadata
            await db.flush()
            await db.refresh(doc)
        return doc
    
    @staticmethod
    async def delete_document(db: AsyncSession, doc_id: int) -> bool:
        """Delete document and its file"""
        doc = await DocumentService.get_document(db, doc_id)
        if doc:
            # Delete physical file
            try:
                file_path = Path(doc.file_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {doc.file_path}: {e}")
            
            # Delete database record
            await db.execute(delete(Document).where(Document.id == doc_id))
            await db.flush()
            return True
        return False
    
    @staticmethod
    async def count_documents(db: AsyncSession) -> int:
        """Count total documents"""
        result = await db.execute(select(Document))
        return len(list(result.scalars().all()))
