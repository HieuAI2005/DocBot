import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.db_manager import get_db
from backend.services.document_service import DocumentService
from backend.services.adaptive_rag import get_adaptive_rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/reindex-documents")
async def reindex_all_documents(db: AsyncSession = Depends(get_db)):
    """Re-index all documents that failed processing"""
    try:
        import PyPDF2
        from backend.config import UPLOAD_DIR
        
        # Get all documents
        docs = await DocumentService.get_all_documents(db, skip=0, limit=1000)
        reindexed = []
        failed = []
        
        for doc in docs:
            try:
                # Check if file exists
                file_path = Path(doc.file_path)
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    failed.append({
                        "id": doc.id,
                        "filename": doc.filename,
                        "error": "File not found"
                    })
                    continue
                
                # Extract text with PyPDF2
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text_pages = []
                    for page in pdf_reader.pages:
                        text_pages.append(page.extract_text())
                    text = "\n\n".join(text_pages)
                
                if text and len(text.strip()) > 50:
                    # Add to RAG index
                    adaptive_rag = get_adaptive_rag()
                    num_chunks = adaptive_rag.add_document(
                        doc_id=str(doc.id),
                        text=text,
                        source_path=str(file_path)
                    )
                    
                    # Update status
                    await DocumentService.update_document_status(
                        db, doc.id,
                        status="completed",
                        num_chunks=num_chunks
                    )
                    await db.commit()
                    
                    reindexed.append({
                        "id": doc.id,
                        "filename": doc.filename,
                        "num_chunks": num_chunks
                    })
                    logger.info(f"Re-indexed document {doc.id}: {num_chunks} chunks")
                else:
                    failed.append({
                        "id": doc.id,
                        "filename": doc.filename,
                        "error": "No text extracted"
                    })
                    
            except Exception as e:
                logger.error(f"Error re-indexing document {doc.id}: {e}")
                failed.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "error": str(e)
                })
        
        return {
            "reindexed": reindexed,
            "failed": failed,
            "total_reindexed": len(reindexed),
            "total_failed": len(failed)
        }
        
    except Exception as e:
        logger.error(f"Error in reindex_all_documents: {e}", exc_info=True)
        raise HTTPException(500, f"Error re-indexing documents: {str(e)}")
