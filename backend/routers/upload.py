import sys
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles
import uuid
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.models.db_manager import get_db
from backend.models.schemas import DocumentCreate, DocumentResponse, DocumentList
from backend.services.document_service import DocumentService
from backend.services.adaptive_rag import get_adaptive_rag
from backend.config import UPLOAD_DIR, MAX_UPLOAD_SIZE, ALLOWED_EXTENSIONS
from backend.process.extract.mineru import extract_one
from backend.process.config import OUTPUT_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a PDF document"""
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {file_ext} not allowed. Only PDF files are supported.")
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_UPLOAD_SIZE // (1024*1024)}MB")
    
    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{unique_id}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Create document record
        doc_data = DocumentCreate(filename=file.filename)
        doc = await DocumentService.create_document(db, doc_data, str(file_path))
        await db.commit()
        
        # Process PDF in background (mark as processing)
        await DocumentService.update_document_status(db, doc.id, "processing")
        await db.commit()
        
        # Extract PDF
        try:
            output_dir = OUTPUT_DIR / unique_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            text = None
            
            # Try mineru extraction first
            try:
                extract_one(file_path)
                md_path = output_dir / "main.md"
                if md_path.exists():
                    text = md_path.read_text(encoding="utf-8", errors="ignore")
                    logger.info(f"Successfully extracted with mineru: {len(text)} chars")
            except Exception as mineru_error:
                logger.warning(f"Mineru extraction failed: {mineru_error}, falling back to PyPDF2")
            
            # Fallback to PyPDF2 if mineru failed
            if not text:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text_pages = []
                        for page in pdf_reader.pages:
                            text_pages.append(page.extract_text())
                        text = "\n\n".join(text_pages)
                    logger.info(f"Successfully extracted with PyPDF2: {len(text)} chars")
                except Exception as pypdf_error:
                    logger.error(f"PyPDF2 extraction also failed: {pypdf_error}")
                    raise Exception("Both mineru and PyPDF2 extraction failed")
            
            # Add to adaptive RAG if we have text
            if text and len(text.strip()) > 0:
                adaptive_rag = get_adaptive_rag()
                num_chunks = adaptive_rag.add_document(
                    doc_id=str(doc.id),  # Use database ID
                    text=text,
                    source_path=str(file_path)
                )
                
                # Update document status
                await DocumentService.update_document_status(
                    db, doc.id,
                    status="completed",
                    num_chunks=num_chunks,
                    doc_metadata={"output_dir": str(output_dir)}
                )
                await db.commit()
                
                logger.info(f"Successfully processed document {doc.id}")
            else:
                raise Exception("Extraction failed: main.md not found")
        
        except Exception as e:
            logger.error(f"Error processing document {doc.id}: {e}", exc_info=True)
            await DocumentService.update_document_status(
                db, doc.id,
                status="failed",
                doc_metadata={"error": str(e)}
            )
            await db.commit()
        
        # Return document info
        await db.refresh(doc)
        return doc
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(500, f"Error uploading document: {str(e)}")


@router.get("", response_model=DocumentList)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all uploaded documents"""
    docs = await DocumentService.get_all_documents(db, skip, limit)
    total = await DocumentService.count_documents(db)
    return DocumentList(documents=docs, total=total)


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get document by ID"""
    doc = await DocumentService.get_document(db, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document"""
    success = await DocumentService.delete_document(db, doc_id)
    if not success:
        raise HTTPException(404, "Document not found")
    await db.commit()
    return {"message": "Document deleted successfully"}
