from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

from backend.services.qa_service import get_qa_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/qa", tags=["qa"])


class MCQRequest(BaseModel):
    question: str
    options: Dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    document_id: Optional[int] = None


class MCQResponse(BaseModel):
    answer: list[str]
    probabilities: Dict[str, float]
    reasoning: str
    confidence: float
    evidence_count: int = 0
    evidence_quality: float = 0.0


@router.post("/answer", response_model=MCQResponse)
async def answer_mcq(request: MCQRequest):
    """
    Answer a multi-choice question using Chain-of-Thought reasoning.
    
    - **question**: The question text
    - **options**: Dictionary with keys A, B, C, D and option text
    - **document_id**: Optional - filter retrieval to specific document
    
    Returns answer(s), probabilities, and CoT reasoning.
    """
    try:
        # Validate options
        if not all(k in request.options for k in ['A', 'B', 'C', 'D']):
            raise HTTPException(400, "Options must include A, B, C, D")
        
        # Get QA service and process
        qa_service = get_qa_service()
        result = await qa_service.answer_mcq(
            question=request.question,
            options=request.options,
            document_id=request.document_id
        )
        
        # Check for errors
        if "error" in result:
            logger.error(f"QA error: {result['error']}")
            # Return partial result instead of raising error
        
        return MCQResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in QA endpoint: {e}", exc_info=True)
        raise HTTPException(500, f"Error processing question: {str(e)}")
