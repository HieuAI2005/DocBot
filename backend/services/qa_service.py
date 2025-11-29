import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.process.qa.answer_mcq import _load_llm, _build_system_prompt, _prompt, _gen_json, _choose, _derive_thresholds, _detect_multi_hint
from backend.process.rag.retrieve import Retriever
from backend.process.qa.reasoning_parser import parse_cot_response

logger = logging.getLogger(__name__)


class QAService:
    """Service for handling multi-choice question answering"""
    
    def __init__(self):
        self.retriever = Retriever()
        self.model = None
        self.tokenizer = None
        # Lazy load LLM when needed
    
    def _ensure_llm_loaded(self):
        """Lazy load LLM on first use"""
        if self.model is None or self.tokenizer is None:
            logger.info("Loading LLM for QA service...")
            self.model, self.tokenizer = _load_llm()
    
    async def answer_mcq(
        self, 
        question: str,
        options: Dict[str, str],
        document_id: Optional[int] = None
    ) -> Dict:
        """
        Answer a multi-choice question using Chain-of-Thought reasoning.
        
        Args:
            question: The question text
            options: Dictionary with keys A, B, C, D and their text
            document_id: Optional document ID to filter retrieval
        
        Returns:
            Dictionary with:
            - answer: List of selected answer(s)
            - probabilities: Dict of probabilities for each option
            - reasoning: Chain-of-thought explanation
            - confidence: Overall confidence score
        """
        try:
            # Ensure LLM is loaded
            self._ensure_llm_loaded()
            
            # Retrieve relevant evidence
            # TODO: Add document_id filtering when retrieve.py supports it
            query = question + " " + " ".join([f"{k}:{v}" for k, v in options.items()])
            evidence = self.retriever.query(query, options=options)
            
            # Get best score and detect multi-hint
            best_score = max((ev.get("score", 0.0) for ev in evidence), default=0.0)
            hint = _detect_multi_hint(question)
            
            # Derive thresholds
            choose_th, fallback_th, multi_th, multi_gap, max_multi = _derive_thresholds(
                hint, best_score, len(evidence)
            )
            
            # Build prompts with CoT
            system_prompt = _build_system_prompt(question, use_cot=True)
            user_prompt = _prompt(question, options, evidence, use_cot=True)
            
            # Generate answer
            probs, response_text = _gen_json(
                self.model, self.tokenizer, 
                system_prompt, user_prompt,
                temperature=0.3, use_cot=True
            )
            
            # Choose final answer
            chosen, choose_meta = _choose(
                probs, choose_th, fallback_th, 
                multi_th, multi_gap, max_multi
            )
            
            # Parse reasoning from response
            parsed = parse_cot_response(response_text)
            
            # Calculate confidence
            top_prob = max(probs.values())
            confidence = top_prob if len(chosen) == 1 else sum(probs[k] for k in chosen) / len(chosen)
            
            return {
                "answer": sorted(chosen),
                "probabilities": probs,
                "reasoning": parsed.get("raw_response", response_text)[:1000],  # Limit reasoning length
                "confidence": round(confidence, 3),
                "evidence_count": len(evidence),
                "evidence_quality": round(best_score, 3)
            }
            
        except Exception as e:
            logger.error(f"Error in QA service: {e}", exc_info=True)
            return {
                "error": str(e),
                "answer": [],
                "probabilities": {k: 0.25 for k in options.keys()},
                "reasoning": "Lỗi khi xử lý câu hỏi",
                "confidence": 0.0
            }


# Singleton instance
_qa_service_instance = None

def get_qa_service() -> QAService:
    """Get or create QA service singleton"""
    global _qa_service_instance
    if _qa_service_instance is None:
        _qa_service_instance = QAService()
    return _qa_service_instance
