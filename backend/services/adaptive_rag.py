import sys
from pathlib import Path
import pickle
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.process.config import (
    OUTPUT_DIR, CHUNK_CHARS, CHUNK_STRIDE, EMB_MODEL,
    TOPK_BM25, TOPK_EMB, KEEP_TOPK
)
from backend.config import (
    VECTOR_STORE_DIR,
    AUTO_INDEX_ON_UPLOAD,
    USE_CONVERSATION_CONTEXT,
    MAX_CONVERSATION_HISTORY,
    MIN_TOPK,
    MAX_TOPK,
    CONFIDENCE_THRESHOLD,
    FEEDBACK_WEIGHT,
)

logger = logging.getLogger(__name__)


class AdaptiveRAG:
    """
    Adaptive RAG system with:
    - Incremental indexing
    - Conversation-aware retrieval
    - Dynamic top-k selection
    - Feedback learning
    """
    
    def __init__(self):
        self.model = SentenceTransformer(EMB_MODEL)
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self.faiss_index: Optional[faiss.Index] = None
        self.feedback_scores: Dict[int, float] = {}  # chunk_id -> score
        
        # Load existing indices if available
        self._load_indices()
    
    def _load_indices(self):
        """Load existing BM25 and FAISS indices"""
        bm25_path = VECTOR_STORE_DIR / "bm25.pkl"
        faiss_path = VECTOR_STORE_DIR / "embeddings.faiss"
        meta_path = VECTOR_STORE_DIR / "emb_meta.pkl"
        
        try:
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    obj = pickle.load(f)
                    self.bm25 = obj["bm25"]
                    self.chunks = obj["chunks"]
                logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")
            
            if faiss_path.exists() and meta_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"Loaded FAISS index")
            
            # Load feedback scores if available
            feedback_path = VECTOR_STORE_DIR / "feedback.pkl"
            if feedback_path.exists():
                with open(feedback_path, "rb") as f:
                    self.feedback_scores = pickle.load(f)
                logger.info(f"Loaded {len(self.feedback_scores)} feedback scores")
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
    
    def _save_indices(self):
        """Save BM25 and FAISS indices"""
        try:
            VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save BM25
            with open(VECTOR_STORE_DIR / "bm25.pkl", "wb") as f:
                pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
            
            # Save FAISS
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(VECTOR_STORE_DIR / "embeddings.faiss"))
            
            # Save metadata
            with open(VECTOR_STORE_DIR / "emb_meta.pkl", "wb") as f:
                pickle.dump(self.chunks, f)
            
            # Save feedback scores
            with open(VECTOR_STORE_DIR / "feedback.pkl", "wb") as f:
                pickle.dump(self.feedback_scores, f)
            
            logger.info("Saved indices successfully")
        except Exception as e:
            logger.error(f"Error saving indices: {e}")
    
    def add_document(self, doc_id: str, text: str, source_path: str):
        """
        Add a new document to the index incrementally
        """
        logger.info(f"Adding document {doc_id} to index")
        
        # Chunk the document
        new_chunks = self._chunk_text(text, doc_id, source_path)
        if not new_chunks:
            logger.warning(f"No chunks created for document {doc_id}")
            return
        
        # Add to chunks list
        start_idx = len(self.chunks)
        self.chunks.extend(new_chunks)
        
        # Update BM25 index
        tokenized_corpus = [c["text"].split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Update FAISS index
        new_texts = [c["text"] for c in new_chunks]
        new_embeddings = self.model.encode(new_texts, normalize_embeddings=True)
        
        if self.faiss_index is None:
            # Create new index
            d = new_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)
        
        self.faiss_index.add(new_embeddings.astype("float32"))
        
        # Save updated indices
        self._save_indices()
        
        logger.info(f"Added {len(new_chunks)} chunks from document {doc_id}")
        return len(new_chunks)
    
    def _chunk_text(self, text: str, doc_id: str, source_path: str) -> List[Dict]:
        """Chunk text with sliding window"""
        chunks = []
        if not text or len(text) < CHUNK_CHARS // 2:
            return [{"doc_id": doc_id, "chunk_id": 0, "text": text, "source": source_path}]
        
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + CHUNK_CHARS
            chunk_text = text[start:end]
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source_path
            })
            chunk_id += 1
            if end >= len(text):
                break
            start += CHUNK_STRIDE
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks with adaptive features:
        - Conversation-aware query expansion
        - Dynamic top-k selection
        - Feedback-based re-ranking
        - Optional document filtering
        """
        if not self.chunks:
            logger.warning("No chunks in index")
            return []
        
        # Conversation-aware query expansion
        if USE_CONVERSATION_CONTEXT and conversation_history:
            query = self._expand_query_with_context(query, conversation_history)
        
        # Determine dynamic top-k
        if top_k is None:
            top_k = KEEP_TOPK
        
        # BM25 retrieval
        bm25_results = self._bm25_retrieve(query, top_k * 2)
        
        # Dense retrieval
        dense_results = self._dense_retrieve(query, top_k * 2)
        
        # Combine results
        combined = self._combine_results(bm25_results, dense_results)
        
        # Filter by document_id if specified
        if document_id:
            # Ensure string comparison (doc_id stored as string in chunks)
            doc_id_str = str(document_id)
            combined = [c for c in combined if str(c.get("doc_id")) == doc_id_str]
            logger.info(f"Filtered to {len(combined)} chunks from document {doc_id_str} (total before filter: {len(self._combine_results(bm25_results, dense_results))})")
        
        # Re-rank with feedback
        combined = self._rerank_with_feedback(combined)
        
        # Return top-k
        return combined[:top_k]
    
    def _expand_query_with_context(self, query: str, history: List[str]) -> str:
        """Expand query with conversation context"""
        if not history:
            return query
        
        # Take last N messages
        recent = history[-MAX_CONVERSATION_HISTORY:]
        context = " ".join(recent)
        
        # Simple expansion: prepend context
        return f"{context} {query}"
    
    def _bm25_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 retrieval"""
        if self.bm25 is None:
            return []
        
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx]
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense retrieval with FAISS"""
        if self.faiss_index is None:
            return []
        
        emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.faiss_index.search(np.array(emb, dtype="float32"), min(top_k, self.faiss_index.ntotal))
        return [(int(i), float(d)) for i, d in zip(I[0], D[0])]
    
    def _combine_results(
        self,
        bm25_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]]
    ) -> List[Dict]:
        """Combine BM25 and dense results with score normalization"""
        candidates = {}
        
        for idx, score in bm25_results:
            candidates.setdefault(idx, {"bm25": 0.0, "dense": 0.0})
            candidates[idx]["bm25"] = max(candidates[idx]["bm25"], score)
        
        for idx, score in dense_results:
            candidates.setdefault(idx, {"bm25": 0.0, "dense": 0.0})
            candidates[idx]["dense"] = max(candidates[idx]["dense"], score)
        
        if not candidates:
            return []
        
        # Normalize scores
        bm25_scores = np.array([v["bm25"] for v in candidates.values()])
        dense_scores = np.array([v["dense"] for v in candidates.values()])
        
        bm25_norm = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-9)
        dense_norm = (dense_scores - dense_scores.min()) / (np.ptp(dense_scores) + 1e-9)
        
        # Combine scores
        results = []
        for (idx, _), bn, dn in zip(candidates.items(), bm25_norm, dense_norm):
            combined_score = 0.5 * bn + 0.5 * dn
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(combined_score)
            chunk["id"] = int(idx)
            results.append(chunk)
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def _rerank_with_feedback(self, chunks: List[Dict]) -> List[Dict]:
        """Re-rank chunks based on user feedback"""
        if not self.feedback_scores:
            return chunks
        
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if chunk_id in self.feedback_scores:
                feedback_boost = self.feedback_scores[chunk_id] * FEEDBACK_WEIGHT
                chunk["score"] = chunk["score"] * (1 + feedback_boost)
        
        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks
    
    def add_feedback(self, chunk_ids: List[int], is_helpful: bool):
        """Add user feedback for chunks"""
        score = 1.0 if is_helpful else -0.5
        
        for chunk_id in chunk_ids:
            if chunk_id in self.feedback_scores:
                # Exponential moving average
                self.feedback_scores[chunk_id] = 0.7 * self.feedback_scores[chunk_id] + 0.3 * score
            else:
                self.feedback_scores[chunk_id] = score
        
        # Save feedback
        self._save_indices()
        logger.info(f"Updated feedback for {len(chunk_ids)} chunks")


# Singleton instance
_adaptive_rag_instance: Optional[AdaptiveRAG] = None


def get_adaptive_rag() -> AdaptiveRAG:
    """Get or create AdaptiveRAG singleton"""
    global _adaptive_rag_instance
    if _adaptive_rag_instance is None:
        _adaptive_rag_instance = AdaptiveRAG()
    return _adaptive_rag_instance
