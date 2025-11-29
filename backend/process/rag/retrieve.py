import pickle, faiss, numpy as np, json, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ..config import (
    INDEX_DIR, TOPK_BM25, TOPK_EMB, KEEP_TOPK, EMB_MODEL,
    QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_WITH_OPTIONS
)

class Retriever:
    def __init__(self):
        with open(INDEX_DIR/"bm25.pkl","rb") as f:
            obj = pickle.load(f)
            self.bm25: BM25Okapi = obj["bm25"]
            self.chunks = obj["chunks"]
        self.model = SentenceTransformer(EMB_MODEL)
        self.faiss = faiss.read_index(str(INDEX_DIR/"embeddings.faiss"))
        with open(INDEX_DIR/"emb_meta.pkl","rb") as f:
            self.meta = pickle.load(f)

    def _expand_query(self, q: str, options: Optional[Dict[str,str]] = None) -> str:
        """
        Expand query with options to improve retrieval.
        
        Args:
            q: Original question
            options: Dictionary of options {A: text, B: text, ...}
        
        Returns:
            Expanded query string
        """
        if not QUERY_EXPANSION_ENABLED:
            return q
        
        expanded = q
        
        # Add option keywords if enabled
        if QUERY_EXPANSION_WITH_OPTIONS and options:
            # Extract key terms from options (first few words)
            option_keywords = []
            for key, text in options.items():
                # Take first 50 chars or first sentence
                snippet = text[:50] if len(text) > 50 else text
                option_keywords.append(snippet)
            
            # Append to query
            expanded = q + " " + " ".join(option_keywords)
        
        return expanded

    def _bm25(self, q: str) -> List[Tuple[int,float]]:
        toks = q.split()
        scores = self.bm25.get_scores(toks)
        idx = np.argsort(scores)[::-1][:TOPK_BM25]
        return [(int(i), float(scores[i])) for i in idx]

    def _dense(self, q: str) -> List[Tuple[int,float]]:
        emb = self.model.encode([q], normalize_embeddings=True)
        D,I = self.faiss.search(np.array(emb, dtype="float32"), TOPK_EMB)
        return [(int(i), float(d)) for i,d in zip(I[0], D[0])]

    def query(self, q: str, options: Optional[Dict[str,str]] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            q: Query string
            options: Optional dictionary of answer options for query expansion
        
        Returns:
            List of top-K chunks with scores
        """
        # Expand query if enabled
        expanded_q = self._expand_query(q, options)
        
        bm = self._bm25(expanded_q)   # [(idx, score)]
        de = self._dense(expanded_q)  # [(idx, score)]

        cand = {}
        for i,s in bm:
            cand.setdefault(i, {"bm25":0.0, "dense":0.0})
            cand[i]["bm25"] = max(cand[i]["bm25"], float(s))
        for i,s in de:
            cand.setdefault(i, {"bm25":0.0, "dense":0.0})
            cand[i]["dense"] = max(cand[i]["dense"], float(s))

        if not cand: return []
        sb = np.array([v["bm25"] for v in cand.values()])
        se = np.array([v["dense"] for v in cand.values()])
        # chuẩn hoá độc lập (NumPy 2.0 compatible)
        sb = (sb - sb.min())/(np.ptp(sb)+1e-9)
        se = (se - se.min())/(np.ptp(se)+1e-9)

        outs=[]
        for (k,(v,sb_n,se_n)) in zip(cand.keys(), zip(cand.values(), sb, se)):
            sc = 0.5*sb_n + 0.5*se_n
            c = self.chunks[k].copy(); c["score"]=float(sc); c["id"]=int(k)
            outs.append(c)
        outs.sort(key=lambda x:x["score"], reverse=True)
        return outs[:KEEP_TOPK]