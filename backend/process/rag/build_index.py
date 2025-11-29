import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..config import OUTPUT_DIR, INDEX_DIR, CHUNK_CHARS, CHUNK_STRIDE, EMB_MODEL

def load_documents() -> List[Dict]:
    """
    Load all main.md files from OUTPUT_DIR.
    Returns a list of dicts: {'id': str, 'text': str, 'path': Path}
    """
    docs = []
    # Assuming structure: outputs/<stem>/main.md
    if not OUTPUT_DIR.exists():
        return docs
        
    for pdf_dir in sorted(OUTPUT_DIR.glob("*")):
        if pdf_dir.is_dir():
            md_path = pdf_dir / "main.md"
            if md_path.exists():
                try:
                    text = md_path.read_text(encoding="utf-8", errors="ignore")
                    docs.append({
                        "id": pdf_dir.name,
                        "text": text,
                        "path": md_path
                    })
                except Exception as e:
                    print(f"Error reading {md_path}: {e}")
    return docs

def chunk_text(text: str, chunk_size: int, stride: int) -> List[str]:
    """
    Simple sliding window chunking.
    """
    chunks = []
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += stride
    return chunks

def build_index():
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print(f"No documents found in {OUTPUT_DIR}")
        return

    print(f"Found {len(docs)} documents. Chunking...")
    all_chunks = []
    
    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_CHARS, CHUNK_STRIDE)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc["id"],
                "chunk_id": i,
                "text": chunk,
                "source": str(doc["path"])
            })
    
    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks created.")
        return
    
    # 1. BM25 Index
    print("Building BM25 index...")
    tokenized_corpus = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save BM25 and chunks (as expected by retrieve.py)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunks": all_chunks 
        }, f)
        
    # 2. Dense Index (FAISS)
    print(f"Loading embedding model: {EMB_MODEL}...")
    model = SentenceTransformer(EMB_MODEL)
    
    print("Encoding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    print("Building FAISS index...")
    d = embeddings.shape[1]
    # Inner Product for normalized embeddings = Cosine Similarity
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings.astype("float32"))
    
    faiss.write_index(index, str(INDEX_DIR / "embeddings.faiss"))
    
    # Save metadata (chunks) separately as well, to satisfy retrieve.py loading it
    with open(INDEX_DIR / "emb_meta.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index built successfully in {INDEX_DIR}")

if __name__ == "__main__":
    build_index()