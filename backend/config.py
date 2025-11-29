from pathlib import Path
import os

# ============ PATHS ============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DATABASE_PATH = DATA_DIR / "database.db"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ============ VIRTUAL ENVIRONMENT ============
VENV_ACTIVATE = "/home/hiwe/environment/nlp_viettel/bin/activate"

# ============ DATABASE ============
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

# ============ API SETTINGS ============
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative frontend port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# ============ UPLOAD SETTINGS ============
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf"}

# ============ ADAPTIVE RAG SETTINGS ============
# Incremental indexing
AUTO_INDEX_ON_UPLOAD = True
REINDEX_THRESHOLD = 10  # Re-index after N documents

# Conversation-aware retrieval
USE_CONVERSATION_CONTEXT = True
MAX_CONVERSATION_HISTORY = 5

# Dynamic retrieval
MIN_TOPK = 3
MAX_TOPK = 10
CONFIDENCE_THRESHOLD = 0.5

# Feedback learning
ENABLE_FEEDBACK = True
FEEDBACK_WEIGHT = 0.2

# ============ CHAT SETTINGS ============
STREAM_CHUNK_SIZE = 512
MAX_RESPONSE_TOKENS = 2048
