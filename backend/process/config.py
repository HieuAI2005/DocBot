from pathlib import Path

# ============ PATHS ============
INPUT_DIR = Path("inputs/training_input")          # thư mục chứa PDF
OUTPUT_DIR = Path("outputs")        # nơi sinh main.md, images/, answer.md
INDEX_DIR = OUTPUT_DIR / "index"    # nơi lưu chỉ mục RAG
QUESTIONS_CSV = Path("inputs/training_input/question.csv")  # use training questions for evaluation

# ============ EXTRACT ============
IMG_DPI = 300                       # xuất ảnh (tăng từ 200 để cải thiện OCR)
MAX_IMG_PER_PAGE = 50               # an toàn
TABLE_MIN_ROWS = 2                  # bảng tối thiểu
PARA_JOIN_THRESHOLD = 6             # ghép dòng < 6 px coi là 1 đoạn
FALLBACK_COPY_ALL_IMAGES = True     # copy tất cả ảnh nếu không tìm thấy trong MD

# ============ RAG ============
CHUNK_CHARS = 1200
CHUNK_STRIDE = 200
TOPK_BM25 = 8
TOPK_EMB = 8
KEEP_TOPK = 6
EMB_MODEL = "intfloat/multilingual-e5-small"

# ============ QA ============
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_4BIT = False  # Disabled - bitsandbytes not available
MAX_CONTEXT_CHARS = 2500
THRESH_CHOOSE = 0.45              # chọn trực tiếp nếu xác suất >= 45%
THRESH_FALLBACK = 0.25            # fallback nếu xác suất cao nhất >= 25%
THRESH_MULTI = 0.25               # ngưỡng để cân nhắc thêm đáp án phụ
THRESH_MULTI_GAP = 0.10           # chênh lệch tối đa so với đáp án cao nhất
MAX_MULTI_ANS = 3                 # giới hạn số đáp án mặc định

# Ngưỡng tuỳ biến theo chất lượng bằng chứng / hint nhiều đáp án
LOW_EVIDENCE_COUNT = 3
LOW_EVIDENCE_SCORE = 0.25
RELAX_CHOOSE_DELTA = 0.08
RELAX_FALLBACK_DELTA = 0.08
RELAX_MULTI_DELTA = 0.08
MULTI_HINT_CHOOSE_DELTA = 0.05
MULTI_HINT_MULTI_DELTA = 0.05

SEED = 42

# ============ QA ENHANCEMENTS ============
# Chain-of-Thought settings
USE_CHAIN_OF_THOUGHT = True
COT_MAX_REASONING_TOKENS = 384  # Increased from 256 to handle complex reasoning

# Self-consistency settings
USE_SELF_CONSISTENCY = False  # Start disabled, enable after testing
SELF_CONSISTENCY_SAMPLES = 3  # Number of reasoning paths
SELF_CONSISTENCY_TEMPERATURE = 0.7  # Temperature for sampling

# Reasoning logging
SAVE_REASONING_TRACES = True
REASONING_TRACE_DIR = OUTPUT_DIR / "debug" / "qa_reasoning"

# ============ RAG ENHANCEMENTS ============
# Query expansion
QUERY_EXPANSION_ENABLED = True
QUERY_EXPANSION_WITH_OPTIONS = True  # Include option text in query

# Retrieval enhancements
USE_SEMANTIC_RERANKING = False  # Disabled initially (requires additional model)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOPK_MULTIPLIER = 2  # Retrieve 2x chunks then rerank

# Smart chunking
SMART_CHUNKING_ENABLED = True
INCLUDE_CONTEXT_CHUNKS = False  # Include ±1 chunk for context (disabled initially)