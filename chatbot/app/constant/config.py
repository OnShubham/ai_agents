from dotenv import load_dotenv
import os

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
OPEN_API_KEY: str = os.getenv("OPEN_API_KEY", "")

# ── Paths ──────────────────────────────────────────────────────────────────────
# NOTE: We use DATA_PATH to avoid colliding with the system PATH env variable.
DATA_PATH: str = os.getenv("DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "app", "data"))
VECTOR_DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNKING_SIZE: int = 1000
CHUNKING_OVERLAP: int = 200

# ── Embedding model (SentenceTransformer) ──────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ── Reranker model (CrossEncoder) ─────────────────────────────────────────────
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Vector DB ─────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "rag_documents"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 5                    # final chunks returned after reranking
TOP_K_CANDIDATES_MULTIPLIER: int = 3   # fetch TOP_K * multiplier before reranking

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str = "gpt-4o-mini"
TEMPERATURE: float = 0.5
MAX_TOKENS: int = 1024
MAX_HISTORY_TURNS: int = 10       # max conversation turns kept in context