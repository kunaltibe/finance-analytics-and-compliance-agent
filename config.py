"""
Centralized configuration management for the Financial Analytics & Compliance Agent.
All settings are loaded from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- PATHS --- 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"

# --- LLM CONFIG (OLLAMA) --- 
# Ollama runs locally, so no API key is needed (we use a dummy one for LangChain compatibility)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE","0.0"))

# --- VECTOR DB (CHROMADB - LOCAL) --- 
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "finance_policies")

# --- RELATION DB (SQLITE - LOCAL) ---
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/finance_bank.db")

# --- RAG CONFIG ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- AGENT CONFIG ---
MAX_AGENT_RETRIES = int(os.getenv("MAX_AGENT_RETRIES", "3"))

# ==========================================
# OPTIONAL: SECRETS (Uncomment when needed)
# ==========================================
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not set in .env")