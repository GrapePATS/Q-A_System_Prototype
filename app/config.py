from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
print(f"BASE_DIR = {BASE_DIR}")
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
VECTOR_DB_DIR = STORAGE_DIR / "faiss_index"
METADATA_FILE = STORAGE_DIR / "metadata_store.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5