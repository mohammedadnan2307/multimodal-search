import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"

# Model settings
CLIP_MODEL = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
BATCH_SIZE = 32
DEVICE = "cuda" if os.environ.get("USE_CUDA") else "cpu"

# Text chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 50

# Search
DEFAULT_TOP_K = 10
SIMILARITY_THRESHOLD = 0.1

# Ensure directories exist
for directory in [DATA_DIR, PDF_DIR, IMAGES_DIR, INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
