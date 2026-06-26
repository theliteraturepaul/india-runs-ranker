# Scoring weights (must sum to 1.0)
SEMANTIC_WEIGHT = 0.45
SIGNAL_WEIGHT = 0.35
LLM_RERANK_WEIGHT = 0.20

# Models
EMBEDDING_MODEL = "models/gemini-embedding-2"
GEMINI_MODEL = "gemini-2.5-flash"

# Top-N shortlist size
TOP_N = 20

# Paths
RAW_DATA_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR = "data/output/"