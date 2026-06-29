# --- Scoring Weights ---
# Must sum to 1.0. Adjust these to favor AI semantic matching vs strict JD rules.
SEMANTIC_WEIGHT = 0.60
SIGNAL_WEIGHT = 0.40

# --- AI Models ---
EMBEDDING_MODEL = "models/gemini-embedding-2"
GEMINI_MODEL = "gemini-2.5-flash"

# --- Pipeline Limits ---
# The number of candidates to retrieve via FAISS before applying the HybridScorer rules.
TOP_N = 20

# --- Directory Paths ---
# Note: Final outputs (CSV/JSON) are generated at the root level by app.py
RAW_DATA_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"