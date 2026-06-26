# India Runs Hackathon — AI Candidate Ranking System
**Goal:** Build an AI system that ranks candidates the way a great recruiter would — by deeply understanding the job description and evaluating the full candidate picture (skills, career history, behavioral signals, platform activity), not just keyword matching.
**Output:** A ranked shortlist CSV that a recruiter can trust, with score breakdowns.

## System Overview
Job Description (text)
        │
        ▼
┌──────────────────┐
│   JD Parser      │  ← Gemini 3.0 Flash extracts structured requirements
│  (jd_parser.py)  │    skills, experience, context, soft signals, deal-breakers
└────────┬─────────┘
         │  Structured JD Profile (JSON)
         ▼
┌──────────────────────────────────────────────────────────┐
│                   Ranking Pipeline                        │
│                                                          │
│  Candidate Data (CSV/JSON)                               │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────┐    ┌──────────────────┐            │
│  │Candidate Profiler│    │  Embedding Engine │            │
│  │(profiler.py)    │───▶│  (embedder.py)   │            │
│  └─────────────────┘    └────────┬─────────┘            │
│                                  │ Semantic Score         │
│         ┌────────────────────────┘                       │
│         │                                                │
│  ┌──────▼──────────┐                                    │
│  │  Signal Scorer  │  ← Career trajectory, activity,    │
│  │(signal_scorer.py)│   skill depth, behavioral signals  │
│  └──────┬──────────┘                                    │
│         │ Signal Score                                   │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │ Hybrid Ranker│  ← Weighted combination of all scores  │
│  │ (ranker.py)  │                                        │
│  └──────┬───────┘                                        │
└─────────┼────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐
│   Output Generator   │  → ranked_output.csv + report
│ (output_generator.py)│    (predefined format + score breakdown)
└──────────────────────┘
          │
          ▼ (Stretch)
┌──────────────────────┐
│    Streamlit UI      │  → Upload JD → View ranked shortlist live
│      (app.py)        │
└──────────────────────┘

## Project File Structure
india-runs-ranker/
├── .env                       # API keys (never commit)
├── .env.example               # Template for .env
├── requirements.txt
├── README.md
├── main.py                    # CLI entry point
├── config.py                  # All weights, model names, constants
├── data/
│   ├── raw/                   # Drop dataset files here (CSV/JSON)
│   ├── processed/             # Auto-generated processed profiles
│   └── output/                # Final ranked output files go here
├── src/
│   ├── __init__.py
│   ├── jd_parser.py
│   ├── candidate_profiler.py
│   ├── embedder.py
│   ├── signal_scorer.py
│   ├── ranker.py
│   └── output_generator.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── helpers.py
├── notebooks/
│   └── eda.ipynb              # EDA after dataset arrives
└── app.py                     # Streamlit UI (Phase 6)

## MVP Features (Must Ship)
* **JD Understanding** — Gemini 3.0 Flash parses JD into structured JSON (skills, seniority, context, soft signals)
* **Semantic Matching** — `gemini-embedding-2` for JD ↔ candidate semantic similarity
* **Signal Scoring** — Rule-based scoring for career trajectory, skill depth, activity signals
* **Hybrid Ranker** — Configurable weighted combination of semantic + signal scores
* **Ranked Output** — CSV file in predefined format with scores

## Tech Stack
| Component | Library | Why |
| :--- | :--- | :--- |
| **LLM** | `google-generativeai` (Gemini 3.0 Flash) | Best structured extraction, blazingly fast |
| **Embeddings** | `google-generativeai` (gemini-embedding-2) | Explicitly requested, natively integrated with the LLM API |
| **Vector Search** | `faiss-cpu` | Fast cosine similarity at scale |
| **Data** | `pandas`, `numpy` | Standard, reliable |
| **UI (stretch)** | `streamlit` | Fastest path to a working demo |
| **Config** | `python-dotenv` | Env management |
| **DX** | `rich`, `tqdm` | Nice logs, progress bars |