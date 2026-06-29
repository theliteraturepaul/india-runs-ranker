## 📊 Data Exploration & Processing (EDA)

Real-world candidate data is deeply nested and structurally complex. Before running the ranking algorithm, we perform rigorous data hygiene and feature extraction to build a clean evaluation matrix.

The exploratory data analysis (`notebooks/eda.ipynb`) executes the following pipeline:

1. **JSON Normalization:** Flattens the nested `data/raw/candidates.json` schema into a tabular format using Pandas.
2. **Schema Mapping & Null Handling:** Enforces strict checks for required fields (`candidate_id`, `skills`, `education`, `career_history`) and maps out missing values.
3. **Feature Engineering:** Extracts actionable metrics across four core pillars:
   * **Profile:** Calculates experience bands and normalizes current job titles and industries.
   * **Skills:** Evaluates not just skill presence, but aggregates proficiency levels, duration of use (months), and total endorsements.
   * **Career History:** Calculates role tenure (duration in years) and weights current versus past roles.
   * **Redrob Signals:** Parses proprietary hackathon signals for advanced candidate matching.

### 🏗️ Data Pipeline Architecture

```mermaid
graph TD
    A[Raw JSON: candidates.json] --> B(Pandas Normalization)
    B --> C{Feature Extraction Pillars}
    
    C -->|Skills| D[Proficiency & Endorsement Scoring]
    C -->|Profile| E[Experience Banding]
    C -->|Career| F[Role Tenure & Current Status]
    C -->|Signals| G[Redrob Signal Parsing]
    
    D --> H[Cleaned Feature Matrix]
    E --> H
    F --> H
    G --> H
    
    H --> I((Ranking Engine))
    
    style A fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    style B fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    style C fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style H fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    style I fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
```




## ⚙️ Core Engine Architecture (`src/`)

The ScoutAI ranking engine does not rely solely on basic keyword matching or pure black-box LLM scoring. Instead, it utilizes a **Hybrid Ranking System**, combining the semantic understanding of Google's Gemini models with high-speed FAISS vector search and strict, rules-based recruiter logic.

The backend is split into four primary modules:

### 1. `candidate_profiler.py` (Data Ingestion)
Transforms raw, unpredictable JSON into structured profiles. Crucially, it dynamically generates a dense `text_representation` for each candidate by synthesizing their profile, skills, career history, and projects into a single cohesive string optimized for vector embedding.

### 2. `jd_parser.py` (LLM Structuring)
Job descriptions are messy. This module uses **Gemini** and **Pydantic** to parse raw text JDs into a strict `JDProfileSchema`. It automatically extracts:
* Required vs. Preferred skills
* Minimum and maximum experience ranges
* Critical "deal-breakers"
* A synthesized embedding text for vector matching

### 3. `embedder.py` (Vector Search)
Powered by `models/gemini-embedding-2`, this module converts the synthesized candidate texts and JD profiles into high-dimensional vectors. We use a **FAISS Inner-Product Index (`IndexFlatIP`)** with L2 normalization to perform lightning-fast, scalable cosine-similarity searches to find the top semantically matched candidates.

### 4. `scorer.py` (The Hybrid Evaluator)
Semantic similarity is not enough (e.g., an LLM might confuse a Junior Data Analyst with a Senior Data Scientist because the vectors are close). The `HybridScorer` takes the Top-K FAISS results and re-ranks them using strict business logic:
* **Signal Scoring:** Evaluates precise matches for required/preferred skills (utilizing a massive built-in synonym/alias map to equate "JS" with "JavaScript" and "K8s" with "Kubernetes").
* **Experience Clamping:** Penalizes candidates who fall outside the parsed JD experience range.
* **Dealbreaker Penalties:** Applies aggressive mathematical penalties for missing non-negotiable requirements.

---

### 🧠 System Architecture & Data Flow

```mermaid
graph TD
    subgraph Input Layer
        A[Raw Job Description]
        B[Cleaned Candidate Profiles]
    end

    subgraph NLP & Embedding Layer
        C[JD Parser<br/><i>Gemini + Pydantic</i>]
        D[Embedding Engine<br/><i>Gemini-Embedding-2</i>]
    end

    subgraph Search & Scoring Layer
        E[(FAISS Vector Index<br/><i>Cosine Similarity</i>)]
        F[Hybrid Scorer<br/><i>Rules + Aliases</i>]
    end

    A -->|Raw Text| C
    C -->|Structured JD Schema| F
    C -->|Synthesized JD Text| D
    B -->|Dense Text Representation| D
    B -->|Structured Candidate Data| F

    D -->|JD Vector| E
    D -->|Candidate Vectors| E

    E -->|Top-K Semantic Matches & Scores| F
    F -->|Applies Weights, Exp. Range & Dealbreakers| G[Final Ranked Output .CSV]

    style A fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    style B fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    style C fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style D fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style E fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    style F fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
    style G fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
```




## 🖥️ Frontend & User Interface (`app.py`)

To make the engine accessible to recruiters and judges, ScoutAI includes a premium, interactive web application built with **Streamlit**. 

Rather than a standard, out-of-the-box layout, the UI is heavily customized using injected CSS to feature a **"Dark Liquid Glass"** aesthetic—utilizing glassmorphic sidebars, glowing cyan metrics, and custom HTML "skill pills" for an enterprise-grade feel.

### ✨ Key Application Features
* **Stateful Execution:** Utilizes Streamlit's `session_state` to cache the heavy AI ranking results in memory, ensuring the app doesn't re-run the entire pipeline when a user clicks the download buttons.
* **Real-time Pipeline Tracking:** Features an animated status box that walks the user through the 4-phase backend execution (Ingestion -> Parsing -> Vector Search -> Hybrid Scoring).
* **Rich Candidate Cards:** The top 10 candidates are rendered in expandable glassmorphic cards containing:
  * A glowing **Match Confidence** progress bar.
  * Explicit **AI Reasoning** explaining exactly *why* the candidate was ranked there.
  * A breakdown of core skills and career trajectory.
* **1-Click Export:** Generates the hackathon-required `output.csv` and a detailed JSON export instantly from the UI.

### 🔄 User Flow Architecture

```mermaid
graph TD
    subgraph Sidebar [User Input - Sidebar]
        A[Upload Candidates JSON]
        B[Paste Job Description]
        C((Initialize Engine))
    end

    subgraph Memory [Streamlit Session State]
        D[Phase 1: Ingest & Clean]
        E[Phase 2: LLM JD Parse]
        F[Phase 3: 3072-D Vector Search]
        G[Phase 4: Hybrid Scoring & Penalties]
    end

    subgraph Dashboard [UI Rendering - Main Dashboard]
        H[Ranked Candidate Cards<br/><i>Confidence Metrics & Reasoning</i>]
        I[CSV Export]
        J[JSON Export]
    end

    A --> C
    B --> C
    C -->|Triggers| D
    D --> E
    E --> F
    F --> G
    
    G -->|Saves to Memory| H
    G -->|Formats Data| I
    G -->|Formats Data| J

    style A fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    style B fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    style C fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    style D fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style E fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style F fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style G fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style H fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    style I fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
    style J fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
```




## ⚙️ Headless Execution (`main.py`)

For pure pipeline testing without the web interface, ScoutAI provides a robust Command Line Interface (CLI) entry point[cite: 10]. 

`main.py` executes the exact same 4-phase architecture as the frontend application but runs entirely headless[cite: 10]. It outputs a clean, formatted leaderboard directly to the console and natively generates the required export files[cite: 10]. This is ideal for rapid local evaluation or server environments where a web UI is unnecessary.

### 🚀 Terminal Usage

Execute the pipeline directly from your terminal:
\`\`\`bash
python main.py
\`\`\`

*Note: Before running, ensure you have your target job description saved as a text file at `data/raw/jd.txt`*[cite: 10].

### ✨ Key Script Features
* **Automated Infrastructure:** The script automatically initializes the environment and ensures the `RAW_DATA_DIR` and `PROCESSED_DIR` directories are created[cite: 10].
* **Full 4-Phase Execution:** It seamlessly runs candidate ingestion, LLM job description parsing, 3072-D FAISS vector generation, and hybrid re-ranking[cite: 10].
* **Output Parity:** Automatically formats and saves the `scoutai_top_candidates.csv` and `scoutai_top_candidates.json` files directly to your root directory for easy submission[cite: 10].
* **Rich CLI Leaderboard:** Prints the Top 10 candidates directly to the terminal, complete with their match percentage, the AI's step-by-step reasoning for the score, and a curated list of their core skills[cite: 10].

### 🖥️ CLI Execution Flow

```mermaid
graph TD
    subgraph Terminal [Terminal Execution]
        A[python main.py]
    end

    subgraph Core [Core Pipeline]
        B[Phase 1: Ingest & Clean Candidates]
        C[Phase 2: Parse JD txt]
        D[Phase 3: FAISS Vector Search]
        E[Phase 4: Hybrid Re-ranking]
    end

    subgraph Output [System Output]
        F[Top 10 Console Leaderboard]
        G[scoutai_top_candidates.csv]
        H[scoutai_top_candidates.json]
    end

    A -->|Initializes| B
    B --> C
    C --> D
    D --> E
    E -->|Prints to stdout| F
    E -->|Saves to disk| G
    E -->|Saves to disk| H

    style A fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    style B fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style C fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style D fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style E fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style F fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    style G fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
    style H fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
```




## 🎛️ System Configuration (`config.py`)

ScoutAI is designed to be highly modular. Rather than hardcoding hyperparameters into the ranking engine, all global variables, model selections, and directory paths are centrally managed in `config.py`. 

This allows recruiters or engineers to easily tune the system—such as adjusting the scoring weights to favor strict skill-matching versus broader semantic discovery—without touching the core backend logic.

### ⚙️ Core Parameters

* **Scoring Weights:** The engine balances AI semantic search (`0.60`) with rules-based recruiter signals (`0.40`). *(Note: These must sum to 1.0)*.
* **AI Models:** Powered by `gemini-2.5-flash` for job description parsing and `models/gemini-embedding-2` for generating the 3072-D vector embeddings.
* **Pipeline Limits:** The `TOP_N` limit dictates that the FAISS index retrieves the top 20 candidates before applying the intensive HybridScorer logic.
* **Directory Management:** Centralized control for the `data/raw/` and `data/processed/` data pipelines.

### 🧠 Configuration Data Flow

```mermaid
graph TD
    subgraph Config [Central Configuration - config.py]
        A["Scoring Weights<br/><i>Semantic 0.60 | Signal 0.40</i>"]
        B["AI Models<br/><i>Gemini 2.5 Flash | Embedding-2</i>"]
        C["Pipeline Limits<br/><i>Top 20 FAISS Retrieval</i>"]
        D["Directory Paths<br/><i>raw/ | processed/</i>"]
    end

    subgraph CoreModules [Engine Modules]
        E[src/scorer.py]
        F[src/jd_parser.py]
        G[src/embedder.py]
        H[main.py & app.py]
    end

    A -.->|Injects Weights| E
    B -.->|Defines Models| F
    B -.->|Defines Models| G
    C -.->|Sets Thresholds| E
    D -.->|Sets File I/O| H

    style A fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    style B fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    style C fill:#f39c12,stroke:#f1c40f,stroke-width:2px,color:#fff
    style D fill:#d63031,stroke:#fab1a0,stroke-width:2px,color:#fff
    style E fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style F fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style G fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    style H fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
```







