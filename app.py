import os
import json
import time
import pandas as pd
import streamlit as st

# Import your core AI engines
from src.candidate_profiler import CandidateProfiler
from src.jd_parser import JDParser
from src.embedder import EmbeddingEngine
from src.scorer import HybridScorer
import config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="ScoutAI | Enterprise Recruiting", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR DARK LIQUID GLASS ---
st.markdown("""
<style>
    /* 1. Hide Streamlit Branding & Footer, but KEEP header transparent for the sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide "Press Ctrl+Enter to apply" hint in text areas */
    div[data-testid="InputInstructions"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 2. Sleek Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        background-attachment: fixed;
        color: #f8fafc;
    }
    
    /* Force base text to be light for dark mode */
    h1, h2, h3, h4, p, span, div {
        color: #f1f5f9;
    }
    
    /* 3. Dark Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* 4. Dark Glassmorphic Expanders (Candidate Cards) */
    div[data-testid="stExpander"] details {
        background-color: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 10px;
    }
    div[data-testid="stExpander"] summary {
        background-color: transparent !important;
    }
    
    /* 5. Dark Glassmorphic Status Box */
    div[data-testid="stStatusWidget"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }

    /* 6. Glowing Cyan Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important; /* Bright Cyber Cyan */
        text-shadow: 0px 0px 10px rgba(56, 189, 248, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Layout
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=60) # Placeholder logo
    st.title("ScoutAI Engine")
    
    st.divider()
    
    st.subheader("1️⃣ Candidate Data")
    uploaded_file = st.file_uploader("Upload Dataset (JSON)", type=["json"])

    st.subheader("2️⃣ Job Description")
    default_jd = """Senior Python Backend Engineer
We are looking for a Senior Engineer with 5+ years of experience to join our core AI team. 
You must have deep expertise in Python, FastAPI, and PostgreSQL. 
Experience with Machine Learning pipelines or FAISS is a huge plus. 
Day-to-day, you will build scalable microservices and optimize our ranking engine. 
We value extreme ownership and clear communication. 
Deal-breakers: AWS, GCP, Docker."""

    jd_text = st.text_area("Paste constraints & requirements", value=default_jd, height=250)

    st.divider()
    run_button = st.button("🚀 Initialize AI Search", use_container_width=True, type="primary")

# 3. Main Dashboard Area
st.title("Talent Discovery Dashboard")
st.markdown("Semantic search and hybrid ranking for professional roles.")

# --- THE EMPTY STATE ---
if not run_button:
    st.divider()
    st.info("👈 **Awaiting Input:** Upload your candidate pool and define your JD constraints in the sidebar to begin.")

# 4. Execution Logic

# --- INITIALIZE SESSION STATE ---
# This prevents Streamlit from deleting your results when a download button is clicked.
if "ranked_candidates" not in st.session_state:
    st.session_state.ranked_candidates = None

if run_button:
    if not uploaded_file:
        st.error("⚠️ Please upload a Candidate JSON file in the sidebar first!")
    elif not jd_text.strip():
        st.error("⚠️ Please paste a Job Description in the sidebar!")
    else:
        # Start the animated status box
        with st.status("🤖 AI is analyzing candidates...", expanded=True) as status:
            try:
                # Setup directories and paths
                os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
                os.makedirs(config.PROCESSED_DIR, exist_ok=True)

                raw_candidates_path = os.path.join(config.RAW_DATA_DIR, "candidates.json")
                processed_candidates_path = os.path.join(config.PROCESSED_DIR, "candidates.json")
                processed_jd_path = os.path.join(config.PROCESSED_DIR, "jd_profile.json")

                # Save the uploaded file
                with open(raw_candidates_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # --- Phase 1: Process Candidates ---
                st.write("📄 Ingesting and cleaning candidate resumes...")
                time.sleep(0.8) # Pacing
                
                profiler = CandidateProfiler()
                cleaned_candidates = profiler.process_all(raw_candidates_path)
                with open(processed_candidates_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_candidates, f, indent=4, ensure_ascii=False)
                
                # --- Phase 2: Parse Job Description ---
                st.write("🧠 Parsing Job Description constraints...")
                time.sleep(0.8) # Pacing
                
                parser = JDParser()
                parsed_jd = parser.parse(jd_text)
                jd_data = parsed_jd.model_dump() if hasattr(parsed_jd, "model_dump") else parsed_jd
                with open(processed_jd_path, "w", encoding="utf-8") as f:
                    json.dump(jd_data, f, indent=4, ensure_ascii=False)

                # --- Phase 3 & 4: Embedding Setup ---
                st.write("🌌 Generating semantic vectors in 3072-D space...")
                
                embedder = EmbeddingEngine()
                scorer = HybridScorer(jd_data)

                valid_candidates = []
                candidate_texts = []
                
                for c in cleaned_candidates:
                    skills_list = c.get("skills", [])
                    extracted_skills = []
                    for s in skills_list:
                        if isinstance(s, str):
                            extracted_skills.append(s)
                        elif isinstance(s, dict):
                            skill_name = s.get("name") or s.get("skill") or s.get("title") or (list(s.values())[0] if s else "")
                            if skill_name:
                                extracted_skills.append(str(skill_name))
                    skills_str = ", ".join(extracted_skills) if extracted_skills else "General software engineering"
                    
                    history = c.get("career_history", [])
                    job_summaries = []
                    for job in history:
                        title = job.get("title", "Engineer")
                        desc = job.get("description", "")
                        if desc:
                            job_summaries.append(f"{title}: {desc}")
                    history_str = " | ".join(job_summaries)
                    
                    dense_summary = f"Skills: {skills_str}. Experience: {history_str}"
                    if dense_summary.strip() and len(dense_summary) > 30:
                        valid_candidates.append(c)
                        candidate_texts.append(dense_summary.strip())

                target_jd_text = str(jd_data.get("embedding_text", "")).strip()

                # Generate Vectors
                candidate_vectors = embedder.embed_texts(candidate_texts)
                embedder.build_index(candidate_vectors)
                
                # --- Re-rank with hard logic ---
                st.write("⚖️ Applying hard logic and dealbreaker penalties...")
                time.sleep(1.2) # Final suspense
                
                indices, scores = embedder.search(target_jd_text, top_k=20)
                ranked_candidates = scorer.score_candidates(indices, scores, valid_candidates)

                # --- SAVE TO MEMORY ---
                st.session_state.ranked_candidates = ranked_candidates

                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error", expanded=True)
                st.error(f"Error Details: {str(e)}")
                st.stop()


# --- UI RENDERING & EXPORT (Only runs if data is in memory) ---
if st.session_state.ranked_candidates is not None:
    ranked_candidates = st.session_state.ranked_candidates
    
    st.success("🎯 Top matches compiled successfully.")
    
    # 1. Prepare JSON Data
    json_export = json.dumps(ranked_candidates, indent=4, ensure_ascii=False, default=str)
    
    # 2. Prepare CSV Data
    csv_export_list = []
    for idx, r in enumerate(ranked_candidates, 1):
        match = r.get("candidate", {})
        profile = match.get("profile", {})
        personal_info = profile.get("personal_info", {})
        
        name = (personal_info.get("name") or 
                profile.get("anonymized_name") or
                profile.get("name") or
                match.get("name") or
                match.get("candidate_id") or 
                f"Candidate {r.get('candidate_index', idx)}")

        csv_export_list.append({
            "Rank": idx,
            "Name_or_ID": name,
            "Final_Score": r.get("final_score"),
            "Semantic_Score": r.get("semantic_score"),
            "Signal_Score": r.get("signal_score"),
            "Experience_Years": r.get("experience_years"),
            "Missing_Dealbreakers": r.get("missing_dealbreakers_count"),
            "AI_Reasoning": r.get("explanation")
        })
        
    csv_export = pd.DataFrame(csv_export_list).to_csv(index=False).encode('utf-8')

    # 3. Render Download Buttons
    dl_col1, dl_col2 = st.columns([1, 1])
    with dl_col1:
        st.download_button(
            label="📥 Export as CSV",
            data=csv_export,
            file_name="scoutai_top_candidates.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary"
        )
    with dl_col2:
        st.download_button(
            label="📥 Export as JSON",
            data=json_export,
            file_name="scoutai_top_candidates.json",
            mime="application/json",
            use_container_width=True,
            type="secondary"
        )
        
    st.divider()

    # 4. Render Expander Cards
    for rank, result in enumerate(ranked_candidates[:10], 1):
        match = result.get("candidate", {})
        profile = match.get("profile", {})
        personal_info = profile.get("personal_info", {})
        
        name = (personal_info.get("name") or 
                profile.get("anonymized_name") or
                profile.get("name") or
                match.get("name") or
                match.get("candidate_id") or 
                f"Candidate {result.get('candidate_index', rank)}")
        
        final_score = result.get("final_score", 0.0)
        
        with st.expander(f"🏅 Rank {rank} | {name}", expanded=(rank == 1)):
            
            col_score, col_why = st.columns([1, 3])
            
            with col_score:
                st.metric(label="Match Confidence", value=f"{final_score * 100:.1f}%")
                st.progress(max(0.0, min(final_score, 1.0)))
                
            with col_why:
                st.info(f"**🧠 AI Reasoning:** {result.get('explanation', 'Strong overall profile match based on skills and experience.')}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🛠️ Core Skills")
                raw_skills = match.get("skills", [])
                print_skills = []
                for s in raw_skills:
                    if isinstance(s, str):
                        print_skills.append(s)
                    elif isinstance(s, dict):
                        skill_name = s.get("name") or s.get("skill") or s.get("title") or (list(s.values())[0] if s else "")
                        if skill_name:
                            print_skills.append(str(skill_name))
                
                if print_skills:
                    # Dark Liquid Glass "Skill Pills"
                    html_skills = "".join([
                        f"<span style='display:inline-block; background-color: rgba(255, 255, 255, 0.08); "
                        f"backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.15); "
                        f"padding:4px 12px; border-radius:20px; margin:4px 4px 8px 0px; font-size:13px; font-weight:600; "
                        f"color:#e2e8f0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);'>{s}</span>" 
                        for s in print_skills
                    ])
                    st.markdown(html_skills, unsafe_allow_html=True)
                else:
                    st.caption("No specific skills extracted.")
                    
            with col2:
                st.markdown("#### 💼 Career Trajectory")
                history = match.get("career_history", [])
                if history:
                    for job in history:
                        title = job.get("title", "Unknown Role")
                        company = job.get("company", "Unknown Company")
                        months = float(job.get("duration_months", 0))
                        st.markdown(f"• **{title}** at {company} *(~{months/12:.1f} yrs)*")
                else:
                    st.caption("No experience history found.")