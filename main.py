import os
import json
import pandas as pd
from dotenv import load_dotenv
import config

# Import your three core engines
from src.candidate_profiler import CandidateProfiler
from src.jd_parser import JDParser
from src.embedder import EmbeddingEngine
from src.scorer import HybridScorer

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file.")
        return

    print("✅ System initialized. Keys loaded.\n")

    # --- Setup Directories (Aligned with app.py) ---
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    # Define paths
    raw_candidates_path = os.path.join(config.RAW_DATA_DIR, "candidates.json")
    raw_jd_path = os.path.join(config.RAW_DATA_DIR, "jd.txt")
    processed_candidates_path = os.path.join(config.PROCESSED_DIR, "candidates.json")
    processed_jd_path = os.path.join(config.PROCESSED_DIR, "jd_profile.json")

    try:
        # --- PHASE 1: Candidates ---
        print("🚀 Phase 1: Ingesting and cleaning candidate resumes...")
        profiler = CandidateProfiler()
        cleaned_candidates = profiler.process_all(raw_candidates_path)
        
        with open(processed_candidates_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_candidates, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Loaded and cleaned {len(cleaned_candidates)} candidates.")

        # --- PHASE 2: Job Description ---
        print("\n🚀 Phase 2: Parsing Job Description constraints...")
        if not os.path.exists(raw_jd_path):
            raise FileNotFoundError(f"Missing {raw_jd_path}. Please create it and paste your JD.")
            
        with open(raw_jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
            
        parser = JDParser()
        parsed_jd = parser.parse(jd_text)
        
        jd_data = parsed_jd.model_dump() if hasattr(parsed_jd, "model_dump") else parsed_jd
        with open(processed_jd_path, "w", encoding="utf-8") as f:
            json.dump(jd_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Loaded JD Profile for: {jd_data.get('role_title', 'Unknown Role')}")

       # --- PHASE 3: Embedding & Semantic Search ---
        print("\n🚀 Phase 3: Generating semantic vectors in 3072-D space...")
        embedder = EmbeddingEngine()
        scorer = HybridScorer(jd_data)

        valid_candidates = []
        candidate_texts = []
        
        # Exact extraction loop aligned with app.py
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

        if not valid_candidates:
            raise ValueError("❌ No valid text found in the candidates data! Check your JSON structure.")

        target_jd_text = str(jd_data.get("embedding_text", "")).strip()

        print(f"   -> Generating embeddings for {len(valid_candidates)} candidates...")
        candidate_vectors = embedder.embed_texts(candidate_texts)
        
        print("   -> Building FAISS Index...")
        embedder.build_index(candidate_vectors)

        print("   -> Searching for Top 20 Semantic Matches...")
        indices, scores = embedder.search(target_jd_text, top_k=20)

        # --- PHASE 4: Hybrid Scoring & Re-ranking ---
        print("\n🚀 Phase 4: Applying hard logic and dealbreaker penalties...")
        ranked_candidates = scorer.score_candidates(indices, scores, valid_candidates)

        # --- OUTPUT PARITY (CSV & JSON Generation exactly like app.py) ---
        print("\n💾 Generating Export Files...")
        
        # 1. JSON Export
        with open("scoutai_top_candidates.json", "w", encoding="utf-8") as f:
            json.dump(ranked_candidates, f, indent=4, ensure_ascii=False, default=str)
            
        # 2. CSV Export Logic
        AI_SKILLS = {
            "python", "machine learning", "deep learning", "nlp", "llm",
            "tensorflow", "pytorch", "scikit-learn", "sql", "data science",
            "computer vision", "transformers", "langchain", "faiss", "rag",
            "generative ai", "huggingface", "keras", "spark", "airflow"
        }

        csv_export_list = []
        for idx, r in enumerate(ranked_candidates, 1):
            match = r.get("candidate", {})
            profile = match.get("profile", {})

            candidate_id = match.get("candidate_id", f"CAND_{idx:07d}")
            title = profile.get("current_title", "Unknown")
            yrs = profile.get("years_of_experience", 0)
            response_rate = match.get("redrob_signals", {}).get("recruiter_response_rate", 0)

            ai_skill_count = sum(
                1 for s in match.get("skills", [])
                if s.get("name", "").lower() in AI_SKILLS
            )

            reasoning = f"{title} with {yrs} yrs; {ai_skill_count} AI core skills; response rate {response_rate:.2f}."

            csv_export_list.append({
                "candidate_id": candidate_id,
                "rank": idx,
                "score": round(r.get("final_score", 0), 4),
                "reasoning": reasoning
            })

        pd.DataFrame(csv_export_list).to_csv("scoutai_top_candidates.csv", index=False)
        print("✅ Saved scoutai_top_candidates.csv and scoutai_top_candidates.json to root directory.")

        # --- CLI LEADERBOARD (Matched to frontend UI cards) ---
        print("\n🏆 FINAL TOP 10 HYBRID LEADERBOARD 🏆")
        print("=" * 60)
        
        for rank, result in enumerate(ranked_candidates[:10], 1):
            match = result.get("candidate", {})
            profile = match.get("profile", {})
            personal_info = profile.get("personal_info", {})
            
            # Exact fallback logic from app.py
            name = (personal_info.get("name") or 
                    profile.get("anonymized_name") or
                    profile.get("name") or
                    match.get("name") or
                    match.get("candidate_id") or 
                    f"Candidate {result.get('candidate_index', rank)}")
            
            final_score = result.get("final_score", 0.0)
            print(f"🏅 Rank {rank} | {name} | Match: {final_score * 100:.1f}%")
            print(f"   🧠 AI Reasoning: {result.get('explanation')}")
            
            # Print core skills summary
            raw_skills = match.get("skills", [])
            print_skills = [
                s if isinstance(s, str) else str(s.get("name") or s.get("skill") or s.get("title") or (list(s.values())[0] if s else ""))
                for s in raw_skills
            ]
            print_skills = [s for s in print_skills if s] # filter empties
            
            if print_skills:
                print(f"   🛠️  Skills: {', '.join(print_skills[:8])}")
            print("-" * 60)
                        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()