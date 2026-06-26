import os
import json
from dotenv import load_dotenv
import config
from src.scorer import HybridScorer

# Import your three engines
from src.candidate_profiler import CandidateProfiler
from src.jd_parser import JDParser
from src.embedder import EmbeddingEngine

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file.")
        return

    print("✅ System initialized. Keys loaded.\n")

    # Define paths
    raw_candidates_path = os.path.join(config.RAW_DATA_DIR, "candidates.json")
    raw_jd_path = os.path.join(config.RAW_DATA_DIR, "jd.txt")
    processed_candidates_path = os.path.join(config.PROCESSED_DIR, "candidates.json")
    processed_jd_path = os.path.join(config.PROCESSED_DIR, "jd_profile.json")

    try:
        # --- PHASE 1: Candidates ---
        if not os.path.exists(processed_candidates_path):
            print("🚀 Phase 1: Processing Candidates...")
            profiler = CandidateProfiler()
            cleaned_candidates = profiler.process_all(raw_candidates_path)
            with open(processed_candidates_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_candidates, f, indent=4, ensure_ascii=False)
        
        with open(processed_candidates_path, "r", encoding="utf-8") as f:
            candidates_data = json.load(f)
        print(f"✅ Loaded {len(candidates_data)} candidates.")

        # --- PHASE 2: Job Description ---
        if not os.path.exists(processed_jd_path):
            print("🚀 Phase 2: Parsing Job Description...")
            with open(raw_jd_path, "r", encoding="utf-8") as f:
                jd_text = f.read()
            parser = JDParser()
            parsed_jd = parser.parse(jd_text)
            
            json_data = parsed_jd.model_dump() if hasattr(parsed_jd, "model_dump") else parsed_jd
            with open(processed_jd_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

        with open(processed_jd_path, "r", encoding="utf-8") as f:
            jd_data = json.load(f)
        print(f"✅ Loaded JD Profile for: {jd_data.get('role_title', 'Unknown Role')}\n")

       # --- PHASE 3: Embedding & Semantic Search ---
        print("🚀 Phase 3: Semantic Vector Search...")
        embedder = EmbeddingEngine()

        # 1. Dynamically build a rich text summary for every candidate
        print("   -> Extracting candidate data...")
        valid_candidates = []
        candidate_texts = []
        
        for c in candidates_data:
            # --- SAFE SKILLS EXTRACTION ---
            skills_list = c.get("skills", [])
            extracted_skills = []
            
            for s in skills_list:
                if isinstance(s, str):
                    extracted_skills.append(s)
                elif isinstance(s, dict):
                    # Try common keys like 'name', 'skill', or 'title'
                    skill_name = s.get("name") or s.get("skill") or s.get("title") or str(list(s.values())[0])
                    if skill_name:
                        extracted_skills.append(str(skill_name))
            
            skills_str = ", ".join(extracted_skills) if extracted_skills else "General software engineering"
            
            # --- SAFE CAREER HISTORY EXTRACTION ---
            history = c.get("career_history", [])
            job_summaries = []
            for job in history:
                title = job.get("title", "Engineer")
                desc = job.get("description", "")
                if desc:
                    job_summaries.append(f"{title}: {desc}")
            
            history_str = " | ".join(job_summaries)
            
            # Smash it all together into one dense paragraph for the AI
            dense_summary = f"Skills: {skills_str}. Experience: {history_str}"
            
            # Only keep candidates that actually have text to embed
            if dense_summary.strip() and len(dense_summary) > 30:
                valid_candidates.append(c)
                candidate_texts.append(dense_summary.strip())

        if not valid_candidates:
            raise ValueError("❌ No valid text found in the candidates data! Check your JSON structure.")

        # Update our dataset to only include the valid ones so our rankings match up
        candidates_data = valid_candidates
        target_jd_text = str(jd_data.get("embedding_text", "")).strip()

        # 2. Generate Vectors
        print(f"   -> Generating embeddings for {len(valid_candidates)} candidates...")
        candidate_vectors = embedder.embed_texts(candidate_texts)
        
        # 3. Build FAISS Index
        print("   -> Building FAISS Index...")
        embedder.build_index(candidate_vectors)

        # 4. Search (Retrieve a wider net, e.g., Top 20, so we have room to re-rank)
        print("   -> Searching for Top 20 Semantic Matches...\n")
        indices, scores = embedder.search(target_jd_text, top_k=20)

        # --- PHASE 4: Hybrid Scoring & Re-ranking ---
        print("🚀 Phase 4: Hybrid Re-ranking...")
        # jd_data is already a dictionary from Phase 2
        scorer = HybridScorer(jd_data) 
        
        # Get the final ranked list
        ranked_candidates = scorer.score_candidates(indices, scores, candidates_data)

        # --- OUTPUT RESULTS ---
        print("\n🏆 FINAL TOP 5 HYBRID LEADERBOARD 🏆")
        print("-" * 50)
        
        # Print only the top 5 after re-ranking
        for rank, result in enumerate(ranked_candidates[:5], 1):
            # Unpack the actual candidate data from the scorer's result wrapper
            match = result.get("candidate", {})
            
            name = match.get("name") or match.get("candidate_id") or f"Candidate {result.get('candidate_index')}"
            
            # Print scores
            final_score = result.get("final_score", 0.0)
            print(f"#{rank} | {name} | Final Score: {final_score:.3f}")
            
            # Print the AI's explanation of exactly why they got this score!
            print(f"    💡 Why: {result.get('explanation')}")
            
            # Extract skills safely for printing
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
                print(f"    Skills: {', '.join(print_skills[:6])}...")
            else:
                print(f"    Skills: None listed")
            print("-" * 50)
                        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()