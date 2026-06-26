# Execution & Submission Details

## Execution Order
1. **Phase 0 (Bootstrap)** — 30 min (Dataset-independent. Start immediately.)
2. **Phase 2 (JD Parser)** — 1.5 hrs (Dataset-independent. Start immediately.)
3. *[Dataset arrives - unblocked]*
4. **Phase 1 (Data Layer)** — 1.5 hrs
5. **Phase 3 (Embeddings)** — 1 hr
6. **Phase 4 (Signals)** — 2 hrs
7. **Phase 5 (Output)** — 1 hr
8. **Phase 6 (UI - Stretch)** — 2 hrs

## Submission Checklist
* [ ] GitHub repo (public) with clean commit history
* [ ] `README.md` covering: problem approach, architecture diagram, tech stack, how to run
* [ ] `ranked_output.csv` — final ranked candidate shortlist in predefined format
* [ ] `summary_report.txt` or equivalent — methodology explanation
* [ ] (Stretch) Streamlit demo running or deployed

## Key Design Decisions
| Decision | Rationale |
| :--- | :--- |
| **Gemini 3.0 Flash for JD parsing** | High speed, cost-effective, native JSON schema enforcement |
| **gemini-embedding-2** | Explicitly required, streamlines architecture (no local model binaries to download) |
| **FAISS for vector search** | Handles large datasets instantly |
| **Hybrid scoring** | Pure semantic misses career signals; pure rule-based misses context |
| **Config-driven weights** | Judges can see we thought about tunability |
| **Streamlit (not React)** | 10x faster to build; demo quality is the same |

*Last updated: June 25, 2026 — India Runs Hackathon, Redrob AI x Hack2Skill*