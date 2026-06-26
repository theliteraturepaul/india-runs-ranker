"""Hybrid semantic and rules-based scoring for candidate ranking."""

from __future__ import annotations

import functools
import re
from typing import Any, Mapping, Sequence

import numpy as np

import config

# ---------------------------------------------------------------------------
# Cross-domain skill synonym / alias map
# Both directions are resolved at the instance level.
# ---------------------------------------------------------------------------
_SKILL_ALIASES: dict[str, list[str]] = {
    # ── CS / Engineering ──────────────────────────────────────────────
    "ml":           ["machine learning"],
    "ai":           ["artificial intelligence"],
    "js":           ["javascript"],
    "ts":           ["typescript"],
    "py":           ["python"],
    "nlp":          ["natural language processing"],
    "cv":           ["computer vision"],
    "dl":           ["deep learning"],
    "rl":           ["reinforcement learning"],
    "llm":          ["large language model", "large language models"],
    "rag":          ["retrieval augmented generation", "retrieval-augmented generation"],
    "k8s":          ["kubernetes"],
    "tf":           ["tensorflow"],
    "sklearn":      ["scikit learn", "scikit-learn"],
    "node":         ["node js", "nodejs"],
    "react":        ["reactjs", "react js"],
    "vue":          ["vuejs", "vue js"],
    "postgres":     ["postgresql"],
    "mongo":        ["mongodb"],
    "ci cd":        ["continuous integration", "continuous delivery", "continuous deployment"],
    "oop":          ["object oriented programming", "object-oriented programming"],
    "aws":          ["amazon web services"],
    "gcp":          ["google cloud platform"],
    "azure":        ["microsoft azure"],
    "rest":         ["rest api", "restful api", "restful"],
    "docker":       ["containerization", "containers"],
    "git":          ["version control", "github", "gitlab", "bitbucket"],
    "agile":        ["scrum", "kanban", "sprint"],
    "nosql":        ["no sql", "non relational"],
    "hadoop":       ["hdfs", "mapreduce"],
    "spark":        ["apache spark", "pyspark"],
    "kafka":        ["apache kafka", "event streaming"],
    "fastapi":      ["fast api"],
    "django":       ["django rest framework", "drf"],
    "data science": ["data scientist", "data analytics", "data analysis"],
    "mlops":        ["ml operations", "machine learning operations", "model deployment"],
    "devops":       ["dev ops", "site reliability", "sre"],
    "genai":        ["generative ai", "gen ai", "generative artificial intelligence"],
    
    # ── Marketing ─────────────────────────────────────────────────────
    "seo":          ["search engine optimization", "search engine optimisation"],
    "sem":          ["search engine marketing"],
    "crm":          ["customer relationship management"],
    "ppc":          ["pay per click", "paid search"],
    "ctr":          ["click through rate"],
    "cpc":          ["cost per click"],
    "b2b":          ["business to business"],
    "b2c":          ["business to consumer"],
    "kpi":          ["key performance indicator", "key performance indicators"],
    "roi":          ["return on investment"],
    "cac":          ["customer acquisition cost"],
    "ltv":          ["lifetime value", "customer lifetime value"],
    "mrr":          ["monthly recurring revenue"],
    "arr":          ["annual recurring revenue"],
    "gtm":          ["go to market"],
    "uiux":         ["ui ux", "user interface", "user experience"],
    "ux":           ["user experience"],
    "ui":           ["user interface"],
    
    # ── Finance / Accounting ──────────────────────────────────────────
    "p l":          ["profit and loss", "profit loss"],
    "dcf":          ["discounted cash flow"],
    "m a":          ["mergers and acquisitions", "mergers acquisitions"],
    "fp a":         ["financial planning and analysis", "financial planning analysis"],
    "gaap":         ["generally accepted accounting principles"],
    "irr":          ["internal rate of return"],
    "npv":          ["net present value"],
    "aum":          ["assets under management"],
    "ebitda":       ["earnings before interest taxes depreciation amortization"],
    "ifrs":         ["international financial reporting standards"],
    
    # ── HR / People ───────────────────────────────────────────────────
    "ats":          ["applicant tracking system"],
    "okr":          ["objectives and key results"],
    "l d":          ["learning and development"],
    "hris":         ["human resources information system"],
    "dei":          ["diversity equity inclusion", "diversity equity and inclusion"],
    
    # ── Sales ─────────────────────────────────────────────────────────
    "sdr":          ["sales development representative"],
    "ae":           ["account executive"],
    "saas":         ["software as a service"],
    "bdr":          ["business development representative"],
    
    # ── General / Cross-domain ────────────────────────────────────────
    "pm":           ["project management", "project manager"],
    "ba":           ["business analysis", "business analyst"],
    "sop":          ["standard operating procedure", "standard operating procedures"],
    "poc":          ["proof of concept"],
    "mvp":          ["minimum viable product"],
    "r d":          ["research and development"],
    "qa":           ["quality assurance"],
    "sql":          ["structured query language"],
    "api":          ["application programming interface"],
}


class HybridScorer:
    """
    Combine FAISS semantic scores with structured recruiter-signal logic.

    Scoring pipeline
    ----------------
    1. Normalize the raw FAISS cosine-similarity batch via percentile
       min-max stretch so the best candidate in the pool always maps to ~1.0
       and the worst to ~0.0.
    2. Compute a signal score per candidate (skills, experience, title).
    3. Blend semantic + signal using config weights.
    4. Apply stacking deal-breaker penalties.
    """

    # --- Signal sub-weights (must sum to 1.0) ---
    REQUIRED_SKILL_WEIGHT  = 0.55
    PREFERRED_SKILL_WEIGHT = 0.15
    EXPERIENCE_WEIGHT      = 0.20
    TITLE_MATCH_WEIGHT     = 0.10

    # --- Deal-breaker penalty ---
    DEALBREAKER_PENALTY_MULTIPLIER = 0.60
    DEALBREAKER_STACKING           = True

    # --- Smooth experience scoring ---
    UNDEREXP_SCORE_AT_ZERO   = 0.0
    OVEREXP_PENALTY_PER_YEAR = 0.08
    OVEREXP_FLOOR            = 0.25

    # --- Semantic normalization ---
    SEMANTIC_FLOOR            = 0.30
    SEMANTIC_PERCENTILE_TOP   = 95
    SEMANTIC_PERCENTILE_BOT   = 5

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        jd_profile: Any,
        extra_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.jd_profile = self._as_dict(jd_profile)

        sem_w = float(config.SEMANTIC_WEIGHT)
        sig_w = float(config.SIGNAL_WEIGHT)
        total = sem_w + sig_w
        if total <= 0:
            raise ValueError("SEMANTIC_WEIGHT + SIGNAL_WEIGHT must be > 0")
        self._norm_sem = sem_w / total
        self._norm_sig = sig_w / total

        # Build instance-level alias maps so callers can extend per-domain
        merged = {k: list(v) for k, v in _SKILL_ALIASES.items()}
        for abbrev, expansions in (extra_aliases or {}).items():
            key = self._normalize_term(abbrev)
            merged.setdefault(key, []).extend(expansions)
        self._aliases: dict[str, list[str]] = merged

        # Reverse lookup: expansion → [abbrev, ...]
        self._alias_reverse: dict[str, list[str]] = {}
        for abbrev, expansions in self._aliases.items():
            for exp in expansions:
                self._alias_reverse.setdefault(
                    self._normalize_term(exp), []
                ).append(abbrev)

        # Pre-compute JD fields once
        self._required_skills  = self._list_field("required_skills")
        self._preferred_skills = self._list_field("preferred_skills")
        self._deal_breakers    = self._list_field("deal_breakers")
        self._target_titles    = self._list_field("target_titles")
        self._min_exp          = float(self.jd_profile.get("min_experience_years") or 0)
        self._max_exp          = float(self.jd_profile.get("max_experience_years") or 0)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def score_candidates(
        self,
        faiss_indices: Sequence[int] | np.ndarray,
        faiss_scores:  Sequence[float] | np.ndarray,
        raw_candidate_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank candidates by blended semantic + signal score."""
        indices     = self._flatten(faiss_indices, cast=int)
        raw_sems    = self._flatten(faiss_scores,  cast=float)

        if len(indices) != len(raw_sems):
            raise ValueError("faiss_indices and faiss_scores must have equal length")

        norm_sems = self._normalize_semantic_batch(raw_sems)
        ranked: list[dict[str, Any]] = []

        for idx, raw_sem, norm_sem in zip(indices, raw_sems, norm_sems):
            if idx < 0 or idx >= len(raw_candidate_data):
                continue

            candidate   = raw_candidate_data[idx]
            signal      = self.calculate_signal_score(candidate)
            combined    = norm_sem * self._norm_sem + signal * self._norm_sig
            final, n_db = self.apply_dealbreaker_penalty(candidate, combined)
            details     = self._candidate_match_details(candidate)

            ranked.append({
                "candidate_index":              idx,
                "candidate_id":                 candidate.get("candidate_id", ""),
                "final_score":                  round(final,     6),
                "semantic_score":               round(norm_sem,  6),
                "raw_semantic_score":           round(raw_sem,   6),
                "signal_score":                 round(signal,    6),
                "combined_score_before_penalty":round(combined,  6),
                "dealbreaker_penalty_applied":  n_db > 0,
                "missing_dealbreakers_count":   n_db,
                "matched_required_skills":      details["matched_required_skills"],
                "missing_required_skills":      details["missing_required_skills"],
                "matched_preferred_skills":     details["matched_preferred_skills"],
                "missing_preferred_skills":     details["missing_preferred_skills"],
                "missing_deal_breakers":        details["missing_deal_breakers"],
                "experience_years":             details["experience_years"],
                "experience_in_range":          details["experience_in_range"],
                "title_match_score":            details["title_match_score"],
                "explanation":                  self._build_explanation(
                                                    signal, norm_sem, final, details
                                                ),
                "candidate": candidate,
            })

        return sorted(ranked, key=lambda r: r["final_score"], reverse=True)

    def calculate_signal_score(self, candidate: dict[str, Any]) -> float:
        """Return a structured recruiter-signal score in [0.0, 1.0]."""
        req   = self._skill_match_score(candidate, self._required_skills)
        pref  = self._skill_match_score(candidate, self._preferred_skills)
        exp   = self._experience_score(candidate)
        title = self._title_match_score(candidate)

        raw = (
            req   * self.REQUIRED_SKILL_WEIGHT
            + pref  * self.PREFERRED_SKILL_WEIGHT
            + exp   * self.EXPERIENCE_WEIGHT
            + title * self.TITLE_MATCH_WEIGHT
        )
        return self._clamp(raw)

    def apply_dealbreaker_penalty(
        self,
        candidate: dict[str, Any],
        current_score: float,
    ) -> tuple[float, int]:
        if not self._deal_breakers:
            return self._clamp(current_score), 0

        missing  = self._missing_terms(candidate, self._deal_breakers)
        n_missing = len(missing)
        if n_missing == 0:
            return self._clamp(current_score), 0

        if self.DEALBREAKER_STACKING:
            penalized = current_score * (self.DEALBREAKER_PENALTY_MULTIPLIER ** n_missing)
        else:
            penalized = current_score * self.DEALBREAKER_PENALTY_MULTIPLIER

        return self._clamp(penalized), n_missing

    # ------------------------------------------------------------------ #
    #  Semantic normalisation                                            #
    # ------------------------------------------------------------------ #

    def _normalize_semantic_batch(self, raw: list[float]) -> list[float]:
        if not raw:
            return []
        if len(raw) == 1:
            return [1.0]

        arr  = np.array(raw, dtype=float)
        arr  = np.where(arr < self.SEMANTIC_FLOOR, self.SEMANTIC_FLOOR, arr)
        low  = float(np.percentile(arr, self.SEMANTIC_PERCENTILE_BOT))
        high = float(np.percentile(arr, self.SEMANTIC_PERCENTILE_TOP))

        if high <= low:
            return [0.5] * len(raw)

        normalized = ((arr - low) / (high - low)).clip(0.0, 1.0)
        return normalized.tolist()

    # ------------------------------------------------------------------ #
    #  Signal sub-scorers                                                #
    # ------------------------------------------------------------------ #

    def _skill_match_score(self, candidate: dict[str, Any], skills: list[str]) -> float:
        if not skills:
            return 1.0
        matched = len(self._matched_terms(candidate, skills))
        return matched / len(skills)

    def _experience_score(self, candidate: dict[str, Any]) -> float:
        years = self._candidate_experience_years(candidate)
        no_min = self._min_exp <= 0
        no_max = self._max_exp <= 0

        if no_min and no_max:
            return 1.0

        below_min = (not no_min) and years < self._min_exp
        above_max = (not no_max) and years > self._max_exp

        if not below_min and not above_max:
            return 1.0

        if below_min:
            ratio = years / self._min_exp if self._min_exp > 0 else 0.0
            return self._clamp(
                ratio * (1.0 - self.UNDEREXP_SCORE_AT_ZERO) + self.UNDEREXP_SCORE_AT_ZERO
            )

        excess = years - self._max_exp
        return self._clamp(max(self.OVEREXP_FLOOR, 1.0 - excess * self.OVEREXP_PENALTY_PER_YEAR))

    def _title_match_score(self, candidate: dict[str, Any]) -> float:
        if not self._target_titles:
            return 0.5

        recent = self._get_recent_title(candidate)
        if not recent:
            return 0.25

        norm_recent = self._normalize_term(recent)
        for target in self._target_titles:
            norm_target = self._normalize_term(target)
            if norm_target == norm_recent:
                return 1.0
            target_words = set(norm_target.split())
            recent_words = set(norm_recent.split())
            overlap = len(target_words & recent_words)
            if overlap >= max(1, len(target_words) // 2):
                return 0.75

        return 0.0

    # ------------------------------------------------------------------ #
    #  Skill-matching helpers (Optimized)                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _get_regex_pattern(alias: str) -> re.Pattern:
        """Cache regex compilation to prevent loop overhead."""
        return re.compile(r"(?<![a-z0-9+#.])" + re.escape(alias) + r"(?![a-z0-9+#.])")

    def _matched_terms(self, candidate: dict[str, Any], terms: list[str]) -> list[str]:
        return [t for t in terms if self._candidate_has_term(candidate, t)]

    def _missing_terms(self, candidate: dict[str, Any], terms: list[str]) -> list[str]:
        return [t for t in terms if not self._candidate_has_term(candidate, t)]

    def _candidate_has_term(self, candidate: dict[str, Any], term: str) -> bool:
        norm_term = self._normalize_term(term)
        if not norm_term:
            return True

        aliases       = self._expand_aliases(norm_term)
        skill_terms   = self._candidate_skill_terms(candidate)

        # 1 & 2: exact match in structured skills
        for alias in aliases:
            if alias in skill_terms:
                return True

        # 3: substring containment in structured skills
        for alias in aliases:
            for skill in skill_terms:
                if alias in skill or skill in alias:
                    return True

        # 4: whole-word search in scoped free text (optimized)
        scoped_text = self._normalize_term(self._candidate_scoped_text(candidate))
        for alias in aliases:
            # NEW: Skip dangerous short-acronym matches in free text
            if len(alias) <= 2 and alias not in ["ai", "ml", "cv", "ui", "ux", "js", "ts"]:
                continue
                
            pattern = self._get_regex_pattern(alias)
            if pattern.search(scoped_text):
                return True

        return False

    def _expand_aliases(self, norm_term: str) -> list[str]:
        """Return the term plus all known forward and reverse aliases via instance maps."""
        aliases = {norm_term}

        if norm_term in self._aliases:
            aliases.update(self._normalize_term(e) for e in self._aliases[norm_term])

        if norm_term in self._alias_reverse:
            for abbrev in self._alias_reverse[norm_term]:
                aliases.add(self._normalize_term(abbrev))
                aliases.update(
                    self._normalize_term(e)
                    for e in self._aliases.get(abbrev, [])
                )

        return [a for a in aliases if a]

    def _candidate_skill_terms(self, candidate: dict[str, Any]) -> set[str]:
        skills = candidate.get("skills", [])
        terms: set[str] = set()
        if not isinstance(skills, list):
            return terms
        for skill in skills:
            if isinstance(skill, Mapping):
                terms.add(self._normalize_term(skill.get("name", "")))
            else:
                terms.add(self._normalize_term(str(skill)))
        return {t for t in terms if t}

    def _candidate_scoped_text(self, candidate: dict[str, Any]) -> str:
        """
        Build and cache a targeted text blob from semantically relevant fields only.
        """
        # Return cached string if we already built it for this candidate
        if "_cached_scoped_text" in candidate:
            return candidate["_cached_scoped_text"]
            
        if candidate.get("text_representation"):
            text = str(candidate["text_representation"])
            candidate["_cached_scoped_text"] = text
            return text

        parts: list[str] = []

        for key in ("summary", "bio", "about", "headline", "objective"):
            if val := candidate.get(key):
                parts.append(str(val))

        for job in candidate.get("career_history", []):
            if isinstance(job, dict):
                for f in ("title", "description", "responsibilities", "technologies"):
                    val = job.get(f, "")
                    if isinstance(val, list):
                        parts.append(" ".join(str(v) for v in val))
                    elif val:
                        parts.append(str(val))

        for edu in candidate.get("education", []):
            if isinstance(edu, dict):
                parts.append(edu.get("degree", ""))
                parts.append(edu.get("field", ""))
                parts.append(edu.get("major", ""))

        for cert in candidate.get("certifications", []):
            parts.append(cert.get("name", "") if isinstance(cert, dict) else str(cert))

        for proj in candidate.get("projects", []):
            if isinstance(proj, dict):
                parts.append(proj.get("name", ""))
                parts.append(proj.get("description", ""))
                techs = proj.get("technologies", [])
                parts.append(" ".join(techs) if isinstance(techs, list) else str(techs))

        text = " ".join(filter(None, parts))
        candidate["_cached_scoped_text"] = text  # Cache for subsequent skill checks
        return text

    def _get_recent_title(self, candidate: dict[str, Any]) -> str:
        history = candidate.get("career_history", [])
        if not history or not isinstance(history, list):
            return str(candidate.get("current_title", ""))

        for job in history:
            if isinstance(job, dict):
                end = str(job.get("end_date", "")).strip().lower()
                if job.get("is_current") or end in ("", "present", "none", "null", "current"):
                    return str(job.get("title", ""))

        first = history[0]
        return str(first.get("title", "")) if isinstance(first, dict) else ""

    # ------------------------------------------------------------------ #
    #  Match details                                                     #
    # ------------------------------------------------------------------ #

    def _candidate_match_details(self, candidate: dict[str, Any]) -> dict[str, Any]:
        years = self._candidate_experience_years(candidate)
        return {
            "matched_required_skills":  self._matched_terms(candidate, self._required_skills),
            "missing_required_skills":  self._missing_terms(candidate, self._required_skills),
            "matched_preferred_skills": self._matched_terms(candidate, self._preferred_skills),
            "missing_preferred_skills": self._missing_terms(candidate, self._preferred_skills),
            "missing_deal_breakers":    self._missing_terms(candidate, self._deal_breakers),
            "experience_years":         years,
            "experience_in_range":      self._experience_in_range(years),
            "title_match_score":        self._title_match_score(candidate),
        }

    # ------------------------------------------------------------------ #
    #  Explanation builder                                               #
    # ------------------------------------------------------------------ #

    def _build_explanation(
        self,
        signal_score:   float,
        semantic_score: float,
        final_score:    float,
        details:        dict[str, Any],
    ) -> str:
        exp_range = (
            f"{self._min_exp:.0f}–{self._max_exp:.0f} yrs"
            if (self._min_exp or self._max_exp) else "unspecified"
        )
        penalty_text = (
            f" Dealbreaker penalty applied – missing: "
            f"{', '.join(details['missing_deal_breakers'])}."
            if details["missing_deal_breakers"]
            else " No dealbreaker penalty."
        )
        title_text = (
            f" Title relevance: {details['title_match_score']:.2f}."
            if self._target_titles else ""
        )
        exp_label = "within" if details["experience_in_range"] else "outside"

        return (
            f"Semantic {semantic_score:.3f} | Signal {signal_score:.3f} | "
            f"Required skills {len(details['matched_required_skills'])}"
            f"/{len(self._required_skills)} | "
            f"Preferred skills {len(details['matched_preferred_skills'])}"
            f"/{len(self._preferred_skills)} | "
            f"Experience {details['experience_years']:.1f} yrs "
            f"({exp_label} JD range {exp_range})."
            f"{title_text}{penalty_text} "
            f"→ Final {final_score:.3f}"
        )

    # ------------------------------------------------------------------ #
    #  Shared utilities                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _candidate_experience_years(candidate: dict[str, Any]) -> float:
        history = candidate.get("career_history", [])
        total_months = sum(
            float(job.get("duration_months", 0.0))
            for job in history
            if isinstance(job, dict) and job.get("duration_months")
        )
        return total_months / 12.0

    def _experience_in_range(self, years: float) -> bool:
        if self._min_exp > 0 and years < self._min_exp:
            return False
        if self._max_exp > 0 and years > self._max_exp:
            return False
        return True

    def _list_field(self, field_name: str) -> list[str]:
        value = self.jd_profile.get(field_name, [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _normalize_term(value: Any) -> str:
        return re.sub(r"[^a-z0-9+#.]+", " ", str(value).lower()).strip()

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError(
            f"jd_profile must be a JDProfileSchema-like object or Mapping, got {type(value)}"
        )

    @staticmethod
    def _flatten(values: Sequence[Any] | np.ndarray, cast: type) -> list[Any]:
        arr = np.asarray(values)
        if arr.ndim == 0:
            return [cast(arr.item())]
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return [cast(v) for v in arr.tolist()]

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))