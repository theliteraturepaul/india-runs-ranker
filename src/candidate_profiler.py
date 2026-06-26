"""Candidate data loading, cleaning, and embedding text preparation."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

try:
    import config
except ImportError:  # pragma: no cover - supports direct module reuse elsewhere.
    config = None


TEXT_DEFAULT = ""
NUMERIC_DEFAULT = 0.0
BOOL_DEFAULT = False


class CandidateProfiler:
    """Prepare raw candidate JSON for scoring and semantic embedding."""

    def __init__(self, raw_data_path: str | Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default_raw_dir = getattr(config, "RAW_DATA_DIR", "data/raw/") if config else "data/raw/"
        self.raw_data_path = Path(raw_data_path) if raw_data_path else project_root / default_raw_dir / "candidates.json"
        if not self.raw_data_path.is_absolute():
            self.raw_data_path = project_root / self.raw_data_path
        self._schema: dict[str, Any] | None = None

    def load_raw_data(self) -> list[dict[str, Any]]:
        """Load candidates from data/raw/candidates.json."""
        with self.raw_data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of candidates in {self.raw_data_path}")
        if not all(isinstance(candidate, dict) for candidate in data):
            raise ValueError("Every candidate record must be a JSON object")

        return data

    def clean(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Return one schema-normalized candidate dictionary."""
        if self._schema is None:
            self._schema = self._infer_schema([candidate])

        cleaned = self._clean_value(candidate, self._schema)
        cleaned["text_representation"] = self.build_text_representation(cleaned)
        return cleaned

    def build_text_representation(self, candidate: dict[str, Any]) -> str:
        """Combine skills, experience, and projects into dense embedding text."""
        profile = candidate.get("profile", {})
        skills = candidate.get("skills", [])
        career_history = candidate.get("career_history", [])
        projects = candidate.get("projects", candidate.get("project_history", []))

        parts: list[str] = []

        profile_text = self._join_text(
            [
                profile.get("current_title"),
                profile.get("headline"),
                profile.get("summary"),
                profile.get("current_industry"),
                profile.get("country"),
                f"{profile.get('years_of_experience', NUMERIC_DEFAULT)} years of experience",
            ]
        )
        if profile_text:
            parts.append(profile_text)

        skill_texts = []
        for skill in skills if isinstance(skills, list) else []:
            if not isinstance(skill, dict):
                continue
            skill_texts.append(
                self._join_text(
                    [
                        skill.get("name"),
                        skill.get("proficiency"),
                        f"{skill.get('duration_months', NUMERIC_DEFAULT)} months",
                        f"{skill.get('endorsements', NUMERIC_DEFAULT)} endorsements",
                    ]
                )
            )
        if skill_texts:
            parts.append("Skills: " + "; ".join(filter(None, skill_texts)))

        experience_texts = []
        for job in career_history if isinstance(career_history, list) else []:
            if not isinstance(job, dict):
                continue
            experience_texts.append(
                self._join_text(
                    [
                        job.get("title"),
                        job.get("company"),
                        job.get("industry"),
                        f"{job.get('duration_months', NUMERIC_DEFAULT)} months",
                        job.get("description"),
                    ]
                )
            )
        if experience_texts:
            parts.append("Experience: " + "; ".join(filter(None, experience_texts)))

        project_texts = []
        for project in projects if isinstance(projects, list) else []:
            if isinstance(project, dict):
                project_texts.append(
                    self._join_text(
                        [
                            project.get("name"),
                            project.get("title"),
                            project.get("summary"),
                            project.get("description"),
                            project.get("tech_stack"),
                            project.get("skills"),
                        ]
                    )
                )
            else:
                project_texts.append(self._normalize_text(project))
        if project_texts:
            parts.append("Projects: " + "; ".join(filter(None, project_texts)))

        return self._normalize_text(" ".join(parts))

    def process_all(self, path: str) -> list[dict[str, Any]]:
        """Load, clean, and enrich every candidate record."""
        raw_candidates = self.load_raw_data()
        self._schema = self._infer_schema(raw_candidates)
        return [self.clean(candidate) for candidate in raw_candidates]

    def _infer_schema(self, values: list[Any]) -> dict[str, Any]:
        non_null_values = [value for value in values if value is not None]

        if any(isinstance(value, dict) for value in non_null_values):
            keys = sorted(
                {
                    key
                    for value in non_null_values
                    if isinstance(value, dict)
                    for key in value.keys()
                }
            )
            return {
                "kind": "dict",
                "fields": {
                    key: self._infer_schema(
                        [value.get(key) for value in non_null_values if isinstance(value, dict)]
                    )
                    for key in keys
                },
            }

        if any(isinstance(value, list) for value in non_null_values):
            items = [
                item
                for value in non_null_values
                if isinstance(value, list)
                for item in value
            ]
            return {"kind": "list", "item": self._infer_schema(items)}

        if any(isinstance(value, str) for value in non_null_values):
            return {"kind": "text"}
        if any(isinstance(value, bool) for value in non_null_values):
            return {"kind": "bool"}
        if any(self._is_number(value) for value in non_null_values):
            return {"kind": "number"}

        return {"kind": "text"}

    def _clean_value(self, value: Any, schema: dict[str, Any]) -> Any:
        kind = schema["kind"]

        if kind == "dict":
            source = value if isinstance(value, dict) else {}
            return {
                key: self._clean_value(source.get(key), child_schema)
                for key, child_schema in schema["fields"].items()
            }

        if kind == "list":
            if not isinstance(value, list):
                return []
            return [self._clean_value(item, schema["item"]) for item in value]

        if kind == "number":
            if value is None or value == "":
                return NUMERIC_DEFAULT
            if self._is_number(value):
                return float(value)
            return NUMERIC_DEFAULT

        if kind == "bool":
            if value is None or value == "":
                return BOOL_DEFAULT
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y"}
            return bool(value)

        if value is None:
            return TEXT_DEFAULT
        return self._normalize_text(value)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return TEXT_DEFAULT
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=True, sort_keys=True)
        return re.sub(r"\s+", " ", str(value)).strip()

    def _join_text(self, values: list[Any]) -> str:
        normalized_values = [self._normalize_text(value) for value in values]
        return " ".join(value for value in normalized_values if value)


if __name__ == "__main__":
    profiler = CandidateProfiler()
    processed = profiler.process_all()
    print(f"Processed {len(processed)} candidates")
