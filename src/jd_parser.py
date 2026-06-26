"""Job description parsing with Gemini structured JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from google import genai
from pydantic import BaseModel, Field

import config


class JDProfileSchema(BaseModel):
    """Structured profile extracted from a raw job description."""

    role_title: str = Field(default="")
    seniority: str = Field(default="")
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = Field(default=0)
    max_experience_years: int = Field(default=0)
    role_context: str = Field(default="")
    soft_signals: list[str] = Field(default_factory=list)
    deal_breakers: list[str] = Field(default_factory=list)
    embedding_text: str = Field(default="")


class JDParser:
    """Parse job descriptions into recruiter-oriented structured profiles."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.GEMINI_MODEL
        self.project_root = Path(__file__).resolve().parents[1]
        self.processed_dir = self.project_root / config.PROCESSED_DIR

    def parse(self, jd_text: str) -> JDProfileSchema:
        """Parse raw JD text into a validated JD profile."""
        if not jd_text or not jd_text.strip():
            raise ValueError("jd_text must not be empty")

        client = genai.Client()
        prompt = self._build_prompt(jd_text)

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JDProfileSchema,
            ),
        )

        if getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            if isinstance(parsed, JDProfileSchema):
                return parsed
            return JDProfileSchema.model_validate(parsed)

        return JDProfileSchema.model_validate_json(response.text)

    def parse_file(
        self,
        jd_file_path: str | Path,
        output_path: str | Path | None = None,
    ) -> JDProfileSchema:
        """Read a JD file, parse it, and save JSON to data/processed/jd_profile.json."""
        jd_path = Path(jd_file_path)
        if not jd_path.is_absolute():
            jd_path = self.project_root / jd_path

        jd_text = jd_path.read_text(encoding="utf-8")
        profile = self.parse(jd_text)
        self.save_profile(profile, output_path)
        return profile

    def save_profile(
        self,
        profile: JDProfileSchema,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save a parsed JD profile as formatted JSON."""
        destination = Path(output_path) if output_path else self.processed_dir / "jd_profile.json"
        if not destination.is_absolute():
            destination = self.project_root / destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(profile.model_dump(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def _build_prompt(jd_text: str) -> str:
        return (
            "Extract a structured job profile from the job description below. "
            "Return only the JSON object matching the provided schema. "
            "Use concise, recruiter-friendly language. "
            "For embedding_text, write one dense paragraph that combines role title, "
            "seniority, required and preferred skills, domain context, experience range, "
            "soft signals, and deal breakers for vector search.\n\n"
            f"Job description:\n{jd_text.strip()}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a JD file into data/processed/jd_profile.json")
    parser.add_argument("jd_file", help="Path to a plain-text job description file")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to data/processed/jd_profile.json",
    )
    args = parser.parse_args()

    output = JDParser().parse_file(args.jd_file, args.output)
    print(json.dumps(output.model_dump(), indent=2, ensure_ascii=True))
