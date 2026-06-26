"""Embedding and vector search utilities for candidate ranking."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from google import genai
from tqdm import tqdm


class EmbeddingEngine:
    """Generate Gemini embeddings and search them with FAISS cosine similarity."""

    def __init__(self, model: str = "models/gemini-embedding-2") -> None:
        self.client = genai.Client()
        self.model = model
        self.index: faiss.IndexFlatIP | None = None

    def get_embedding(self, text: str) -> list[float]:
        """Return a Gemini embedding vector for one text string."""
        normalized_text = text.strip() if isinstance(text, str) else str(text).strip()
        if not normalized_text:
            raise ValueError("Cannot embed empty text")

        response = self.client.models.embed_content(
            model="models/gemini-embedding-2",
            contents=normalized_text,
        )
        return list(response.embeddings[0].values)

    def embed_candidates(
        self,
        candidates: list[dict[str, Any]],
        text_key: str = "text_representation",
    ) -> np.ndarray:
        """Embed every candidate text representation into a NumPy array."""
        embeddings = []

        for candidate in tqdm(candidates, desc="Embedding candidates"):
            text = candidate.get(text_key, "")
            embeddings.append(self.get_embedding(text))

        return np.asarray(embeddings, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of arbitrary texts into a NumPy array."""
        embeddings = [
            self.get_embedding(text)
            for text in tqdm(texts, desc="Embedding texts")
        ]
        return np.asarray(embeddings, dtype=np.float32)

    def build_index(self, candidate_embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build a FAISS inner-product index over L2-normalized embeddings."""
        embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("candidate_embeddings must be a 2D array")
        if embeddings.shape[0] == 0:
            raise ValueError("candidate_embeddings must contain at least one vector")

        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        return index

    def search(
        self,
        query: str,
        top_k: int = 10,
        index: faiss.IndexFlatIP | None = None,
    ) -> tuple[list[int], list[float]]:
        """Embed a query and return top-K candidate indices and similarity scores."""
        active_index = index or self.index
        if active_index is None:
            raise ValueError("No FAISS index available. Call build_index first or pass an index.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = np.asarray([self.get_embedding(query)], dtype=np.float32)
        query_embedding = np.ascontiguousarray(query_embedding)
        faiss.normalize_L2(query_embedding)

        scores, indices = active_index.search(query_embedding, top_k)
        return indices[0].tolist(), scores[0].tolist()
