"""Embedding backends. All return L2-normalised float32 matrices so a FAISS
inner-product index gives cosine similarity."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import requests


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> "Embedder":
        return self


class SentenceTransformerEmbedder(Embedder):
    """Dense local embeddings (e.g. all-MiniLM-L6-v2, bge-small-en-v1.5).
    Downloads the model from Hugging Face on first use (~90 MB)."""
    name = "st"

    _cache: dict = {}   # one model in memory, shared by every corpus

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        if model_name not in self._cache:
            self._cache[model_name] = SentenceTransformer(model_name)
        self.model = self._cache[model_name]

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False,
                                 normalize_embeddings=True)
        return _normalise(vecs)


class OpenAICompatEmbedder(Embedder):
    """Any OpenAI-compatible /embeddings endpoint: OpenAI, Ollama
    (base_url=http://localhost:11434/v1, model=nomic-embed-text), etc."""
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str, timeout: int = 60):
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            r = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key or 'none'}"},
                json={"model": self.model, "input": texts[i:i + 64]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return _normalise(np.array(out))


class TfidfEmbedder(Embedder):
    """Lexical baseline: TF-IDF with unigrams+bigrams. Needs no model download,
    which also makes it a useful comparison point against dense embeddings."""
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                   stop_words="english", min_df=1)

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def embed(self, texts):
        return _normalise(self.vec.transform(texts).toarray())

    def save(self, path: Path):
        with open(path / "tfidf.pkl", "wb") as f:
            pickle.dump(self.vec, f)

    def load(self, path: Path):
        with open(path / "tfidf.pkl", "rb") as f:
            self.vec = pickle.load(f)
        return self


def make_embedder(cfg, kind: str | None = None) -> Embedder:
    kind = kind or cfg.embedder
    if kind == "st":
        try:
            return SentenceTransformerEmbedder(cfg.st_model)
        except ImportError as e:
            raise SystemExit("sentence-transformers is not installed. Run "
                             "`pip install sentence-transformers`, or set GT_EMBEDDER=tfidf "
                             "to use the no-download baseline.") from e
    if kind == "openai":
        return OpenAICompatEmbedder(cfg.embed_model, cfg.embed_base_url,
                                    cfg.embed_api_key, cfg.llm_timeout)
    if kind == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder '{kind}'. Use 'st', 'openai' or 'tfidf'.")
