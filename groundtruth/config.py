"""Central configuration. Every setting can be overridden with an environment
variable or a line in a `.env` file at the repo root (KEY=value)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # --- Corpus -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: ROOT / _env("GT_DATA_DIR", "data/raw"))
    # Base URL used to turn file paths into clickable source links.
    source_base_url: str = _env("GT_SOURCE_BASE_URL", "https://handbook.tts.gsa.gov")

    # --- Chunking -----------------------------------------------------------
    chunker: str = _env("GT_CHUNKER", "section")          # "section" | "fixed"
    chunk_size: int = int(_env("GT_CHUNK_SIZE", "220"))    # words
    chunk_overlap: int = int(_env("GT_CHUNK_OVERLAP", "40"))  # words

    # --- Embeddings ---------------------------------------------------------
    # "st"     = sentence-transformers, runs locally (recommended)
    # "openai" = any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, ...)
    # "tfidf"  = sparse lexical baseline, no download needed
    embedder: str = _env("GT_EMBEDDER", "st")
    st_model: str = _env("GT_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model: str = _env("GT_EMBED_MODEL", "text-embedding-3-small")
    # For embedder="openai". Default to the LLM endpoint/key when unset.
    embed_base_url: str = _env("GT_EMBED_BASE_URL", _env("LLM_BASE_URL", ""))
    embed_api_key: str = _env("GT_EMBED_API_KEY", _env("LLM_API_KEY", ""))

    # --- Retrieval ----------------------------------------------------------
    top_k: int = int(_env("GT_TOP_K", "5"))
    # If the best passage scores below this, we answer "I don't know" without
    # calling the LLM. Calibrate per embedder with eval/calibrate_threshold.py.
    # An empty value means "use the per-embedder default below".
    min_score: str = _env("GT_MIN_SCORE", "")

    # --- Generation ---------------------------------------------------------
    # "openai"     = any OpenAI-compatible chat endpoint (OpenAI, Groq, Ollama,
    #                Gemini's OpenAI endpoint, Together, ...)
    # "anthropic"  = Anthropic Messages API
    # "extractive" = no LLM; quotes the best-matching sentences (offline fallback)
    llm_provider: str = _env("LLM_PROVIDER", "extractive")
    llm_model: str = _env("LLM_MODEL", "")
    llm_base_url: str = _env("LLM_BASE_URL", "")
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0"))
    llm_timeout: int = int(_env("LLM_TIMEOUT", "60"))

    # --- Paths --------------------------------------------------------------
    index_dir: Path = field(default_factory=lambda: ROOT / _env("GT_INDEX_DIR", "index"))

    # Default refusal thresholds by embedder (cosine similarity of best chunk).
    # These are starting points; calibrate_threshold.py prints better values
    # chosen on the DEV set (never on the test set).
    DEFAULT_MIN_SCORE = {"st": 0.35, "openai": 0.30, "tfidf": 0.12}

    @property
    def refusal_threshold(self) -> float:
        if self.min_score.strip():
            return float(self.min_score)
        return self.DEFAULT_MIN_SCORE.get(self.embedder, 0.3)

    def index_path(self, chunker: str | None = None, embedder: str | None = None,
                   chunk_size: int | None = None) -> Path:
        name = f"{chunker or self.chunker}-{chunk_size or self.chunk_size}-{embedder or self.embedder}"
        return self.index_dir / name


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg
