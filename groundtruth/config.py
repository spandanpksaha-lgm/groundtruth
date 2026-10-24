"""Central configuration. Every setting can be overridden with an environment
variable or a line in a `.env` file at the repo root (KEY=value)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embedder_label(embedder: str, st_model: str = DEFAULT_ST_MODEL) -> str:
    """Short human name for an embedder, used in tables and charts."""
    if embedder != "st":
        return {"tfidf": "TF-IDF", "openai": "OpenAI-compatible"}.get(embedder, embedder)
    tail = st_model.split("/")[-1].lower()
    for key, name in (("minilm", "MiniLM"), ("bge-small", "BGE-small"), ("bge-base", "BGE-base"),
                      ("mpnet", "MPNet"), ("e5", "E5")):
        if key in tail:
            return name
    return tail


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = re.split(r"\s+#", value, maxsplit=1)[0]  # drop inline comments
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # --- Corpus -------------------------------------------------------------
    # Which document set: "tts" (TTS Travel & Leave) or "uc" (UC Student
    # Handbook). See groundtruth/corpora.py. data_dir and source_base_url are
    # filled in from the corpus unless set explicitly.
    corpus: str = _env("GT_CORPUS", "tts")
    data_dir: Path | None = None
    source_base_url: str | None = None

    # --- Chunking -----------------------------------------------------------
    chunker: str = _env("GT_CHUNKER", "section")          # "section" | "fixed"
    chunk_size: int = int(_env("GT_CHUNK_SIZE", "220"))    # words
    chunk_overlap: int = int(_env("GT_CHUNK_OVERLAP", "40"))  # words

    # --- Embeddings ---------------------------------------------------------
    # "st"     = sentence-transformers, runs locally (recommended)
    # "openai" = any OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, ...)
    # "tfidf"  = sparse lexical baseline, no download needed
    embedder: str = _env("GT_EMBEDDER", "st")
    st_model: str = _env("GT_ST_MODEL", DEFAULT_ST_MODEL)
    embed_model: str = _env("GT_EMBED_MODEL", "text-embedding-3-small")
    # For embedder="openai". Default to the LLM endpoint/key when unset.
    embed_base_url: str = _env("GT_EMBED_BASE_URL", _env("LLM_BASE_URL", ""))
    embed_api_key: str = _env("GT_EMBED_API_KEY", _env("LLM_API_KEY", ""))

    # --- Retrieval ----------------------------------------------------------
    top_k: int = int(_env("GT_TOP_K", "5"))
    # If the best passage scores below this, we answer "I don't know" without
    # calling the LLM. Calibrate per embedder with eval/calibrate_threshold.py.
    # An empty value means "use the per-embedder default below".
    # Per corpus: GT_MIN_SCORE_UC=..., GT_MIN_SCORE_TTS=... (GT_MIN_SCORE
    # alone still applies to tts, for older .env files).
    min_score: str = ""

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

    def resolve(self) -> "Config":
        from .corpora import corpus_spec
        spec = corpus_spec(self.corpus)
        if self.data_dir is None:
            rel = _env("GT_DATA_DIR", spec["data_dir"]) if self.corpus == "tts" else spec["data_dir"]
            self.data_dir = ROOT / rel
        if self.source_base_url is None:
            self.source_base_url = (_env("GT_SOURCE_BASE_URL", spec["base_url"])
                                    if self.corpus == "tts" else spec["base_url"])
        return self

    @property
    def threshold_env_key(self) -> str:
        return "GT_MIN_SCORE" if self.corpus == "tts" else f"GT_MIN_SCORE_{self.corpus.upper()}"

    @property
    def cell_env_key(self) -> str:
        """Threshold key for one (corpus, chunker, embedding model) cell, e.g.
        GT_MIN_SCORE_UC_FIXED_BGE_SMALL. Similarity scores are on a different
        scale for each embedding model, so each cell gets its own cut-off."""
        label = re.sub(r"[^A-Z0-9]+", "_", self.embedder_label.upper()).strip("_")
        return f"GT_MIN_SCORE_{self.corpus.upper()}_{self.chunker.upper()}_{label}"

    @property
    def refusal_threshold(self) -> float:
        for v in (self.min_score, os.environ.get(self.cell_env_key, ""),
                  os.environ.get(f"GT_MIN_SCORE_{self.corpus.upper()}", ""),
                  os.environ.get("GT_MIN_SCORE", "") if self.corpus == "tts" else ""):
            if v and v.strip():
                return float(v)
        return self.DEFAULT_MIN_SCORE.get(self.embedder, 0.3)

    def index_path(self, chunker: str | None = None, embedder: str | None = None,
                   chunk_size: int | None = None) -> Path:
        emb = embedder or self.embedder
        name = f"{self.corpus}-{chunker or self.chunker}-{chunk_size or self.chunk_size}-{emb}"
        if emb == "st" and self.st_model != DEFAULT_ST_MODEL:
            name += "-" + self.st_model.split("/")[-1]   # e.g. -bge-small-en-v1.5
        return self.index_dir / name

    @property
    def embedder_label(self) -> str:
        return embedder_label(self.embedder, self.st_model)

    def questions_path(self, split: str) -> Path:
        stem = f"questions_{split}" if self.corpus == "tts" else f"questions_{self.corpus}_{split}"
        return ROOT / "eval" / f"{stem}.jsonl"

    @property
    def results_prefix(self) -> str:
        return "" if self.corpus == "tts" else f"{self.corpus}-"


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg.resolve()
