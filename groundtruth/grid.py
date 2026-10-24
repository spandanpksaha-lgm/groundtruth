"""The 2x2 experiment grid: two chunking strategies x two embedding models.

Each cell is a complete, separate GroundTruth system (own index, own
calibrated "I don't know" threshold). The app's UC tabs and the 2x2 benchmark
runs both use these cells, so the demo and the numbers always match.

Override the two embedding models with GT_GRID_MODELS, e.g.
    GT_GRID_MODELS=sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5
(the special value "tfidf" uses the keyword baseline instead of a dense model).
"""
from __future__ import annotations

import copy
import os

from .config import DEFAULT_ST_MODEL, embedder_label, get_config

CHUNKERS = ["section", "fixed"]
BGE_SMALL = "BAAI/bge-small-en-v1.5"


def grid_models() -> list[str]:
    raw = os.environ.get("GT_GRID_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()] or [DEFAULT_ST_MODEL, BGE_SMALL]
    return models[:2]


def cell_label(model: str) -> str:
    return embedder_label("tfidf") if model == "tfidf" else embedder_label("st", model)


def cell_config(corpus: str, chunker: str, model: str, **overrides):
    cfg = get_config(corpus=corpus, chunker=chunker, **overrides)
    cfg = copy.copy(cfg)
    if model == "tfidf":
        cfg.embedder = "tfidf"
    else:
        cfg.embedder = "st"
        cfg.st_model = model
    return cfg


def cells(corpus: str = "uc"):
    """[(chunker, model, label)] in display order: rows = chunking, columns = model."""
    return [(ch, m, f"{ch.capitalize()} chunks · {cell_label(m)}") for ch in CHUNKERS for m in grid_models()]


def run_tag(corpus: str, split: str, chunker: str, model: str, size: int = 220) -> str:
    """Results folder for a benchmark run of one cell. The default-model cells
    keep the names used before the grid existed (…-main)."""
    suffix = "main" if model == DEFAULT_ST_MODEL else cell_label(model).lower().replace("-", "")
    return f"{corpus}-{split}-{chunker}{size}-{suffix}"
