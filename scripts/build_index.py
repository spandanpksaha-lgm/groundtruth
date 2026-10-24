"""Build (or rebuild) the vector index.

    python scripts/build_index.py                       # uses .env settings (tts)
    python scripts/build_index.py --corpus uc           # UC Student Handbook
    python scripts/build_index.py --corpus uc --chunker fixed --st-model BAAI/bge-small-en-v1.5
    python scripts/build_index.py --chunker fixed --chunk-size 400 --embedder tfidf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--corpus", choices=["tts", "uc"])
p.add_argument("--chunker", choices=["section", "fixed"])
p.add_argument("--chunk-size", type=int)
p.add_argument("--embedder", choices=["st", "openai", "tfidf"])
p.add_argument("--st-model", help="dense model, e.g. BAAI/bge-small-en-v1.5")
a = p.parse_args()

cfg = get_config(corpus=a.corpus, chunker=a.chunker, chunk_size=a.chunk_size, embedder=a.embedder,
                 st_model=a.st_model)
store = VectorStore.load_or_build(cfg, rebuild=True)
print(f"Saved to {cfg.index_path()}")
