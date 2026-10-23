"""Build (or rebuild) the vector index.

    python scripts/build_index.py                       # uses .env settings
    python scripts/build_index.py --chunker fixed --chunk-size 400 --embedder tfidf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--chunker", choices=["section", "fixed"])
p.add_argument("--chunk-size", type=int)
p.add_argument("--embedder", choices=["st", "openai", "tfidf"])
a = p.parse_args()

cfg = get_config(chunker=a.chunker, chunk_size=a.chunk_size, embedder=a.embedder)
store = VectorStore.load_or_build(cfg, rebuild=True)
print(f"Saved to {cfg.index_path()}")
