"""Offline smoke test: no API key, no model download. Run before demo day.

    python tests/smoke_test.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GT_EMBEDDER", "tfidf")
os.environ.setdefault("LLM_PROVIDER", "extractive")
os.environ.setdefault("GT_INDEX_DIR", "index/_smoke")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.rag import GroundTruth, parse_citations  # noqa: E402

gt = GroundTruth(get_config(), rebuild=True)

a = gt.ask("How much annual leave can I carry over into the next leave year?")
assert not a.refused and "240" in a.text, a.text
assert parse_citations(a.text, len(a.hits)), "answer must cite a passage"

b = gt.ask("What is the capital of France?")
assert b.refused and b.refusal_reason == "low_retrieval", b

assert parse_citations("x [1][3] y [2, 9]", 5) == [1, 3, 2]
print("smoke test passed")
