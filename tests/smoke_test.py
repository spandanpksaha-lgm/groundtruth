"""Offline smoke test: no API key, no model download. Run before demo day.

    python tests/smoke_test.py
"""
import os
import sys
from pathlib import Path

# Fixed offline settings, so the test never depends on .env (thresholds there are
# calibrated for the dense embedder and would be far too strict for TF-IDF).
os.environ["GT_EMBEDDER"] = "tfidf"
os.environ["LLM_PROVIDER"] = "extractive"
os.environ["GT_INDEX_DIR"] = "index/_smoke"
for _k in ("GT_MIN_SCORE", "GT_MIN_SCORE_TTS", "GT_MIN_SCORE_UC"):
    os.environ[_k] = "0.12"
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

# Second, fully separate corpus: UC Student Handbook (PDF converted to Markdown).
uc = GroundTruth(get_config(corpus="uc"), rebuild=True)
c = uc.ask("How long after the last day of class do I have to file a formal written appeal of a grade?")
assert not c.refused and "four weeks" in c.text.lower(), c.text
assert all(h.chunk.page for h in c.hits), "UC chunks must carry page numbers"
assert all(h.chunk.doc_id.endswith("student-handbook.md") for h in c.hits), "corpora must not mix"
d = gt.ask("How long after the last day of class do I have to file a formal written appeal of a grade?")
assert all(not h.chunk.doc_id.endswith("student-handbook.md") for h in d.hits), "corpora must not mix"
print("smoke test passed")
