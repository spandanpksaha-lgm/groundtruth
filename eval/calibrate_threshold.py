"""Pick the retrieval-score threshold below which GroundTruth refuses without
calling the LLM. Calibrated on the DEV set only, so the test set stays clean.

    python eval/calibrate_threshold.py                 # current .env settings
    python eval/calibrate_threshold.py --embedder tfidf

This gate only needs to catch clearly out-of-scope questions; plausible but
unanswerable ones ("What's the M&IE rate in Boston?") still reach the LLM,
which must decline them itself. So we pick the threshold that never refuses a
dev question the documents can answer, then report how many unanswerable
questions the gate alone catches.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.evaluation import load_questions  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402


def auroc(pos, neg):
    """P(score of a random answerable question > a random unanswerable one)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--chunker", choices=["section", "fixed"])
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--embedder", choices=["st", "openai", "tfidf"])
    p.add_argument("--margin", type=float, default=0.9,
                   help="threshold = margin x lowest answerable top-score (safety buffer)")
    p.add_argument("--write", action="store_true", help="save the threshold to .env")
    a = p.parse_args(argv)

    cfg = get_config(chunker=a.chunker, chunk_size=a.chunk_size, embedder=a.embedder)
    store = VectorStore.load_or_build(cfg)
    qs = load_questions(ROOT / "eval" / "questions_dev.jsonl")
    pos, neg = [], []
    for q in qs:
        top = store.search(q["question"], 1)[0].score
        (pos if q["answerable"] else neg).append(top)
        print(f"  {'ANS ' if q['answerable'] else 'UNANS'} {top:.3f}  {q['question'][:70]}")

    thr = round(min(pos) * a.margin, 3)
    caught = sum(n < thr for n in neg)
    print(f"\nembedder={cfg.embedder}  answerable top-scores: min={min(pos):.3f} "
          f"mean={sum(pos)/len(pos):.3f} | unanswerable: max={max(neg):.3f} mean={sum(neg)/len(neg):.3f}")
    print(f"AUROC of top-score as an answerable/unanswerable signal: {auroc(pos, neg):.2f}")
    print(f"Suggested threshold: GT_MIN_SCORE={thr}  "
          f"(gate alone would catch {caught}/{len(neg)} unanswerable dev questions; "
          f"current default {cfg.refusal_threshold})")
    print("Note: the dev set is small (9 questions). Treat this as a sanity check, "
          "not a precise optimum.")
    if a.write:
        env = ROOT / ".env"
        lines = [l for l in (env.read_text().splitlines() if env.exists() else [])
                 if not l.startswith("GT_MIN_SCORE=")]
        lines.append(f"GT_MIN_SCORE={thr}")
        env.write_text("\n".join(lines) + "\n")
        print(f"Wrote GT_MIN_SCORE={thr} to .env")
    return thr


if __name__ == "__main__":
    main()
