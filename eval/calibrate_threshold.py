"""Pick the retrieval-score threshold below which GroundTruth refuses without
calling the LLM. Calibrated on the DEV set only, so the test set stays clean.

    python eval/calibrate_threshold.py                 # current .env settings
    python eval/calibrate_threshold.py --embedder tfidf
    python eval/calibrate_threshold.py --corpus uc --write
    python eval/calibrate_threshold.py --corpus uc --grid --write   # all 4 cells of the 2x2

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
from groundtruth.grid import cell_config, cells  # noqa: E402
from groundtruth.evaluation import load_questions  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402


def auroc(pos, neg):
    """P(score of a random answerable question > a random unanswerable one)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def write_env(key, value, keep_existing=False):
    import os
    env = ROOT / ".env"
    if keep_existing and os.environ.get(key):
        print(f"Kept existing {key}={os.environ[key]} (finished runs used it)")
        return
    lines = [l for l in (env.read_text().splitlines() if env.exists() else []) if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    env.write_text("\n".join(lines) + "\n")
    print(f"Wrote {key}={value} to .env")


def calibrate(cfg, margin=0.9, write=False, quiet=False, keep_existing=False):
    store = VectorStore.load_or_build(cfg)
    qs = load_questions(cfg.questions_path("dev"))
    pos, neg = [], []
    for q in qs:
        top = store.search(q["question"], 1)[0].score
        (pos if q["answerable"] else neg).append(top)
        if not quiet:
            print(f"  {'ANS ' if q['answerable'] else 'UNANS'} {top:.3f}  {q['question'][:70]}")

    thr = round(min(pos) * margin, 3)
    caught = sum(n < thr for n in neg)
    au = auroc(pos, neg)
    print(f"\n[{cfg.corpus}] {cfg.chunker} chunks, {cfg.embedder_label}: answerable top-scores min={min(pos):.3f} "
          f"mean={sum(pos)/len(pos):.3f} | unanswerable max={max(neg):.3f} mean={sum(neg)/len(neg):.3f} "
          f"| AUROC {au:.2f}")
    # The default (section) cell also keeps the corpus-wide key, so older commands still work.
    keys = [cfg.cell_env_key]
    if cfg.chunker == "section" and cfg.embedder_label == "MiniLM":
        keys.append(cfg.threshold_env_key)
    print(f"Suggested threshold: {keys[0]}={thr}  (gate alone would catch {caught}/{len(neg)} "
          f"unanswerable dev questions; current {cfg.refusal_threshold})")
    if write:
        for key in keys:
            write_env(key, thr, keep_existing)
    return thr


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["tts", "uc"])
    p.add_argument("--chunker", choices=["section", "fixed"])
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--embedder", choices=["st", "openai", "tfidf"])
    p.add_argument("--st-model", help="dense model, e.g. BAAI/bge-small-en-v1.5")
    p.add_argument("--grid", action="store_true", help="calibrate every cell of the 2x2 grid")
    p.add_argument("--margin", type=float, default=0.9,
                   help="threshold = margin x lowest answerable top-score (safety buffer)")
    p.add_argument("--write", action="store_true", help="save the threshold to .env")
    p.add_argument("--keep-existing", action="store_true",
                   help="with --write: don't change a threshold that is already set in .env")
    a = p.parse_args(argv)

    if a.grid:
        corpus = a.corpus or "uc"
        for ch, model, _ in cells(corpus):
            calibrate(cell_config(corpus, ch, model, chunk_size=a.chunk_size), a.margin, a.write, quiet=True,
                      keep_existing=a.keep_existing)
        print("Note: the dev set is small. Treat these as sanity checks, not precise optima.")
        return
    cfg = get_config(corpus=a.corpus, chunker=a.chunker, chunk_size=a.chunk_size, embedder=a.embedder,
                     st_model=a.st_model)
    thr = calibrate(cfg, a.margin, a.write)
    print("Note: the dev set is small (9 questions). Treat this as a sanity check, "
          "not a precise optimum.")
    return thr


if __name__ == "__main__":
    main()
