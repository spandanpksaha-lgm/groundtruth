"""Stretch goal: compare chunking strategies (and optionally embedders) with numbers.

Retrieval-only, so it needs no LLM and runs in seconds:

    python eval/compare_chunking.py                       # embedder from .env
    python eval/compare_chunking.py --embedders st tfidf  # also compare embedders

For each config it reports, on the TEST questions:
  hit@1 / hit@3 / hit@5  gold evidence in the top-k passages
  MRR                    mean reciprocal rank of the evidence
  AUROC                  how well the top retrieval score separates answerable
                         from unanswerable questions (0.5 = coin flip)
Writes eval/results/chunking_comparison.{csv,md,png}.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibrate_threshold import auroc  # noqa: E402
from groundtruth.config import get_config  # noqa: E402
from groundtruth.evaluation import evidence_rank, load_questions  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402

OUT = ROOT / "eval" / "results"


def evaluate(store, qs, k=5):
    ranks, pos, neg = [], [], []
    for q in qs:
        hits = store.search(q["question"], k)
        if q["answerable"]:
            ranks.append(evidence_rank(hits, q["evidence"]))
            pos.append(hits[0].score)
        else:
            neg.append(hits[0].score)
    n = len(ranks)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
        "hit@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "auroc": round(auroc(pos, neg), 3),
        "missed": ",".join(q["id"] for q, r in zip([q for q in qs if q["answerable"]], ranks) if r is None),
    }


def chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    labels = [f"{r['chunker']}\n{r['chunk_size']}w · {r['embedder']}" for r in rows]
    metrics = [("hit@1", "#2a78d6"), ("hit@5", "#eb6834"), ("mrr", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(max(7, 1.25 * len(rows) + 2), 4.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    w = 0.26
    for j, (m, color) in enumerate(metrics):
        xs = [i + (j - 1) * w for i in range(len(rows))]
        vals = [r[m] for r in rows]
        ax.bar(xs, vals, width=w - 0.03, color=color, label=m.upper() if m == "mrr" else m,
               edgecolor="#fcfcfb", linewidth=1)
        if m == "hit@5":
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=muted)
    ax.set_xticks(range(len(rows)), labels, fontsize=8, color=ink)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=8, colors=muted)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.set_title("Retrieval quality by chunking strategy (test set, answerable questions)",
                 fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.13))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--embedders", nargs="+", help="default: the embedder in .env")
    p.add_argument("--sizes", nargs="+", type=int, default=[120, 220, 400])
    p.add_argument("--chunkers", nargs="+", default=["fixed", "section"])
    a = p.parse_args(argv)

    cfg = get_config()
    embedders = a.embedders or [cfg.embedder]
    qs = load_questions(ROOT / "eval" / "questions_test.jsonl")
    rows = []
    for emb in embedders:
        for ch in a.chunkers:
            for size in a.sizes:
                store = VectorStore.load_or_build(cfg, chunker=ch, embedder=emb, chunk_size=size,
                                                  verbose=False)
                r = {"embedder": emb, "chunker": ch, "chunk_size": size,
                     "n_chunks": len(store.chunks), **evaluate(store, qs)}
                rows.append(r)
                print(f"  {emb:6} {ch:8} {size:4}w  chunks={r['n_chunks']:4}  hit@1={r['hit@1']:.2f} "
                      f"hit@5={r['hit@5']:.2f}  MRR={r['mrr']:.2f}  AUROC={r['auroc']:.2f}  missed={r['missed']}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "chunking_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["| Embedder | Chunker | Size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | AUROC | Missed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['embedder']} | {r['chunker']} | {r['chunk_size']} | {r['n_chunks']} | "
                  f"{r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} | {r['mrr']:.2f} | "
                  f"{r['auroc']:.2f} | {r['missed'] or '—'} |")
    (OUT / "chunking_comparison.md").write_text("\n".join(md) + "\n")
    chart(rows, OUT / "chunking_comparison.png")
    print(f"\nSaved {OUT.relative_to(ROOT)}/chunking_comparison.{{csv,md,png}}")


if __name__ == "__main__":
    main()
