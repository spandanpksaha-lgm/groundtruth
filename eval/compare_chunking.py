"""Stretch goal: compare chunking strategies AND embedding models, with numbers.

Retrieval-only, so it needs no LLM:

    python eval/compare_chunking.py --corpus uc                       # embedder from .env
    python eval/compare_chunking.py --corpus uc --embedders st tfidf \\
        --st-models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5

Grid: {section, fixed} chunking x {120, 220, 400} words x each embedder
(each --st-models entry is its own dense model; tfidf is the keyword baseline).

For each config, on the TEST questions:
  hit@1 / hit@3 / hit@5  gold evidence in the top-k passages
  MRR                    mean reciprocal rank of the evidence
  AUROC                  how well the top similarity score separates answerable
                         from unanswerable questions (0.5 = coin flip); this is
                         the signal the "I don't know" gate relies on
Writes eval/results/chunking_comparison[_uc].{csv,md,png}.
"""
import argparse
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_threshold import auroc  # noqa: E402
from groundtruth.config import embedder_label, get_config  # noqa: E402
from groundtruth.evaluation import evidence_rank, load_questions  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402

OUT = ROOT / "eval" / "results"
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"section": "#2a78d6", "fixed": "#eb6834"}


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
    answerable = [q for q in qs if q["answerable"]]
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
        "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 3),
        "hit@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "auroc": round(auroc(pos, neg), 3) if neg else None,
        "missed": ",".join(q["id"] for q, r in zip(answerable, ranks) if r is None),
    }


def chart(rows, path, corpus="tts"):
    """Small multiples: one column per embedder, rows = Hit@5 and MRR;
    in each panel, section vs fixed chunking at each chunk size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from groundtruth.corpora import corpus_spec

    embs = list(dict.fromkeys(r["embedder"] for r in rows))
    sizes = sorted({r["chunk_size"] for r in rows})
    metrics = [("hit@5", "Hit@5"), ("mrr", "MRR")]
    fig, axes = plt.subplots(len(metrics), len(embs), figsize=(max(7.5, 3.2 * len(embs) + 1), 5.6),
                             dpi=200, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURF)
    w = 0.38
    for j, emb in enumerate(embs):
        for i, (m, mlabel) in enumerate(metrics):
            ax = axes[i][j]
            ax.set_facecolor(SURF)
            for k, ch in enumerate(["section", "fixed"]):
                vals = [next((r[m] for r in rows if r["embedder"] == emb and r["chunker"] == ch
                              and r["chunk_size"] == s), 0) for s in sizes]
                xs = [x + (k - 0.5) * w for x in range(len(sizes))]
                ax.bar(xs, vals, width=w - 0.04, color=COLORS[ch], edgecolor=SURF, linewidth=1,
                       label=f"{ch} chunks" if (i, j) == (0, 0) else None)
                for x, v in zip(xs, vals):
                    ax.text(x, v + 0.02, f"{v:.0%}" if m != "mrr" else f"{v:.2f}", ha="center",
                            va="bottom", fontsize=6.5, color=MUTED)
            ax.set_xticks(range(len(sizes)), [f"{s}w" for s in sizes], fontsize=7.5, color=INK)
            ax.set_ylim(0, 1.15)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(axis="y", labelsize=7, colors=MUTED)
            if i == 0:
                ax.set_title(emb, fontsize=9.5, color=INK)
            if j == 0:
                ax.set_ylabel(mlabel, fontsize=8.5, color=INK)
    fig.suptitle(f"{corpus_spec(corpus)['name']}: retrieval by chunking strategy and embedding model\n"
                 f"test set, answerable questions; higher is better", fontsize=10, color=INK,
                 x=0.02, ha="left", y=0.985)
    fig.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["tts", "uc"])
    p.add_argument("--embedders", nargs="+", help="st and/or tfidf (default: the embedder in .env)")
    p.add_argument("--st-models", nargs="+", help="dense models to compare when 'st' is listed "
                   "(default: GT_ST_MODEL)")
    p.add_argument("--sizes", nargs="+", type=int, default=[120, 220, 400])
    p.add_argument("--chunkers", nargs="+", default=["section", "fixed"])
    a = p.parse_args(argv)

    base = get_config(corpus=a.corpus)
    sfx = "" if base.corpus == "tts" else f"_{base.corpus}"
    embedders = a.embedders or [base.embedder]
    qs = load_questions(base.questions_path("test"))
    rows = []
    for emb in embedders:
        for model in (a.st_models or [base.st_model]) if emb == "st" else [None]:
            cfg = copy.copy(base)
            cfg.embedder = emb
            if model:
                cfg.st_model = model
            label = embedder_label(emb, cfg.st_model)
            for ch in a.chunkers:
                for size in a.sizes:
                    store = VectorStore.load_or_build(cfg, chunker=ch, embedder=emb, chunk_size=size,
                                                      verbose=False)
                    r = {"embedder": label, "chunker": ch, "chunk_size": size,
                         "n_chunks": len(store.chunks), **evaluate(store, qs)}
                    rows.append(r)
                    au = "  n/a" if r["auroc"] is None else f"{r['auroc']:.2f}"
                    print(f"  {label:10} {ch:8} {size:4}w  chunks={r['n_chunks']:4}  hit@1={r['hit@1']:.2f} "
                          f"hit@5={r['hit@5']:.2f}  MRR={r['mrr']:.2f}  AUROC={au}  missed={r['missed']}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"chunking_comparison{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    md = ["| Embedder | Chunker | Size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | AUROC | Missed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        au = "—" if r["auroc"] is None else f"{r['auroc']:.2f}"
        md.append(f"| {r['embedder']} | {r['chunker']} | {r['chunk_size']} | {r['n_chunks']} | "
                  f"{r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} | {r['mrr']:.2f} | "
                  f"{au} | {r['missed'] or '—'} |")
    (OUT / f"chunking_comparison{sfx}.md").write_text("\n".join(md) + "\n")
    chart(rows, OUT / f"chunking_comparison{sfx}.png", base.corpus)
    print(f"\nSaved {OUT.relative_to(ROOT)}/chunking_comparison{sfx}.{{csv,md,png}}")


if __name__ == "__main__":
    main()
