"""Course connection: what the embedding (latent) space looks like.

    python eval/plot_latent_space.py --corpus uc
    python eval/plot_latent_space.py --corpus uc --st-model BAAI/bge-small-en-v1.5

Writes two charts to eval/results/:
  latent_space[_uc].png    every handbook chunk projected from the embedding
                           space to 2D (PCA), colored by section; test questions
                           placed in the same space, each answerable question
                           joined to the chunk that holds its answer.
  similarity_gap[_uc].png  cosine similarity between each test question and its
                           nearest chunk, answerable vs unanswerable, with the
                           "I don't know" threshold. Retrieval is nearest-neighbour
                           search in this space; the threshold is a distance cut.
"""
import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.config import embedder_label, get_config  # noqa: E402
from groundtruth.evaluation import _contains, load_questions  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402

OUT = ROOT / "eval" / "results"
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER = "#c9c8c2"


def group_of(chunk, multi_doc: bool) -> str:
    if multi_doc:
        return chunk.title
    parts = [p.strip() for p in chunk.section.split(">")]
    # Skip container headings like "Policies and Regulations" to reach the topic.
    if len(parts) >= 2 and parts[0] in {"Policies and Regulations", "Student Life"}:
        return parts[1]
    return parts[0]


def short(s, n=34):
    return s if len(s) <= n else s[: n - 1] + "…"


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["tts", "uc"])
    p.add_argument("--embedder", choices=["st", "tfidf", "openai"])
    p.add_argument("--st-model")
    p.add_argument("--chunker", default="section")
    p.add_argument("--chunk-size", type=int, default=220)
    p.add_argument("--out-suffix", default="", help="added to file names, e.g. _bge")
    a = p.parse_args(argv)

    cfg = get_config(corpus=a.corpus, embedder=a.embedder)
    if a.st_model:
        cfg = copy.copy(cfg)
        cfg.st_model = a.st_model
    sfx = ("" if cfg.corpus == "tts" else f"_{cfg.corpus}") + a.out_suffix
    store = VectorStore.load_or_build(cfg, chunker=a.chunker, chunk_size=a.chunk_size, verbose=False)
    label = embedder_label(cfg.embedder, cfg.st_model)
    qs = load_questions(cfg.questions_path("test"))
    chunks, V = store.chunks, store.vectors
    Q = store.embedder.embed([q["question"] for q in qs])

    # --- 2D projection -----------------------------------------------------
    pca = PCA(n_components=2, random_state=0).fit(V)
    C2, Q2 = pca.transform(V), pca.transform(Q)
    var = pca.explained_variance_ratio_.sum()
    multi = len({c.doc_id for c in chunks}) > 1
    groups = [group_of(c, multi) for c in chunks]
    top = [g for g, _ in Counter(groups).most_common(len(PALETTE))]
    color = {g: PALETTE[i] for i, g in enumerate(top)}

    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    other = [i for i, g in enumerate(groups) if g not in color]
    if other:
        ax.scatter(C2[other, 0], C2[other, 1], s=10, color=OTHER, linewidths=0, label="Other sections")
    for g in top:
        idx = [i for i, x in enumerate(groups) if x == g]
        ax.scatter(C2[idx, 0], C2[idx, 1], s=12, color=color[g], linewidths=0, label=short(g))
    # answerable questions: joined to the chunk that holds their answer
    for qi, q in enumerate(qs):
        if not q["answerable"]:
            continue
        gold = next((ci for ci, c in enumerate(chunks)
                     if all(_contains(c.text, e) for e in q["evidence"])), None)
        if gold is not None:
            ax.plot([Q2[qi, 0], C2[gold, 0]], [Q2[qi, 1], C2[gold, 1]], color=INK, linewidth=0.5, alpha=0.5)
    ans = [i for i, q in enumerate(qs) if q["answerable"]]
    una = [i for i, q in enumerate(qs) if not q["answerable"]]
    ax.scatter(Q2[ans, 0], Q2[ans, 1], s=34, facecolor=SURF, edgecolor=INK, linewidths=1.2, zorder=3,
               label="Test question (answerable)")
    ax.scatter(Q2[una, 0], Q2[una, 1], s=40, marker="x", color=INK, linewidths=1.4, zorder=3,
               label="Test question (should say I don't know)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_xlabel("principal component 1", fontsize=8, color=MUTED)
    ax.set_ylabel("principal component 2", fontsize=8, color=MUTED)
    ax.set_title(f"{label} embedding space: {len(chunks)} chunks and {len(qs)} test questions "
                 f"(2D PCA keeps {var:.0%} of the variance)", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.01, 0.01, "Lines join each answerable question to the chunk holding its answer. "
             "Distances in 2D are approximate; retrieval uses cosine similarity in the full space.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out1 = OUT / f"latent_space{sfx}.png"
    fig.savefig(out1, facecolor=SURF)
    plt.close(fig)

    # --- similarity gap ------------------------------------------------------
    top_sim = (Q @ V.T).max(axis=1)
    thr = cfg.refusal_threshold
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for row, (name, idx, mk) in enumerate([("Answerable", ans, "o"), ("Should say\nI don't know", una, "x")]):
        y = row + rng.uniform(-0.12, 0.12, len(idx))
        if mk == "o":
            ax.scatter(top_sim[idx], y, s=30, facecolor="#2a78d6", edgecolor=SURF, linewidths=1, zorder=3)
        else:
            ax.scatter(top_sim[idx], y, s=36, marker="x", color="#eb6834", linewidths=1.6, zorder=3)
    # Each embedding model scores on its own scale, so the line is drawn only when
    # this model has its own calibrated threshold (calibrate_threshold.py --grid).
    import os
    show_thr = not a.st_model or bool(os.environ.get(cfg.cell_env_key))
    if show_thr:
        ax.axvline(thr, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(thr, 1.55, f" refusal threshold {thr:.2f}", fontsize=7.5, color=INK, va="center")
    ax.set_yticks([0, 1], ["Answerable", "Should say\nI don't know"], fontsize=8, color=INK)
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlabel("cosine similarity between the question and its nearest chunk", fontsize=8, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ov = sum(top_sim[i] >= thr for i in una)
    title = (f"{label}: trick questions still land close to real passages "
             f"({ov} of {len(una)} pass the threshold, so the LLM must decline them)" if show_thr else
             f"{label}: how close answerable vs trick questions land to their nearest passage")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    out2 = OUT / f"similarity_gap{sfx}.png"
    fig.savefig(out2, facecolor=SURF)
    plt.close(fig)
    print(f"[{cfg.corpus}] {label}: PCA variance kept {var:.0%}; "
          f"answerable top-sim mean {top_sim[ans].mean():.3f}, unanswerable {top_sim[una].mean():.3f}; "
          f"{ov}/{len(una)} unanswerable above threshold {thr:.2f}")
    print(f"Wrote {out1.relative_to(ROOT)} and {out2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
