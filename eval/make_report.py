"""Turn one or more eval runs into slide-ready charts and a Markdown report.

    python eval/make_report.py                     # every run in eval/results/
    python eval/make_report.py test-section220-st-openai test-fixed220-st-openai

    python eval/make_report.py --corpus uc         # every uc-test-* run

Writes eval/results/report.md, metrics.png and outcomes.png
(for uc: report_uc.md, metrics_uc.png, outcomes_uc.png).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "eval" / "results"
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
METRICS = [("answer_correctness", "Answer\ncorrectness"),
           ("citation_faithfulness", "Citation\nfaithfulness"),
           ("refusal_accuracy", "Refusal accuracy\n(unanswerable)"),
           ("false_refusal_rate", "False refusals\n(lower is better)"),
           ("retrieval_hit@5", "Retrieval\nhit@5")]
OUTCOMES = [("correct", "Correct", "#0ca30c"),
            ("correct_refusal", "Correct refusal", "#1baf7a"),
            ("correct_bad_citation", "Right answer, wrong citation", "#fab219"),
            ("false_refusal", "False refusal", "#ec835a"),
            ("wrong_answer", "Wrong answer", "#d03b3b"),
            ("retrieval_miss", "Retrieval miss", "#a3a29c"),
            ("hallucinated", "Hallucinated (should refuse)", "#7a1f1f")]


def _style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def label(s):
    c = s["config"]
    return f"{c['chunker']} {c['chunk_size']}w · {c['embedder']} · {c['llm_model'] or c['llm_provider']}"


def _desc(runs):
    s = runs[0][1]
    return (f"{s['n_questions']} {s['config'].get('split', 'test')} questions "
            f"({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable)")


def _corpus_name(runs):
    from groundtruth.corpora import corpus_spec
    return corpus_spec(runs[0][1]["config"].get("corpus", "tts"))["name"]


def metrics_chart(runs, path):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    n = len(runs)
    w = 0.8 / n
    for j, (tag, s) in enumerate(runs):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(METRICS))]
        vals = [s.get(m) or 0 for m, _ in METRICS]
        ax.bar(xs, vals, width=w - 0.03, color=SERIES[j % len(SERIES)], label=label(s),
               edgecolor=SURF, linewidth=1)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(METRICS)), [l for _, l in METRICS], fontsize=8, color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(f"{_corpus_name(runs)} — {_desc(runs)}",
                 fontsize=10, color=INK, loc="left")
    if n > 1:
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def outcomes_chart(runs, path):
    fig, ax = plt.subplots(figsize=(9, 1.6 + 0.5 * len(runs)), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for i, (tag, s) in enumerate(runs):
        left = 0
        for key, name, color in OUTCOMES:
            v = s["outcomes"].get(key, 0)
            if not v:
                continue
            ax.barh(i, v, left=left, color=color, edgecolor=SURF, linewidth=2, height=0.6,
                    label=name if i == 0 or name not in ax.get_legend_handles_labels()[1] else None)
            ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if key not in ("correct_bad_citation",) else INK)
            left += v
    ax.set_yticks(range(len(runs)), [label(s) for _, s in runs], fontsize=7, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(s["n_questions"] for _, s in runs))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=7)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.set_title(f"{_corpus_name(runs)}: where each question landed (failure taxonomy)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.legend(seen.values(), seen.keys(), frameon=False, fontsize=7, ncol=4,
               loc="lower left", bbox_to_anchor=(0.02, 0.0))
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def failures_md(tag):
    rows = list(csv.DictReader(open(RES / tag / "per_question.csv")))
    bad = [r for r in rows if r["outcome"] not in ("correct", "correct_refusal")]
    out = [f"### Failures — `{tag}`", "",
           "| ID | Outcome | Question | What it said | Top score |", "|---|---|---|---|---|"]
    for r in bad:
        ans = r["answer"].replace("|", "/").replace("\n", " ")
        out.append(f"| {r['id']} | {r['outcome']} | {r['question']} | {ans[:160]}{'…' if len(ans) > 160 else ''} | {r['top_score']} |")
    return "\n".join(out)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--corpus", default="tts", choices=["tts", "uc"])
    ap.add_argument("--split", default="test", help="which question set's runs to collect")
    a = ap.parse_args(argv)
    sfx = ("" if a.corpus == "tts" else f"_{a.corpus}") + ("" if a.split == "test" else f"_{a.split}")
    prefix = a.split if a.corpus == "tts" else f"{a.corpus}-{a.split}"
    tags = a.tags or sorted(p.parent.name for p in RES.glob("*/summary.json")
                            if p.parent.name.startswith(prefix))
    runs = [(t, json.loads((RES / t / "summary.json").read_text())) for t in tags]
    if not runs:
        sys.exit("No runs found. Run eval/run_eval.py first.")
    metrics_chart(runs, RES / f"metrics{sfx}.png")
    outcomes_chart(runs, RES / f"outcomes{sfx}.png")

    keys = [m for m, _ in METRICS] + ["sentence_support_rate", "mrr", "harm_weighted_error",
                                      "judge_answer_correctness", "judge_faithfulness", "avg_latency_s"]
    md = [f"# GroundTruth evaluation report — {_corpus_name(runs)}", "",
          f"Question set: {_desc(runs)}, hand-written before looking at model output. "
          "Definitions are in `groundtruth/evaluation.py`.", "",
          "| Metric | " + " | ".join(f"`{t}`" for t, _ in runs) + " |",
          "|---|" + "---|" * len(runs)]
    for k in keys:
        if all(s.get(k) is None for _, s in runs):
            continue
        md.append(f"| {k} | " + " | ".join("—" if s.get(k) is None else str(s[k]) for _, s in runs) + " |")
    md += ["", f"![metrics](metrics{sfx}.png)", "", f"![outcomes](outcomes{sfx}.png)", ""]
    if (RES / f"chunking_comparison{sfx}.md").exists():
        md += ["## Chunking comparison (retrieval only)", "",
               (RES / f"chunking_comparison{sfx}.md").read_text(),
               f"![chunking](chunking_comparison{sfx}.png)", ""]
    md += ["## Failure log", ""] + [failures_md(t) + "\n" for t, _ in runs]
    (RES / f"report{sfx}.md").write_text("\n".join(md))
    print(f"Wrote {RES.relative_to(ROOT)}/report{sfx}.md, metrics{sfx}.png, outcomes{sfx}.png")


if __name__ == "__main__":
    main(sys.argv[1:])
