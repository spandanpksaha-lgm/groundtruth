"""Which setup wins on which kind of question? A heatmap for the 2x2 grid.

    python eval/plot_grid_by_type.py

Rows: the team's question groups (primary stress, then question type).
Columns: the four systems (chunking x embedding model) that have been run.
Cell: share of questions answered correctly (correct answer with a supporting
citation, or a correct refusal), with the count. Writes
eval/results/grid_by_type_uc.png and grid_by_type_uc.csv.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from groundtruth.grid import cell_label, cells, run_tag  # noqa: E402

RES = ROOT / "eval" / "results"
GOOD = {"correct", "correct_refusal"}
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STRESS_NAMES = {"Fixed": "tests fixed chunking", "Section": "tests section chunking",
                "Refusal": "should refuse (not in handbook)"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    qs = {q["id"]: q for q in map(json.loads, open(ROOT / "eval" / "questions_uc_hard.jsonl"))}
    runs = []
    for ch, model, _ in cells("uc"):
        path = RES / run_tag("uc", "hard", ch, model) / "per_question.csv"
        if path.exists():
            runs.append((f"{ch.capitalize()}\n{cell_label(model)}",
                         {r["id"]: r["outcome"] in GOOD for r in csv.DictReader(open(path))}))
    if not runs:
        sys.exit("No hard-set runs yet. Run bash run_grid_uc.sh first.")

    groups = defaultdict(list)
    for qid, q in qs.items():
        groups["All 40 questions"].append(qid)
        stress = q.get("primary_stress") or "Refusal"
        groups[f"Stress: {STRESS_NAMES.get(stress, stress)}"].append(qid)
        cat = q["category"]
        if cat != "unanswerable_team":   # already its own row via "Stress: Refusal"
            groups["Type: " + cat.replace("_", " ").capitalize()].append(qid)
    # Fold one-question types into a single row so the chart stays readable.
    for k in [k for k in groups if k.startswith("Type") and len(groups[k]) < 2]:
        groups["Type: other (1 question each)"] += groups.pop(k)
    order = (["All 40 questions"] + sorted(k for k in groups if k.startswith("Stress"))
             + sorted((k for k in groups if k.startswith("Type")),
                      key=lambda k: (k.endswith("each)"), -len(groups[k]), k)))

    M = [[sum(ok.get(q, False) for q in groups[g]) / len(groups[g]) for _, ok in runs] for g in order]
    with open(RES / "grid_by_type_uc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n"] + [n.replace("\n", " + ") for n, _ in runs])
        for g, row in zip(order, M):
            w.writerow([g, len(groups[g])] + [round(v, 3) for v in row])

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    fig, ax = plt.subplots(figsize=(2.2 + 1.55 * len(runs), 0.42 * len(order) + 1.6), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for i, (g, row) in enumerate(zip(order, M)):
        n = len(groups[g])
        for j, v in enumerate(row):
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, color=cmap(v), linewidth=0))
            k = round(v * n)
            ax.text(j + 0.5, i + 0.5, f"{k}/{n}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else INK, fontweight="bold" if i == 0 else "normal")
        best = max(row)
        if best > min(row):
            for j, v in enumerate(row):
                if v == best:
                    ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, fill=False,
                                               edgecolor=INK, linewidth=1.2))
    ax.set_xlim(0, len(runs))
    ax.set_ylim(len(order), 0)
    ax.set_xticks([j + 0.5 for j in range(len(runs))], [n for n, _ in runs], fontsize=8, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(order))], [f"{g}  (n={len(groups[g])})" for g in order],
                  fontsize=8, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for y in (1, 1 + sum(k.startswith("Stress") for k in order)):
        ax.axhline(y, color=MUTED, linewidth=0.8)
    ax.set_title("Team benchmark: questions answered correctly, by question group and setup\n"
                 "darker = more correct; outlined = best setup for that group (if not a tie)",
                 fontsize=9.5, color=INK, loc="left", pad=34)
    fig.text(0.01, 0.01, "Small groups (n<5) are anecdotes, not evidence. Correct = right answer with a "
             "supporting citation, or a correct refusal.", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = RES / "grid_by_type_uc.png"
    fig.savefig(out, facecolor=SURF)
    print(f"Wrote {out.relative_to(ROOT)} and grid_by_type_uc.csv")


if __name__ == "__main__":
    main()
