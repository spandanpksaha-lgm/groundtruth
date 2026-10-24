"""Re-score saved runs with the current scoring rules, without calling the LLM.

    python eval/rescore.py --corpus uc --split hard

Use after fixing a scorer bug or a too-strict key fact: answers, retrieval
ranks and citation checks are kept as they were; only `correct` and the
outcome label are recomputed. Prints every question whose label changed.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.evaluation import key_facts_present, load_questions, load_rows, save, summarise  # noqa: E402


def outcome_for(q, r):
    if not q["answerable"]:
        return r.outcome
    if r.correct and r.cites_evidence:
        return "correct"
    if r.correct:
        return "correct_bad_citation"
    if not r.retrieval_hit:
        return "retrieval_miss"
    if r.refused:
        return "false_refusal"
    return "wrong_answer"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="uc", choices=["tts", "uc"])
    p.add_argument("--split", default="hard")
    a = p.parse_args(argv)
    cfg = get_config(corpus=a.corpus)
    qs = {q["id"]: q for q in load_questions(cfg.questions_path(a.split))}
    prefix = f"{cfg.results_prefix}{a.split}-"
    for d in sorted((ROOT / "eval" / "results").glob(f"{prefix}*")):
        if not (d / "per_question.csv").exists():
            continue
        rows = load_rows(d / "per_question.csv")
        changed = []
        for r in rows:
            q = qs.get(r.id)
            if q is None or not q["answerable"]:
                continue
            r.correct = (not r.refused) and key_facts_present(r.answer, q["key_facts"])
            new = outcome_for(q, r)
            if new != r.outcome:
                changed.append(f"{r.id}: {r.outcome} -> {new}")
                r.outcome = new
        meta = json.loads((d / "summary.json").read_text())
        save(rows, summarise(rows, meta["config"].get("top_k", 5)), d, meta["config"])
        print(f"{d.name}: {len(changed)} changed" + ("".join(f"\n   {c}" for c in changed)))


if __name__ == "__main__":
    main()
