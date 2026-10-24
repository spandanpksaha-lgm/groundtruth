"""Run the full evaluation: retrieval + generation + scoring.

    python eval/run_eval.py                         # test set, settings from .env
    python eval/run_eval.py --chunker fixed --tag fixed-baseline
    python eval/run_eval.py --judge                 # add LLM-as-judge columns
    python eval/run_eval.py --split dev             # only for tuning, never report
    python eval/run_eval.py --corpus uc             # UC Student Handbook
    python eval/run_eval.py --corpus uc --split hard --chunker fixed \
        --st-model BAAI/bge-small-en-v1.5 --tag uc-hard-fixed220-bgesmall

Writes eval/results/<tag>/{per_question.csv, summary.json}.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.evaluation import LLMJudge, load_questions, load_rows, run, save, summarise  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402
from groundtruth.llm import DailyLimitError, make_llm  # noqa: E402
from groundtruth.rag import GroundTruth  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["tts", "uc"], help="document set (default: tts)")
    p.add_argument("--split", default="test", choices=["test", "dev", "hard"])
    p.add_argument("--chunker", choices=["section", "fixed"])
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--embedder", choices=["st", "openai", "tfidf"])
    p.add_argument("--st-model", help="dense model, e.g. BAAI/bge-small-en-v1.5")
    p.add_argument("--llm", choices=["openai", "anthropic", "extractive"], help="override LLM_PROVIDER")
    p.add_argument("--k", type=int)
    p.add_argument("--min-score", help="override refusal threshold")
    p.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    p.add_argument("--tag", help="results folder name")
    p.add_argument("--retry-errors", action="store_true",
                   help="re-run only the questions that failed with llm_error (e.g. rate limits) in an "
                        "existing --tag run, and update that run's results in place")
    p.add_argument("--skip-if-done", action="store_true",
                   help="if --tag already has results with the same settings, don't re-run it "
                        "(only re-run questions that hit llm_error)")
    a = p.parse_args(argv)

    cfg = get_config(corpus=a.corpus, chunker=a.chunker, chunk_size=a.chunk_size, embedder=a.embedder,
                     st_model=a.st_model, llm_provider=a.llm, min_score=a.min_score)
    k = a.k or cfg.top_k
    store = VectorStore.load_or_build(cfg)
    gt = GroundTruth(cfg, store=store)
    judge = LLMJudge(make_llm(cfg)) if a.judge and cfg.llm_provider != "extractive" else None

    tag = a.tag or f"{cfg.results_prefix}{a.split}-{cfg.chunker}{cfg.chunk_size}-{cfg.embedder}-{cfg.llm_provider}"
    config = {"corpus": cfg.corpus, "split": a.split, "chunker": cfg.chunker, "chunk_size": cfg.chunk_size,
              "embedder": cfg.embedder, "st_model": cfg.st_model if cfg.embedder == "st" else None,
              "llm_provider": cfg.llm_provider, "llm_model": getattr(gt.llm, "model", ""),
              "top_k": k, "refusal_threshold": gt.threshold, "judge": bool(judge)}
    print(f"\n== {tag} ==  {json.dumps(config)}")

    qpath = cfg.questions_path(a.split)
    if not qpath.exists():
        raise SystemExit(f"No question file {qpath.relative_to(ROOT)} for corpus '{cfg.corpus}'.")
    qs = load_questions(qpath)
    out_dir = ROOT / "eval" / "results" / tag
    if a.skip_if_done and (out_dir / "summary.json").exists() and not a.retry_errors:
        old_cfg = json.loads((out_dir / "summary.json").read_text()).get("config", {})
        same = all(old_cfg.get(key) == config[key] for key in
                   ("corpus", "split", "chunker", "chunk_size", "embedder", "st_model", "llm_model",
                    "top_k", "refusal_threshold"))
        if same:
            if any(r.refusal_reason == "llm_error" for r in load_rows(out_dir / "per_question.csv")):
                a.retry_errors = True
            else:
                print(f"{tag}: already done with these settings, skipping.")
                return
        else:
            changed = [key for key in ("chunker", "embedder", "st_model", "llm_model", "refusal_threshold")
                       if old_cfg.get(key) != config[key]]
            print(f"{tag}: settings changed ({', '.join(changed)}), re-running all questions.")
    if a.retry_errors:
        old = load_rows(out_dir / "per_question.csv")
        failed = {r.id for r in old if r.refusal_reason == "llm_error"}
        print(f"Re-running {len(failed)} question(s) that hit llm_error: {', '.join(sorted(failed)) or 'none'}")
        new, _ = run(gt, [q for q in qs if q["id"] in failed], k, judge)
        by_id = {r.id: r for r in new}
        rows = [by_id.get(r.id, r) for r in old]
        summary = summarise(rows, k)
    else:
        rows, summary = run(gt, qs, k, judge)
    save(rows, summary, out_dir, config)
    left = sum(r.refusal_reason == "llm_error" for r in rows)
    if left:
        print(f"\nNOTE: {left} question(s) failed because the LLM could not be reached (rate limit?). "
              f"Wait a minute, then re-run them only:\n  python eval/run_eval.py {' '.join(sys.argv[1:])} --retry-errors")

    keys = ["answer_correctness", "citation_faithfulness", "sentence_support_rate",
            "refusal_accuracy", "false_refusal_rate", f"retrieval_hit@{k}", "mrr",
            "harm_weighted_error"]
    print("\n" + "\n".join(f"  {key:24} {summary.get(key)}" for key in keys))
    print(f"  {'outcomes':24} {summary['outcomes']}")
    print(f"\nSaved eval/results/{tag}/")
    return summary


if __name__ == "__main__":
    try:
        main()
    except DailyLimitError as e:
        sys.exit(f"\nSTOPPED, nothing saved for this run (earlier results are untouched).\n{e}")
