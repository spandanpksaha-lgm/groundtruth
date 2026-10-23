"""Run the full evaluation: retrieval + generation + scoring.

    python eval/run_eval.py                         # test set, settings from .env
    python eval/run_eval.py --chunker fixed --tag fixed-baseline
    python eval/run_eval.py --judge                 # add LLM-as-judge columns
    python eval/run_eval.py --split dev             # only for tuning, never report

Writes eval/results/<tag>/{per_question.csv, summary.json}.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.evaluation import LLMJudge, load_questions, run, save  # noqa: E402
from groundtruth.index import VectorStore  # noqa: E402
from groundtruth.llm import make_llm  # noqa: E402
from groundtruth.rag import GroundTruth  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--chunker", choices=["section", "fixed"])
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--embedder", choices=["st", "openai", "tfidf"])
    p.add_argument("--llm", choices=["openai", "anthropic", "extractive"], help="override LLM_PROVIDER")
    p.add_argument("--k", type=int)
    p.add_argument("--min-score", help="override refusal threshold")
    p.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    p.add_argument("--tag", help="results folder name")
    a = p.parse_args(argv)

    cfg = get_config(chunker=a.chunker, chunk_size=a.chunk_size, embedder=a.embedder,
                     llm_provider=a.llm, min_score=a.min_score)
    k = a.k or cfg.top_k
    store = VectorStore.load_or_build(cfg)
    gt = GroundTruth(cfg, store=store)
    judge = LLMJudge(make_llm(cfg)) if a.judge and cfg.llm_provider != "extractive" else None

    tag = a.tag or f"{a.split}-{cfg.chunker}{cfg.chunk_size}-{cfg.embedder}-{cfg.llm_provider}"
    config = {"split": a.split, "chunker": cfg.chunker, "chunk_size": cfg.chunk_size,
              "embedder": cfg.embedder, "st_model": cfg.st_model if cfg.embedder == "st" else None,
              "llm_provider": cfg.llm_provider, "llm_model": getattr(gt.llm, "model", ""),
              "top_k": k, "refusal_threshold": gt.threshold, "judge": bool(judge)}
    print(f"\n== {tag} ==  {json.dumps(config)}")

    qs = load_questions(ROOT / "eval" / f"questions_{a.split}.jsonl")
    rows, summary = run(gt, qs, k, judge)
    save(rows, summary, ROOT / "eval" / "results" / tag, config)

    keys = ["answer_correctness", "citation_faithfulness", "sentence_support_rate",
            "refusal_accuracy", "false_refusal_rate", f"retrieval_hit@{k}", "mrr",
            "harm_weighted_error"]
    print("\n" + "\n".join(f"  {key:24} {summary.get(key)}" for key in keys))
    print(f"  {'outcomes':24} {summary['outcomes']}")
    print(f"\nSaved eval/results/{tag}/")
    return summary


if __name__ == "__main__":
    main()
