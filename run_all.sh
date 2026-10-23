#!/usr/bin/env bash
# One command to produce every number for the slides.
# Needs .env with your LLM settings (see .env.example). Takes ~5-10 minutes,
# mostly LLM calls (30 questions x 3 configs, plus judge calls).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 Build indexes (section + fixed chunking) =="
python scripts/build_index.py --chunker section
python scripts/build_index.py --chunker fixed

echo "== 2/5 Calibrate the refusal threshold on the DEV set =="
python eval/calibrate_threshold.py --write

echo "== 3/5 Stretch goal: chunking x embedder comparison (retrieval only) =="
python eval/compare_chunking.py --embedders st tfidf

echo "== 4/5 Full evaluation on the TEST set =="
python eval/run_eval.py --chunker section --judge --tag test-section220-main
python eval/run_eval.py --chunker fixed   --judge --tag test-fixed220-baseline
python eval/run_eval.py --chunker section --llm extractive --tag test-section220-nollm

echo "== 5/5 Report and charts =="
python eval/make_report.py test-section220-main test-fixed220-baseline test-section220-nollm
echo "Done. Open eval/results/report.md"
