#!/usr/bin/env bash
# Every number for the UC Student Handbook slides, in one command.
# Needs .env with your LLM settings. Takes ~10 minutes (Groq free tier pauses
# when it hits its per-minute limit; that's normal).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/6 Build UC indexes (section + fixed chunking, MiniLM + BGE-small) =="
python scripts/build_index.py --corpus uc --chunker section
python scripts/build_index.py --corpus uc --chunker fixed
python scripts/build_index.py --corpus uc --chunker section --st-model BAAI/bge-small-en-v1.5
python scripts/build_index.py --corpus uc --chunker fixed --st-model BAAI/bge-small-en-v1.5

echo "== 2/6 Calibrate a refusal threshold for each of the 4 systems on the UC DEV set =="
python eval/calibrate_threshold.py --corpus uc --grid --write --keep-existing

echo "== 3/6 Stretch goal: 2 chunking strategies x 3 sizes x 3 embedders (retrieval only) =="
# Downloads BGE-small (~130 MB) the first time.
python eval/compare_chunking.py --corpus uc --embedders st tfidf \
    --st-models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
echo "== Course connection: the embedding (latent) space =="
python eval/plot_latent_space.py --corpus uc
python eval/plot_latent_space.py --corpus uc --st-model BAAI/bge-small-en-v1.5 --out-suffix _bge

echo "== 4/6 Full evaluation on the UC TEST set =="
python eval/run_eval.py --corpus uc --chunker section --tag uc-test-section220-main
python eval/run_eval.py --corpus uc --chunker fixed   --tag uc-test-fixed220-baseline
python eval/run_eval.py --corpus uc --chunker section --llm extractive --tag uc-test-section220-nollm

echo "== 5/6 Team's 40-question hard benchmark through all 4 systems (2x2) =="
python eval/run_eval.py --corpus uc --split hard --chunker section --tag uc-hard-section220-main --skip-if-done
python eval/run_eval.py --corpus uc --split hard --chunker fixed   --tag uc-hard-fixed220-main --skip-if-done
python eval/run_eval.py --corpus uc --split hard --chunker section --st-model BAAI/bge-small-en-v1.5 --tag uc-hard-section220-bgesmall --skip-if-done
python eval/run_eval.py --corpus uc --split hard --chunker fixed   --st-model BAAI/bge-small-en-v1.5 --tag uc-hard-fixed220-bgesmall --skip-if-done
python eval/rescore.py --corpus uc --split hard
python eval/fill_benchmark_sheet.py
python eval/plot_grid_by_type.py

echo "== 6/6 Reports and charts =="
python eval/make_report.py --corpus uc uc-test-section220-main uc-test-fixed220-baseline uc-test-section220-nollm
python eval/make_report.py --corpus uc --split hard uc-hard-section220-main uc-hard-fixed220-main uc-hard-section220-bgesmall uc-hard-fixed220-bgesmall
echo "Done. Open eval/results/report_uc.md, report_uc_hard.md and UC_benchmark_results.xlsx"
