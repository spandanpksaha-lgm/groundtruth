#!/usr/bin/env bash
# The 2x2 grid for the UC Student Handbook: {section, fixed} chunking x
# {MiniLM, BGE-small} embeddings, on the team's 40-question hard benchmark.
# Safe to re-run: finished runs with unchanged settings are skipped, and only
# questions that hit Groq's rate limit are retried.
set -euo pipefail
cd "$(dirname "$0")"
BGE=BAAI/bge-small-en-v1.5

echo "== 1/4 Build the four UC indexes (downloads BGE-small, ~130 MB, the first time) =="
python scripts/build_index.py --corpus uc --chunker section
python scripts/build_index.py --corpus uc --chunker fixed
python scripts/build_index.py --corpus uc --chunker section --st-model $BGE
python scripts/build_index.py --corpus uc --chunker fixed --st-model $BGE

echo "== 2/4 Calibrate an 'I don't know' threshold for each of the four systems (DEV set) =="
python eval/calibrate_threshold.py --corpus uc --grid --write --keep-existing

echo "== 3/4 Team benchmark (40 questions) through each system =="
python eval/run_eval.py --corpus uc --split hard --chunker section --tag uc-hard-section220-main --skip-if-done
python eval/run_eval.py --corpus uc --split hard --chunker fixed --tag uc-hard-fixed220-main --skip-if-done
python eval/run_eval.py --corpus uc --split hard --chunker section --st-model $BGE --tag uc-hard-section220-bgesmall --skip-if-done
python eval/run_eval.py --corpus uc --split hard --chunker fixed --st-model $BGE --tag uc-hard-fixed220-bgesmall --skip-if-done

echo "== 4/4 Fill the team's sheet with all four columns and the 2x2 table =="
python eval/rescore.py --corpus uc --split hard
python eval/fill_benchmark_sheet.py
python eval/plot_grid_by_type.py
echo "Done. Open eval/results/UC_benchmark_results.xlsx (tab: Paired comparison) and eval/results/grid_by_type_uc.png."
