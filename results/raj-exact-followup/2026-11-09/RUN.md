# Reproducing this local follow-up

Run from the existing language-aware-moe-placement repository, with its pinned Python environment, imported data/per_sequence files and earlier published count-analysis files present:

1. `.venv/bin/python exact_followup/01_exact_shuffle.py`
2. `.venv/bin/python exact_followup/02_tradeoff_curves.py`
3. `.venv/bin/python exact_followup/03_tokens_rebuild.py`
4. `.venv/bin/python exact_followup/04_prior_curve.py`
5. `.venv/bin/python exact_followup/plots_shuffle.py`
6. `.venv/bin/python exact_followup/validate.py > results/raj-exact-followup/2026-11-09/validation.txt`
7. `.venv/bin/python exact_followup/report.py`

The shuffle source writes all 100 fit/evaluation assignment vectors per condition, enabling replay. The curve script reuses completed per-dataset/GPU rows if they already exist; use a fresh output directory for a clean rerun after changing numerical logic. The preceding results/raj-followup tables supply the older Qwen3 Wikipedia curve and synthetic-rebuild measurements, with hashes recorded. The ZIP is an analysis artifact, not a standalone dataset or serving system. No text, model weights or GPU runs are included.
