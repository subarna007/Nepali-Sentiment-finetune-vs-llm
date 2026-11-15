#!/usr/bin/env bash
# Runs every experiment and builds the results table. Needs a GPU for steps 2-4.
set -e
python src/baseline_tfidf.py
python src/train_transformer.py --model xlm-roberta-base
python src/zero_shot_llm.py --shots 0
python src/zero_shot_llm.py --shots 2
python src/make_report.py
python src/make_figures.py
