#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Activate virtual environment
source /home/hiwe/environment/nlp_viettel/bin/activate

# Run QA
python -m src.qa.answer_mcq

echo "[OK] Wrote outputs/answer.md"