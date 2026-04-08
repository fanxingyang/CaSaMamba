#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python -c "from predict import test_main; test_main('./data/6mA_C.elegans/test_pos.txt','./best_mamba_model.pth','./result/output_logits.xlsx')"
