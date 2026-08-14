#!/usr/bin/env bash
set -u

ROOT="${QUDOR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${QUDOR_PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINT="${QUDOR_PROFILE_CHECKPOINT:-$ROOT/runs/official_rules_v2_batched/best.pt}"
OUT="${QUDOR_PROFILE_OUT:-$ROOT/bench/batch_profiles}"
cd "$ROOT" || exit 1
mkdir -p "$OUT"

for threads in 1 2 4 6 8; do
  for concurrent in 16 32 64 128; do
    echo "threads=$threads concurrent=$concurrent"
    "$PYTHON" bench/profile_server.py --checkpoint "$CHECKPOINT" \
      --games 128 --concurrent-games "$concurrent" --threads "$threads" \
      > "$OUT/t${threads}_c${concurrent}.json"
  done
done
