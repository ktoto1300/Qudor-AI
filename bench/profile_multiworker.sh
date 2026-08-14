#!/usr/bin/env bash
set -u

ROOT="${QUDOR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${QUDOR_PYTHON:-$ROOT/.venv/bin/python}"
CHECKPOINT="${QUDOR_PROFILE_CHECKPOINT:-$ROOT/runs/official_rules_v2_batched/best.pt}"
OUT="${QUDOR_PROFILE_OUT:-$ROOT/bench/multi_profiles}"
cd "$ROOT" || exit 1
mkdir -p "$OUT"

run_group() {
  workers=$1
  games=$2
  start=$(date +%s%N)
  for worker in $(seq 1 "$workers"); do
    "$PYTHON" bench/profile_server.py --checkpoint "$CHECKPOINT" \
      --games "$games" --concurrent-games "$games" --threads 1 \
      > "$OUT/w${workers}_${worker}.json" &
  done
  wait
  end=$(date +%s%N)
  "$PYTHON" -c "print(($end - $start) / 1e9)" > "$OUT/w${workers}_wall.txt"
}

run_group 2 128
run_group 4 64
run_group 6 43
