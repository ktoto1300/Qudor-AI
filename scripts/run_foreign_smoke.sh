#!/usr/bin/env bash
set -u

ROOT="${QUDOR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${QUDOR_PYTHON:-$ROOT/.venv/bin/python}"
NET="${QUDOR_NET:-$ROOT/runs/official_rules_v2_batched/latest.pt}"
OUT="${QUDOR_FOREIGN_OUT:-$ROOT/results/foreign_smoke}"
export BOTS_DIR="${BOTS_DIR:-/root/bots}"
export QUDOR_REPO="$ROOT"
export CRYER_PLAYOUTS="${CRYER_PLAYOUTS:-40}"
export CRYER_ROLLOUT_CAP="${CRYER_ROLLOUT_CAP:-60}"
export GORISANSON_SIMS="${GORISANSON_SIMS:-20000}"

cd "$ROOT" || exit 1
mkdir -p "$OUT"
: > "$OUT/run.log"

for opponent in dimi marcobt15 gorisanson cryer berlioz vader; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $opponent" | tee -a "$OUT/run.log"
  "$PYTHON" -u tools/foreign_arena.py \
    --net "$NET" --opponent "$opponent" --games 2 --sims 16 \
    --max-plies 60 --seed 20260814 --device cuda \
    --output "$OUT/$opponent.json" > "$OUT/$opponent.log" 2>&1
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finish $opponent rc=$rc" | tee -a "$OUT/run.log"
done
