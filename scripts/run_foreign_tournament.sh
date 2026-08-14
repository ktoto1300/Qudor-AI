#!/usr/bin/env bash
set -u

ROOT="${QUDOR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${QUDOR_PYTHON:-$ROOT/.venv/bin/python}"
NET="${QUDOR_NET:-$ROOT/runs/official_rules_v2_batched/latest.pt}"
OUT="${QUDOR_FOREIGN_OUT:-$ROOT/results/foreign_full}"
GAMES="${QUDOR_FOREIGN_GAMES:-20}"
SIMS="${QUDOR_FOREIGN_SIMS:-64}"
export BOTS_DIR="${BOTS_DIR:-/root/bots}"
export QUDOR_REPO="$ROOT"
export GORISANSON_SIMS="${GORISANSON_SIMS:-60000}"
export CRYER_PLAYOUTS="${CRYER_PLAYOUTS:-60}"
export CRYER_ROLLOUT_CAP="${CRYER_ROLLOUT_CAP:-60}"
export BERLIOZ_ITER="${BERLIOZ_ITER:-250}"
export BERLIOZ_ROLLOUT_CAP="${BERLIOZ_ROLLOUT_CAP:-20}"

cd "$ROOT" || exit 1
mkdir -p "$OUT"

for opponent in gorisanson dimi marcobt15 cryer berlioz; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $opponent" | tee -a "$OUT/tournament.log"
  "$PYTHON" -u tools/foreign_arena.py \
    --net "$NET" --opponent "$opponent" --games "$GAMES" --sims "$SIMS" \
    --max-plies 160 --seed 20260814 --device cuda \
    --output "$OUT/$opponent.json" > "$OUT/$opponent.log" 2>&1
  rc=$?
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finish $opponent rc=$rc" | tee -a "$OUT/tournament.log"
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tournament complete" | tee -a "$OUT/tournament.log"
