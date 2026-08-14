#!/usr/bin/env bash
set -u

ROOT="${QUDOR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${QUDOR_OUTPUT:-$ROOT/runs/official_rules_v2_batched}"
CONFIG="${QUDOR_CONFIG:-$ROOT/configs/rtx3060_official_batched.json}"
INIT="${QUDOR_INIT:-$ROOT/runs/official_fixed_batched/latest.pt}"
PYTHON="${QUDOR_PYTHON:-$ROOT/.venv/bin/python}"

cd "$ROOT" || exit 1
mkdir -p "$OUT"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/started_at.txt"
rm -f "$OUT/finished_at.txt" "$OUT/exit_code.txt"

if [ -f "$OUT/latest.pt" ]; then
  "$PYTHON" -u -m quoridor_ai.az_train \
    --config "$CONFIG" --output "$OUT" --device cuda \
    >> "$OUT/training.log" 2>&1
else
  if [ ! -f "$INIT" ]; then
    echo "Initial checkpoint does not exist: $INIT" >&2
    exit 2
  fi
  "$PYTHON" -u -m quoridor_ai.az_train \
    --config "$CONFIG" --output "$OUT" --init "$INIT" \
    --no-resume --device cuda \
    > "$OUT/training.log" 2>&1
fi

rc=$?
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/finished_at.txt"
echo "$rc" > "$OUT/exit_code.txt"
exit "$rc"
