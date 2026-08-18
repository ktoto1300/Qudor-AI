#!/usr/bin/env python3
"""CLI tool for reproducible baseline evaluations against deterministic bots.

Evaluates any checkpoint against `rusher` (shortest path baseline) and/or
`greedy` (1-ply heuristic blocker) with alternating colors and confidence intervals.

Usage examples:
    python scripts/eval_baseline.py --net checkpoints/gen69_best.pt --bot all --games 100
    python scripts/eval_baseline.py --net checkpoints/gen69_best.pt --bot greedy --games 50 --json results.json --md results.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quoridor_ai.baseline import main

if __name__ == "__main__":
    main()
