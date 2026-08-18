#!/usr/bin/env python3
"""CLI tool to scan experiment results and build unified metrics manifests.

Usage:
    python tools/metrics_manifest.py
    python tools/metrics_manifest.py --results-dir results/ --output-json results/MANIFEST.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quoridor_ai.results_manifest import main


if __name__ == "__main__":
    sys.exit(main())
