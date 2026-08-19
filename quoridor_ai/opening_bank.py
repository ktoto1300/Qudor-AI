"""Versioned, legality-checked opening lines for opt-in self-play seeding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core.encoding import MIRROR
from .core.engine import State, apply_unchecked, legal_actions

OPENING_BANK_VERSION = 1


def load_opening_bank(path, *, rules_version=2):
    """Load and validate a versioned bank, returning immutable action tuples."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != OPENING_BANK_VERSION:
        raise ValueError(f"opening bank version must be {OPENING_BANK_VERSION}")
    if payload.get("rules_version", rules_version) != rules_version:
        raise ValueError("opening bank rules_version does not match the engine")
    lines = payload.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError("opening bank lines must be a non-empty list")
    checked = []
    for line in lines:
        actions = line.get("actions") if isinstance(line, dict) else line
        if not isinstance(actions, list) or not actions:
            raise ValueError("each opening line must contain actions")
        state = State()
        clean = []
        for action in actions:
            if not isinstance(action, int) or action not in legal_actions(state):
                raise ValueError(f"illegal opening action {action!r}")
            clean.append(action)
            state = apply_unchecked(state, action)
        checked.append(tuple(clean))
    return tuple(checked)


def mirror_line(line):
    """Return the exact left-right reflection of an action line."""
    return tuple(int(MIRROR[action]) for action in line)


def select_opening(bank, seed, *, mirror=False):
    """Select a deterministic line from a loaded bank."""
    if not bank:
        return ()
    rng = np.random.default_rng(seed)
    line = bank[int(rng.integers(len(bank)))]
    return mirror_line(line) if mirror else line
