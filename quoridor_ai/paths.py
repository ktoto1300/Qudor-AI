"""Where the project and its external dependencies live on disk.

Six bridge scripts and the foreign arena all needed the same two directories, and each
had spelled out an absolute Windows path of its own as the default. That is one hardcoded
path per file to fix when the tree moves, and a clone on any other machine fails six
different ways - each with a `ModuleNotFoundError` from deep inside a third-party repo
rather than a message about a missing directory.

Both locations are now derived from this file's own position, so a checkout works
wherever it sits, and both can still be overridden by environment variable for a layout
that does not follow the default.
"""
from __future__ import annotations

import os
from pathlib import Path

# quoridor_ai/paths.py -> quoridor_ai -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Third-party Quoridor engines, one subdirectory each. They are not part of this project
# and are deliberately outside it: they carry their own git history, weights and
# dependencies. A sibling of the repo is the layout the bridges were written against.
_DEFAULT_BOTS = REPO_ROOT.parent / 'bots'

BOTS_ENV = 'BOTS_DIR'
REPO_ENV = 'QUDOR_REPO'


def repo_root() -> Path:
    """This project's root. `QUDOR_REPO` overrides the location derived from this file."""
    override = os.environ.get(REPO_ENV)
    return Path(override).resolve() if override else REPO_ROOT


def bots_dir(require: bool = False) -> Path:
    """Directory holding the third-party engines. `BOTS_DIR` overrides the default.

    With `require=True` a missing directory raises instead of being handed to a bridge
    that would fail on an import several frames deeper.
    """
    override = os.environ.get(BOTS_ENV)
    path = Path(override).resolve() if override else _DEFAULT_BOTS
    if require and not path.is_dir():
        raise FileNotFoundError(
            f'foreign-bot directory not found: {path}. Clone the engines there, '
            f'or point {BOTS_ENV} at them.')
    return path


def bot_repo(name: str, require: bool = False) -> Path:
    """One third-party engine's directory.

    With `require=True` a missing engine names itself in the error, which is the
    difference between a fixable message and an import failure inside someone else's code.
    """
    path = bots_dir(require=require) / name
    if require and not path.is_dir():
        raise FileNotFoundError(
            f'foreign engine {name!r} not found at {path}. Clone it there, '
            f'or point {BOTS_ENV} at the directory that holds it.')
    return path
