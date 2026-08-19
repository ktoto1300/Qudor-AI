"""Pure value-target helpers."""
from __future__ import annotations

import numpy as np


def alternating_td_lambda(values, rewards, dones, *, gamma=1.0, lam=0.8):
    """Compute TD(lambda) targets for values whose player alternates each ply.

    ``values[t]`` is from the player-to-move perspective at ply ``t``. Therefore the
    bootstrap value at the next ply is negated. ``values`` has one extra bootstrap
    value; ``rewards`` and ``dones`` have one entry per transition.
    """
    values = np.asarray(values, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    if values.ndim != 1 or rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError("values, rewards, and dones must be one-dimensional")
    if len(values) != len(rewards) + 1 or len(rewards) != len(dones):
        raise ValueError("values must have one more entry than rewards and dones")
    if not 0 <= lam <= 1 or gamma < 0:
        raise ValueError("gamma must be non-negative and lam must be in [0, 1]")
    targets = values[:-1].copy()
    advantage = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        bootstrap = 0.0 if dones[t] else -values[t + 1]
        delta = rewards[t] + gamma * bootstrap - values[t]
        advantage = delta + (0.0 if dones[t] else gamma * lam * advantage)
        targets[t] = values[t] + advantage
    return targets.astype(np.float32)
