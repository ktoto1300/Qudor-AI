"""Head-to-head evaluation with search, used to gate promotions.

Comparing raw policy heads (what arena.py does) measures the wrong thing: what plays the
game is policy *plus* search, and a net with a slightly worse prior can easily search
better. Gating on the deployed configuration is the only comparison that predicts strength.

Games are batched across boards for the same reason self-play is: one forward pass per
round over all games in flight rather than per position.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .az_selfplay import (Node, _Sched, _backup, _evaluate, _improved_policy, _select,
                          _select_gumbel, _terminal_value)
from .core.encoding import is_canonical, version_for_planes
from .core.engine import State, apply_unchecked
from .model import net_from_checkpoint
from .runtime import resolve_device
from .safe_loader import load_checkpoint


class _Duel:
    def __init__(self, swap: bool, seed: int):
        self.s = State()
        self.swap = swap            # True: net B moves first
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.sched = None           # Sequential Halving state, Gumbel mode only
        self.result = None

    def mover_is_a(self):
        return (self.s.player == 0) != self.swap


def _search_round(net, group, device, enc, sims, c_puct, temp, max_plies, gumbel, gumbel_cap):
    """Search, then move, for every duel in `group` - one batched pass per round.

    Shared by the promotion gate and by the baseline harness, so that a win rate against a
    hand-written bot is produced by exactly the same move-selection rule as a win rate
    against the reigning champion. Two rules differing in temperature, or in what the final
    move is read from, would not be comparable numbers.
    """
    canon = is_canonical(enc)
    for d in group:
        d.root = Node(d.s)
    _evaluate(net, [d.root for d in group], device, enc, canon)
    if gumbel:
        for d in group:
            d.sched = _Sched(d.root, d.rng, sims, gumbel_cap)
        rounds = max(d.sched.budget for d in group)
    else:
        rounds = sims
    spent = {id(d): 0 for d in group}
    for _ in range(rounds):
        pending, paths = [], []
        for d in group:
            if spent[id(d)] >= rounds:
                continue
            if gumbel:
                leaf, path = _select_gumbel(d, max_plies)
                if leaf is None:            # halving schedule exhausted early
                    spent[id(d)] = rounds
                    continue
            else:
                leaf, path = _select(d.root, c_puct, max_plies)
            spent[id(d)] += 1
            if leaf.terminal is not None:
                _backup(path, leaf.terminal)
            else:
                pending.append(leaf)
                paths.append(path)
        _evaluate(net, pending, device, enc, canon)
        for leaf, path in zip(pending, paths):
            _backup(path, leaf.terminal if leaf.terminal is not None else leaf.value)
    for d in group:
        if gumbel:
            # Act on the improved policy itself. Using the Gumbel winner here would carry
            # training-time exploration noise into the match: measured on a trained net it
            # agrees with a 400-sim search only 54% of the time, against 66% for plain PUCT
            # at the same budget.
            probs = _improved_policy(d.root).astype(np.float64)
        else:
            probs = d.root.n.astype(np.float64)
            if probs.sum() <= 0:
                probs = d.root.p.astype(np.float64)
        probs = probs / probs.sum()
        # A little temperature early keeps the games from being byte-identical, which would
        # make the win rate meaningless.
        if temp > 0 and d.s.ply < 16:
            p = probs ** (1.0 / temp)
            i = int(d.rng.choice(len(p), p=p / p.sum()))
        else:
            i = int(np.argmax(probs))
        d.s = apply_unchecked(d.s, d.root.acts[i])


def _play_batch(a, b, device, games, sims, c_puct, temp, max_plies, seed, enc_a, enc_b,
                gumbel=False, gumbel_cap=16):
    duels = [_Duel(i % 2 == 1, seed * 104729 + i) for i in range(games)]
    live = list(duels)
    while live:
        # Split by which net is to move: each net gets its own batched search round.
        for net, enc, want_a in ((a, enc_a, True), (b, enc_b, False)):
            # A game can finish inside the first group's move, so the finished ones must be
            # dropped here too - searching a terminal board has no legal actions and crashes.
            group = [d for d in live if d.mover_is_a() == want_a
                     and d.s.winner is None and d.s.ply < max_plies]
            if group:
                _search_round(net, group, device, enc, sims, c_puct, temp, max_plies,
                              gumbel, gumbel_cap)
        still = []
        for d in live:
            if d.s.winner is not None:
                d.result = 1.0 if (d.s.winner == 0) != d.swap else 0.0
            elif d.s.ply >= max_plies:
                d.result = 0.5
            else:
                still.append(d)
                continue
        live = still
    return [d.result for d in duels]


def compare(a, b, device, games=40, sims=100, c_puct=1.6, temp=0.6, max_plies=220, seed=0,
            gumbel=False, gumbel_cap=16):
    """Score net `a` against net `b`. Returns a dict; win_rate counts draws as a half."""
    enc_a = version_for_planes(a.planes)
    enc_b = version_for_planes(b.planes)
    scores = _play_batch(a, b, device, games, sims, c_puct, temp, max_plies, seed, enc_a, enc_b,
                         gumbel, gumbel_cap)
    wr = sum(scores) / max(1, len(scores))
    elo = 400 * math.log10(max(1e-4, wr) / max(1e-4, 1 - wr))
    return {'games': len(scores), 'wins': sum(1 for s in scores if s == 1),
            'draws': sum(1 for s in scores if s == 0.5),
            'losses': sum(1 for s in scores if s == 0),
            'win_rate': wr, 'elo_delta': elo, 'sims': sims, 'temperature': temp,
            'gumbel': bool(gumbel), 'seed': seed}


def run(candidate, best, games, out, sims=100, temp=0.6, seed=0, gumbel=False, device=None):
    d = resolve_device(device)
    a = net_from_checkpoint(load_checkpoint(candidate, map_location=d), d)
    b = net_from_checkpoint(load_checkpoint(best, map_location=d), d)
    r = compare(a, b, d, games, sims, temp=temp, seed=seed, gumbel=gumbel)
    Path(out).write_text(json.dumps(r, indent=2))
    print(r)
    return r


def main():
    p = argparse.ArgumentParser(description='MCTS head-to-head between two checkpoints')
    p.add_argument('--candidate', required=True)
    p.add_argument('--best', required=True)
    p.add_argument('--games', type=int, default=40)
    p.add_argument('--sims', type=int, default=100)
    p.add_argument('--temp', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gumbel', action='store_true', help='Gumbel search instead of PUCT')
    p.add_argument('--output', default='arena.json')
    p.add_argument('--device', help="'cpu' or 'cuda'; overrides autodetection")
    a = p.parse_args()
    run(a.candidate, a.best, a.games, a.output, a.sims, a.temp, a.seed, a.gumbel, a.device)


if __name__ == '__main__':
    main()
