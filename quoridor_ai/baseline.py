"""Absolute strength check: the trained net against hand-written opponents.

Every Elo number this project produces is relative - a candidate against the reigning
champion, both from the same lineage. A self-play system can climb honestly on its own
scale while standing still in absolute terms, because "beat the previous version of
myself" and "play Quoridor well" are not the same objective. Nothing in the run so far
distinguishes the two.

These bots are the missing fixed point. They never learn, so their strength is constant
across the whole project's life, and a win rate against them means the same thing at
generation 15 as it will at generation 40. They are deliberately simple: the engine
already computes BFS distance to goal for the v3 encoder, and the whole game is a race
between two shortest paths, so a competent beginner is a handful of lines on top of
`dist_to_goal`.

Usage:  python -m quoridor_ai.baseline --net runs/az_15gb/best.pt --bot greedy --games 100
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .az_arena import _Duel, _search_round
from .core.encoding import version_for_planes
from .core.engine import apply_unchecked, dist_field, legal_actions
from .model import net_from_checkpoint
from .runtime import configure_threads, resolve_device
from .safe_loader import load_checkpoint

WALL_WEIGHT = 0.25      # a wall in hand is worth a quarter of a step of progress
BLOCK_WEIGHT = 1.5      # delaying the opponent must beat racing one square ourselves


def _dists(s):
    """(distance for player 0, distance for player 1). Two BFS sweeps, walls only."""
    return dist_field(s, 0)[s.p0], dist_field(s, 8)[s.p1]


def _score(s, p):
    """How good `s` looks for player `p`. Higher is better.

    The race term dominates, with an opponent delay worth a little more than our own
    progress. Otherwise every pawn step scores above every one-step wall delay and this
    supposed blocker silently becomes another rusher. Walls in hand break close ties, so
    it still will not spend one for no gain.
    """
    if s.winner is not None:
        return float('inf') if s.winner == p else float('-inf')
    d0, d1 = _dists(s)
    dme, dopp = (d0, d1) if p == 0 else (d1, d0)
    wme, wopp = (s.walls0, s.walls1) if p == 0 else (s.walls1, s.walls0)
    return BLOCK_WEIGHT * dopp - dme + WALL_WEIGHT * (wme - wopp)


def _pick(cands, rng):
    """Argmax over (action, score) with random tie-breaking, so games differ."""
    best = max(sc for _, sc in cands)
    return int(rng.choice([a for a, sc in cands if sc >= best - 1e-9]))


def rusher(s, rng):
    """Walk the shortest path to the goal. Never places a wall.

    The floor of the scale: this is a person who has just been told the rules and has not
    yet noticed that walls exist. Any trained net must beat it near-100%; failing to is
    proof that something is broken rather than merely weak.
    """
    p = s.player
    goal = 0 if p == 0 else 8
    field = dist_field(s, goal)
    moves = [a for a in legal_actions(s) if a < 81]
    return _pick([(a, -field[a]) for a in moves], rng)


def greedy(s, rng):
    """One-ply best move over every legal action, pawn moves and walls alike.

    Places a wall whenever it costs the opponent more than a step of its own progress,
    which makes it a genuine if crude opponent: it blocks, it races, and it never wastes
    a wall. It has no lookahead at all, so it walls itself into bad shapes and cannot see
    a trap one move away.
    """
    p = s.player
    return _pick([(a, _score(apply_unchecked(s, a), p)) for a in legal_actions(s)], rng)


BOTS = {'rusher': rusher, 'greedy': greedy}


def play(net, bot, device, games=100, sims=64, c_puct=1.6, temp=0.6, max_plies=220, seed=0,
         gumbel=True, gumbel_cap=16):
    """Play `net`, with search, against `bot`. Same result shape as `az_arena.compare`.

    Colours alternate game by game. Player 0 has a real first-move advantage in Quoridor, so
    a single-colour match would measure that advantage as much as it measures strength.
    """
    enc = version_for_planes(net.planes)
    duels = [_Duel(i % 2 == 1, seed * 104729 + i) for i in range(games)]
    live = list(duels)
    while live:
        group = [d for d in live if d.mover_is_a()
                 and d.s.winner is None and d.s.ply < max_plies]
        if group:
            _search_round(net, group, device, enc, sims, c_puct, temp, max_plies,
                          gumbel, gumbel_cap)
        for d in live:
            # Re-checked here rather than reusing `group`: a game can end on the net's move,
            # and a finished board has no legal action for the bot to choose from.
            if not d.mover_is_a() and d.s.winner is None and d.s.ply < max_plies:
                d.s = apply_unchecked(d.s, bot(d.s, d.rng))
        still = []
        for d in live:
            if d.s.winner is not None:
                d.result = 1.0 if (d.s.winner == 0) != d.swap else 0.0
            elif d.s.ply >= max_plies:
                d.result = 0.5
            else:
                still.append(d)
        live = still
    scores = [d.result for d in duels]
    first = [d.result for d in duels if not d.swap]
    second = [d.result for d in duels if d.swap]
    wr = sum(scores) / max(1, len(scores))
    elo = 400 * math.log10(max(1e-4, wr) / max(1e-4, 1 - wr))
    return {'games': len(scores), 'wins': sum(1 for s in scores if s == 1),
            'draws': sum(1 for s in scores if s == 0.5),
            'losses': sum(1 for s in scores if s == 0),
            'win_rate': wr, 'elo_delta': elo,
            'win_rate_as_p0': sum(first) / max(1, len(first)),
            'win_rate_as_p1': sum(second) / max(1, len(second)),
            'avg_plies': sum(d.s.ply for d in duels) / max(1, len(duels)),
            'sims': sims, 'temperature': temp, 'gumbel': bool(gumbel), 'seed': seed}


def main():
    p = argparse.ArgumentParser(description='Play a checkpoint against hand-written bots')
    p.add_argument('--net', required=True)
    p.add_argument('--bot', default='greedy', choices=sorted(BOTS))
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--sims', type=int, default=64, help='matches the play-time default')
    p.add_argument('--temp', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-plies', type=int, default=220)
    p.add_argument('--puct', action='store_true', help='PUCT search instead of Gumbel')
    p.add_argument('--threads', type=int, help='intra-op threads; defaults to every core')
    p.add_argument('--output', help='write the result dict here as JSON')
    p.add_argument('--device', help="'cpu' or 'cuda'; overrides autodetection")
    a = p.parse_args()

    dev = resolve_device(a.device)
    configure_threads(a.threads)
    ck = load_checkpoint(a.net, map_location=dev)
    net = net_from_checkpoint(ck, dev)
    r = play(net, BOTS[a.bot], dev, a.games, a.sims, temp=a.temp, seed=a.seed,
             max_plies=a.max_plies, gumbel=not a.puct)
    r.update(bot=a.bot, net=str(a.net), generation=ck.get('generation'),
             iteration=ck.get('iteration'), device=str(dev))
    if a.output:
        Path(a.output).write_text(json.dumps(r, indent=2))
    print(f"gen {r['generation']} vs {a.bot}: {r['wins']}W {r['draws']}D {r['losses']}L "
          f"of {r['games']}  ->  win rate {r['win_rate']:.3f}")
    print(f"  as player 0 {r['win_rate_as_p0']:.3f} | as player 1 {r['win_rate_as_p1']:.3f} "
          f"| avg length {r['avg_plies']:.1f} plies")
    return r


if __name__ == '__main__':
    main()
