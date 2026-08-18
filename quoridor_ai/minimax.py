"""CPU minimax opponent using the engine's distance heuristic.

A fixed, non-learning opponent with actual lookahead. `baseline.py`'s greedy bot scores
one ply and cannot see a trap one move away; this one searches `depth` plies with
alpha-beta over the same `_score` evaluation, so it punishes exactly the shapes a
one-ply bot walks into.

Search is plain minimax from the root player's fixed viewpoint - `_score` is always
"how good is this for the root" - rather than negamax, because the evaluation is not
zero-sum-symmetric in its wall term and flipping its sign per ply would not reproduce
the opponent's own preference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .az_arena import _Duel, _search_round, summarise
from .baseline import _score
from .core.encoding import version_for_planes
from .core.engine import apply_unchecked, legal_actions
from .model import net_from_checkpoint
from .runtime import configure_threads, resolve_device
from .safe_loader import load_checkpoint

# Bigger than any _score can reach, so a forced win always outranks a heuristic gain.
# _score is bounded by BLOCK_WEIGHT * 127 (the unreachable sentinel) plus wall terms.
WIN_SCORE = 1e9


def _moves(s, player, width):
    """The `width` most promising (action, resulting state) pairs, best first.

    The child states are returned alongside the actions because ordering them already
    required building every one of them: `_score` needs the position, and rebuilding it
    in the caller would double the cost of the most expensive part of this search (two
    BFS sweeps per candidate).
    """
    scored = []
    for a in legal_actions(s):
        child = apply_unchecked(s, a)
        scored.append((_score(child, player), a, child))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(a, child) for _, a, child in scored[:width]]


def choose(s, rng, depth=2, width=12):
    """Best action for `s.player` under `depth`-ply alpha-beta search.

    Raises ValueError on a position with no legal action - a finished board or a
    stalemate. A bot cannot answer that question and silently returning something
    would hide the caller's real mistake.
    """
    if depth < 1:
        raise ValueError(f'depth must be at least 1, got {depth}')
    if width < 1:
        raise ValueError(f'width must be at least 1, got {width}')
    root = s.player

    def search(pos, left, alpha, beta):
        if pos.winner is not None:
            # `left` breaks ties toward the shorter mate: a win with plies to spare
            # scores above one found at the search horizon. Without it every winning
            # line looks equal and the bot dawdles in a won position.
            return (WIN_SCORE + left) if pos.winner == root else -(WIN_SCORE + left)
        moves = _moves(pos, pos.player, width) if left > 0 else []
        if not moves:
            # Horizon, or a position with no legal action. Either way the heuristic is
            # the best answer available; recursing on `left - 1` past zero would run to
            # the end of the game.
            return _score(pos, root)
        maximising = pos.player == root
        best = -math.inf if maximising else math.inf
        for _a, child in moves:
            val = search(child, left - 1, alpha, beta)
            if maximising:
                best = max(best, val)
                alpha = max(alpha, best)
            else:
                # `best` starts at +inf here. Seeding it with -inf - as this once did -
                # pins every min node at -inf and collapses the whole search: every
                # root candidate then scores -inf and the tie-break below picks at
                # random. tests/test_minimax.py guards against that specifically.
                best = min(best, val)
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    candidates = _moves(s, root, width)
    if not candidates:
        raise ValueError('no legal action in this position')
    scored = [(a, search(child, depth - 1, -math.inf, math.inf)) for a, child in candidates]
    best = max(v for _, v in scored)
    # Random tie-breaking keeps a match from being one game repeated `games` times.
    return int(rng.choice([a for a, v in scored if v >= best - 1e-9]))


def play(net, device, games=100, sims=64, depth=2, width=12, seed=0, max_plies=220,
         c_puct=1.6, temp=0.6, gumbel=True, gumbel_cap=16):
    """Play `net`, with search, against minimax. Same result shape as `baseline.play`.

    Colours alternate game by game: player 0 has a real first-move advantage in
    Quoridor, so a single-colour match measures that as much as it measures strength.
    """
    if games < 0:
        raise ValueError('games must be non-negative')
    if depth < 1:
        raise ValueError(f'depth must be at least 1, got {depth}')
    if width < 1:
        raise ValueError(f'width must be at least 1, got {width}')
    duels = [_Duel(i % 2 == 1, seed * 104729 + i) for i in range(games)]
    # Deliberately after the duels are built and before the net is touched: `games=0`
    # is a legitimate degenerate call, and it must not require a loaded network.
    enc = version_for_planes(net.planes) if duels else None
    live = list(duels)
    while live:
        group = [d for d in live if d.mover_is_a()
                 and d.s.winner is None and d.s.ply < max_plies]
        if group:
            _search_round(net, group, device, enc, sims, c_puct, temp, max_plies,
                          gumbel, gumbel_cap)
        for d in live:
            # Re-checked rather than reusing `group`: a game can end on the net's move,
            # and a finished board has no legal action for minimax to choose from.
            if not d.mover_is_a() and d.s.winner is None and d.s.ply < max_plies:
                d.s = apply_unchecked(d.s, choose(d.s, d.rng, depth, width))
        still = []
        for d in live:
            if d.s.winner is not None:
                d.result = 1.0 if (d.s.winner == 0) != d.swap else 0.0
            elif d.s.ply >= max_plies:
                d.result = 0.5
            else:
                still.append(d)
        live = still
    return summarise(duels, bot='minimax', depth=depth, width=width, sims=sims,
                     temperature=temp, gumbel=bool(gumbel), seed=seed)


def main():
    p = argparse.ArgumentParser(description='Play a checkpoint against CPU minimax')
    p.add_argument('--net', required=True)
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--sims', type=int, default=64, help='matches the play-time default')
    p.add_argument('--depth', type=int, default=2, help='minimax plies, at least 1')
    p.add_argument('--width', type=int, default=12, help='candidates kept per node')
    p.add_argument('--temp', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-plies', type=int, default=220)
    p.add_argument('--puct', action='store_true', help='PUCT search instead of Gumbel')
    p.add_argument('--threads', type=int, help='intra-op threads; minimax is single-game')
    p.add_argument('--output', help='write the result dict here as JSON')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.games <= 0:
        p.error('--games must be positive')
    if a.depth < 1:
        p.error('--depth must be at least 1')
    if a.width < 1:
        p.error('--width must be at least 1')

    configure_threads(a.threads or 1)
    dev = resolve_device(a.device)
    ck = load_checkpoint(a.net, map_location=dev)
    net = net_from_checkpoint(ck, dev)
    r = play(net, dev, games=a.games, sims=a.sims, depth=a.depth, width=a.width,
             seed=a.seed, max_plies=a.max_plies, temp=a.temp, gumbel=not a.puct)
    r.update(net=str(a.net), generation=ck.get('generation'),
             iteration=ck.get('iteration'), device=str(dev))
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + '\n')
    print(f"gen {r['generation']} vs minimax(d{a.depth}w{a.width}): "
          f"{r['wins']}W {r['draws']}D {r['losses']}L of {r['games']} "
          f"->  win rate {r['win_rate']:.3f}")
    print(f"  as player 0 {r['win_rate_as_p0']:.3f} | as player 1 {r['win_rate_as_p1']:.3f} "
          f"| avg length {r['avg_plies']:.1f} plies")
    return r


if __name__ == '__main__':
    main()
