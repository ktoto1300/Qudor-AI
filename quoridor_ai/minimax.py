"""CPU minimax opponent using the engine's distance heuristic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .baseline import _score
from .core.engine import apply_unchecked, legal_actions
from .model import net_from_checkpoint
from .runtime import configure_threads, resolve_device
from .safe_loader import load_checkpoint
from .az_arena import _Duel, _search_round
from .core.encoding import version_for_planes


def _moves(s, player, width):
    acts = legal_actions(s)
    ranked = sorted(acts, key=lambda a: _score(apply_unchecked(s, a), player), reverse=True)
    return ranked[:width]


def choose(s, rng, depth=2, width=12):
    root = s.player

    def search(pos, left, alpha, beta):
        if pos.winner is not None:
            return 1e9 if pos.winner == root else -1e9
        if left == 0:
            return _score(pos, root)
        best = -math.inf
        for act in _moves(pos, pos.player, width):
            val = search(apply_unchecked(pos, act), left - 1, alpha, beta)
            # Ordinary minimax: _score is always from root's perspective.
            best = max(best, val) if pos.player == root else min(best, val)
            if pos.player == root:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    scored = []
    for act in _moves(s, root, width):
        val = search(apply_unchecked(s, act), depth - 1, -math.inf, math.inf)
        scored.append((act, val))
    best = max(v for _, v in scored)
    return int(rng.choice([a for a, v in scored if v >= best - 1e-9]))


def main():
    p = argparse.ArgumentParser(description='Play a checkpoint against CPU minimax')
    p.add_argument('--net', required=True); p.add_argument('--games', type=int, default=100)
    p.add_argument('--sims', type=int, default=64); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--width', type=int, default=12); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output'); p.add_argument('--device', default='cpu')
    a=p.parse_args(); configure_threads(1); dev=resolve_device(a.device)
    ck=load_checkpoint(a.net,map_location=dev); net=net_from_checkpoint(ck,dev)
    enc=version_for_planes(net.planes)
    duels=[_Duel(i%2==1,a.seed*104729+i) for i in range(a.games)]; live=list(duels)
    while live:
        group=[d for d in live if d.mover_is_a() and d.s.winner is None and d.s.ply<220]
        if group: _search_round(net,group,dev,enc,a.sims,1.6,0.6,220,True,16)
        for d in live:
            if not d.mover_is_a() and d.s.winner is None and d.s.ply<220:
                d.s=apply_unchecked(d.s,choose(d.s,d.rng,a.depth,a.width))
        rest=[]
        for d in live:
            if d.s.winner is not None:d.result=1.0 if (d.s.winner==0)!=d.swap else 0.0
            elif d.s.ply>=220:d.result=.5
            else:rest.append(d)
        live=rest
    r={'games':a.games,'wins':sum(d.result==1 for d in duels),'draws':sum(d.result==.5 for d in duels),'losses':sum(d.result==0 for d in duels),'win_rate':sum(d.result for d in duels)/a.games,'bot':'minimax','depth':a.depth,'width':a.width,'net':a.net,'generation':ck.get('generation'),'iteration':ck.get('iteration'),'device':str(dev)}
    if a.output: Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__': main()
