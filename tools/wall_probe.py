"""How many walls does the net actually spend, and when?

The earlier measurement in this session played the net against a copy of itself with a
greedy policy. That is a mirror: both sides pick the same move for the same reason, and
the game collapses into a wall-dumping equilibrium that says little about play against a
real opponent. This plays the net against a racer instead - an opponent that always takes
the pawn move closest to its goal and never walls, which is roughly what a human beginner
does - and reports walls spent by ply 20, the metric proposed for tracking progress.

Run: python tools/wall_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quoridor_ai.az_selfplay import search
from quoridor_ai.core.engine import State, apply_unchecked, dist_to_goal, legal_actions
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.safe_loader import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
DEV = torch.device("cpu")
SIMS = 64
MAX_PLIES = 220


def racer_move(s: State) -> int:
    """The pawn move that gets closest to this player's goal. Never places a wall."""
    best, best_d = None, 10**9
    for a in legal_actions(s):
        if a >= 81:
            continue
        d = dist_to_goal(apply_unchecked(s, a), s.player)
        if d < best_d:
            best, best_d = a, d
    return best


def net_move(net, s: State, seed: int) -> int:
    actions, probs, _v = search(net, s, DEV, encoding=3, sims=SIMS,
                                max_plies=MAX_PLIES, seed=seed)
    return int(actions[int(probs.argmax())])


def play(net, opponent, net_seat: int, seed: int) -> dict:
    """One game. `opponent` is a net or None, in which case the racer plays that seat."""
    s = State()
    walls_at_20 = None
    while s.winner is None and s.ply < MAX_PLIES:
        if s.ply == 20 and walls_at_20 is None:
            walls_at_20 = 10 - (s.walls0 if net_seat == 0 else s.walls1)
        if not legal_actions(s):
            break
        if s.player == net_seat:
            a = net_move(net, s, seed + s.ply)
        elif opponent is not None:
            a = net_move(opponent, s, seed + 500 + s.ply)
        else:
            a = racer_move(s)
        s = apply_unchecked(s, a)
    if walls_at_20 is None:                      # game ended before ply 20
        walls_at_20 = 10 - (s.walls0 if net_seat == 0 else s.walls1)
    left = s.walls0 if net_seat == 0 else s.walls1
    return {"plies": s.ply, "winner": s.winner, "net_seat": net_seat,
            "walls_by_20": walls_at_20, "walls_left": left,
            "net_won": s.winner == net_seat}


def load(rel: str):
    net = net_from_checkpoint(load_checkpoint(ROOT / rel), DEV)
    net.eval()
    return net


def report(title: str, games: list[dict]) -> None:
    n = len(games)
    print(f"\n{title}")
    for g in games:
        seat = "синий" if g["net_seat"] == 0 else "оранжевый"
        res = "выиграла" if g["net_won"] else ("ничья" if g["winner"] is None else "проиграла")
        print(f"   за {seat:<10} потратила {g['walls_by_20']:>2} стен к 20-му полуходу, "
              f"осталось {g['walls_left']:>2}, {g['plies']:>3} полуходов, сеть {res}")
    print(f"   среднее: {sum(g['walls_by_20'] for g in games)/n:.1f} стен к 20-му полуходу, "
          f"{sum(g['walls_left'] for g in games)/n:.1f} осталось в конце, "
          f"побед {sum(g['net_won'] for g in games)}/{n}")


def main() -> None:
    gen8, gen0 = load("runs/az_15gb/best.pt"), load("runs/az_15gb/best_gen0_backup.pt")
    print(f"{SIMS} симуляций на ход, поиск жадный (без температуры)")

    report("Обученная сеть (поколение 8) против бегуна:",
           [play(gen8, None, seat, 100 + seat) for seat in (0, 1)])
    report("Необученная сеть (поколение 0) против бегуна:",
           [play(gen0, None, seat, 300 + seat) for seat in (0, 1)])
    report("Поколение 8 против самой себя (то, что я мерял раньше):",
           [play(gen8, gen8, 0, 700)])


if __name__ == "__main__":
    main()
