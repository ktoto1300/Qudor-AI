"""Bounded exact solver for Quoridor positions with no walls remaining."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from quoridor_ai.core.engine import State, apply_unchecked, legal_actions


class Outcome(str, Enum):
    WIN = "WIN"
    DRAW = "DRAW"
    LOSS = "LOSS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SolveResult:
    outcome: Outcome
    remaining_plies: int
    node_budget: int
    nodes: int
    optimal_actions: tuple[int, ...]
    pv: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome.value, "remaining_plies": self.remaining_plies,
                "node_budget": self.node_budget, "nodes": self.nodes,
                "optimal_actions": list(self.optimal_actions), "pv": list(self.pv)}


def solve(state: State, remaining_plies: int, node_budget: int = 100_000) -> SolveResult:
    if state.walls0 != 0 or state.walls1 != 0:
        raise ValueError("no_wall_solver requires walls0=walls1=0")
    if remaining_plies < 0 or node_budget < 1:
        raise ValueError("remaining_plies must be non-negative and node_budget positive")
    nodes = 0
    exhausted = False
    root_scores: list[tuple[Outcome, int, tuple[int, ...]]] = []
    cache: dict[tuple[int, int, int, int, int, int], tuple[Outcome, tuple[int, ...]]] = {}

    def search(s: State, depth: int) -> tuple[Outcome, tuple[int, ...]]:
        nonlocal nodes, exhausted
        key = (s.p0, s.p1, s.h, s.v, s.player, depth)
        if key in cache:
            return cache[key]
        if nodes >= node_budget:
            exhausted = True
            return Outcome.UNKNOWN, ()
        nodes += 1
        if s.winner is not None:
            return (Outcome.WIN if s.winner == s.player else Outcome.LOSS), ()
        if depth == 0:
            return Outcome.DRAW, ()
        actions = legal_actions(s)
        if not actions:
            return Outcome.DRAW, ()
        scored: list[tuple[Outcome, int, tuple[int, ...]]] = []
        unknown = False
        for action in actions:
            child, child_pv = search(apply_unchecked(s, action), depth - 1)
            if child is Outcome.UNKNOWN:
                unknown = True
                continue
            perspective = Outcome.LOSS if child is Outcome.WIN else Outcome.WIN if child is Outcome.LOSS else Outcome.DRAW
            scored.append((perspective, action, child_pv))
        if s is state:
            root_scores.extend(scored)
        if not scored:
            return Outcome.UNKNOWN if exhausted else Outcome.DRAW, ()
        rank = {Outcome.WIN: 2, Outcome.DRAW: 1, Outcome.LOSS: 0}
        best = max(rank[item[0]] for item in scored)
        chosen = [item for item in scored if rank[item[0]] == best]
        first = chosen[0]
        if unknown:
            return Outcome.UNKNOWN, ()
        result = first[0], (first[1],) + first[2]
        cache[key] = result
        return result

    outcome, pv = search(state, remaining_plies)
    actions: tuple[int, ...] = ()
    if outcome is not Outcome.UNKNOWN:
        rank = {Outcome.WIN: 2, Outcome.DRAW: 1, Outcome.LOSS: 0}
        actions = tuple(action for candidate, action, _ in root_scores if rank[candidate] == rank[outcome])
    return SolveResult(outcome, remaining_plies, node_budget, nodes, actions, pv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=int, default=76)
    parser.add_argument("--p1", type=int, default=4)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--h", type=int, default=0)
    parser.add_argument("--v", type=int, default=0)
    parser.add_argument("--ply", type=int, default=0)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--node-budget", type=int, default=100_000)
    args = parser.parse_args(argv)
    state = State(p0=args.p0, p1=args.p1, walls0=0, walls1=0, h=args.h, v=args.v,
                  player=args.player, ply=args.ply)
    result = solve(state, args.plies, args.node_budget)
    print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
