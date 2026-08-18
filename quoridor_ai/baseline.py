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

Usage:
  python -m quoridor_ai.baseline --net checkpoints/gen69_best.pt --bot all --games 100
  qudor-eval-baseline --net checkpoints/gen69_best.pt --bot greedy --games 100 --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .az_arena import _Duel, _search_round, elo_delta, summarise
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


def calc_confidence_intervals(scores: Sequence[Any], z: float = 1.95996) -> dict[str, Any]:
    """Calculate 95% confidence intervals for win rate and Elo delta.

    Draws count as 0.5. Sample variance accounts for draws directly.
    Elo delta CI is obtained by mapping win rate CI bounds through elo_delta().
    """
    if len(scores) > 0 and hasattr(scores[0], 'result'):
        raw_scores = [d.result for d in scores if d.result is not None]
    else:
        raw_scores = [float(s) for s in scores]

    n = len(raw_scores)
    if n == 0:
        return {
            'win_rate_ci_95': [0.0, 0.0],
            'win_rate_se': 0.0,
            'elo_delta_ci_95': [0.0, 0.0],
            'elo_delta_se': 0.0,
        }

    wr = sum(raw_scores) / n
    if n == 1:
        e = elo_delta(wr)
        return {
            'win_rate_ci_95': [round(wr, 4), round(wr, 4)],
            'win_rate_se': 0.0,
            'elo_delta_ci_95': [round(e, 2), round(e, 2)],
            'elo_delta_se': 0.0,
        }

    sample_var = sum((s - wr) ** 2 for s in raw_scores) / (n - 1)
    wr_se = math.sqrt(max(0.0, sample_var) / n)

    wr_low = max(0.0, wr - z * wr_se)
    wr_high = min(1.0, wr + z * wr_se)

    elo_low = elo_delta(wr_low)
    elo_high = elo_delta(wr_high)

    p_eff = min(max(wr, 1e-4), 1.0 - 1e-4)
    elo_se = (400.0 / (math.log(10.0) * p_eff * (1.0 - p_eff))) * wr_se

    return {
        'win_rate_ci_95': [round(wr_low, 4), round(wr_high, 4)],
        'win_rate_se': round(wr_se, 4),
        'elo_delta_ci_95': [round(elo_low, 2), round(elo_high, 2)],
        'elo_delta_se': round(elo_se, 2),
    }


def play(net, bot, device, games=100, sims=64, c_puct=1.6, temp=0.6, max_plies=220, seed=0,
         gumbel=True, gumbel_cap=16, tanh_value_transform=False, root_visit_compensation=False):
    """Play `net`, with search, against `bot`. Same result shape as `az_arena.compare`.

    Colours alternate game by game. Player 0 has a real first-move advantage in Quoridor, so
    a single-colour match would measure that advantage as much as it measures strength.
    """
    if games < 0:
        raise ValueError('games must be non-negative')
    enc = version_for_planes(net.planes)
    duels = [_Duel(i % 2 == 1, seed * 104729 + i) for i in range(games)]
    live = list(duels)
    while live:
        group = [d for d in live if d.mover_is_a()
                 and d.s.winner is None and d.s.ply < max_plies]
        if group:
            _search_round(net, group, device, enc, sims, c_puct, temp, max_plies,
                          gumbel, gumbel_cap, tanh_value_transform=tanh_value_transform,
                          root_visit_compensation=root_visit_compensation)
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
    res = summarise(duels, sims=sims, temperature=temp, gumbel=bool(gumbel), seed=seed)
    ci = calc_confidence_intervals(duels)
    res.update(ci)
    return res


def format_markdown_summary(results: list[dict[str, Any]] | dict[str, Any]) -> str:
    """Format baseline evaluation results into a Markdown summary table."""
    items = [results] if isinstance(results, dict) else list(results)
    lines = [
        "| Checkpoint | Bot | Games | W / D / L | Win Rate (95% CI) | Elo Delta (95% CI) | P0 / P1 Win Rate | Avg Plies | Search |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in items:
        net_name = Path(str(r.get("net", "model"))).name if r.get("net") else "model"
        bot = str(r.get("bot", "unknown"))
        games = int(r.get("games", 0))
        w = int(r.get("wins", 0))
        d = int(r.get("draws", 0))
        loss = int(r.get("losses", 0))
        wr = float(r.get("win_rate", 0.0))
        wr_ci = r.get("win_rate_ci_95", [wr, wr])
        elo = float(r.get("elo_delta", 0.0))
        elo_ci = r.get("elo_delta_ci_95", [elo, elo])
        p0 = float(r.get("win_rate_as_p0", 0.0))
        p1 = float(r.get("win_rate_as_p1", 0.0))
        avg_plies = float(r.get("avg_plies", 0.0))
        sims = int(r.get("sims", 64))
        mode = "Gumbel" if r.get("gumbel", True) else "PUCT"
        search_desc = f"{mode} ({sims} sims)"

        wr_str = f"{wr * 100:.1f}% [{wr_ci[0] * 100:.1f}%, {wr_ci[1] * 100:.1f}%]"
        elo_sign = "+" if elo > 0 else ""
        elo_low_sign = "+" if elo_ci[0] > 0 else ""
        elo_high_sign = "+" if elo_ci[1] > 0 else ""
        elo_str = f"{elo_sign}{elo:.1f} [{elo_low_sign}{elo_ci[0]:.1f}, {elo_high_sign}{elo_ci[1]:.1f}]"
        p0_p1_str = f"{p0 * 100:.1f}% / {p1 * 100:.1f}%"

        lines.append(
            f"| `{net_name}` | `{bot}` | {games} | {w} / {d} / {loss} | {wr_str} | {elo_str} | {p0_p1_str} | {avg_plies:.1f} | {search_desc} |"
        )
    return "\n".join(lines)


def evaluate_checkpoint(
    net_path: str | Path,
    bot_names: Sequence[str] = ("greedy",),
    games: int = 100,
    sims: int = 64,
    temp: float = 0.6,
    seed: int = 0,
    max_plies: int = 220,
    gumbel: bool = True,
    gumbel_cap: int = 16,
    device: Any = None,
    threads: int | None = None,
    tanh_value_transform: bool = False,
    root_visit_compensation: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a checkpoint against one or more deterministic baseline bots."""
    dev = resolve_device(device)
    if threads is not None:
        configure_threads(threads)
    ck = load_checkpoint(net_path, map_location=dev)
    net = net_from_checkpoint(ck, dev)

    results: list[dict[str, Any]] = []
    for bot_name in bot_names:
        if bot_name not in BOTS:
            raise ValueError(f"Unknown bot '{bot_name}'. Available: {sorted(BOTS)}")
        r = play(
            net,
            BOTS[bot_name],
            dev,
            games=games,
            sims=sims,
            temp=temp,
            seed=seed,
            max_plies=max_plies,
            gumbel=gumbel,
            gumbel_cap=gumbel_cap,
            tanh_value_transform=tanh_value_transform,
            root_visit_compensation=root_visit_compensation,
        )
        r.update(
            bot=bot_name,
            net=str(net_path),
            generation=ck.get("generation"),
            iteration=ck.get("iteration"),
            rules_version=ck.get("rules_version"),
            device=str(dev),
        )
        results.append(r)
    return results


def main(args=None):
    p = argparse.ArgumentParser(description="Play a checkpoint against hand-written baseline bots")
    p.add_argument("--net", required=True, help="Path to checkpoint .pt file")
    p.add_argument(
        "--bot",
        default="greedy",
        help="Bot to play against ('greedy', 'rusher', or 'all'/'both' for all bots; default: greedy)",
    )
    p.add_argument("--games", type=int, default=100, help="Number of games to evaluate (default: 100)")
    p.add_argument("--sims", type=int, default=64, help="Simulations per move (default: 64)")
    p.add_argument("--temp", type=float, default=0.6, help="Opening move temperature (default: 0.6)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    p.add_argument("--max-plies", type=int, default=220, help="Max plies before draw (default: 220)")
    p.add_argument("--puct", action="store_true", help="Use PUCT search instead of Gumbel")
    p.add_argument("--search", choices=["gumbel", "puct"], help="Explicit search algorithm choice")
    p.add_argument("--threads", type=int, help="PyTorch intra-op threads")
    p.add_argument("--device", help="'cpu' or 'cuda'; overrides autodetection")
    p.add_argument("--output", "-o", "--json", dest="output", help="Write evaluation results to JSON file")
    p.add_argument("--markdown", "--md", dest="markdown", help="Write Markdown summary table to file")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose stdout logs")
    a = p.parse_args(args)

    gumbel = (a.search == "gumbel") if a.search else (not a.puct)
    if a.bot in ("all", "both"):
        selected_bots = sorted(BOTS)
    elif "," in a.bot:
        selected_bots = [b.strip() for b in a.bot.split(",") if b.strip()]
    else:
        selected_bots = [a.bot]

    results = evaluate_checkpoint(
        net_path=a.net,
        bot_names=selected_bots,
        games=a.games,
        sims=a.sims,
        temp=a.temp,
        seed=a.seed,
        max_plies=a.max_plies,
        gumbel=gumbel,
        device=a.device,
        threads=a.threads,
    )

    md_summary = format_markdown_summary(results)

    if not a.quiet:
        for r in results:
            print(
                f"gen {r.get('generation')} vs {r['bot']}: {r['wins']}W {r['draws']}D {r['losses']}L "
                f"of {r['games']}  ->  win rate {r['win_rate']:.3f} (95% CI: [{r['win_rate_ci_95'][0]:.3f}, {r['win_rate_ci_95'][1]:.3f}]) "
                f"Elo delta: {r['elo_delta']:+.1f} (95% CI: [{r['elo_delta_ci_95'][0]:+.1f}, {r['elo_delta_ci_95'][1]:+.1f}])"
            )
            print(
                f"  as player 0 {r['win_rate_as_p0']:.3f} | as player 1 {r['win_rate_as_p1']:.3f} "
                f"| avg length {r['avg_plies']:.1f} plies"
            )
        print("\nMarkdown Summary:\n" + md_summary)

    if a.output:
        out_payload = results[0] if len(results) == 1 else results
        Path(a.output).write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    if a.markdown:
        Path(a.markdown).write_text(md_summary + "\n", encoding="utf-8")

    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    main()
