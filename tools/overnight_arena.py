r"""Overnight foreign-engine tournament for one checkpoint.

Plays our net against every third-party engine that actually moves, in rounds, until a
wall-clock deadline. State is written after every batch so a crash, a reboot, or Ctrl-C
never loses more than the batch in flight: on restart it reads the same result files and
keeps adding to the totals.

    python tools/overnight_arena.py --net runs/cloud_metrics/gen37_best.pt --hours 8

Each opponent gets its own results/overnight/<opponent>.json with cumulative W/D/L and a
per-round history; results/overnight/summary.json is the leaderboard across all of them.
vader is excluded by default because its bridge forfeits on move one - counting that as a
win measures nothing. Pass --include-vader to keep it anyway.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from quoridor_ai.model import net_from_checkpoint          # noqa: E402
from quoridor_ai.runtime import configure_threads, resolve_device  # noqa: E402
from quoridor_ai.safe_loader import load_checkpoint        # noqa: E402

import tools.foreign_arena as fa                           # noqa: E402

# Per-move search budget for the slower engines, so one bot cannot eat the whole night on
# a single position. gorisanson's own default is 60000 playouts (minutes per move); this
# keeps it to seconds without changing the bridge.
BUDGETS = {'gorisanson': '2000'}


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _blank(opponent: str, net_name: str, generation) -> dict:
    # win_rate/avg_plies are seeded here, not only in _merge: _summary reads them for every
    # opponent each round, including ones whose first batch has not run yet. Without the
    # seed the very first summary raises KeyError on the still-blank opponents.
    return {'opponent': opponent, 'net': net_name, 'generation': generation,
            'games': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'opp_forfeits': 0,
            'plies_sum': 0.0, 'win_rate': 0.0, 'avg_plies': 0.0, 'rounds': []}


def _merge(acc: dict, r: dict, seconds: float) -> dict:
    """Fold one batch result into the cumulative record for an opponent."""
    acc['games'] += r['games']
    acc['wins'] += r['wins']
    acc['draws'] += r['draws']
    acc['losses'] += r['losses']
    acc['opp_forfeits'] += r['opp_forfeits']
    acc['plies_sum'] += r['avg_plies'] * r['games']
    acc['win_rate'] = acc['wins'] / acc['games'] if acc['games'] else 0.0
    acc['avg_plies'] = acc['plies_sum'] / acc['games'] if acc['games'] else 0.0
    acc['rounds'].append({'at': _now(), 'seed': r['seed'], 'games': r['games'],
                          'wins': r['wins'], 'draws': r['draws'], 'losses': r['losses'],
                          'opp_forfeits': r['opp_forfeits'], 'seconds': round(seconds, 1)})
    acc['updated'] = _now()
    return acc


def _summary(records: dict[str, dict], net_name: str, generation, deadline_at: str) -> dict:
    board = []
    for opp, acc in sorted(records.items()):
        clean = acc['games'] - acc['opp_forfeits']
        board.append({
            'opponent': opp, 'games': acc['games'],
            'wins': acc['wins'], 'draws': acc['draws'], 'losses': acc['losses'],
            'opp_forfeits': acc['opp_forfeits'],
            'win_rate': round(acc.get('win_rate', 0.0), 4),
            # A win only earned over the board, ignoring forfeits, is the honest number.
            'decisive_games': clean,
            'avg_plies': round(acc.get('avg_plies', 0.0), 1)})
    board.sort(key=lambda b: (-b['win_rate'], -b['games']))
    total_games = sum(b['games'] for b in board)
    return {'net': net_name, 'generation': generation, 'updated': _now(),
            'deadline': deadline_at, 'total_games': total_games, 'leaderboard': board}


def main() -> int:
    p = argparse.ArgumentParser(description='Overnight arena against foreign engines')
    p.add_argument('--net', required=True)
    p.add_argument('--hours', type=float, default=8.0, help='wall-clock budget')
    p.add_argument('--batch', type=int, default=10, help='games per opponent per round')
    p.add_argument('--sims', type=int, default=48, help='our search simulations per move')
    p.add_argument('--temp', type=float, default=0.5)
    p.add_argument('--outdir', default='results/overnight')
    p.add_argument('--include-vader', action='store_true',
                   help='keep vader even though its bridge forfeits on move one')
    p.add_argument('--only', help='comma-separated subset of opponents to play')
    p.add_argument('--device')
    p.add_argument('--threads', type=int)
    a = p.parse_args()

    opponents = [o for o in fa.OPPONENTS if a.include_vader or o != 'vader']
    if a.only:
        wanted = {o.strip() for o in a.only.split(',') if o.strip()}
        unknown = wanted - set(fa.OPPONENTS)
        if unknown:
            p.error(f'unknown opponents: {sorted(unknown)}; known: {sorted(fa.OPPONENTS)}')
        opponents = [o for o in opponents if o in wanted]

    dev = resolve_device(a.device)
    configure_threads(a.threads)
    ck = load_checkpoint(a.net, map_location=dev)
    net = net_from_checkpoint(ck, dev)
    generation = ck.get('generation')
    net_name = Path(a.net).name

    outdir = (_REPO / a.outdir) if not Path(a.outdir).is_absolute() else Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = outdir / 'bridge_logs'

    # Resume: reuse whatever totals already exist so a restart adds to the night's work.
    records: dict[str, dict] = {}
    for opp in opponents:
        prior = _load(outdir / f'{opp}.json')
        records[opp] = prior if prior and prior.get('rounds') is not None \
            else _blank(opp, net_name, generation)

    deadline = time.monotonic() + a.hours * 3600.0
    deadline_at = _now()
    print(f'[{_now()}] net={net_name} gen={generation} dev={dev} '
          f'opponents={opponents} batch={a.batch} sims={a.sims} '
          f'budget={a.hours:.1f}h', flush=True)

    # The per-opponent seed advances by the games already recorded, so a resumed run does
    # not replay the same games it played before the restart.
    rnd = 0
    while time.monotonic() < deadline:
        rnd += 1
        for opp in opponents:
            if time.monotonic() >= deadline:
                break
            if opp in BUDGETS:
                import os
                os.environ['GORISANSON_SIMS'] = BUDGETS[opp]
            seed = 100000 + records[opp]['games']
            t0 = time.monotonic()
            try:
                r = fa.play(net, opp, dev, games=a.batch, sims=a.sims, temp=a.temp,
                            seed=seed, gumbel=True, logdir=logdir)
            except Exception as e:               # a bot exploding must not end the night
                print(f'[{_now()}] {opp}: batch failed: {type(e).__name__}: {e}',
                      flush=True)
                continue
            dt = time.monotonic() - t0
            _merge(records[opp], r, dt)
            _atomic_write(outdir / f'{opp}.json', records[opp])
            _atomic_write(outdir / 'summary.json',
                          _summary(records, net_name, generation, deadline_at))
            acc = records[opp]
            print(f'[{_now()}] r{rnd} {opp}: +{r["wins"]}W/{r["draws"]}D/{r["losses"]}L '
                  f'(ff {r["opp_forfeits"]}) in {dt:.0f}s | total {acc["wins"]}/'
                  f'{acc["games"]} = {acc["win_rate"]:.3f}', flush=True)

    print(f'[{_now()}] done. summary -> {outdir / "summary.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
