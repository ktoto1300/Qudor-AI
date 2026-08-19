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
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import tools.foreign_arena as fa
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.runtime import configure_threads, resolve_device
from quoridor_ai.safe_loader import load_checkpoint

# Per-move search budget for the slower engines, so one bot cannot eat the whole night on
# a single position. gorisanson's own default is 60000 playouts (minutes per move); this
# keeps it to seconds without changing the bridge.
BUDGETS = {'gorisanson': '2000'}
KNOWN_BROKEN = {
    'cryer': 'bridge repair cannot be verified without the external bot repository',
    'pavlosdais': 'native bridge repair cannot be verified without bot sources/binary',
    'vader': 'known move-one bridge forfeit; external repository/weights required to repair',
}
EXCLUDED_NAMES = {'rusher', 'greedy'}
PROFILES = {
    'cpu-laptop': {'device': 'cpu', 'threads': 6, 'sims': 16, 'batch': 1,
                   'min_ram_gb': 6, 'seconds_per_game': 240,
                   'host': 'Intel Core i5-1155G7 laptop'},
    'cpu-smoke': {'device': 'cpu', 'threads': 2, 'sims': 4, 'batch': 1,
                  'min_ram_gb': 4, 'seconds_per_game': 60, 'host': 'CPU smoke'},
    'cuda-8gb': {'device': 'cuda', 'threads': 4, 'sims': 64, 'batch': 4,
                 'min_ram_gb': 8, 'seconds_per_game': 90, 'host': '8 GB CUDA GPU'},
    'cuda-16gb': {'device': 'cuda', 'threads': 6, 'sims': 128, 'batch': 8,
                  'min_ram_gb': 16, 'seconds_per_game': 60, 'host': '16 GB CUDA GPU'},
}


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
            'plies_sum': 0.0, 'win_rate': 0.0, 'avg_plies': 0.0, 'rounds': [],
            'technical_streak': 0,
            **{outcome: 0 for outcome in fa.TECHNICAL_OUTCOMES}}


def _normalise(acc: dict, opponent: str, net_name: str, generation) -> dict:
    """Fill fields absent from older overnight JSON without discarding its totals."""
    clean = _blank(opponent, net_name, generation)
    clean.update(acc)
    clean['rounds'] = list(clean.get('rounds') or [])
    clean['plies_sum'] = float(
        acc['plies_sum'] if 'plies_sum' in acc
        else clean.get('avg_plies', 0.0) * clean.get('games', 0))
    for outcome in fa.TECHNICAL_OUTCOMES:
        clean[outcome] = int(clean.get(outcome, 0))
    clean['technical_streak'] = int(clean.get('technical_streak', 0))
    classified = sum(clean[outcome] for outcome in fa.TECHNICAL_OUTCOMES)
    missing = max(0, int(clean.get('games', 0)) - classified)
    if missing:
        old_forfeits = min(missing, int(clean.get('opp_forfeits', 0)))
        clean['bridge_error'] += old_forfeits
        clean['honest_game'] += missing - old_forfeits
    return clean


def _merge(acc: dict, r: dict, seconds: float) -> dict:
    """Fold one batch result into the cumulative record for an opponent."""
    acc['games'] += r['games']
    acc['wins'] += r['wins']
    acc['draws'] += r['draws']
    acc['losses'] += r['losses']
    acc['opp_forfeits'] += r['opp_forfeits']
    acc['plies_sum'] += r['avg_plies'] * r['games']
    for outcome in fa.TECHNICAL_OUTCOMES:
        acc[outcome] = acc.get(outcome, 0) + r.get(outcome, 0)
    if r.get('honest_game', 0):
        acc['technical_streak'] = 0
    elif r.get('games', 0):
        acc['technical_streak'] = acc.get('technical_streak', 0) + r['games']
    acc['win_rate'] = acc['wins'] / acc['games'] if acc['games'] else 0.0
    acc['avg_plies'] = acc['plies_sum'] / acc['games'] if acc['games'] else 0.0
    acc['rounds'].append({'at': _now(), 'seed': r['seed'], 'games': r['games'],
                           'wins': r['wins'], 'draws': r['draws'], 'losses': r['losses'],
                           'opp_forfeits': r['opp_forfeits'],
                           **{outcome: r.get(outcome, 0)
                              for outcome in fa.TECHNICAL_OUTCOMES},
                           'seconds': round(seconds, 1)})
    acc['updated'] = _now()
    return acc


def _wilson95(successes: float, games: int) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def _summary(records: dict[str, dict], net_name: str, generation, deadline_at: str,
             excluded: dict[str, str] | None = None) -> dict:
    board = []
    for opp, acc in sorted(records.items()):
        honest_games = acc.get('honest_game', acc['games'] - acc['opp_forfeits'])
        honest_wins = max(0, acc['wins'] - acc['opp_forfeits'])
        honest_score = honest_wins + 0.5 * acc['draws']
        honest_rate = honest_score / honest_games if honest_games else 0.0
        board.append({
            'opponent': opp, 'games': acc['games'],
            'wins': acc['wins'], 'draws': acc['draws'], 'losses': acc['losses'],
            'opp_forfeits': acc['opp_forfeits'],
            **{outcome: acc.get(outcome, 0) for outcome in fa.TECHNICAL_OUTCOMES},
            'win_rate': round(acc.get('win_rate', 0.0), 4),
            'honest_games': honest_games,
            'honest_win_rate': round(honest_rate, 4),
            'honest_win_rate_wilson95': _wilson95(honest_score, honest_games),
            'avg_plies': round(acc.get('avg_plies', 0.0), 1)})
    board.sort(key=lambda b: (-b['honest_win_rate'], -b['honest_games']))
    total_games = sum(b['games'] for b in board)
    return {'net': net_name, 'generation': generation, 'updated': _now(),
            'deadline': deadline_at, 'total_games': total_games,
            'excluded': excluded or {}, 'leaderboard': board}


def _print_preflight(report: dict, plan: dict | None = None) -> None:
    if plan:
        print(f"Profile: {plan['profile']} ({plan['host']}) | device={plan['device']} "
              f"threads={plan['threads']} sims={plan['sims']} batch={plan['batch']}")
        print(f"Planned games: ~{plan['planned_games']} in ~{plan['rough_hours']:.2f}h "
              f"({plan['seconds_per_game']}s/game estimate, serial={plan['serial']})")
        for opponent, reason in plan.get('excluded', {}).items():
            print(f'Excluded {opponent}: {reason}')
    for req in report.get('host_requirements', []):
        print(f"Host {req['kind']}: {'OK' if req['ok'] else 'MISSING'} - "
              f"{req['value']} ({req['detail']})")
    print('Selected opponents: ' + ', '.join(
        item['opponent'] for item in report['opponents']))
    for item in report['opponents']:
        print(f"  {item['opponent']}: {' '.join(item['command'])}")
        for req in item['requirements']:
            detail = f" ({req['detail']})" if req['detail'] else ''
            print(f"    {'OK' if req['ok'] else 'MISSING'} {req['kind']}: "
                  f"{req['value']}{detail}")
    print(report.get('external_assets', fa.EXTERNAL_ASSET_NOTICE))


def _add_run_requirements(report: dict, net: str, device: str) -> None:
    checkpoint = Path(net)
    if not checkpoint.is_absolute():
        checkpoint = _REPO / checkpoint
    requirements = report.setdefault('host_requirements', [])
    requirements.append(fa._requirement('checkpoint', checkpoint, checkpoint.is_file()))
    if str(device).startswith('cuda'):
        requirements.append(fa._requirement(
            'cuda', device, torch.cuda.is_available(),
            ('CUDA is available' if torch.cuda.is_available()
             else 'PyTorch cannot access a CUDA device')))
    report['ok'] = report['ok'] and all(req['ok'] for req in requirements)


def _plan(profile_name: str, hours: float, device, threads, sims, batch) -> dict:
    profile = PROFILES[profile_name]
    effective = {'profile': profile_name,
                 'host': profile['host'],
                 'device': device or profile['device'],
                 'threads': threads if threads is not None else profile['threads'],
                 'sims': sims if sims is not None else profile['sims'],
                 'batch': batch if batch is not None else profile['batch'],
                 'seconds_per_game': profile['seconds_per_game']}
    effective['planned_games'] = max(0, int(hours * 3600 / profile['seconds_per_game']))
    effective['rough_hours'] = (effective['planned_games'] * profile['seconds_per_game']
                                / 3600)
    effective['serial'] = effective['batch'] == 1
    return effective


def main() -> int:
    p = argparse.ArgumentParser(description='Overnight arena against foreign engines')
    p.add_argument('--net', required=True)
    p.add_argument('--hours', type=float, default=8.0, help='wall-clock budget')
    p.add_argument('--batch', type=int, help='games per opponent per round (default: 10)')
    p.add_argument('--sims', type=int, help='our search simulations per move (default: 48)')
    p.add_argument('--temp', type=float, default=0.5)
    p.add_argument('--outdir', default='results/overnight')
    p.add_argument('--include-known-broken', action='store_true',
                   help='override the known-broken exclusion')
    p.add_argument('--include-vader', action='store_true', dest='include_known_broken',
                   help=argparse.SUPPRESS)
    p.add_argument('--only', help='comma-separated subset of opponents to play')
    p.add_argument('--device')
    p.add_argument('--threads', type=int)
    p.add_argument('--profile', choices=tuple(PROFILES), default='cpu-laptop')
    p.add_argument('--gumbel-value-mixture', choices=('adapted', 'canonical'), default='adapted')
    p.add_argument('--preset', choices=('laptop',), help=argparse.SUPPRESS)
    p.add_argument('--technical-failure-threshold', type=int, default=3,
                   help='exclude an engine after this many consecutive technical games')
    p.add_argument('--dry-run', action='store_true',
                   help='check selected bridges without loading the checkpoint')
    a = p.parse_args()

    if a.preset == 'laptop':
        a.profile = 'cpu-laptop'
    plan = _plan(a.profile, a.hours, a.device, a.threads, a.sims, a.batch)
    a.device, a.threads = plan['device'], plan['threads']
    a.sims, a.batch = plan['sims'], plan['batch']
    if a.batch <= 0:
        p.error('--batch must be positive')
    if a.sims <= 0:
        p.error('--sims must be positive')
    if a.hours < 0:
        p.error('--hours must be non-negative')
    if a.technical_failure_threshold <= 0:
        p.error('--technical-failure-threshold must be positive')

    if EXCLUDED_NAMES & set(fa.OPPONENTS):
        p.error('rusher and greedy must not be registered as foreign opponents')
    excluded = {} if a.include_known_broken else dict(KNOWN_BROKEN)
    plan['excluded'] = excluded
    opponents = [o for o in fa.OPPONENTS if o not in excluded]
    if a.only:
        wanted = {o.strip() for o in a.only.split(',') if o.strip()}
        unknown = wanted - set(fa.OPPONENTS)
        if unknown:
            p.error(f'unknown opponents: {sorted(unknown)}; known: {sorted(fa.OPPONENTS)}')
        opponents = [o for o in opponents if o in wanted]
    if not opponents:
        p.error('no opponents selected (known-broken engines require '
                '--include-known-broken)')

    report = fa.preflight(opponents, min_ram_gb=PROFILES[a.profile]['min_ram_gb'])
    _add_run_requirements(report, a.net, a.device)
    if a.dry_run:
        _print_preflight(report, plan)
        return 0 if report['ok'] else 1
    failed_host = [req for req in report.get('host_requirements', []) if not req['ok']]
    if failed_host:
        _print_preflight(report, plan)
        p.error('host preflight failed')
    for item in report['opponents']:
        if not item['ok']:
            missing = ', '.join(req['kind'] for req in item['missing'])
            excluded[item['opponent']] = f'dependency preflight failed: {missing}'
    opponents = [opponent for opponent in opponents if opponent not in excluded]
    if not opponents:
        _print_preflight(report, plan)
        p.error('no dependency-ready opponents remain')

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
        if prior and (prior.get('net') != net_name or prior.get('generation') != generation):
            p.error(f'{opp}.json belongs to net={prior.get("net")} '
                    f'generation={prior.get("generation")}; use a different --outdir')
        records[opp] = (_normalise(prior, opp, net_name, generation) if prior
                        else _blank(opp, net_name, generation))

    deadline = time.monotonic() + a.hours * 3600.0
    deadline_at = _now()
    print(f'[{_now()}] net={net_name} gen={generation} dev={dev} '
           f'opponents={opponents} batch={a.batch} sims={a.sims} '
           f'budget={a.hours:.1f}h planned_games~{plan["planned_games"]} '
           f'rough_duration={plan["rough_hours"]:.2f}h', flush=True)

    # The per-opponent seed advances by the games already recorded, so a resumed run does
    # not replay the same games it played before the restart.
    rnd = 0
    while time.monotonic() < deadline:
        rnd += 1
        for opp in opponents:
            if time.monotonic() >= deadline:
                break
            if records[opp].get('technical_streak', 0) >= a.technical_failure_threshold:
                excluded[opp] = (f'technical failure streak '
                                 f'{records[opp]["technical_streak"]} reached threshold '
                                 f'{a.technical_failure_threshold}')
                _atomic_write(outdir / 'summary.json',
                              _summary(records, net_name, generation, deadline_at, excluded))
                continue
            if opp in BUDGETS:
                os.environ['GORISANSON_SIMS'] = BUDGETS[opp]
            seed = 100000 + records[opp]['games']
            t0 = time.monotonic()
            persisted_at = t0

            def persist_game(game_result, *, opponent=opp):
                nonlocal persisted_at
                now = time.monotonic()
                _merge(records[opponent], game_result, now - persisted_at)
                persisted_at = now
                _atomic_write(outdir / f'{opponent}.json', records[opponent])
                _atomic_write(outdir / 'summary.json',
                              _summary(records, net_name, generation, deadline_at, excluded))

            try:
                r = fa.play(net, opp, dev, games=a.batch, sims=a.sims, temp=a.temp,
                            seed=seed, gumbel=True, logdir=logdir,
                            gumbel_value_mixture=a.gumbel_value_mixture,
                            on_game_complete=persist_game)
            except Exception as e:  # noqa: BLE001 - one foreign bot must not end the night
                records[opp]['technical_streak'] += 1
                records[opp]['updated'] = _now()
                _atomic_write(outdir / f'{opp}.json', records[opp])
                if records[opp]['technical_streak'] >= a.technical_failure_threshold:
                    excluded[opp] = (f'runtime failure streak '
                                     f'{records[opp]["technical_streak"]} reached threshold '
                                     f'{a.technical_failure_threshold}')
                _atomic_write(outdir / 'summary.json',
                              _summary(records, net_name, generation, deadline_at, excluded))
                print(f'[{_now()}] {opp}: batch failed: {type(e).__name__}: {e}',
                      flush=True)
                continue
            dt = time.monotonic() - t0
            acc = records[opp]
            print(f'[{_now()}] r{rnd} {opp}: +{r["wins"]}W/{r["draws"]}D/{r["losses"]}L '
                  f'(ff {r["opp_forfeits"]}) in {dt:.0f}s | total {acc["wins"]}/'
                   f'{acc["games"]} = {acc["win_rate"]:.3f}', flush=True)

        if all(opp in excluded for opp in opponents):
            print(f'[{_now()}] all opponents excluded; stopping', flush=True)
            break

    print(f'[{_now()}] done. summary -> {outdir / "summary.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
