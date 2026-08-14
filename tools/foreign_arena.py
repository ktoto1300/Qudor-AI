r"""Match a checkpoint against foreign bots downloaded into C:\Users\kamil\Desktop\bots.

Each foreign bot runs as a separate process talking a tiny JSON-lines protocol over
stdin/stdout, with the bridge publishing OUR action encoding (a<81 pawn dest, 81+r*8+c
horizontal wall, 145+r*8+c vertical wall) so the main process never depends on a lens.

Protocol:
  driver -> bridge: {"type":"hello","seed":N}            bridge: {"ok":true,"name":"..."}
  driver -> bridge: {"type":"state","p0":int,"p1":int,
                     "w0":int,"w1":int,"h":[slot..],"v":[slot..],
                     "player":0|1,"ply":int}
                     bridge: {"a":int}  or  {"a":int,"illegal":"note"} or {"forfeit":"note"}
  driver -> bridge: {"type":"bye"}
Everything non-JSON on stdout is ignored (must be diagnosed via stderr).

Usage:
  python tools/foreign_arena.py --net checkpoints/official_rules_best.pt --opponent vader \
      --games 20 --sims 64 --output results/foreign_vader.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(os.environ.get('QUDOR_REPO', r'C:\Users\kamil\Desktop\Qudor'))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quoridor_ai.az_arena import _Duel, _search_round
from quoridor_ai.core.encoding import version_for_planes
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.runtime import configure_threads, resolve_device
from quoridor_ai.safe_loader import load_checkpoint

BOTS = Path(os.environ.get('BOTS_DIR', r'C:\Users\kamil\Desktop\bots'))
BRIDGES = REPO / 'tools' / 'bridges'

OPPONENTS = {
    'vader': [sys.executable, str(BRIDGES / 'vader_bridge.py')],
    'berlioz': [sys.executable, str(BRIDGES / 'berlioz_bridge.py')],
    'marcobt15': [sys.executable, str(BRIDGES / 'marcobt15_bridge.py')],
    'cryer': [sys.executable, str(BRIDGES / 'cryer_bridge.py')],
    'dimi': [sys.executable, str(BRIDGES / 'dimi_bridge.py')],
    'gorisanson': ['node', str(BRIDGES / 'gorisanson_bridge.js')],
}

FORFEIT = {}


def _slots(mask: int):
    out = []
    while mask:
        b = mask & -mask
        out.append(b.bit_length() - 1)
        mask ^= b
    return out


class Bridge:
    """Persistent subprocess speaking the JSON-lines protocol."""

    def __init__(self, argv: list[str], logfile: Path):
        self.argv = argv
        self.log = logfile
        self.proc = None
        self._lock = threading.Lock()

    def start(self, seed: int):
        self.log.unlink(missing_ok=True)
        self.proc = subprocess.Popen(
            self.argv, cwd=str(Path(self.argv[-1]).resolve().parent),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.log, 'a', encoding='utf-8'),
            text=True, encoding='utf-8', errors='replace')
        self._hello(seed)

    def _hello(self, seed: int):
        reply = self._ask({'type': 'hello', 'seed': seed})
        if not reply or not reply.get('ok'):
            raise RuntimeError(f'bridge {self.argv[0]} failed handshake: {reply}')

    def move(self, s: State):
        req = {'type': 'state', 'p0': s.p0, 'p1': s.p1, 'w0': s.walls0, 'w1': s.walls1,
               'h': _slots(s.h), 'v': _slots(s.v), 'player': s.player, 'ply': s.ply}
        reply = self._ask(req)
        if reply is None:
            raise RuntimeError('bridge closed stdin')
        if 'forfeit' in reply:
            return FORFEIT, reply['forfeit']
        a = reply['a']
        if isinstance(a, bool) or not isinstance(a, int):
            raise RuntimeError(f'bridge returned non-int action {a!r}')
        return a, reply.get('illegal')

    def _ask(self, obj):
        with self._lock:
            self.proc.stdin.write(json.dumps(obj) + '\n')
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                return None
            return json.loads(line)

    def close(self):
        if self.proc is None:
            return
        try:
            with self._lock:
                self.proc.stdin.write(json.dumps({'type': 'bye'}) + '\n')
                self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=5)


class ForeignBot:
    def __init__(self, argv: list[str], logdir: Path, seed: int):
        self.bridge = Bridge(argv, logdir / f'{Path(argv[-1]).stem}.log')
        self.bridge.start(seed)

    def __call__(self, s: State, rng):
        """Returns an int action, or FORFEIT[reason] when the bridge gave up/erred."""
        try:
            a, illegal = self.bridge.move(s)
        except Exception as e:                      # crashed bridge: bot loses
            return FORFEIT, str(e)
        if a is FORFEIT:
            return a, illegal
        if a in legal_actions(s):
            if illegal:
                print(f'  WARN bridge {Path(self.bridge.argv[-1]).name}: self-reported {illegal}', flush=True)
            return a, None
        dump = REPO / 'runs' / 'foreign_logs' / 'illegal_dump.jsonl'
        with dump.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'bot': Path(self.bridge.argv[-1]).name,
                                'state': {'p0': s.p0, 'p1': s.p1, 'w0': s.walls0, 'w1': s.walls1,
                                          'h': _slots(s.h), 'v': _slots(s.v),
                                          'player': s.player, 'ply': s.ply},
                                'action': a, 'note': illegal}) + '\n')
        return FORFEIT, f'illegal action {a} (illegal={illegal})'

    def close(self):
        self.bridge.close()


def play(net, opponent, device, games=40, sims=64, c_puct=1.6, temp=0.6, max_plies=220,
         seed=0, gumbel=True, gumbel_cap=16, logdir=REPO / 'runs' / 'foreign_logs'):
    enc = version_for_planes(net.planes)
    logdir.mkdir(parents=True, exist_ok=True)
    bot = ForeignBot(OPPONENTS[opponent], logdir, seed)
    duels = [_Duel(i % 2 == 1, seed * 104729 + i) for i in range(games)]
    live = list(duels)
    try:
        while live:
            group = [d for d in live if d.mover_is_a() and d.s.winner is None and d.s.ply < max_plies]
            if group:
                _search_round(net, group, device, enc, sims, c_puct, temp, max_plies, gumbel, gumbel_cap)
            for d in live:
                if not d.mover_is_a() and d.s.winner is None and d.s.ply < max_plies:
                    m = _opponent_move(bot, d)
                    if m[0] is FORFEIT:
                        d.result = 1.0                     # foreign bot forfeits; we win
                        d.forfeit = m[1]
                    else:
                        d.s = apply_unchecked(d.s, m[0])
            still = []
            for d in live:
                if d.result is not None or d.s.winner is not None:
                    if d.result is None:
                        d.result = 1.0 if (d.s.winner == 0) != d.swap else 0.0
                elif d.s.ply >= max_plies:
                    d.result = 0.5
                else:
                    still.append(d)
            live = still
    finally:
        bot.close()
    scores = [d.result for d in duels]
    first = [d.result for d in duels if not d.swap]
    second = [d.result for d in duels if d.swap]
    wr = sum(scores) / max(1, len(scores))
    elo = 400 * math.log10(max(1e-4, wr) / max(1e-4, 1 - wr))
    forfeits = sum(1 for d in duels if getattr(d, 'forfeit', None) is not None)
    return {'games': len(scores), 'wins': sum(1 for x in scores if x == 1),
            'draws': sum(1 for x in scores if x == 0.5),
            'losses': sum(1 for x in scores if x == 0),
            'opp_forfeits': forfeits,
            'win_rate': wr, 'elo_delta': elo,
            'win_rate_as_p0': sum(first) / max(1, len(first)),
            'win_rate_as_p1': sum(second) / max(1, len(second)),
            'avg_plies': sum(d.s.ply for d in duels) / max(1, len(duels)),
            'sims': sims, 'temperature': temp, 'gumbel': bool(gumbel), 'seed': seed}


def _opponent_move(bot, d):
    for _ in range(3):                     # bridges can glitch; retry a few times
        m = bot(d.s, d.rng)
        if m[0] is not FORFEIT:
            return m
        reason = m[1]
        if 'crashed' in str(reason).lower() or 'closed' in str(reason).lower():
            break
    return m


def main():
    p = argparse.ArgumentParser(description='Foreign-bot arena')
    p.add_argument('--net', required=True)
    p.add_argument('--opponent', required=True, choices=sorted(OPPONENTS))
    p.add_argument('--games', type=int, default=40)
    p.add_argument('--sims', type=int, default=64)
    p.add_argument('--temp', type=float, default=0.6)
    p.add_argument('--gumbel', action='store_true', default=True)
    p.add_argument('--puct', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-plies', type=int, default=220)
    p.add_argument('--threads', type=int)
    p.add_argument('--device')
    p.add_argument('--output', default='')
    a = p.parse_args()

    dev = resolve_device(a.device)
    configure_threads(a.threads)
    ck = load_checkpoint(a.net, map_location=dev)
    net = net_from_checkpoint(ck, dev)
    r = play(net, a.opponent, dev, a.games, a.sims, temp=a.temp, seed=a.seed,
             max_plies=a.max_plies, gumbel=not a.puct)
    r.update(opponent=a.opponent, net=str(a.net), generation=ck.get('generation'),
             iteration=ck.get('iteration'), device=str(dev), net_name=Path(a.net).name)
    if a.output:
        Path(a.output).write_text(json.dumps(r, indent=2))
    print(f"gen {r['generation']} vs {a.opponent}: {r['wins']}W {r['draws']}D {r['losses']}L "
          f"of {r['games']} (opp forfeits {r['opp_forfeits']}) -> win rate {r['win_rate']:.3f}")
    print(f"  as player 0 {r['win_rate_as_p0']:.3f} | as player 1 {r['win_rate_as_p1']:.3f} "
          f"| avg length {r['avg_plies']:.1f} plies")
    return r


if __name__ == '__main__':
    main()
