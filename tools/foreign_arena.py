r"""Match a checkpoint against third-party Quoridor engines.

Every number this project produces otherwise comes from inside its own world: the
promotion gate plays a candidate against its own ancestor, and the baseline bots are
written against the same engine and the same `_score`. Both can agree that a net is
improving while it is only getting better at the shapes this codebase happens to
produce. Foreign engines are the one measurement with no shared assumptions - different
board representation, different search, different author - so they are the check that
catches a systematic blind spot rather than a regression.

Each foreign bot runs as a separate process talking a tiny JSON-lines protocol over
stdin/stdout, with the bridge publishing OUR action encoding (a<81 pawn dest, 81+r*8+c
horizontal wall, 145+r*8+c vertical wall) so the main process never depends on a lens.
Out-of-process because these engines disagree about numpy and torch versions and two of
them call `os.chdir`; in-process they would fight each other and this repo.

Protocol:
  driver -> bridge: {"type":"hello","seed":N}            bridge: {"ok":true,"name":"..."}
  driver -> bridge: {"type":"state","p0":int,"p1":int,
                     "w0":int,"w1":int,"h":[slot..],"v":[slot..],
                     "player":0|1,"ply":int}
                     bridge: {"a":int}  or  {"a":int,"illegal":"note"} or {"forfeit":"note"}
  driver -> bridge: {"type":"bye"}
Everything non-JSON on stdout is ignored (bridges must diagnose via stderr, which is
captured to a per-bot log file).

The engines themselves live outside this repo - see quoridor_ai/paths.py.

Usage:
  python tools/foreign_arena.py --net checkpoints/gen13_best.pt --opponent vader \
      --games 20 --sims 64 --output results/foreign_vader.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

# Running as a script rather than through the package, so the repo has to be importable
# before anything from it can be. paths.py is loaded directly for the same reason.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quoridor_ai.az_arena import _Duel, _search_round, summarise
from quoridor_ai.core.encoding import version_for_planes
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.paths import repo_root
from quoridor_ai.runtime import configure_threads, resolve_device
from quoridor_ai.safe_loader import load_checkpoint

REPO = repo_root()
BRIDGES = Path(__file__).resolve().parent / 'bridges'

OPPONENTS = {
    'vader': [sys.executable, str(BRIDGES / 'vader_bridge.py')],
    'berlioz': [sys.executable, str(BRIDGES / 'berlioz_bridge.py')],
    'marcobt15': [sys.executable, str(BRIDGES / 'marcobt15_bridge.py')],
    'cryer': [sys.executable, str(BRIDGES / 'cryer_bridge.py')],
    'dimi': [sys.executable, str(BRIDGES / 'dimi_bridge.py')],
    'gorisanson': ['node', str(BRIDGES / 'gorisanson_bridge.js')],
    # AlphaZero net (torch, in-process); the two C engines below run their compiled
    # binary as a sub-subprocess, so the bridge itself is still a Python script.
    'sigma': [sys.executable, str(BRIDGES / 'sigma_bridge.py')],
    'pavlosdais': [sys.executable, str(BRIDGES / 'pavlosdais_bridge.py')],
    'giorgos': [sys.executable, str(BRIDGES / 'giorgos_bridge.py')],
}

BOT_REPOS = {
    'vader': 'v-ade-r_QuoridorAI-AlphaZero',
    'berlioz': 'berlioz10_quoridor-monte-carlo',
    'marcobt15': 'marcobt15_Quoridor_Reinforcement_Learning',
    'cryer': 'cryer_AlphaZero_Quoridor',
    'dimi': 'dimitrijekaranfilovic_quoridor',
    'gorisanson': 'gorisanson_quoridor-ai',
    'sigma': 'bartolomeo3000_SigmaQuoridor',
    'pavlosdais': 'pavlosdais_Quoridor',
    'giorgos': 'giorgosnikolaou_Quoridor-Engine',
}
TECHNICAL_OUTCOMES = (
    'honest_game', 'explicit_forfeit', 'illegal_move', 'timeout', 'bridge_error')
EXTERNAL_ASSET_NOTICE = (
    'External bot folders and weights are not stored in Qudor and cannot be restored '
    'automatically; clone or recover them separately and set BOTS_DIR if needed.')

# Sentinel for "this bot is not playing on": a crash, an illegal move, or its own
# surrender. A unique object rather than a magic int, because every int in this module
# is a legal action index somewhere.
FORFEIT = object()

# How long to wait for one move before giving up on the bridge. Their searches range
# from 60 playouts to 60000 simulations, so this is generous; it exists to turn a
# deadlocked subprocess into a forfeit instead of a hung match.
MOVE_TIMEOUT = 300.0
HANDSHAKE_TIMEOUT = 120.0


def _slots(mask: int):
    """Set bit indices of a wall bitboard, low to high."""
    out = []
    while mask:
        b = mask & -mask
        out.append(b.bit_length() - 1)
        mask ^= b
    return out


class BridgeError(RuntimeError):
    """The bridge subprocess cannot be talked to. Always a forfeit, never a crash."""


class BridgeTimeout(BridgeError):
    """The bridge failed to answer before its protocol deadline."""


class ForfeitReason:
    __slots__ = ('message', 'outcome', 'retryable')

    def __init__(self, outcome: str, message: str, retryable: bool = False):
        self.outcome = outcome
        self.message = message
        self.retryable = retryable

    def __str__(self):
        return self.message

    def __contains__(self, text):
        return text in self.message


def _requirement(kind: str, value, ok: bool, detail: str = '') -> dict:
    return {'kind': kind, 'value': str(value), 'ok': bool(ok), 'detail': detail}


def _total_ram_bytes() -> int | None:
    """Physical RAM without adding a runtime dependency such as psutil."""
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('memory_load', ctypes.c_ulong),
                        ('total_phys', ctypes.c_ulonglong),
                        ('avail_phys', ctypes.c_ulonglong),
                        ('total_page_file', ctypes.c_ulonglong),
                        ('avail_page_file', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong),
                        ('avail_virtual', ctypes.c_ulonglong),
                        ('avail_extended_virtual', ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            return status.total_phys if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)) else None
        except (AttributeError, OSError):
            return None
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, OSError, ValueError):
        return None


def _build_tool_requirement(binary_exists: bool, label: str, commands: tuple[str, ...]) -> dict:
    found = None
    for command in commands:
        found = shutil.which(command)
        if found:
            break
    ok = binary_exists or bool(found)
    if found:
        detail = f'available for an external build ({Path(found).resolve()})'
    elif binary_exists:
        detail = 'not required because the platform-compatible engine binary exists'
    else:
        detail = f'{" or ".join(commands)} was not found on PATH; build externally'
    return _requirement('build_tool', found or label, ok, detail)


def preflight(opponents, min_ram_gb: float = 0) -> dict:
    """Check bridge dependencies without importing or starting foreign engines."""
    selected = list(opponents)
    unknown = set(selected) - set(OPPONENTS)
    if unknown:
        raise ValueError(f'unknown opponents: {sorted(unknown)}; known: {sorted(OPPONENTS)}')
    bots = Path(os.environ.get('BOTS_DIR', REPO.parent / 'bots')).resolve()
    total_ram = _total_ram_bytes()
    host_requirements = []
    if min_ram_gb:
        actual = total_ram / 1024 ** 3 if total_ram is not None else None
        host_requirements.append(_requirement(
            'ram', f'{min_ram_gb:g} GiB minimum',
            actual is not None and actual >= min_ram_gb,
            ('physical RAM could not be detected' if actual is None
             else f'{actual:.1f} GiB physical RAM detected')))
    statuses = []
    for name in selected:
        command = list(OPPONENTS[name])
        script = Path(command[-1])
        requirements = [_requirement('bridge_script', script, script.is_file())]
        runtime = command[0]
        resolved_runtime = (str(Path(runtime).resolve()) if Path(runtime).is_file()
                            else shutil.which(runtime))
        requirements.append(_requirement(
            'runtime', resolved_runtime or runtime, bool(resolved_runtime),
            f'{runtime} was not found on PATH' if not resolved_runtime else ''))
        if resolved_runtime:
            command[0] = resolved_runtime

        repo = bots / BOT_REPOS[name]
        requirements.append(_requirement('external_repo', repo, repo.is_dir()))
        if name == 'vader':
            weight = repo / 'temp' / 'best.pth.tar'
            requirements.append(_requirement('weights', weight, weight.is_file()))
        elif name == 'marcobt15':
            weight = repo / os.environ.get(
                'MARCOBT15_MODEL', 'quoridor_aec_v6_runs_and_walls.zip')
            requirements.append(_requirement('weights', weight, weight.is_file()))
        elif name == 'sigma':
            weight = repo / 'runs' / 'models_9x9_heads' / 'best.pt'
            requirements.append(_requirement('weights', weight, weight.is_file()))
        elif name in ('pavlosdais', 'giorgos'):
            base = repo / 'bin' / 'ipquoridor' if name == 'pavlosdais' else repo / 'main'
            candidates = ([Path(f'{base}.exe')] if os.name == 'nt' else [base])
            binary = next((path for path in candidates if path.is_file()), None)
            platform_name = 'Windows PE (.exe)' if os.name == 'nt' else platform.system()
            requirements.append(_requirement(
                'binary', binary or candidates[0], binary is not None,
                (f'{platform_name} binary required; preflight never builds foreign engines'
                 if binary is None else f'platform-compatible {platform_name} binary')))
            make_commands = ('mingw32-make', 'make') if os.name == 'nt' else ('make',)
            requirements.append(_build_tool_requirement(
                binary is not None, ' or '.join(make_commands), make_commands))
            requirements.append(_build_tool_requirement(
                binary is not None, 'gcc or clang', ('gcc', 'clang')))
        missing = [req for req in requirements if not req['ok']]
        statuses.append({'opponent': name, 'command': command, 'ok': not missing,
                         'requirements': requirements, 'missing': missing})
    return {'ok': (all(req['ok'] for req in host_requirements)
                   and all(item['ok'] for item in statuses)),
            'host_requirements': host_requirements, 'opponents': statuses,
            'external_assets': EXTERNAL_ASSET_NOTICE}


def smoke(opponent: str, logdir=None, seed: int = 0) -> dict:
    """Handshake, request one legal opening move, and close a dependency-ready bridge."""
    report = preflight([opponent])
    if not report['ok']:
        return {'opponent': opponent, 'ok': False, 'ran': False,
                'error': 'dependencies failed', 'preflight': report}
    logdir = Path(logdir) if logdir else REPO / 'runs' / 'foreign_logs'
    bridge = Bridge(OPPONENTS[opponent], logdir / f'{opponent}_smoke.log')
    try:
        bridge.start(seed)
        action, note = bridge.move(State())
        if action is FORFEIT:
            raise BridgeError(f'bridge forfeited opening move: {note}')
        if action not in legal_actions(State()):
            raise BridgeError(f'bridge returned illegal opening action {action}')
        return {'opponent': opponent, 'ok': True, 'ran': True, 'action': action}
    except BridgeError as exc:
        return {'opponent': opponent, 'ok': False, 'ran': True, 'error': str(exc)}
    finally:
        bridge.close()


class Bridge:
    """Persistent subprocess speaking the JSON-lines protocol.

    A context manager, because the alternative is what this used to be: a `stderr` file
    handle opened inline in `Popen` and never closed, plus a `terminate()`/`wait()` pair
    that raises `TimeoutExpired` on a wedged process and leaks it. A foreign engine that
    ignores SIGTERM would then outlive the match and keep its log file locked on Windows.
    """

    def __init__(self, argv: list[str], logfile: Path):
        self.argv = list(argv)
        self.log = Path(logfile)
        self.proc = None
        self._err = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return Path(self.argv[-1]).name

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False

    def start(self, seed: int):
        script = Path(self.argv[-1])
        if not script.is_file():
            raise BridgeError(f'bridge script not found: {script}')
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.unlink(missing_ok=True)
        # The bridge's own directory as cwd: two of these engines resolve weights
        # relative to it, and one calls os.chdir on import.
        self._err = self.log.open('a', encoding='utf-8')
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=str(script.resolve().parent),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                text=True, encoding='utf-8', errors='replace')
        except OSError as exc:
            self._close_err()
            raise BridgeError(f'cannot start bridge {self.name}: {exc}') from exc
        try:
            self._hello(seed)
        except BaseException:
            self.close()
            raise

    def _hello(self, seed: int):
        reply = self._ask({'type': 'hello', 'seed': seed}, HANDSHAKE_TIMEOUT)
        if not isinstance(reply, dict) or not reply.get('ok'):
            raise BridgeError(f'bridge {self.name} failed handshake: {reply!r}')

    def move(self, s: State):
        """(action, note) or (FORFEIT, reason). Never raises for a bot's own fault."""
        req = {'type': 'state', 'p0': s.p0, 'p1': s.p1, 'w0': s.walls0, 'w1': s.walls1,
               'h': _slots(s.h), 'v': _slots(s.v), 'player': s.player, 'ply': s.ply}
        reply = self._ask(req, MOVE_TIMEOUT)
        if reply is None:
            raise BridgeError(f'bridge {self.name} closed its stdout')
        if not isinstance(reply, dict):
            raise BridgeError(f'bridge {self.name} sent {type(reply).__name__}, not an object')
        if 'forfeit' in reply:
            return FORFEIT, str(reply['forfeit'])
        if 'a' not in reply:
            raise BridgeError(f'bridge {self.name} sent no action: {reply!r}')
        a = reply['a']
        # bool is an int subclass and True would silently become action 1.
        if isinstance(a, bool) or not isinstance(a, int):
            raise BridgeError(f'bridge {self.name} returned non-int action {a!r}')
        return a, reply.get('illegal')

    def _ask(self, obj, timeout: float):
        """One request/response round trip. None when the bridge closed its stdout."""
        if self.proc is None:
            raise BridgeError(f'bridge {self.name} is not running')
        with self._lock:
            if self.proc.poll() is not None:
                raise BridgeError(
                    f'bridge {self.name} exited with code {self.proc.returncode}; '
                    f'see {self.log}')
            try:
                self.proc.stdin.write(json.dumps(obj) + '\n')
                self.proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise BridgeError(f'bridge {self.name} closed its stdin: {exc}') from exc
            line = self._readline(timeout)
            if not line:
                return None
            try:
                return json.loads(line)
            except ValueError as exc:
                raise BridgeError(
                    f'bridge {self.name} sent non-JSON {line[:120]!r}: {exc}') from exc

    def _readline(self, timeout: float):
        """Read one line, treating a timeout as a dead bridge.

        `readline` on a pipe has no timeout of its own, so it runs on a helper thread.
        The thread is a daemon and the process is killed on timeout, which is what
        lets it end: a blocked read on a killed process returns.
        """
        result = {}

        def read():
            try:
                result['line'] = self.proc.stdout.readline()
            except (OSError, ValueError) as exc:
                result['error'] = exc

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self._kill()
            raise BridgeTimeout(f'bridge {self.name} did not answer within {timeout:.0f}s')
        if 'error' in result:
            raise BridgeError(f'bridge {self.name} read failed: {result["error"]}')
        return result.get('line')

    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _close_err(self):
        if self._err is not None:
            self._err.close()
            self._err = None

    def close(self):
        """Shut the bridge down. Safe to call twice, and never raises."""
        proc = self.proc
        if proc is None:
            self._close_err()
            return
        try:
            if proc.poll() is None:
                try:
                    with self._lock:
                        proc.stdin.write(json.dumps({'type': 'bye'}) + '\n')
                        proc.stdin.flush()
                except (OSError, ValueError):
                    pass                       # already gone; the kill below settles it
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._kill()           # ignores SIGTERM: stop asking
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        finally:
            self.proc = None
            self._close_err()


class ForeignBot:
    """One foreign engine as a callable move function, with forfeits instead of crashes."""

    def __init__(self, argv: list[str], logdir: Path, seed: int):
        self.dumpfile = Path(logdir) / 'illegal_dump.jsonl'
        self.bridge = Bridge(argv, Path(logdir) / f'{Path(argv[-1]).stem}.log')
        self.bridge.start(seed)

    def __call__(self, s: State, rng):
        """(action, note) or (FORFEIT, reason) when the bridge gave up or erred."""
        try:
            a, illegal = self.bridge.move(s)
        except BridgeTimeout as e:
            return FORFEIT, ForfeitReason('timeout', str(e))
        except BridgeError as e:                     # a bot that cannot answer loses
            return FORFEIT, ForfeitReason('bridge_error', str(e))
        except Exception as e:  # noqa: BLE001 - foreign code cannot abort the arena
            return FORFEIT, ForfeitReason(
                'bridge_error', f'{type(e).__name__}: {e}', retryable=True)
        if a is FORFEIT:
            return a, ForfeitReason('explicit_forfeit', str(illegal))
        if a in legal_actions(s):
            if illegal:
                print(f'  WARN bridge {self.bridge.name}: self-reported {illegal}', flush=True)
            return a, None
        # An illegal move is a translation bug in the bridge far more often than a bad
        # engine, so the exact position is recorded to make it reproducible.
        self._dump(s, a, illegal)
        return FORFEIT, ForfeitReason(
            'illegal_move', f'illegal action {a} (illegal={illegal})')

    def _dump(self, s: State, a, illegal):
        record = {'bot': self.bridge.name,
                  'state': {'p0': s.p0, 'p1': s.p1, 'w0': s.walls0, 'w1': s.walls1,
                            'h': _slots(s.h), 'v': _slots(s.v),
                            'player': s.player, 'ply': s.ply},
                  'action': a, 'note': illegal}
        try:
            self.dumpfile.parent.mkdir(parents=True, exist_ok=True)
            with self.dumpfile.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except OSError as exc:
            # Losing the dump must not lose the match: the forfeit is the real result.
            print(f'  WARN could not write {self.dumpfile}: {exc}', flush=True)

    def close(self):
        self.bridge.close()


def _opponent_move(bot, duel, attempts=3):
    """One opponent move, retrying a recoverable glitch.

    Bridges do occasionally return a garbled line under load. A retry is worth it when
    the process is still alive, and pointless once it is gone - retrying a dead pipe
    turns a lost game into three identical failures.
    """
    move = (FORFEIT, 'no attempt made')
    for _ in range(max(1, attempts)):
        move = bot(duel.s, duel.rng)
        if move[0] is not FORFEIT:
            return move
        reason = move[1]
        if isinstance(reason, ForfeitReason) and not reason.retryable:
            break
    return move


def play(net, opponent, device, games=40, sims=64, c_puct=1.6, temp=0.6, max_plies=220,
         seed=0, gumbel=True, gumbel_cap=16, logdir=None, gumbel_value_mixture='adapted',
         on_game_complete: Callable[[dict], None] | None = None):
    """Play `net`, with search, against a foreign engine. Colours alternate per game."""
    if games < 0:
        raise ValueError('games must be non-negative')
    if opponent not in OPPONENTS:
        raise ValueError(f'unknown opponent {opponent!r}; known: {sorted(OPPONENTS)}')
    logdir = Path(logdir) if logdir else REPO / 'runs' / 'foreign_logs'
    duels = [_Duel(i % 2 == 1, seed * 104729 + i) for i in range(games)]
    if not duels:
        # No games means no bridge to start: an empty match must not require the
        # foreign engine to be installed at all.
        return summarise(duels, opponent=opponent, opp_forfeits=0, sims=sims,
                         temperature=temp, gumbel=bool(gumbel), seed=seed,
                         gumbel_value_mixture=gumbel_value_mixture,
                         **{outcome: 0 for outcome in TECHNICAL_OUTCOMES})
    enc = version_for_planes(net.planes)
    logdir.mkdir(parents=True, exist_ok=True)
    bot = ForeignBot(OPPONENTS[opponent], logdir, seed)
    live = list(duels)
    try:
        while live:
            group = [d for d in live if d.mover_is_a()
                     and d.s.winner is None and d.s.ply < max_plies]
            if group:
                _search_round(net, group, device, enc, sims, c_puct, temp, max_plies,
                               gumbel, gumbel_cap, gumbel_value_mixture=gumbel_value_mixture)
            for d in live:
                # Re-checked rather than reusing `group`: a game can end on our move, and
                # a finished board has no legal action for the foreign bot to answer with.
                if not d.mover_is_a() and d.s.winner is None and d.s.ply < max_plies:
                    action, note = _opponent_move(bot, d)
                    if action is FORFEIT:
                        d.result = 1.0                 # the foreign bot abandoned; we win
                        d.forfeit = note
                        d.technical_outcome = (
                            note.outcome if isinstance(note, ForfeitReason)
                            else 'bridge_error')
                    else:
                        d.s = apply_unchecked(d.s, action)
            still = []
            for d in live:
                if d.result is not None:
                    pass                               # decided by forfeit above
                elif d.s.winner is not None:
                    d.result = 1.0 if (d.s.winner == 0) != d.swap else 0.0
                    d.technical_outcome = 'honest_game'
                elif d.s.ply >= max_plies:
                    d.result = 0.5
                    d.technical_outcome = 'honest_game'
                else:
                    still.append(d)
                    continue
                if on_game_complete is not None:
                    outcome = d.technical_outcome
                    on_game_complete(summarise(
                        [d], opponent=opponent,
                        opp_forfeits=int(d.forfeit is not None), sims=sims,
                         temperature=temp, gumbel=bool(gumbel), seed=seed,
                         gumbel_value_mixture=gumbel_value_mixture,
                        **{name: int(name == outcome) for name in TECHNICAL_OUTCOMES}))
            live = still
    finally:
        bot.close()
    forfeits = sum(1 for d in duels if d.forfeit is not None)
    if forfeits:
        # A win rate inflated by forfeits measures the bridge, not the network.
        print(f'  NOTE {forfeits} of {len(duels)} games ended in an opponent forfeit; '
              f'they count as wins. See {logdir}', flush=True)
    counts = {name: sum(getattr(d, 'technical_outcome', None) == name for d in duels)
              for name in TECHNICAL_OUTCOMES}
    return summarise(duels, opponent=opponent, opp_forfeits=forfeits, sims=sims,
                     temperature=temp, gumbel=bool(gumbel), seed=seed,
                     gumbel_value_mixture=gumbel_value_mixture, **counts)


def main():
    p = argparse.ArgumentParser(description='Foreign-bot arena')
    p.add_argument('--net')
    p.add_argument('--opponent', required=True, choices=sorted(OPPONENTS))
    p.add_argument('--games', type=int, default=40)
    p.add_argument('--sims', type=int, default=64)
    p.add_argument('--temp', type=float, default=0.6)
    p.add_argument('--puct', action='store_true', help='PUCT search instead of Gumbel')
    p.add_argument('--gumbel-value-mixture', choices=('adapted', 'canonical'), default='adapted')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-plies', type=int, default=220)
    p.add_argument('--threads', type=int, help='intra-op threads; defaults to every core')
    p.add_argument('--logdir', help='where bridge stderr and illegal-move dumps go')
    p.add_argument('--device', help="'cpu' or 'cuda'; overrides autodetection")
    p.add_argument('--output', default='', help='write the result dict here as JSON')
    p.add_argument('--dry-run', action='store_true', help='list requirements and exit')
    p.add_argument('--smoke', action='store_true',
                   help='handshake, request one legal move, close, and exit')
    a = p.parse_args()
    if a.games <= 0 and not (a.dry_run or a.smoke):
        p.error('--games must be positive')

    report = preflight([a.opponent])
    if a.dry_run:
        print(json.dumps(report, indent=2))
        return 0 if report['ok'] else 1
    if not report['ok']:
        for req in report['opponents'][0]['missing']:
            detail = f" ({req['detail']})" if req['detail'] else ''
            print(f"missing {req['kind']}: {req['value']}{detail}", file=sys.stderr)
        p.error(f'foreign bridge preflight failed for {a.opponent}')
    if a.smoke:
        result = smoke(a.opponent, a.logdir, a.seed)
        print(json.dumps(result, indent=2))
        return 0 if result['ok'] else 1
    if not a.net:
        p.error('--net is required unless --dry-run or --smoke is used')

    dev = resolve_device(a.device)
    configure_threads(a.threads)
    ck = load_checkpoint(a.net, map_location=dev)
    net = net_from_checkpoint(ck, dev)
    r = play(net, a.opponent, dev, a.games, a.sims, temp=a.temp, seed=a.seed,
             max_plies=a.max_plies, gumbel=not a.puct, logdir=a.logdir,
             gumbel_value_mixture=a.gumbel_value_mixture)
    r.update(net=str(a.net), generation=ck.get('generation'),
             iteration=ck.get('iteration'), device=str(dev), net_name=Path(a.net).name)
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2) + '\n', encoding='utf-8')
    print(f"gen {r['generation']} vs {a.opponent}: {r['wins']}W {r['draws']}D {r['losses']}L "
          f"of {r['games']} (opp forfeits {r['opp_forfeits']}) -> win rate {r['win_rate']:.3f}")
    print(f"  as player 0 {r['win_rate_as_p0']:.3f} | as player 1 {r['win_rate_as_p1']:.3f} "
          f"| avg length {r['avg_plies']:.1f} plies")
    return r


if __name__ == '__main__':
    main()
