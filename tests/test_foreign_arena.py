import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quoridor_ai.core.engine import State, legal_actions

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'foreign_arena', ROOT / 'tools' / 'foreign_arena.py')
foreign_arena = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(foreign_arena)

BRIDGE_SCRIPT = r'''
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if msg['type'] == 'hello':
        print(json.dumps({'ok': True, 'name': 'probe'}), flush=True)
    elif msg['type'] == 'state':
        print(json.dumps({'a': 42}), flush=True)
    elif msg['type'] == 'bye':
        break
'''


def _write_bridge(tmp_path):
    script = tmp_path / 'probe_bridge.py'
    script.write_text(BRIDGE_SCRIPT, encoding='utf-8')
    return script


def test_every_foreign_opponent_has_an_existing_bridge():
    for argv in foreign_arena.OPPONENTS.values():
        assert Path(argv[-1]).is_file()


def test_bridge_uses_the_bridge_directory_as_cwd(monkeypatch, tmp_path):
    script = tmp_path / 'bridge.py'
    script.write_text('')
    seen = {}

    class Process:
        stdin = stdout = None

        @staticmethod
        def poll():
            return 0                      # already dead: close() skips the kill path

    def popen(argv, **kwargs):
        seen['cwd'] = kwargs['cwd']
        return Process()

    monkeypatch.setattr(foreign_arena.subprocess, 'Popen', popen)
    monkeypatch.setattr(foreign_arena.Bridge, '_hello', lambda self, seed: None)
    bridge = foreign_arena.Bridge(['python', str(script)], tmp_path / 'bridge.log')
    bridge.start(1)
    try:
        assert Path(seen['cwd']) == tmp_path
        bridge.close()                    # must tolerate a process that already died
    finally:
        bridge.close()


@pytest.mark.integration
def test_bridge_runs_a_real_process_round_trip(tmp_path):
    script = _write_bridge(tmp_path)
    log = tmp_path / 'bridge.log'
    bridge = foreign_arena.Bridge([sys.executable, str(script)], log)
    bridge.start(1)
    try:
        a, note = bridge.move(State())
        assert a == 42 and note is None
        assert bridge.proc.poll() is None    # still alive between requests
    finally:
        bridge.close()
    assert bridge.proc is None               # close() forgets it; double close is safe
    bridge.close()
    assert log.is_file()                     # stderr was captured, not leaked


@pytest.mark.integration
def test_bridge_timeout_kills_a_wedged_process(monkeypatch, tmp_path):
    script = tmp_path / 'sleepy_bridge.py'
    script.write_text('import json, sys, time\n'
                      'for line in sys.stdin:\n'
                      '    print(\'{"ok": true, "name": "sleepy"}\', flush=True)\n'
                      '    time.sleep(60)\n'
                      '    break\n', encoding='utf-8')
    monkeypatch.setattr(foreign_arena, 'MOVE_TIMEOUT', 0.5)
    bridge = foreign_arena.Bridge([sys.executable, str(script)], tmp_path / 's.log')
    bridge.start(1)
    try:
        with pytest.raises(foreign_arena.BridgeError, match='did not answer'):
            bridge.move(State())
        assert bridge.proc.poll() is not None      # killed, not left wedged
    finally:
        bridge.close()


def test_bridge_refuses_a_missing_script(tmp_path):
    bridge = foreign_arena.Bridge(['python', str(tmp_path / 'ghost.py')],
                                  tmp_path / 'g.log')
    with pytest.raises(foreign_arena.BridgeError, match='not found'):
        bridge.start(1)


@pytest.mark.integration
def test_bridge_rejects_a_non_int_action(tmp_path):
    script = tmp_path / 'bad_bridge.py'
    script.write_text('import json, sys\n'
                      'print(\'{"ok": true, "name": "bad"}\', flush=True)\n'
                      'for line in sys.stdin:\n'
                      '    print(\'{"a": true}\', flush=True)\n', encoding='utf-8')
    bridge = foreign_arena.Bridge([sys.executable, str(script)], tmp_path / 'b.log')
    bridge.start(1)
    try:
        with pytest.raises(foreign_arena.BridgeError, match='non-int'):
            bridge.move(State())
    finally:
        bridge.close()



def test_foreign_bot_can_warn_about_a_legal_fallback(capsys):
    bot = object.__new__(foreign_arena.ForeignBot)

    class Bridge:
        argv = ['python', 'example_bridge.py']
        name = 'example_bridge.py'

        @staticmethod
        def move(state):
            return legal_actions(state)[0], 'used fallback'

    bot.bridge = Bridge()
    action, note = bot(State(), None)
    assert action in legal_actions(State()) and note is None
    assert 'example_bridge.py' in capsys.readouterr().out


def test_foreign_bot_forfeits_and_dumps_an_illegal_action(tmp_path):
    bot = object.__new__(foreign_arena.ForeignBot)
    bot.dumpfile = tmp_path / 'illegal_dump.jsonl'

    class Bridge:
        name = 'naughty.py'

        @staticmethod
        def move(state):
            return 9999, None

    bot.bridge = Bridge()
    a, reason = bot(State(), None)
    assert a is foreign_arena.FORFEIT
    assert 'illegal action 9999' in reason
    record = json.loads(bot.dumpfile.read_text(encoding='utf-8'))
    assert record['bot'] == 'naughty.py' and record['action'] == 9999


def test_foreign_bot_turns_a_bridge_error_into_a_forfeit():
    bot = object.__new__(foreign_arena.ForeignBot)

    class Bridge:
        name = 'broken.py'

        @staticmethod
        def move(state):
            raise foreign_arena.BridgeError('no such engine')

    bot.bridge = Bridge()
    a, reason = bot(State(), None)
    assert a is foreign_arena.FORFEIT and 'no such engine' in reason


def test_foreign_bot_survives_a_failed_dump(tmp_path, capsys):
    bot = object.__new__(foreign_arena.ForeignBot)
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('', encoding='utf-8')
    bot.dumpfile = blocker / 'illegal_dump.jsonl'     # its parent is a file

    class Bridge:
        name = 'dump_fail.py'

        @staticmethod
        def move(state):
            return 9999, None

    bot.bridge = Bridge()
    a, reason = bot(State(), None)
    assert a is foreign_arena.FORFEIT and 'illegal action' in reason
    assert 'could not write' in capsys.readouterr().out


def test_opponent_move_retries_recoverable_glitches():
    calls = []
    duel = SimpleNamespace(s=None, rng=None)

    class Bot:
        def __call__(self, *_):
            calls.append(1)
            if len(calls) < 3:
                return foreign_arena.FORFEIT, 'garbled line'
            return 5, None

    action, note = foreign_arena._opponent_move(Bot(), duel)
    assert action == 5 and len(calls) == 3


def test_opponent_move_gives_up_immediately_on_a_dead_bridge():
    calls = []
    duel = SimpleNamespace(s=None, rng=None)

    class Bot:
        def __call__(self, *_):
            calls.append(1)
            return foreign_arena.FORFEIT, 'bridge exited with code 1; see ...'

    action, reason = foreign_arena._opponent_move(Bot(), duel, attempts=3)
    assert action is foreign_arena.FORFEIT and len(calls) == 1


def test_play_with_games_zero_needs_no_bots():
    r = foreign_arena.play(None, 'vader', None, games=0)
    assert r['games'] == 0 and r['win_rate'] == 0.0 and r['opponent'] == 'vader'


def test_play_rejects_negative_games_and_unknown_opponents():
    with pytest.raises(ValueError, match='games'):
        foreign_arena.play(None, 'vader', None, games=-1)
    with pytest.raises(ValueError, match='unknown opponent'):
        foreign_arena.play(None, 'nobody', None, games=0)