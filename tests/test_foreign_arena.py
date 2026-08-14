import importlib.util
from pathlib import Path

from quoridor_ai.core.engine import State, legal_actions


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('foreign_arena', ROOT / 'tools' / 'foreign_arena.py')
foreign_arena = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(foreign_arena)


def test_every_foreign_opponent_has_an_existing_bridge():
    for argv in foreign_arena.OPPONENTS.values():
        assert Path(argv[-1]).is_file()


def test_bridge_uses_the_bridge_directory_as_cwd(monkeypatch, tmp_path):
    script = tmp_path / 'bridge.py'
    script.write_text('')
    seen = {}

    class Process:
        stdin = stdout = None

    def popen(argv, **kwargs):
        seen['cwd'] = kwargs['cwd']
        return Process()

    monkeypatch.setattr(foreign_arena.subprocess, 'Popen', popen)
    monkeypatch.setattr(foreign_arena.Bridge, '_hello', lambda self, seed: None)
    bridge = foreign_arena.Bridge(['python', str(script)], tmp_path / 'bridge.log')
    bridge.start(1)
    assert Path(seen['cwd']) == tmp_path


def test_foreign_bot_can_warn_about_a_legal_fallback(capsys):
    bot = object.__new__(foreign_arena.ForeignBot)

    class Bridge:
        argv = ['python', 'example_bridge.py']

        @staticmethod
        def move(state):
            return legal_actions(state)[0], 'used fallback'

    bot.bridge = Bridge()
    action, note = bot(State(), None)
    assert action in legal_actions(State()) and note is None
    assert 'example_bridge.py' in capsys.readouterr().out
