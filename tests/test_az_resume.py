"""Cross-device and cross-config resume.

A run that moves between a GPU config and a CPU one must continue, not silently restart
its schedule. Nothing here crashes when it breaks - the run just trains at the wrong
learning rate for hours, which is indistinguishable from "training plateaued".
"""
import json

import pytest
import torch

from quoridor_ai.az_train import _lr_at, run
from quoridor_ai.runtime import configure_threads, resolve_device


def test_cpu_can_be_forced_while_cuda_is_available(monkeypatch):
    """A Colab CPU runtime does not burn GPU quota, so 'cpu' must be a choice, not a fallback."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert resolve_device('cpu').type == 'cpu'
    assert resolve_device(None).type == 'cuda'


def test_asking_for_a_missing_gpu_is_an_error_not_a_downgrade(monkeypatch):
    """Silently landing a T4 config on two vCPUs looks like a hang, not a misconfiguration."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA is not available'):
        resolve_device('cuda')
    assert resolve_device(None).type == 'cpu'


def test_configure_threads_defaults_to_every_core():
    before = torch.get_num_threads()
    try:
        assert configure_threads(2) == 2 and torch.get_num_threads() == 2
        assert configure_threads(None) >= 1
    finally:
        torch.set_num_threads(before)


def test_lr_depends_on_global_step_not_the_iteration_number():
    """The bug this guards: resuming a 160-step config under an 8-step one.

    Deriving the schedule position from iteration * steps would put iteration 4 at step 32
    instead of 640, restarting warmup and cutting the learning rate ~19x.
    """
    kw = dict(total=80000, base=0.0015, warmup=800)
    assert _lr_at(640, **kw) == pytest.approx(1.202e-3, rel=1e-2)
    assert _lr_at(4 * 8, **kw) < _lr_at(640, **kw) / 10


def _config(tmp_path, **over):
    c = dict(seed=1, encoding=3, iterations=2, channels=8, blocks=1, se=True, device='cpu',
             gumbel=True, gumbel_cap=4, games=2, sims=4, fast_sims=2, full_frac=1.0,
             max_plies=12, temp_moves=2, resign_v=-0.95, steps=2, batch=4, lr=0.001,
             warmup_steps=2, replay=500, checkpoint_replay=100, gate_every=0)
    c.update(over)
    p = tmp_path / 'c.json'
    p.write_text(json.dumps(c))
    return str(p)


def test_a_resumed_run_continues_the_step_count(tmp_path):
    """Two 2-step iterations then two more must reach step 8, not restart at 4."""
    out = tmp_path / 'run'
    run(_config(tmp_path), str(out), resume=False)
    first = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert first['global_step'] == 4

    run(_config(tmp_path, iterations=4), str(out), resume=True)
    second = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert second['iteration'] == 3
    assert second['global_step'] == 8


def test_a_checkpoint_without_global_step_is_reconstructed_from_its_own_config(tmp_path):
    """Checkpoints predating global_step carry the config they ran under; use its steps."""
    out = tmp_path / 'run'
    run(_config(tmp_path), str(out), resume=False)
    d = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    del d['global_step']                       # what a pre-fix GPU checkpoint looks like
    d['config']['steps'] = 160                 # ...saved under a much larger config
    torch.save(d, out / 'latest.pt')

    run(_config(tmp_path, iterations=3, steps=2), str(out), resume=True)
    after = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert after['global_step'] == 2 * 160 + 2, 'should resume at 320, not at the new steps'


def test_save_every_skips_intermediate_writes_but_never_the_last(tmp_path):
    """A ~150 MB latest.pt written every few minutes would swamp Drive's I/O quota."""
    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=3, save_every=5), str(out), resume=False)
    d = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert d['iteration'] == 2, 'the final iteration must always be saved'


def test_promoted_champion_is_recorded_against_a_fixed_bot(tmp_path, monkeypatch):
    """Relative arena wins are not an absolute measure; promotions must get one."""
    import csv
    import quoridor_ai.az_train as az_train

    monkeypatch.setattr(az_train, 'compare', lambda *args, **kwargs: {
        'games': 2, 'wins': 2, 'draws': 0, 'losses': 0, 'win_rate': 1.0, 'elo_delta': 999,
    })
    monkeypatch.setattr(az_train, 'play_baseline', lambda *args, **kwargs: {
        'games': 3, 'wins': 2, 'draws': 0, 'losses': 1, 'win_rate': 2 / 3,
        'elo_delta': 120, 'avg_plies': 20, 'sims': kwargs['sims'],
        'temperature': kwargs['temp'], 'gumbel': kwargs['gumbel'], 'seed': kwargs['seed'],
    })
    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=1, gate_every=1, gate_games=2, gate_threshold=0,
                baseline_bots=['rusher'], baseline_games=3, baseline_sims=2), str(out),
        resume=False)
    with (out / 'baseline.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]['bot'] == 'rusher'
    assert rows[0]['generation'] == '1'
    assert rows[0]['win_rate'] == str(2 / 3)
