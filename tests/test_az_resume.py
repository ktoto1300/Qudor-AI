"""Cross-device and cross-config resume.

A run that moves between a GPU config and a CPU one must continue, not silently restart
its schedule. Nothing here crashes when it breaks - the run just trains at the wrong
learning rate for hours, which is indistinguishable from "training plateaued".
"""
import json
from pathlib import Path

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


def test_checkpoints_record_the_rules_version(tmp_path):
    from quoridor_ai.core.engine import RULES_VERSION

    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=1), str(out), resume=False)
    latest = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    assert latest['rules_version'] == best['rules_version'] == RULES_VERSION


def test_resume_rejects_checkpoint_from_unknown_rules(tmp_path):
    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=1), str(out), resume=False)
    checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    del checkpoint['rules_version']
    torch.save(checkpoint, out / 'latest.pt')
    with pytest.raises(ValueError, match='rules_version'):
        run(_config(tmp_path, iterations=2), str(out), resume=True)


def test_init_is_weights_only_and_starts_a_fresh_lineage(tmp_path):
    source = tmp_path / 'source'
    run(_config(tmp_path, iterations=2), str(source), resume=False)
    old = torch.load(source / 'latest.pt', map_location='cpu', weights_only=False)
    target = tmp_path / 'target'
    run(_config(tmp_path, iterations=1), str(target), resume=False,
        init=str(source / 'latest.pt'))
    fresh = torch.load(target / 'latest.pt', map_location='cpu', weights_only=False)
    assert fresh['iteration'] == 0
    assert fresh['generation'] == 0
    assert fresh['global_step'] == 2
    assert len(fresh['replay']) < len(old['replay'])


def test_missing_init_checkpoint_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='does not exist'):
        run(_config(tmp_path, iterations=0), str(tmp_path / 'run'), resume=False,
            init=str(tmp_path / 'missing.pt'))


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


def test_training_forwards_the_concurrent_game_limit(tmp_path, monkeypatch):
    import quoridor_ai.az_train as az_train

    seen = {}

    def fake_selfplay(net, device, workers, kwargs):
        seen['concurrent_games'] = kwargs['concurrent_games']
        x = np.zeros((16, 9, 9), np.float32)
        pi = np.full(209, 1 / 209, np.float32)
        return [(x, pi, 0.0, 0.0)], {
            'games': 1, 'samples': 1, 'avg_plies': 1.0, 'p0_wins': 0, 'draws': 1,
        }

    import numpy as np
    monkeypatch.setattr(az_train, '_generate_selfplay', fake_selfplay)
    run(_config(tmp_path, iterations=1, concurrent_games=1),
        str(tmp_path / 'run'), resume=False)
    assert seen['concurrent_games'] == 1


def test_no_resume_refuses_to_mix_with_existing_run(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'best.pt').write_bytes(b'old champion')
    with pytest.raises(FileExistsError, match='empty output directory'):
        run(_config(tmp_path, iterations=0), str(out), resume=False)


def test_init_refuses_to_reuse_an_old_champion(tmp_path):
    source = tmp_path / 'source'
    run(_config(tmp_path, iterations=1), str(source), resume=False)
    out = tmp_path / 'target'
    out.mkdir()
    (out / 'best.pt').write_bytes(b'old champion')
    with pytest.raises(FileExistsError, match='empty output directory'):
        run(_config(tmp_path, iterations=2), str(out), init=str(source / 'latest.pt'))


def test_worker_concurrency_is_a_global_limit(monkeypatch, tmp_path):
    import quoridor_ai.az_train as az_train

    payloads = []

    class Pool:
        def __init__(self, workers):
            self.workers = workers
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def map(self, fn, items):
            payloads.extend(items)
            return [([], {'games': item[3]['games'], 'samples': 0, 'avg_plies': 0,
                          'p0_wins': 0, 'draws': item[3]['games']}) for item in items]

    class Context:
        def Pool(self, workers):
            return Pool(workers)

    monkeypatch.setattr(az_train.multiprocessing, 'get_context', lambda method: Context())
    net = az_train.PolicyValueNet(8, 1, 16)
    kwargs = dict(games=20, concurrent_games=7, encoding=3, sims=1, fast_sims=1,
                  full_frac=1, max_plies=1, _weights_dir=str(tmp_path))
    _, stats = az_train._generate_selfplay(net, torch.device('cpu'), 4, kwargs)
    assert stats['games'] == 20
    assert sum(item[3]['concurrent_games'] for item in payloads) == 7
    assert not list(Path(tmp_path).glob('.selfplay_weights_*.pt'))


def test_ema_promotion_resets_optimizer_moments(tmp_path, monkeypatch):
    import quoridor_ai.az_train as az_train

    scores = iter((0.0, 1.0))
    monkeypatch.setattr(az_train, 'compare', lambda *args, **kwargs: {
        'games': 2, 'wins': 2, 'draws': 0, 'losses': 0,
        'win_rate': next(scores), 'elo_delta': 100,
    })
    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=1, gate_every=1, gate_games=2,
                gate_threshold=0.5), str(out), resume=False)
    checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['optimizer']['state'] == {}
    for name, value in checkpoint['model'].items():
        if value.dtype.is_floating_point:
            assert torch.equal(value.float(), checkpoint['ema'][name])
