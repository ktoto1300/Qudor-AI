import json

import numpy as np
import pytest
import torch

from quoridor_ai.az_train import run
from quoridor_ai.replay import DiskReplay


def _config(tmp_path, **over):
    c = dict(seed=1, encoding=3, iterations=2, channels=8, blocks=1, se=True, device='cpu',
             gumbel=True, gumbel_cap=4, games=2, sims=4, fast_sims=2, full_frac=1.0,
             max_plies=12, temp_moves=2, resign_v=-0.95, steps=2, batch=4, lr=0.001,
             warmup_steps=2, replay=500, checkpoint_replay=100, gate_every=0)
    c.update(over)
    path = tmp_path / 'disk.json'
    path.write_text(json.dumps(c))
    return str(path)


def test_training_can_keep_its_replay_buffer_on_disk(tmp_path):
    out = tmp_path / 'run'
    run(_config(tmp_path, replay_on_disk=True), str(out), resume=False)

    buffer_dir = out / 'replay'
    assert buffer_dir.is_dir()
    assert list(buffer_dir.glob('x_*.npy'))

    replay = DiskReplay(buffer_dir, capacity=500, planes=16, actions=209)
    assert len(replay) > 0
    x, pi, z, q = replay[0]
    assert x.shape == (16, 9, 9)
    assert float(pi.sum()) == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= z <= 1.0 and -1.0 <= q <= 1.0

    checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['replay']
    assert np.asarray(checkpoint['replay'][0][0]).shape == (16, 9, 9)


def test_a_disk_backed_run_resumes_from_its_buffer(tmp_path):
    out = tmp_path / 'run'
    run(_config(tmp_path, replay_on_disk=True), str(out), resume=False)
    first = len(DiskReplay(out / 'replay', capacity=500, planes=16, actions=209))

    run(_config(tmp_path, iterations=4, replay_on_disk=True), str(out), resume=True)
    second = len(DiskReplay(out / 'replay', capacity=500, planes=16, actions=209))
    assert second > first
