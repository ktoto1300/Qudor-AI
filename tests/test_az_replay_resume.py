"""Disk-backed replay must not double-count what a checkpoint already holds."""
import json

import pytest
import torch

from quoridor_ai.az_train import run
from quoridor_ai.replay import DiskReplay

pytestmark = pytest.mark.integration


def _config(tmp_path, **over):
    c = dict(seed=1, encoding=3, iterations=2, channels=8, blocks=1, se=True, device='cpu',
             gumbel=True, gumbel_cap=4, games=2, sims=4, fast_sims=2, full_frac=1.0,
             max_plies=12, temp_moves=2, resign_v=-0.95, steps=2, batch=4, lr=0.001,
             warmup_steps=2, replay=500, checkpoint_replay=100, gate_every=0,
             replay_on_disk=True)
    c.update(over)
    path = tmp_path / 'resume.json'
    path.write_text(json.dumps(c))
    return str(path)


def _buffer(out):
    return DiskReplay(out / 'replay', capacity=500, planes=16, actions=209)


def test_resume_does_not_reappend_the_checkpoint_tail(tmp_path):
    out = tmp_path / 'run'
    run(_config(tmp_path), str(out), resume=False)
    after_first = len(_buffer(out))
    checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    tail = len(checkpoint['replay'])
    assert tail > 0

    run(_config(tmp_path, iterations=3), str(out), resume=True)
    after_resume = len(_buffer(out))
    third = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    grown = third['iteration'] == 2

    assert grown
    # One more iteration was generated, so the buffer grew by that iteration alone -
    # not by the iteration plus a second copy of the checkpoint tail.
    assert after_resume < after_first + tail


def test_status_json_is_valid_json_without_a_gate(tmp_path):
    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=1), str(out), resume=False)
    text = (out / 'status.json').read_text(encoding='utf-8')

    def reject(literal):
        raise ValueError(f'non-standard JSON literal: {literal}')

    status = json.loads(text, parse_constant=reject)
    assert status['gate_win_rate'] is None
    assert status['gate_elo'] is None
    assert status['iteration'] == 0


def test_a_fresh_run_refuses_to_inherit_an_existing_disk_buffer(tmp_path):
    out = tmp_path / 'run'
    run(_config(tmp_path, iterations=1), str(out), resume=False)
    for name in ('latest.pt', 'best.pt', 'metrics.csv', 'status.json'):
        (out / name).unlink(missing_ok=True)

    with pytest.raises(FileExistsError, match='replay'):
        run(_config(tmp_path, iterations=1), str(out), resume=False)
