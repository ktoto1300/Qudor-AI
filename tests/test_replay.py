"""The replay buffer is the trainer's largest object; on disk it must behave the same."""
import numpy as np
import pytest

from quoridor_ai.replay import DiskReplay


def _sample(i, planes=16, actions=209):
    x = np.full((planes, 9, 9), float(i), np.float32)
    pi = np.zeros(actions, np.float32)
    pi[i % actions] = 1.0
    return x, pi, float(i % 3) - 1.0, float(i) / 100.0


def test_disk_replay_reads_back_what_it_stored(tmp_path):
    replay = DiskReplay(tmp_path / 'replay', capacity=8, planes=4, actions=6)
    replay.extend([_sample(i, 4, 6) for i in range(3)])
    assert len(replay) == 3
    for i in range(3):
        x, pi, z, q = replay[i]
        assert np.array_equal(x, np.full((4, 9, 9), float(i), np.float32))
        assert int(pi.argmax()) == i % 6
        assert z == pytest.approx(float(i % 3) - 1.0)
        assert q == pytest.approx(float(i) / 100.0)


def test_disk_replay_drops_the_oldest_samples_like_a_deque(tmp_path):
    replay = DiskReplay(tmp_path / 'replay', capacity=4, planes=4, actions=6)
    replay.extend([_sample(i, 4, 6) for i in range(6)])
    assert len(replay) == 4
    # Sample 0 and 1 were overwritten; index 0 is now the oldest survivor.
    assert replay[0][0][0, 0, 0] == pytest.approx(2.0)
    assert replay[3][0][0, 0, 0] == pytest.approx(5.0)


def test_disk_replay_tail_returns_the_most_recent_samples(tmp_path):
    replay = DiskReplay(tmp_path / 'replay', capacity=8, planes=4, actions=6)
    replay.extend([_sample(i, 4, 6) for i in range(5)])
    tail = replay.tail(2)
    assert [row[0][0, 0, 0] for row in tail] == [3.0, 4.0]
    assert replay.tail(99)[0][0][0, 0, 0] == pytest.approx(0.0)


def test_disk_replay_reattaches_after_a_restart(tmp_path):
    first = DiskReplay(tmp_path / 'replay', capacity=8, planes=4, actions=6)
    first.extend([_sample(i, 4, 6) for i in range(5)])
    second = DiskReplay(tmp_path / 'replay', capacity=8, planes=4, actions=6)
    assert len(second) == 5
    assert second[4][0][0, 0, 0] == pytest.approx(4.0)


def test_disk_replay_ignores_a_buffer_with_different_geometry(tmp_path):
    first = DiskReplay(tmp_path / 'replay', capacity=8, planes=4, actions=6)
    first.extend([_sample(i, 4, 6) for i in range(5)])
    reshaped = DiskReplay(tmp_path / 'replay', capacity=8, planes=16, actions=6)
    assert len(reshaped) == 0


def test_disk_replay_rejects_out_of_range_indices(tmp_path):
    replay = DiskReplay(tmp_path / 'replay', capacity=4, planes=4, actions=6)
    replay.extend([_sample(0, 4, 6)])
    with pytest.raises(IndexError):
        replay[1]
