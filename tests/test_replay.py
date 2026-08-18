"""DiskReplay ring-buffer semantics and buffer-geometry guards."""
import numpy as np
import pytest

from quoridor_ai.replay import DiskReplay

PLANES, ACTIONS, CAP = 11, 209, 8

X = np.random.rand(CAP, PLANES, 9, 9).astype(np.float32)
PI = np.random.rand(CAP, ACTIONS).astype(np.float32)
Z = np.random.rand(CAP).astype(np.float32)
Q = np.random.rand(CAP).astype(np.float32)


def _samples(n):
    return [(X[i % CAP], PI[i % CAP], float(Z[i % CAP]), float(Q[i % CAP]))
            for i in range(n)]


def test_append_then_read_keeps_insertion_order(tmp_path):
    r = DiskReplay(tmp_path, CAP, PLANES, ACTIONS)
    r.extend(_samples(3))
    assert len(r) == 3
    for i in range(3):
        x, pi, z, q = r[i]
        assert np.array_equal(x, X[i]) and np.array_equal(pi, PI[i])
        assert z == float(Z[i]) and q == float(Q[i])


def test_ring_overwrites_the_oldest_when_full(tmp_path):
    r = DiskReplay(tmp_path, CAP, PLANES, ACTIONS)
    r.extend(_samples(CAP + 5))
    assert len(r) == CAP
    assert np.array_equal(r[0][0], X[5])          # index 0 is the oldest survivor


def test_reattach_restores_size_and_head(tmp_path):
    r = DiskReplay(tmp_path, CAP, PLANES, ACTIONS)
    r.extend(_samples(5))
    r2 = DiskReplay(tmp_path, CAP, PLANES, ACTIONS)
    assert len(r2) == 5
    for i in range(5):
        assert np.array_equal(r2[i][0], X[i]) and np.array_equal(r2[i][1], PI[i])


def test_geometry_mismatch_in_a_matching_file_is_refused(tmp_path):
    tag = f'x_{CAP}x{PLANES}x{ACTIONS}.npy'
    np.lib.format.open_memmap(tmp_path / tag, mode='w+', dtype=np.float32,
                              shape=(CAP, 5, 9, 9)).flush()
    with pytest.raises(ValueError, match='replay buffer'):
        DiskReplay(tmp_path, CAP, PLANES, ACTIONS)


def test_garbage_in_a_buffer_file_is_refused(tmp_path):
    tag = f'x_{CAP}x{PLANES}x{ACTIONS}.npy'
    (tmp_path / tag).write_bytes(b'not a numpy file at all')
    with pytest.raises(ValueError, match='unreadable replay buffer'):
        DiskReplay(tmp_path, CAP, PLANES, ACTIONS)


def test_extend_rejects_wrong_sample_shapes(tmp_path):
    r = DiskReplay(tmp_path, CAP, PLANES, ACTIONS)
    with pytest.raises(ValueError, match='sample state'):
        r.extend([(np.zeros((5, 9, 9), np.float32), PI[0], 0.0, 0.0)])
    with pytest.raises(ValueError, match='sample policy'):
        r.extend([(X[0], np.zeros(100, np.float32), 0.0, 0.0)])