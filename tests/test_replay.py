"""DiskReplay ring-buffer semantics and buffer-geometry guards."""
import numpy as np
import pytest

from quoridor_ai.replay import DiskReplay, PrioritizedReplay

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


def test_priorities_are_uniform_when_disabled_or_all_zero():
    sampler = PrioritizedReplay(4, seed=3)
    sampler.extend([1, 20, 3, 4])
    assert np.array_equal(sampler.probabilities(), np.full(4, 0.25))

    sampler.enabled = True
    sampler.update(range(4), np.zeros(4))
    assert np.array_equal(sampler.probabilities(), np.full(4, 0.25))


def test_priorities_update_and_follow_circular_overwrite():
    sampler = PrioritizedReplay(3, enabled=True, alpha=1.0)
    sampler.extend([1, 2, 3, 4])
    assert np.allclose(sampler.probabilities(), np.asarray([2, 3, 4]) / 9)

    sampler.update([0, 2], [8, 1])
    assert np.allclose(sampler.probabilities(), np.asarray([8, 3, 1]) / 12)


def test_priority_sampling_is_seeded_and_reports_probabilities():
    first = PrioritizedReplay(4, enabled=True, alpha=1.0, seed=17)
    second = PrioritizedReplay(4, enabled=True, alpha=1.0, seed=17)
    first.extend([1, 2, 4, 8])
    second.extend([1, 2, 4, 8])

    indices, selected = first.sample(20, replace=True)
    other_indices, other_selected = second.sample(20, replace=True)
    assert np.array_equal(indices, other_indices)
    assert np.array_equal(selected, other_selected)
    assert np.array_equal(selected, first.probabilities()[indices])


def test_priority_sidecar_exports_imports_and_resumes_rng(tmp_path):
    sampler = PrioritizedReplay(
        3, enabled=True, alpha=0.5, seed=11, directory=tmp_path
    )
    sampler.extend([1, 4, 9, 16])
    sampler.sample(2, replace=True)

    resumed = PrioritizedReplay(3, directory=tmp_path)
    assert resumed.enabled
    assert resumed.alpha == 0.5
    assert np.array_equal(resumed.probabilities(), sampler.probabilities())
    expected = sampler.sample(12, replace=True)
    actual = resumed.sample(12, replace=True)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])

    exported = tmp_path / 'manual.npz'
    sampler.export(exported)
    imported = PrioritizedReplay(3)
    imported._load(exported)
    assert np.array_equal(imported.probabilities(), sampler.probabilities())


def test_priority_sidecar_rejects_invalid_updates_and_capacity(tmp_path):
    sampler = PrioritizedReplay(2, enabled=True, directory=tmp_path)
    sampler.extend([1, 2])
    with pytest.raises(ValueError, match='same length'):
        sampler.update([0], [1, 2])
    with pytest.raises(ValueError, match='non-negative'):
        sampler.update([0], [-1])
    with pytest.raises(IndexError):
        sampler.update([2], [1])
    with pytest.raises(ValueError, match='capacity'):
        PrioritizedReplay(3).import_state(sampler.state_dict())
