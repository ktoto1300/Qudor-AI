"""Replay buffer backed by files on disk.

The AlphaZero replay buffer is the largest object in the trainer: at 500k samples a
v3 encoded position plus its policy target is ~6 KB, so an in-memory deque reaches
roughly 3 GB. On a box with 8 GB of RAM shared with four self-play workers that is
what eventually kills the run, and a container without CAP_SYS_ADMIN cannot be given
swap to absorb it.

The samples are only ever touched in two ways - appended at the end of an iteration
and read at random indices to build a training batch - so they do not need to be
resident. Each field lives in its own memory-mapped ring buffer and the OS page cache
decides what stays in RAM, which is exactly the behaviour a swap file would have
provided, except bounded to this one structure and backed by a file we control.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class PrioritizedReplay:
    """Experimental persistent priority sampler for a same-capacity replay ring.

    This sidecar deliberately does not own samples or integrate with ``DiskReplay``.
    Call ``add`` whenever the corresponding replay receives one sample, and use the
    returned logical indices with the replay's insertion-order indexing.
    """

    def __init__(self, capacity: int, *, enabled: bool = False, alpha: float = 0.6,
                 seed: int | None = None, directory=None):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError('capacity must be positive')
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and non-negative')
        self.size = 0
        self.head = 0
        self._priorities = np.zeros(self.capacity, dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self.path = None
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f'priorities_{self.capacity}.npz'
            if self.path.exists():
                self._load(self.path)

    def __len__(self):
        return self.size

    def _physical_index(self, logical_index: int):
        if not 0 <= logical_index < self.size:
            raise IndexError(logical_index)
        return (self.head - self.size + logical_index) % self.capacity

    def _logical_priorities(self):
        return np.asarray([
            self._priorities[self._physical_index(i)] for i in range(self.size)
        ])

    @staticmethod
    def _validate_priority(priority):
        priority = float(priority)
        if not np.isfinite(priority) or priority < 0:
            raise ValueError('priorities must be finite and non-negative')
        return priority

    def add(self, priority: float | None = None):
        """Add one priority, overwriting the oldest entry when the ring is full."""
        if priority is None:
            current = self._logical_priorities()
            priority = float(current.max()) if np.any(current > 0) else 1.0
        priority = self._validate_priority(priority)
        self._priorities[self.head] = priority
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.flush()

    def extend(self, priorities):
        for priority in priorities:
            if priority is None:
                current = self._logical_priorities()
                priority = float(current.max()) if np.any(current > 0) else 1.0
            priority = self._validate_priority(priority)
            self._priorities[self.head] = priority
            self.head = (self.head + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        self.flush()

    def update(self, indices, priorities):
        indices = list(indices)
        priorities = list(priorities)
        if len(indices) != len(priorities):
            raise ValueError('indices and priorities must have the same length')
        validated = [self._validate_priority(priority) for priority in priorities]
        physical = [self._physical_index(int(index)) for index in indices]
        for index, priority in zip(physical, validated, strict=True):
            self._priorities[index] = priority
        self.flush()

    def probabilities(self):
        """Return probabilities in the same oldest-to-newest order as DiskReplay."""
        if self.size == 0:
            return np.empty(0, dtype=np.float64)
        priorities = self._logical_priorities()
        if not self.enabled or not np.any(priorities > 0):
            return np.full(self.size, 1.0 / self.size, dtype=np.float64)
        weights = np.power(priorities, self.alpha)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(self.size, 1.0 / self.size, dtype=np.float64)
        return weights / total

    def sample(self, count: int, *, replace: bool = False):
        """Return logical replay indices and their current sampling probabilities."""
        count = int(count)
        if count < 0:
            raise ValueError('sample count must be non-negative')
        if not replace:
            count = min(count, self.size)
        elif self.size == 0 and count:
            raise ValueError('cannot sample from an empty replay')
        probabilities = self.probabilities()
        indices = self._rng.choice(
            self.size, size=count, replace=replace, p=probabilities
        ).astype(np.int64, copy=False)
        self.flush()
        return indices, probabilities[indices]

    def state_dict(self):
        return {
            'version': 1,
            'capacity': self.capacity,
            'enabled': self.enabled,
            'alpha': self.alpha,
            'size': self.size,
            'head': self.head,
            'priorities': self._priorities.copy(),
            'rng_state': self._rng.bit_generator.state,
        }

    def import_state(self, state):
        """Import an exported state without changing the sidecar's capacity."""
        if int(state.get('version', 0)) != 1:
            raise ValueError('unsupported priority sidecar version')
        if int(state.get('capacity', -1)) != self.capacity:
            raise ValueError('priority sidecar capacity does not match')
        priorities = np.asarray(state['priorities'], dtype=np.float64)
        if priorities.shape != (self.capacity,):
            raise ValueError('priority sidecar has an invalid priorities shape')
        if np.any(~np.isfinite(priorities)) or np.any(priorities < 0):
            raise ValueError('priority sidecar contains invalid priorities')
        size = int(state['size'])
        head = int(state['head'])
        if not 0 <= size <= self.capacity or not 0 <= head < self.capacity:
            raise ValueError('priority sidecar has invalid ring metadata')
        alpha = float(state['alpha'])
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError('priority sidecar has invalid alpha')
        self.enabled = bool(state['enabled'])
        self.alpha = alpha
        self.size = size
        self.head = head
        self._priorities[:] = priorities
        self._rng.bit_generator.state = state['rng_state']
        self.flush()

    def export(self, path=None):
        """Atomically export priorities, ring metadata, settings, and RNG state."""
        path = self.path if path is None else Path(path)
        if path is None:
            raise ValueError('an export path is required without a sidecar directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('wb') as file:
            np.savez(
                file,
                version=np.asarray(state['version'], dtype=np.int64),
                capacity=np.asarray(state['capacity'], dtype=np.int64),
                enabled=np.asarray(state['enabled'], dtype=np.bool_),
                alpha=np.asarray(state['alpha'], dtype=np.float64),
                size=np.asarray(state['size'], dtype=np.int64),
                head=np.asarray(state['head'], dtype=np.int64),
                priorities=state['priorities'],
                rng_state=np.asarray(json.dumps(state['rng_state'])),
            )
        os.replace(tmp, path)

    def _load(self, path):
        try:
            with np.load(path, allow_pickle=False) as saved:
                state = {
                    'version': int(saved['version']),
                    'capacity': int(saved['capacity']),
                    'enabled': bool(saved['enabled']),
                    'alpha': float(saved['alpha']),
                    'size': int(saved['size']),
                    'head': int(saved['head']),
                    'priorities': saved['priorities'],
                    'rng_state': json.loads(str(saved['rng_state'])),
                }
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f'unreadable priority sidecar {path}: {exc}') from exc
        self.import_state(state)

    def flush(self):
        if self.path is not None:
            self.export(self.path)


class DiskReplay:
    """Fixed-capacity ring buffer of (state, policy, z, q) samples held in memmaps."""

    def __init__(self, directory, capacity: int, planes: int, actions: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.meta_path = self.dir / 'meta.json'
        # The geometry is part of the filename: a run that changes encoder or capacity
        # then simply maps a different file instead of trying to delete one that another
        # process may still hold open.
        tag = f'{self.capacity}x{planes}x{actions}'
        self._x = self._open(f'x_{tag}.npy', np.float32, (self.capacity, planes, 9, 9))
        self._pi = self._open(f'pi_{tag}.npy', np.float32, (self.capacity, actions))
        self._z = self._open(f'z_{tag}.npy', np.float32, (self.capacity,))
        self._q = self._open(f'q_{tag}.npy', np.float32, (self.capacity,))
        self.size = 0
        self.head = 0
        self._load_meta(planes, actions)

    def _open(self, name, dtype, shape):
        """Map an existing buffer of this exact geometry, or create it.

        A file whose header disagrees with the requested geometry is a corrupt or
        foreign buffer. It is refused with an error that names the files involved
        rather than truncated: deleting it loses every previous run's games, and the
        tag embeds the geometry precisely so an honest geometry change maps a
        different filename and never reaches this path.
        """
        path = self.dir / name
        if path.exists():
            try:
                arr = np.lib.format.open_memmap(path, mode='r+')
            except (ValueError, OSError) as exc:     # not even a readable buffer
                raise ValueError(
                    f'unreadable replay buffer {path}: {exc}. The geometry tag says '
                    f'this run wants {dtype.__name__}{shape}; remove the file only if '
                    f'the run that created it is over') from exc
            if arr.dtype != dtype or arr.shape != shape:
                raise ValueError(
                    f'replay buffer {path} has {arr.dtype}{arr.shape}, but this run '
                    f'wants {dtype.__name__}{shape}. The geometry tag embeds the '
                    f'desired layout, so an existing file with this name was created '
                    f'by the same layout and is corrupt; delete it and restart')
            return arr
        return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)

    def _load_meta(self, planes: int, actions: int):
        """Reattach to a buffer written by an earlier process, or start empty."""
        if not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text(encoding='utf-8'))
        if (meta.get('capacity') != self.capacity or meta.get('planes') != planes
                or meta.get('actions') != actions):
            return                      # geometry changed: the old file is unusable
        self.size = min(int(meta.get('size', 0)), self.capacity)
        self.head = int(meta.get('head', 0)) % self.capacity

    def _save_meta(self):
        self.meta_path.write_text(json.dumps({
            'capacity': self.capacity, 'planes': int(self._x.shape[1]),
            'actions': int(self._pi.shape[1]), 'size': self.size, 'head': self.head,
        }), encoding='utf-8')

    def __len__(self):
        return self.size

    def __getitem__(self, i: int):
        if not 0 <= i < self.size:
            raise IndexError(i)
        # Index 0 is the oldest surviving sample, so callers see insertion order
        # regardless of where the ring currently wraps. The arrays are copied out of
        # the mapping: a returned memmap view would keep the file alive and cannot be
        # serialised into a checkpoint.
        j = (self.head - self.size + i) % self.capacity
        return (np.asarray(self._x[j]).copy(), np.asarray(self._pi[j]).copy(),
                float(self._z[j]), float(self._q[j]))

    def extend(self, samples):
        for x, pi, z, q in samples:
            x = np.asarray(x, np.float32)
            pi = np.asarray(pi, np.float32)
            if x.shape != self._x.shape[1:]:
                raise ValueError(
                    f'sample state has shape {x.shape}, buffer wants '
                    f'{self._x.shape[1:]}')
            if pi.shape != self._pi.shape[1:]:
                raise ValueError(
                    f'sample policy has shape {pi.shape}, buffer wants '
                    f'{self._pi.shape[1:]}')
            j = self.head
            self._x[j] = x
            self._pi[j] = pi
            self._z[j] = float(z)
            self._q[j] = float(q)
            self.head = (j + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        self.flush()

    def tail(self, n: int):
        """The n most recent samples as plain arrays, for the checkpoint."""
        n = max(0, min(int(n), self.size))
        return [tuple(self[i]) for i in range(self.size - n, self.size)]

    def flush(self):
        for array in (self._x, self._pi, self._z, self._q):
            array.flush()
        self._save_meta()
