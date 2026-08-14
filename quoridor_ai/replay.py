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
from pathlib import Path

import numpy as np


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
        """Map an existing buffer of this exact geometry, or create it."""
        path = self.dir / name
        if path.exists():
            try:
                return np.lib.format.open_memmap(path, mode='r+')
            except (ValueError, OSError):
                pass
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
            j = self.head
            self._x[j] = x
            self._pi[j] = pi
            self._z[j] = z
            self._q[j] = q
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
