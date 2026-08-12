"""Device and thread selection, shared by the trainer, the arena and the CPU generator.

Autodetection alone is not enough once a run can legitimately want the CPU while a GPU is
present: a Colab CPU runtime does not burn the GPU quota, so generating self-play data on
the CPU is a real mode rather than a fallback. Every entry point therefore takes an explicit
preference and only falls back to autodetection when none is given.
"""
import os

import torch


def resolve_device(pref=None):
    """Device to run on. `pref` ('cpu'/'cuda'/'cuda:1'/None) overrides autodetection.

    Asking for CUDA when it is unavailable is an error rather than a silent downgrade to the
    CPU: a run configured for a T4 that quietly lands on two vCPUs looks like a hang.
    """
    if pref is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(pref)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'device={pref!r} requested but CUDA is not available')
    return dev


def configure_threads(n=None):
    """Intra-op thread count; defaults to every logical core. Returns what was set."""
    n = int(n) if n else (os.cpu_count() or 1)
    n = max(1, n)
    torch.set_num_threads(n)
    return n
