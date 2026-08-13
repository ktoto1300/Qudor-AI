"""Static int8 quantisation for CPU inference.

The viewer's per-move cost on a CPU is almost entirely the network forward pass: a
float32 forward is ~25 ms here, and a Gumbel search at sims=64 makes ~64 of them.
Static int8 quantisation of the convolutions on the oneDNN backend measured ~2.3x
faster with top-1 policy agreement 100% on the real checkpoint - a change invisible
next to search-time variance.

Quantisation is graph-level (FX), so the quantised model is cached next to the
checkpoint and keyed by the checkpoint's SHA-256: a re-trained checkpoint that
reuses the same filename re-quantises automatically on the next load.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core.encoding import PLANES_BY_VERSION, encode_batch
from .core.engine import State, apply_unchecked, legal_actions

_CALIB_POSITIONS = 256
_CALIB_BATCH = 32


def calibration_states(n: int = _CALIB_POSITIONS, seed: int = 0) -> list[State]:
    """Mid-game positions to calibrate against: enough walls for the convs to see both
    kinds of input, and enough pawn progress for the goal planes to be populated."""
    rng = np.random.default_rng(seed)
    out, s = [], State()
    while len(out) < n:
        acts = legal_actions(s)
        if s.ply >= 220 or s.winner is not None or not acts:
            s = State()
            continue
        s = apply_unchecked(s, int(rng.choice(acts)))
        if 8 <= s.ply <= 100 and s.walls0 + s.walls1 > 0:
            out.append(s)
    return out


def _example_input(states: list[State], encoding: int) -> torch.Tensor:
    x = torch.from_numpy(encode_batch(states, encoding))
    return x.to(memory_format=torch.channels_last)


def _build_quantized(net: nn.Module, encoding: int, calibrate: bool) -> nn.Module:
    """Static-int8 FX graph of `net`. `net` itself is never modified."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    net = net.eval()  # conv-bn fusion runs in eval mode only
    torch.backends.quantized.engine = "onednn"
    prepared = prepare_fx(net, get_default_qconfig_mapping("onednn"),
                          example_inputs=(_example_input(calibration_states(1), encoding),))
    if calibrate:
        x = _example_input(calibration_states(), encoding)
        with torch.inference_mode():
            for i in range(0, len(x), _CALIB_BATCH):
                prepared(x[i:i + _CALIB_BATCH])
    return convert_fx(prepared).eval()


def quantize_net(net: nn.Module, encoding: int, states: list[State] | None = None) -> nn.Module:
    """Static-int8 FX quantisation of `net` on CPU. Returns a new module.

    `states` overrides the default mid-game calibration data. Falls back to `net`
    itself when the build has no oneDNN backend, so callers can always treat the
    result as "the net, possibly quantised".
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    try:
        states = calibration_states() if states is None else states
        net = net.eval()  # conv-bn fusion runs in eval mode only
        torch.backends.quantized.engine = "onednn"
        prepared = prepare_fx(net, get_default_qconfig_mapping("onednn"),
                              example_inputs=(_example_input(states[:1], encoding),))
        x = _example_input(states, encoding)
        with torch.inference_mode():
            for i in range(0, len(x), _CALIB_BATCH):
                prepared(x[i:i + _CALIB_BATCH])
        return convert_fx(prepared).eval()
    except Exception:
        return net


def quantized_for(path: Path, net: nn.Module, encoding: int) -> nn.Module:
    """`net` loaded from `path`, quantised to int8 - from a cache beside the checkpoint.

    The cache is keyed by the checkpoint's SHA-256 and holds only the quantised state
    dict, so loading it never touches the weights. On any failure (no oneDNN, a bad
    cache, a torch without FX support) the float `net` is returned unchanged.
    """
    cache = Path(str(path) + ".int8.pt")
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return net
    try:
        if cache.exists():
            blob = torch.load(cache, map_location="cpu", weights_only=True)
            if isinstance(blob, dict) and blob.get("sha") == digest:
                qnet = _build_quantized(net, encoding, False)
                qnet.load_state_dict(blob["state"])
                return qnet
        qnet = _build_quantized(net, encoding, True)
        torch.save({"sha": digest, "state": qnet.state_dict()}, cache)
        return qnet
    except Exception:
        return net
