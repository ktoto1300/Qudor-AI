import numpy as np
import pytest
import torch

from quoridor_ai.core.encoding import PLANES_BY_VERSION, encode_batch
from quoridor_ai.model import PolicyValueNet
from quoridor_ai.quant import calibration_states, quantize_net, quantized_for


def _inputs(n, seed=0):
    return torch.from_numpy(encode_batch(calibration_states(n, seed), 3)).to(memory_format=torch.channels_last)


def test_quantize_net_tracks_the_float_net():
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    q = quantize_net(net, 3, states=calibration_states(24, seed=1))
    if q is net:
        pytest.skip("static int8 quantisation unavailable on this build")
    x = _inputs(8, seed=2)
    with torch.inference_mode():
        p, v = net(x)
        pq, vq = q(x)
    assert pq.shape == p.shape and vq.shape == v.shape
    assert float((v - vq).abs().mean()) < 0.3
    soft = torch.softmax(p, 1)
    softq = torch.softmax(pq, 1)
    assert float((soft - softq).abs().mean()) < 0.05


def test_quantized_for_caches_by_checkpoint_hash(tmp_path):
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    f = tmp_path / "best.pt"
    f.write_bytes(b"not really a checkpoint")
    q1 = quantized_for(f, net, 3)
    if q1 is net:
        pytest.skip("static int8 quantisation unavailable on this build")
    assert (tmp_path / "best.pt.int8.pt").exists()
    q2 = quantized_for(f, net, 3)
    x = _inputs(4, seed=3)
    with torch.inference_mode():
        p1, v1 = q1(x)
        p2, v2 = q2(x)
    assert torch.equal(p1, p2) and torch.equal(v1, v2)


def test_quantized_for_requantises_when_the_checkpoint_changes(tmp_path):
    net = PolicyValueNet(8, 1, PLANES_BY_VERSION[3])
    f = tmp_path / "best.pt"
    f.write_bytes(b"version one")
    q1 = quantized_for(f, net, 3)
    if q1 is net:
        pytest.skip("static int8 quantisation unavailable on this build")
    x = _inputs(2, seed=4)
    with torch.inference_mode():
        p1, v1 = q1(x)
    # A re-trained checkpoint is a different net with the same name: the cache key is
    # the file's hash, so the quantised weights must be rebuilt and the outputs change.
    f.write_bytes(b"version two - new weights, same name")
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.5)
    q2 = quantized_for(f, net, 3)
    with torch.inference_mode():
        p2, v2 = q2(x)
    assert not torch.equal(p1, p2)