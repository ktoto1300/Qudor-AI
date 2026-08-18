"""Legacy single-tree MCTS, kept for anyone importing `quoridor_ai.mcts.search`.

Everything here predates the batched search and shares none of its code; the training
pipeline uses `az_selfplay` and the viewer uses `batched_mcts`. This module now matches
their semantics so it cannot be imported into a broken state: the encoding is derived
from the network's planes (it used to hardcode v1 and crash on a v3 checkpoint), the
root of a finished game returns a zero policy instead of raising over an empty softmax,
canonical encodings have their policies unmapped back to the real action frame, the
root noise is reproducible when a seed is given, and the net leaves in eval mode during
the search.
"""
import math

import numpy as np
import torch

from .core.encoding import canonical_actions, encode_batch, is_canonical, version_for_planes
from .core.engine import ACTION_SIZE, apply_unchecked, legal_actions


class Node:
    def __init__(self, s, prior=1):
        self.s = s
        self.prior = prior
        self.n = 0
        self.w = 0.
        self.children = {}

    @property
    def q(self):
        return self.w / self.n if self.n else 0


def _expand(net, nodes, device, encoding=1):
    """Batch-evaluate nodes, attach priors, return per-node values.

    Terminal nodes are skipped: with no legal actions there is nothing to softmax over,
    and the caller scores them from the game result instead.
    """
    live = [n for n in nodes if n.s.winner is None and legal_actions(n.s)]
    if not live:
        return [0.] * len(nodes)
    ss = [n.s for n in live]
    x = torch.from_numpy(encode_batch(ss, encoding)).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                enabled=device.type == 'cuda',
                                                dtype=torch.float16):
        logits, vals = net(x)
    vals = vals.float().cpu().tolist()
    for i, n in enumerate(live):
        acts = legal_actions(n.s)
        policy_acts = (canonical_actions(acts, n.s.player) if is_canonical(encoding)
                       else acts)
        z = logits[i, policy_acts].float().cpu().numpy()
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        n.children = {a: Node(apply_unchecked(n.s, a), float(pr))
                      for a, pr in zip(acts, p, strict=True)}
    return vals


def search(net, state, device, sims=64, c_puct=1.5, noise=True, encoding=None,
           max_plies=220, seed=None):
    """PUCT search for one position; the policy is in the engine's real action frame.

    A finished position (a winner, no legal actions, or already past `max_plies`)
    returns a zero policy. `encoding` defaults to what the network's stem expects.
    `seed` is honoured by the root noise, so a fixed seed makes a search reproducible;
    it is ignored when `noise` is False.
    """
    pi = np.zeros(ACTION_SIZE, np.float32)
    if state.winner is not None or state.ply >= max_plies:
        return pi
    enc = version_for_planes(net.planes) if encoding is None else encoding
    rng = np.random.default_rng(seed)
    was_training = net.training
    net.eval()
    try:
        root = Node(state)
        _expand(net, [root], device, enc)
        if noise and root.children:
            d = rng.dirichlet([.3] * len(root.children))
            for k, x in zip(root.children, d, strict=True):
                root.children[k].prior = .75 * root.children[k].prior + .25 * x
        for _ in range(sims):
            n = root
            path = []
            while n.children and n.s.ply < max_plies:
                a, ch = max(n.children.items(),
                            key=lambda kv: -kv[1].q + c_puct * kv[1].prior
                            * math.sqrt(n.n + 1) / (1 + kv[1].n))
                path.append(n)
                n = ch
            if n.s.winner is not None:
                v = -1.
            else:
                v = _expand(net, [n], device, enc)[0]
            n.n += 1
            n.w += v
            for p in reversed(path):
                v = -v
                p.n += 1
                p.w += v
        for a, ch in root.children.items():
            pi[a] = ch.n
        if pi.sum():
            pi /= pi.sum()
        return pi
    finally:
        net.train(was_training)