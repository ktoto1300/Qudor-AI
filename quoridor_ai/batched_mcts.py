"""Batched PUCT MCTS, one tree per state, for the interactive viewer.

A lighter variant of the training search: no Gumbel trick, no Sequential Halving - just
PUCT selection with batch-evaluated leaves. It exists so the viewer can search several
positions at once on a GPU that would be idle for a single-board search.
"""
import math

import numpy as np
import torch

from .core.encoding import canonical_actions, encode_batch, is_canonical
from .core.engine import ACTION_SIZE, apply_unchecked, legal_actions


class Node:
    def __init__(self, s, p=1.):
        self.s = s
        self.p = p
        self.n = 0
        self.w = 0.
        self.children = {}

    @property
    def q(self):
        return self.w / self.n if self.n else 0.


def _expand(net, nodes, device, encoding=1):
    """Batch-evaluate nodes, attach priors as children, return {id(node): value}.

    Terminal nodes (and any node with no legal action) are dropped before the batch:
    softmax over an empty action list raises, so they must never reach the network.
    Their value is decided by the caller from the game result instead.
    """
    live = [n for n in nodes if n.s.winner is None and legal_actions(n.s)]
    if not live:
        return {}
    x = torch.from_numpy(encode_batch([n.s for n in live], encoding)).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                enabled=device.type == 'cuda',
                                                dtype=torch.float16):
        logits, values = net(x)
    vals = values.float().cpu().numpy()
    out = {}
    for i, n in enumerate(live):
        acts = legal_actions(n.s)
        policy_acts = (canonical_actions(acts, n.s.player) if is_canonical(encoding)
                       else acts)
        z = logits[i, policy_acts].float().cpu().numpy()
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        kids = [Node(apply_unchecked(n.s, a), float(v))
                for a, v in zip(acts, p, strict=True)]
        n.children = dict(zip(acts, kids, strict=True))
        n._child_nodes = kids
        n._p = p.astype(np.float64)
        n._child_n = np.zeros(len(kids), dtype=np.float64)
        n._child_w = np.zeros(len(kids), dtype=np.float64)
        out[id(n)] = float(vals[i])
    return out


def batched_search(net, states, device, sims=64, c_puct=1.5, encoding=1,
                   max_plies=220):
    """Batched MCTS with PUCT selection, one tree per state.

    encoding: encoder version matching the network's input planes; default 1 for
              backward compatibility. The returned policy is always in the engine's
              real action frame. Search stops a node at `max_plies` plies and scores
              it a draw, so a position past the cap (or an empty list of states) yields
              a zero policy instead of a crash.
    """
    if not states:
        return np.zeros((0, ACTION_SIZE), np.float32)
    # Training mode is the caller's default for nets that were just created: BatchNorm
    # then normalises over the inference batch and Dropout crops the priors. eval()
    # restores after the search, since the same net object is often trained afterwards.
    was_training = net.training
    net.eval()
    try:
        roots = [Node(s) for s in states]
        # A root already at the cap is a draw and never searches: it gets no children,
        # so its policy stays zero and no network call is spent on it.
        _expand(net, [r for r in roots if r.s.ply < max_plies], device, encoding)
        for _ in range(sims):
            paths = []
            for root in roots:
                # The root belongs in the path: it needs its visit count incremented
                # too, otherwise the exploration term sqrt(parent.n+1) stays pinned at
                # 1 for every root child.
                path = [root]
                child_indices = []
                n = root
                while n.children and n.s.ply < max_plies:
                    if getattr(n, '_child_n', None) is None or len(n._child_n) != len(n.children):
                        kids = list(n.children.values())
                        n._child_nodes = kids
                        n._p = np.array([ch.p for ch in kids], dtype=np.float64)
                        n._child_n = np.array([ch.n for ch in kids], dtype=np.float64)
                        n._child_w = np.array([ch.w for ch in kids], dtype=np.float64)
                    q = np.where(n._child_n > 0, -n._child_w / np.maximum(n._child_n, 1), 0.0)
                    u = c_puct * n._p * (math.sqrt(n.n + 1) / (1.0 + n._child_n))
                    idx = int(np.argmax(q + u))
                    child_indices.append(idx)
                    n = n._child_nodes[idx]
                    path.append(n)
                paths.append((path, child_indices))
            vals = _expand(net, [p[0][-1] for p in paths], device, encoding)
            for path, child_indices in paths:
                leaf = path[-1]
                if leaf.s.winner is not None:
                    v = -1.                     # the side to move there has already lost
                elif leaf.s.ply >= max_plies:
                    v = 0.                      # the game is called a draw
                else:
                    v = vals.get(id(leaf), 0.)
                leaf.n += 1
                leaf.w += v
                v = -v
                for parent, idx in zip(reversed(path[:-1]), reversed(child_indices), strict=True):
                    parent.n += 1
                    parent.w += v
                    if getattr(parent, '_child_n', None) is not None:
                        parent._child_n[idx] += 1
                        parent._child_w[idx] += -v
                    v = -v
        out = []
        for root in roots:
            pi = np.zeros(ACTION_SIZE, np.float32)
            for a, n in root.children.items():
                pi[a] = n.n
            if pi.sum():
                pi /= pi.sum()
            out.append(pi)
        return np.stack(out)
    finally:
        net.train(was_training)