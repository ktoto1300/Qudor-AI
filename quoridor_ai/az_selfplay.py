"""AlphaZero self-play: MCTS visit counts as the policy target.

The previous selfplay.py trained the network on its own temperature-scaled softmax. That
is self-distillation: the target contains no information the network did not already have,
so the policy can only sharpen its existing opinions, never correct them. Search is what
supplies new information - a visit distribution after N simulations is strictly better than
the prior that seeded it, and that gap is the entire learning signal in AlphaZero.

Three efficiency devices make this affordable on one GPU:

  * Many games are searched at once. Every tree contributes one leaf per round, so the
    network sees a batch of `len(active games)` positions per forward pass instead of one.
  * Playout cap randomisation (KataGo): most moves get a cheap search and are played but
    NOT recorded; a minority get the full search and become training samples. Game length
    and sample quality are decoupled, which buys roughly 2-3x more games per hour at equal
    target quality.
  * Tree reuse: the subtree under the chosen move becomes the next root, so the previous
    search is not thrown away.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .core.encoding import MIRROR, encode_batch, is_canonical
from .core.engine import State, ACTION_SIZE, apply_unchecked, legal_actions


class Node:
    """One search node: a state plus per-edge statistics as parallel numpy arrays.

    Arrays rather than per-child objects because selection is an argmax over all edges and
    Quoridor branching runs to ~130 actions; a numpy argmax there is worth ~10x a Python loop.
    """
    __slots__ = ('s', 'acts', 'cacts', 'p', 'n', 'w', 'kids', 'total', 'terminal', 'value')

    def __init__(self, s: State):
        self.s = s
        self.acts = None        # real action ids, index-aligned with the stat arrays
        self.cacts = None       # the same actions in the canonical (encoder) frame
        self.p = self.n = self.w = None
        self.kids = None
        self.total = 0          # sum of n, kept incrementally
        self.terminal = None    # value from this node's mover's view once known
        self.value = 0.0        # network value of this node, for the value target


def _terminal_value(s: State, max_plies: int):
    """Value from the perspective of the player to move, or None if the game continues."""
    if s.winner is not None:
        return -1.0             # the mover is facing a board the opponent already won on
    if s.ply >= max_plies:
        return 0.0              # ply cap: scored as a draw, same as the trainer and the UI
    return None


def _expand(node: Node, logits_row, value, canon: bool):
    """Attach priors to a node from one row of network output."""
    acts = legal_actions(node.s)
    node.acts = acts
    if not acts:
        node.terminal = 0.0
        return
    cacts = [int(MIRROR[a]) for a in acts] if (canon and node.s.player == 1) else list(acts)
    node.cacts = cacts
    z = logits_row[cacts]
    z = z - z.max()
    p = np.exp(z)
    node.p = (p / p.sum()).astype(np.float32)
    k = len(acts)
    node.n = np.zeros(k, np.int32)
    node.w = np.zeros(k, np.float32)
    node.kids = [None] * k
    node.value = float(value)


def _select(root: Node, c_puct: float, max_plies: int):
    """Walk from the root to an unexpanded (or terminal) node, returning the edge path."""
    node, path = root, []
    while True:
        if node.terminal is not None or node.acts is None:
            return node, path
        # PUCT. Unvisited edges get Q=0, which in a [-1,1] value scale means "even" - the
        # standard AlphaZero choice and the reason first-play-urgency is not needed here.
        q = np.where(node.n > 0, node.w / np.maximum(node.n, 1), 0.0)
        u = c_puct * node.p * math.sqrt(node.total + 1) / (1 + node.n)
        i = int(np.argmax(q + u))
        kid = node.kids[i]
        if kid is None:
            kid = Node(apply_unchecked(node.s, node.acts[i]))
            kid.terminal = _terminal_value(kid.s, max_plies)
            node.kids[i] = kid
        path.append((node, i))
        node = kid
        if node.terminal is not None or node.acts is None:
            return node, path


def _backup(path, v: float):
    """Propagate a leaf value up the path, flipping sign at every ply."""
    for node, i in reversed(path):
        v = -v
        node.n[i] += 1
        node.w[i] += v
        node.total += 1


class _Game:
    """One self-play game in flight."""

    def __init__(self, rng, resign: bool):
        self.root = Node(State())
        self.samples = []       # (state, pi, root_q, player)
        self.rng = rng
        self.result = None      # +1 / -1 / 0 from player 0's perspective
        self.resign = resign
        self.bad_streak = 0
        self.sims_left = 0
        self.full = False


def _root_noise(node: Node, rng, frac: float, alpha_scale: float):
    """Dirichlet noise on the root prior - the only source of exploration at the top."""
    k = len(node.acts)
    alpha = max(0.12, alpha_scale / k)
    node.p = ((1 - frac) * node.p + frac * rng.dirichlet([alpha] * k)).astype(np.float32)


def _evaluate(net, nodes, device, encoding, canon):
    """One batched forward pass; expands every node in place."""
    if not nodes:
        return
    x = torch.from_numpy(encode_batch([n.s for n in nodes], encoding)).to(
        device, non_blocking=True, memory_format=torch.channels_last)
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                enabled=device.type == 'cuda', dtype=torch.float16):
        logits, values = net(x)
    logits = logits.float().cpu().numpy()
    values = values.float().cpu().numpy()
    for i, n in enumerate(nodes):
        _expand(n, logits[i], values[i], canon)


def selfplay(net, device, games=64, encoding=3, sims=200, fast_sims=50, full_frac=0.25,
             c_puct=1.6, max_plies=220, temp_moves=20, temp=1.0, noise_frac=0.25,
             alpha_scale=10.0, resign_v=-0.95, resign_streak=4, resign_skip=0.1,
             seed=0, progress=None):
    """Play `games` games with MCTS and return (samples, stats).

    Each sample is (encoded_state, pi, z, q): pi is the normalised root visit distribution in
    the canonical action frame, z is the final result from that position's mover's view, and q
    is the root's search value. The trainer blends z and q - z is unbiased but extremely noisy
    at 200 plies of credit assignment, q is biased toward the current net but low variance.
    """
    canon = is_canonical(encoding)
    rng = np.random.default_rng(seed)
    live = [_Game(np.random.default_rng(seed * 7919 + i), rng.random() >= resign_skip)
            for i in range(games)]
    done = []

    _evaluate(net, [g.root for g in live], device, encoding, canon)
    for g in live:
        g.full = g.rng.random() < full_frac
        g.sims_left = sims if g.full else fast_sims
        if g.full and g.root.acts:
            _root_noise(g.root, g.rng, noise_frac, alpha_scale)

    while live:
        # --- search phase: every live game contributes one leaf to a shared batch ---
        rounds = max(g.sims_left for g in live)
        for _ in range(rounds):
            pending, paths = [], []
            for g in live:
                if g.sims_left <= 0:
                    continue
                g.sims_left -= 1
                leaf, path = _select(g.root, c_puct, max_plies)
                if leaf.terminal is not None:
                    _backup(path, leaf.terminal)
                else:
                    pending.append(leaf)
                    paths.append(path)
            _evaluate(net, pending, device, encoding, canon)
            for leaf, path in zip(pending, paths):
                _backup(path, leaf.terminal if leaf.terminal is not None else leaf.value)

        # --- move phase ---
        still = []
        for g in live:
            root = g.root
            visits = root.n.astype(np.float64)
            if visits.sum() <= 0:                       # search never left the root
                visits = root.p.astype(np.float64)
            q_root = float(root.w.sum() / max(1, root.total))

            if g.full:
                pi = np.zeros(ACTION_SIZE, np.float32)
                pi[root.cacts] = (visits / visits.sum()).astype(np.float32)
                g.samples.append((root.s, pi, q_root, root.s.player))

            # Temperature only early: late-game randomness just throws away won positions.
            if root.s.ply < temp_moves and temp > 0:
                probs = visits ** (1.0 / temp)
                probs /= probs.sum()
                i = int(g.rng.choice(len(probs), p=probs))
            else:
                i = int(np.argmax(visits))

            # Resignation. A game whose search value stays hopeless for several full
            # searches in a row is decided; playing it out only burns simulations.
            if g.resign and g.full and q_root < resign_v:
                g.bad_streak += 1
            elif g.full:
                g.bad_streak = 0
            if g.bad_streak >= resign_streak:
                g.result = -1.0 if root.s.player == 0 else 1.0
                done.append(g)
                continue

            kid = root.kids[i]
            if kid is None:
                kid = Node(apply_unchecked(root.s, root.acts[i]))
                kid.terminal = _terminal_value(kid.s, max_plies)
            if kid.terminal is not None:
                # kid.terminal is from kid's mover's view; +1 means the player who just
                # moved won, i.e. root.s.player.
                if kid.s.winner is not None:
                    g.result = 1.0 if kid.s.winner == 0 else -1.0
                else:
                    g.result = 0.0
                done.append(g)
                continue

            g.root = kid                                 # tree reuse
            g.full = g.rng.random() < full_frac
            g.sims_left = (sims if g.full else fast_sims) - int(kid.total)
            if kid.acts is None:
                _evaluate(net, [kid], device, encoding, canon)
            if g.full and kid.acts:
                _root_noise(kid, g.rng, noise_frac, alpha_scale)
            if g.sims_left <= 0:
                g.sims_left = 1
            still.append(g)
        live = still
        if progress:
            progress(len(done), games)

    out, lengths = [], []
    for g in done:
        lengths.append(g.samples[-1][0].ply if g.samples else 0)
        for s, pi, q, player in g.samples:
            z = g.result if player == 0 else -g.result
            out.append((encode_batch([s], encoding)[0], pi, float(z), float(q)))
    stats = {'games': len(done), 'samples': len(out),
             'avg_plies': float(np.mean(lengths)) if lengths else 0.0,
             'p0_wins': sum(1 for g in done if g.result > 0),
             'draws': sum(1 for g in done if g.result == 0)}
    return out, stats
