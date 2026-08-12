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
from types import SimpleNamespace

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


# --- Gumbel AlphaZero -----------------------------------------------------------
# Danihelka et al., "Policy improvement by planning with Gumbel" (ICLR 2022).
#
# Plain AlphaZero needs a fairly large simulation budget before the visit counts
# become a usable policy target. Measured over random mid-game Quoridor positions the
# branching factor is ~46, so at 24 simulations most edges are never visited once and
# the target degenerates into a spiky histogram that carries almost no information.
# Gumbel replaces three pieces to fix exactly that:
#
#   * Root candidates are drawn by Gumbel-Top-k (sampling without replacement)
#     instead of the prior being perturbed with Dirichlet noise - the exploration
#     lives in the draw itself.
#   * The budget is spent by Sequential Halving over those candidates rather than by
#     PUCT, so it is spread evenly instead of piling onto one edge.
#   * The target is the *completed* improved policy softmax(logits + sigma(Q)),
#     which gives every legal action a probability, including ones search never
#     touched.
#
# That last point is what makes tiny budgets work at all: the paper trains from 2
# simulations where plain MuZero still fails at 16.

_C_VISIT = 50.0      # paper defaults; sigma only has to be monotone in q
_C_SCALE = 1.0


def _sigma(q, max_n: int):
    return (_C_VISIT + max_n) * _C_SCALE * q


def _completed_q(node: Node):
    """Q for every edge, with never-visited edges filled in by v_mix.

    v_mix interpolates between the network's own value for the node and the
    prior-weighted average Q over the edges search did visit, with the weight
    shifting toward search as visits accumulate.
    """
    n, w, p = node.n, node.w, node.p
    seen = n > 0
    q = np.zeros(len(n), np.float32)
    np.divide(w, np.maximum(n, 1), out=q, where=seen)
    total = int(node.total)
    v_mix = node.value
    if total > 0:
        ps = p[seen]
        s = float(ps.sum())
        if s > 1e-9:
            v_mix = (node.value + total * float((ps * q[seen]).sum()) / s) / (1.0 + total)
    q[~seen] = v_mix
    return q


def _improved_policy(node: Node):
    """softmax(logits + sigma(completedQ)) over this node's legal actions."""
    z = np.log(np.maximum(node.p, 1e-9))
    z = z + _sigma(_completed_q(node), int(node.n.max()) if node.n.size else 0)
    z -= z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def _considered(k: int, budget: int, cap: int):
    """Largest power-of-two candidate count the budget can actually halve.

    Sequential Halving over m candidates needs m*log2(m) visits to give each one at
    least one visit per phase, so m has to shrink when the budget is small.
    """
    m = 1
    while m * 2 <= min(k, cap):
        m *= 2
    while m > 1 and m * math.ceil(math.log2(m)) > budget:
        m //= 2
    return max(1, m)


def _descend(node: Node, max_plies: int, path):
    """Walk below the root, keeping visit proportions on track with pi'."""
    while node.terminal is None and node.acts is not None:
        pi = _improved_policy(node)
        i = int(np.argmax(pi - node.n / (1.0 + node.total)))
        kid = node.kids[i]
        if kid is None:
            kid = Node(apply_unchecked(node.s, node.acts[i]))
            kid.terminal = _terminal_value(kid.s, max_plies)
            node.kids[i] = kid
        path.append((node, i))
        node = kid
    return node


class _Sched:
    """Sequential Halving schedule for one root."""

    __slots__ = ('cands', 'score', 'per', 'seen', 'budget', 'phases_left')

    def __init__(self, node: Node, rng, budget: int, cap: int):
        k = len(node.acts)
        # Gumbel-Top-k: argtop(g + logits) is an exact sample of k distinct actions
        # drawn without replacement from softmax(logits).
        self.score = (rng.gumbel(size=k) + np.log(np.maximum(node.p, 1e-9))).astype(np.float32)
        m = _considered(k, budget, cap)
        self.cands = np.argsort(-self.score)[:m].copy()
        self.budget = max(budget, m)
        self.phases_left = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        self._phase()

    def _phase(self):
        self.per = max(1, self.budget // max(1, self.phases_left * len(self.cands)))
        self.seen = np.zeros(len(self.cands), np.int32)

    def _halve(self, node: Node):
        keep = max(1, len(self.cands) // 2)
        self.cands = self.cands[np.argsort(-self._rank(node))[:keep]].copy()
        self.phases_left = max(1, self.phases_left - 1)
        self._phase()

    def _rank(self, node: Node):
        return self.score[self.cands] + _sigma(_completed_q(node)[self.cands],
                                               int(node.n.max()))

    def next(self, node: Node):
        """Edge index to visit next, or None once the budget is spent."""
        while self.budget > 0:
            if len(self.cands) == 1:
                self.budget -= 1
                return int(self.cands[0])
            j = int(np.argmin(self.seen))
            if self.seen[j] >= self.per:
                self._halve(node)
                continue
            self.seen[j] += 1
            self.budget -= 1
            return int(self.cands[j])
        return None

    def winner(self, node: Node):
        if len(self.cands) == 1:
            return int(self.cands[0])
        return int(self.cands[int(np.argmax(self._rank(node)))])


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
        self.sched = None        # Sequential Halving state, Gumbel mode only
        self.final_ply = 0       # actual game length; recorded samples are only a subset of moves


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


def _select_gumbel(g: _Game, max_plies: int):
    """One Sequential-Halving simulation: the root edge is scheduled, the rest is greedy."""
    root = g.root
    i = g.sched.next(root)
    if i is None:
        return None, None
    kid = root.kids[i]
    if kid is None:
        kid = Node(apply_unchecked(root.s, root.acts[i]))
        kid.terminal = _terminal_value(kid.s, max_plies)
        root.kids[i] = kid
    path = [(root, i)]
    return _descend(kid, max_plies, path), path


def search(net, s: State, device, encoding=3, sims=64, gumbel=True, gumbel_cap=16,
           c_puct=1.6, max_plies=220, seed=None):
    """Search one position. Returns (actions, probabilities, root value).

    Self-play and the arena batch many games through each forward pass because they have
    many in flight; an interactive caller has exactly one position, so this trades that
    throughput for a single call. The distribution returned is the same quantity the
    trainer learns from - the improved policy under Gumbel, normalised visit counts under
    PUCT - so a UI showing it is showing what the network is actually taught rather than a
    separate display heuristic.

    Unlike self-play there is no Dirichlet noise and, in Gumbel mode, the caller is
    expected to take the argmax rather than the Gumbel winner. Both exist to explore, and
    an opponent that explores is just an opponent playing below its strength.
    """
    root = Node(s)
    canon = is_canonical(encoding)
    _evaluate(net, [root], device, encoding, canon)
    if not root.acts:
        return [], np.zeros(0, np.float32), 0.0

    holder = SimpleNamespace(root=root, sched=None)
    if gumbel:
        holder.sched = _Sched(root, np.random.default_rng(seed), sims, gumbel_cap)
        rounds = holder.sched.budget      # next() consumes budget, so read it before looping
    else:
        rounds = sims

    for _ in range(rounds):
        if gumbel:
            leaf, path = _select_gumbel(holder, max_plies)
            if leaf is None:              # halving schedule exhausted early
                break
        else:
            leaf, path = _select(root, c_puct, max_plies)
        if leaf.terminal is None:
            _evaluate(net, [leaf], device, encoding, canon)
        _backup(path, leaf.terminal if leaf.terminal is not None else leaf.value)

    if gumbel:
        probs = _improved_policy(root).astype(np.float64)
    else:
        probs = root.n.astype(np.float64)
        if probs.sum() <= 0:              # every simulation hit a terminal edge
            probs = root.p.astype(np.float64)
    probs = probs / probs.sum()
    return list(root.acts), probs.astype(np.float32), float(root.w.sum() / max(1, root.total))


def selfplay(net, device, games=64, encoding=3, sims=200, fast_sims=50, full_frac=0.25,
             c_puct=1.6, max_plies=220, temp_moves=20, temp=1.0, noise_frac=0.25,
             alpha_scale=10.0, resign_v=-0.95, resign_streak=4, resign_skip=0.1,
             seed=0, progress=None, gumbel=False, gumbel_cap=16):
    """Play `games` games with MCTS and return (samples, stats).

    Each sample is (encoded_state, pi, z, q): pi is the improved root policy in the
    canonical action frame, z is the final result from that position's mover's view, and q
    is the root's search value. The trainer blends z and q - z is unbiased but extremely noisy
    at 200 plies of credit assignment, q is biased toward the current net but low variance.

    With `gumbel=True` the root uses Gumbel-Top-k + Sequential Halving and pi is the
    completed improved policy; `sims` can then be an order of magnitude smaller. With
    `gumbel=False` it is textbook AlphaZero: PUCT everywhere, visit counts as the target.
    """
    canon = is_canonical(encoding)
    rng = np.random.default_rng(seed)
    live = [_Game(np.random.default_rng(seed * 7919 + i), rng.random() >= resign_skip)
            for i in range(games)]
    done = []

    def _arm(g, node):
        """Pick this move's playout budget and set up its search state."""
        g.full = g.rng.random() < full_frac
        budget = sims if g.full else fast_sims
        if not node.acts:               # stalemate; Quoridor's rules forbid it
            g.sims_left, g.sched = 0, None
            return
        if gumbel:
            # Visits carried over by tree reuse stay in the tree and sharpen Q, but the
            # halving schedule is drawn fresh - these are new Gumbel candidates.
            g.sched = _Sched(node, g.rng, budget, gumbel_cap)
            g.sims_left = g.sched.budget
        else:
            g.sims_left = max(1, budget - int(node.total))
            if g.full:
                _root_noise(node, g.rng, noise_frac, alpha_scale)

    _evaluate(net, [g.root for g in live], device, encoding, canon)
    for g in live:
        _arm(g, g.root)

    while live:
        # --- search phase: every live game contributes one leaf to a shared batch ---
        rounds = max(g.sims_left for g in live)
        for _ in range(rounds):
            pending, paths = [], []
            for g in live:
                if g.sims_left <= 0:
                    continue
                g.sims_left -= 1
                if gumbel:
                    leaf, path = _select_gumbel(g, max_plies)
                    if leaf is None:            # halving schedule exhausted early
                        g.sims_left = 0
                        continue
                else:
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
            if not root.acts:                   # stalemate; Quoridor's rules forbid it
                g.result = 0.0
                g.final_ply = root.s.ply
                done.append(g)
                continue
            q_root = float(root.w.sum() / max(1, root.total))

            if gumbel:
                target = _improved_policy(root)
                i = g.sched.winner(root)
            else:
                visits = root.n.astype(np.float64)
                if visits.sum() <= 0:                       # search never left the root
                    visits = root.p.astype(np.float64)
                target = (visits / visits.sum()).astype(np.float32)
                # Temperature only early: late randomness just throws away won positions.
                if root.s.ply < temp_moves and temp > 0:
                    probs = visits ** (1.0 / temp)
                    probs /= probs.sum()
                    i = int(g.rng.choice(len(probs), p=probs))
                else:
                    i = int(np.argmax(visits))

            if g.full:
                pi = np.zeros(ACTION_SIZE, np.float32)
                pi[root.cacts] = target
                g.samples.append((root.s, pi, q_root, root.s.player))

            # Resignation. A game whose search value stays hopeless for several full
            # searches in a row is decided; playing it out only burns simulations.
            if g.resign and g.full and q_root < resign_v:
                g.bad_streak += 1
            elif g.full:
                g.bad_streak = 0
            if g.bad_streak >= resign_streak:
                g.result = -1.0 if root.s.player == 0 else 1.0
                g.final_ply = root.s.ply
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
                g.final_ply = kid.s.ply
                done.append(g)
                continue

            g.root = kid                                 # tree reuse
            still.append(g)

        # One batched forward pass for every new root, instead of one call per game.
        _evaluate(net, [g.root for g in still if g.root.acts is None], device, encoding, canon)
        for g in still:
            _arm(g, g.root)
        live = still
        if progress:
            progress(len(done), games)

    out, lengths = [], []
    for g in done:
        lengths.append(g.final_ply)
        for s, pi, q, player in g.samples:
            z = g.result if player == 0 else -g.result
            out.append((encode_batch([s], encoding)[0], pi, float(z), float(q)))
    stats = {'games': len(done), 'samples': len(out),
             'avg_plies': float(np.mean(lengths)) if lengths else 0.0,
             'p0_wins': sum(1 for g in done if g.result > 0),
             'draws': sum(1 for g in done if g.result == 0)}
    return out, stats
