"""Is the cheap search's *training target* close to an expensive search's?

The move finally played matters less than the policy target pi, because pi is what the
network is trained on. A target that agrees with a strong search on the top move but
assigns zero to most legal actions teaches far less than a dense one.

Reference = plain PUCT at a large budget. Run this against a *trained* checkpoint: a random
network has no prior and no value signal, so every search degenerates and the comparison
says nothing.

    python bench/search_quality.py [checkpoint] [positions]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run without installing

import numpy as np, torch, quoridor_ai.az_selfplay as sp
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.safe_loader import load_checkpoint
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'local_cpu_az/best.pt'
NPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

torch.set_num_threads(8)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ck = load_checkpoint(CKPT, map_location=dev)
net = net_from_checkpoint(ck, dev)
net.eval()

COUNT = [0]
_real = sp._evaluate
def counting(n_, nodes, *a, **k):
    COUNT[0] += len(nodes)
    return _real(n_, nodes, *a, **k)
sp._evaluate = counting


def positions(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = State()
        for _ in range(int(rng.integers(6, 40))):
            acts = legal_actions(s)
            if not acts or s.winner is not None:
                break
            s = apply_unchecked(s, int(acts[rng.integers(len(acts))]))
        if s.winner is None and legal_actions(s):
            out.append(s)
    return out


def search(s, budget, gumbel, seed):
    """Returns (target over legal actions, exploring pick, greedy pick)."""
    root = sp.Node(s)
    sp._evaluate(net, [root], dev, 3, True)
    rng = np.random.default_rng(seed)
    if gumbel:
        g = sp._Game(rng, False)
        g.root = root
        g.sched = sp._Sched(root, rng, budget, 16)
        for _ in range(g.sched.budget):
            leaf, path = sp._select_gumbel(g, 220)
            if leaf is None:
                break
            if leaf.terminal is None:
                sp._evaluate(net, [leaf], dev, 3, True)
            sp._backup(path, leaf.terminal if leaf.terminal is not None else leaf.value)
        pi = sp._improved_policy(root)
        # winner() carries the Gumbel exploration noise (what self-play wants); argmax of
        # the improved policy is the greedy actor (what the arena and the UI want).
        return pi, g.sched.winner(root), int(np.argmax(pi))
    for _ in range(budget):
        leaf, path = sp._select(root, 1.65, 220)
        if leaf.terminal is None:
            sp._evaluate(net, [leaf], dev, 3, True)
        sp._backup(path, leaf.terminal if leaf.terminal is not None else leaf.value)
    v = root.n.astype(np.float64)
    v = v / v.sum() if v.sum() > 0 else root.p.astype(np.float64)
    return v.astype(np.float32), int(np.argmax(v)), int(np.argmax(v))


POS = positions(NPOS, seed=3)
print(f'net: {CKPT} gen{ck.get("generation", "?")} on {dev}, {len(POS)} positions, '
      f'avg {np.mean([len(legal_actions(s)) for s in POS]):.1f} legal actions\n')

COUNT[0] = 0
ref = [search(s, 400, False, 99) for s in POS]
print(f'reference PUCT 400 sims: {COUNT[0] / len(POS):.0f} net evals/move\n')
print(f'{"":14s} {"top-1":>7s} {"greedy":>7s} {"KL(ref|pi)":>11s} {"support":>8s} {"evals":>7s}')

for label, budget, gum in (('PUCT   24', 24, False), ('PUCT   48', 48, False),
                           ('PUCT  128', 128, False), ('PUCT  192', 192, False),
                           ('Gumbel 24', 24, True), ('Gumbel 48', 48, True)):
    COUNT[0] = 0
    top = greedy = kl = sup = 0.0
    for i, s in enumerate(POS):
        pi, pick, gpick = search(s, budget, gum, 1000 + i)
        rp, rpick, _ = ref[i]
        top += pick == rpick
        greedy += gpick == rpick
        m = rp > 0
        kl += float((rp[m] * np.log(rp[m] / np.maximum(pi[m], 1e-9))).sum())
        sup += float((pi > 1e-6).sum())
    n = len(POS)
    print(f'{label} sims: {top / n:6.1%} {greedy / n:7.1%} {kl / n:11.3f} '
          f'{sup / n:8.1f} {COUNT[0] / n:7.1f}')
