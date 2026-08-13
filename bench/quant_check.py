"""Does int8 quantisation pay off for the 128x10 net on a 2-core CPU?

Two questions, and the second one is the one that matters: how much faster, and does the
quantised net still play the same moves. A net that is 3x faster but disagrees with itself
on one move in ten is not a speedup, it is a different and weaker player.
"""
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
torch.set_num_threads(2)

from quoridor_ai.core.encoding import encode_batch
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.safe_loader import load_checkpoint

BATCH = 32


def positions(n, seed=0):
    """Mid-game positions: the opening is walled-off enough to be unrepresentative."""
    rng = np.random.default_rng(seed)
    out, s = [], State()
    while len(out) < n:
        acts = legal_actions(s)
        if not acts:
            s = State()
            continue
        s = apply_unchecked(s, int(rng.choice(acts)))
        if s.ply > 20:
            out.append(s)
        if s.winner is not None or s.ply > 120:
            s = State()
    return out


def bench(f, x, reps=8, rounds=3):
    """Best-of-N wall clock per position. Best-of, not mean: the machine has other work."""
    best = float('inf')
    with torch.inference_mode():
        for _ in range(rounds):
            f(x)
            t = time.perf_counter()
            for _ in range(reps):
                f(x)
            best = min(best, (time.perf_counter() - t) / reps / x.shape[0] * 1e6)
    return best


def quality(ref_p, ref_v, p, v):
    """Agreement with the float net, in the terms the search actually consumes."""
    ref_soft, soft = torch.softmax(ref_p, 1), torch.softmax(p, 1)
    kl = float((ref_soft * (torch.log(ref_soft + 1e-9) - torch.log(soft + 1e-9))).sum(1).mean())
    top1 = float((ref_p.argmax(1) == p.argmax(1)).float().mean())
    # Top-4 matters more than top-1: Gumbel search samples several root moves, so a
    # disagreement outside the top of the list still changes what gets explored.
    ref_top4 = ref_p.topk(4, dim=1).indices
    top4 = p.topk(4, dim=1).indices
    same4 = float(np.mean([len(set(a.tolist()) & set(b.tolist())) / 4 for a, b in zip(ref_top4, top4)]))
    return kl, top1, same4, float((ref_v - v).abs().max()), float((ref_v - v).abs().mean())


def main():
    dev = torch.device('cpu')
    net = net_from_checkpoint(load_checkpoint(r'runs\Checkpoints\best.pt', map_location=dev), dev)
    net.eval()

    st = positions(BATCH)
    calib = positions(256, seed=99)          # separate draw: calibrating on the test set flatters
    x = torch.from_numpy(encode_batch(st, 3)).to(memory_format=torch.channels_last)
    xc = torch.from_numpy(encode_batch(calib, 3)).to(memory_format=torch.channels_last)

    with torch.inference_mode():
        ref_p, ref_v = net(x)

    base = bench(net, x)
    print(f'float32 (как сейчас): {base:8.0f} мкс/поз   эталон')
    print()

    rows = []

    # --- dynamic: weights int8, activations stay float. Only Linear is supported. ---
    dyn = torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8).eval()
    with torch.inference_mode():
        p, v = dyn(x)
    rows.append(('динамическое (только Linear)', bench(dyn, x), quality(ref_p, ref_v, p, v)))

    # --- static FX graph mode: this is the one that can reach the convolutions. ---
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        torch.backends.quantized.engine = 'onednn'
        qmap = get_default_qconfig_mapping('onednn')
        prepared = prepare_fx(net, qmap, example_inputs=(x,))
        with torch.inference_mode():
            for i in range(0, xc.shape[0], BATCH):      # calibration pass
                prepared(xc[i:i + BATCH])
        qnet = convert_fx(prepared).eval()
        with torch.inference_mode():
            p, v = qnet(x)
        rows.append(('статическое int8 (свёртки)', bench(qnet, x), quality(ref_p, ref_v, p, v)))
    except Exception as e:
        print(f'статическое int8 не собралось: {type(e).__name__}: {str(e)[:200]}')

    # --- bfloat16: not quantisation, but the same trade and no calibration needed. ---
    try:
        with torch.inference_mode():
            def bf(t):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    return net(t)
            p, v = bf(x)
        rows.append(('bfloat16 (autocast)', bench(bf, x), quality(ref_p, ref_v, p.float(), v.float())))
    except Exception as e:
        print(f'bfloat16 не вышло: {str(e)[:120]}')

    print(f'{"способ":32s} {"мкс/поз":>9s} {"ускор":>6s} {"KL":>8s} {"топ-1":>7s} {"топ-4":>7s} {"ошибка оценки":>14s}')
    print('-' * 90)
    for name, t, (kl, top1, top4, vmax, vmean) in rows:
        print(f'{name:32s} {t:9.0f} {base/t:5.2f}x {kl:8.4f} {top1:6.1%} {top4:6.1%}  {vmean:.4f} (макс {vmax:.3f})')


if __name__ == '__main__':
    main()
