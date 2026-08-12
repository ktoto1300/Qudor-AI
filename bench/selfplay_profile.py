"""What fraction of self-play is the network, and what fraction is plain Python?

This is the number that decides whether a GPU is worth buying. A GPU only makes the
forward pass cheaper. Everything else - building the tree, walking it, the Gumbel
schedule, encoding positions - runs on the CPU no matter what card is in the box.
If the net is 90% of self-play, a fast card nearly doubles the run twice over; if it
is 50%, the ceiling is 2x however much money you spend.
"""
import json
import time
import warnings
from collections import Counter

import numpy as np
import torch

warnings.filterwarnings('ignore')
torch.set_num_threads(2)

from quoridor_ai import az_selfplay
from quoridor_ai.core.encoding import encode_batch
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.safe_loader import load_checkpoint

T = Counter()          # seconds by stage
N = Counter()          # call / position counts
SIZES = []


def timed_evaluate(net, nodes, device, encoding, canon):
    """Same body as az_selfplay._evaluate, with a stopwatch between the stages."""
    if not nodes:
        return
    N['calls'] += 1
    N['positions'] += len(nodes)
    SIZES.append(len(nodes))

    t = time.perf_counter()
    x = torch.from_numpy(encode_batch([n.s for n in nodes], encoding)).to(
        device, non_blocking=True, memory_format=torch.channels_last)
    T['encode'] += time.perf_counter() - t

    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                enabled=device.type == 'cuda', dtype=torch.float16):
        logits, values = net(x)
    T['net'] += time.perf_counter() - t

    t = time.perf_counter()
    logits = logits.float().cpu().numpy()
    values = values.float().cpu().numpy()
    for i, n in enumerate(nodes):
        az_selfplay._expand(n, logits[i], values[i], canon)
    T['expand'] += time.perf_counter() - t


def main():
    c = json.load(open(r'configs\colab_az_cpu.json', encoding='utf-8'))
    dev = torch.device('cpu')
    net = net_from_checkpoint(load_checkpoint(r'runs\az_15gb\best.pt', map_location=dev), dev)
    net.eval()

    az_selfplay._evaluate = timed_evaluate

    t0 = time.perf_counter()
    _, stats = az_selfplay.selfplay(
        net, dev, games=c['games'], encoding=3,
        sims=c['sims'], fast_sims=c['fast_sims'], full_frac=c['full_frac'],
        max_plies=c['max_plies'], temp_moves=c['temp_moves'],
        gumbel=c['gumbel'], gumbel_cap=c['gumbel_cap'], seed=1)
    total = time.perf_counter() - t0

    known = T['encode'] + T['net'] + T['expand']
    tree = total - known

    print(f'{c["games"]} partiy, {stats["samples"]} semplov, {stats["games"]} zaversheno')
    print(f'vsego: {total:.1f} s   ({total / max(stats["samples"], 1):.2f} s na sempl)')
    print()
    print(f'{"stadiya":22s} {"sekund":>8s} {"dolya":>7s}')
    print('-' * 40)
    for k, label in (('net', 'set (forward)'), ('encode', 'kodirovka'),
                     ('expand', 'razbor vyhoda'), (None, 'derevo MCTS (python)')):
        sec = tree if k is None else T[k]
        print(f'{label:22s} {sec:8.1f} {sec / total:6.1%}')
    print()
    print(f'vyzovov seti: {N["calls"]}   pozitsiy: {N["positions"]}   '
          f'sredniy batch: {N["positions"] / max(N["calls"], 1):.1f}')
    print(f'batch: min {min(SIZES)}  mediana {int(np.median(SIZES))}  max {max(SIZES)}')

    # Amdahl: what the whole iteration looks like if only the forward pass gets faster.
    print()
    print(f'{"esli set uskorit v":>20s} {"self-play":>10s} {"ot vsego":>9s}')
    print('-' * 42)
    for k in (2, 5, 10, 30, 100):
        new = total - T['net'] + T['net'] / k
        print(f'{k:18d}x {new:9.1f}s {total / new:8.2f}x')


if __name__ == '__main__':
    main()
