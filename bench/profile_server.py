"""Profile self-play encoding, network inference, expansion and tree overhead."""
import argparse
import json
import time
from collections import Counter

import numpy as np
import torch

from quoridor_ai import az_selfplay
from quoridor_ai.core.encoding import encode_batch
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.safe_loader import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--games', type=int, default=32)
    parser.add_argument('--concurrent-games', type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    net = net_from_checkpoint(load_checkpoint(args.checkpoint, map_location=device), device).eval()
    times, counts, sizes = Counter(), Counter(), []
    original_expand = az_selfplay._expand

    def evaluate(net, nodes, device, encoding, canon):
        if not nodes:
            return
        counts['calls'] += 1
        counts['positions'] += len(nodes)
        sizes.append(len(nodes))
        start = time.perf_counter()
        encoded = encode_batch([node.s for node in nodes], encoding)
        times['encode'] += time.perf_counter() - start
        start = time.perf_counter()
        x = torch.from_numpy(encoded).to(device, non_blocking=True,
                                         memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                     enabled=device.type == 'cuda',
                                                     dtype=torch.float16):
            logits, values = net(x)
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy()
        times['net'] += time.perf_counter() - start
        start = time.perf_counter()
        for i, node in enumerate(nodes):
            original_expand(node, logits[i], values[i], canon)
        times['expand'] += time.perf_counter() - start

    az_selfplay._evaluate = evaluate
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    _, stats = az_selfplay.selfplay(
        net, device, games=args.games, encoding=3, sims=24, fast_sims=8,
        full_frac=0.5, max_plies=80, temp_moves=24, resign_v=-0.95,
        gumbel=True, gumbel_cap=16, seed=20260814,
        concurrent_games=args.concurrent_games,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    known = sum(times.values())
    print(json.dumps({
        'games': stats['games'], 'samples': stats['samples'], 'seconds': total,
        'games_per_sec': stats['games'] / total,
        'positions_per_sec': stats['samples'] / total,
        'encode_seconds': times['encode'], 'net_seconds': times['net'],
        'expand_seconds': times['expand'], 'tree_and_output_seconds': total - known,
        'calls': counts['calls'], 'evaluated_positions': counts['positions'],
        'batch_mean': counts['positions'] / counts['calls'],
        'batch_median': float(np.median(sizes)), 'batch_max': max(sizes),
        'threads': args.threads, 'concurrent_games': args.concurrent_games or args.games,
    }, indent=2))


if __name__ == '__main__':
    main()
