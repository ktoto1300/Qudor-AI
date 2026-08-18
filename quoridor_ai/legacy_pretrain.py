"""Pretrain a fresh net on the 7-plane legacy replays.

The legacy samples were collected by the old, pre-bitboard self-play loop, and two of
their features made them dangerous to feed to the current encoder:

1. Wall planes 5/6 are h/v-swapped: plane 5 holds vertical walls and plane 6 holds
   horizontal walls, the opposite of encode_v1. This was verified empirically - the
   swapped layout re-parses into consistent, reachable positions for ~98% of the
   parseable samples while the transposed layout fails for ~97% of them. The loader
   swaps the planes back and re-parses the 2-cell wall paintings into slots instead of
   trusting the raw pixels, so a position whose walls do not add up is skipped rather
   than fed to the network.

2. Every value target in the legacy data is exactly 0.0: the old loop never recorded
   game results. Training the value head on that would tell the net that every game is
   a draw - actively wrong instead of neutral. When no non-zero target exists the value
   term is dropped and the run warns loudly.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .core.encoding import PLANES_BY_VERSION, encode_v1
from .core.engine import State
from .model import PolicyValueNet
from .safe_loader import load_checkpoint

LEGACY_PLANES = 7

# Legacy dumps only carry planes 0..6. Planes 8..10 of v1 are recoverable - plane 8 is
# 1-player (the constant plane 2 holds `player`), planes 9/10 are the fixed goal rows -
# so they are reconstructed rather than left as zeros; feeding the net a plane pattern
# that encode_v1 never produces would put pretraining off-distribution. Plane 7
# (ply/200) is genuinely unknown here and stays zero.


def _wall_slots(plane, vertical: bool):
    """Slot indices out of one legacy wall plane, or None for an unpaired cell.

    Legacy walls are painted exactly like encode_v1's: a vertical wall slot (r,c) marks
    cells (r,c) and (r+1,c), a horizontal one marks (r,c) and (r,c+1). The old ruleset
    allowed walls to touch, which merges neighbouring pairs into longer runs; each
    painted adjacency still contributes exactly one wall, so scanning adjacency pairs
    recovers the wall set even when the plane shows a 3+ cell run. A cell that is not
    part of any painted pair is an artifact (the legacy data has a few) and refuses the
    sample rather than letting a phantom wall through.
    """
    a = plane > 0.5
    if vertical:
        peers = a[:-1, :] & a[1:, :]
        covered = np.zeros_like(a)
        covered[:-1, :] |= peers
        covered[1:, :] |= peers
        out = [r * 8 + c for r in range(8) for c in range(9) if peers[r, c]]
    else:
        peers = a[:, :-1] & a[:, 1:]
        covered = np.zeros_like(a)
        covered[:, :-1] |= peers
        covered[:, 1:] |= peers
        out = [r * 8 + c for r in range(9) for c in range(8) if peers[r, c]]
    if (a & ~covered).any():
        return None
    return out


def _constant_plane(plane: np.ndarray) -> float | None:
    """The value of a constant plane, None if it is not constant."""
    v = float(plane.reshape(-1)[0])
    return v if np.allclose(plane, v) else None


def _walls_left(value: float | None) -> int | None:
    """A walls/10 plane as an integer wall count, None if it is not constant/even."""
    if value is None:
        return None
    n = round(value * 10)
    return n if abs(value * 10 - n) < 1e-3 else None


def lift_sample(src) -> State | None:
    """A legacy 7-plane sample as a `State`, or None when its walls do not parse.

    Returning the parsed `State` instead of a lifted tensor lets this module refuse
    corrupted positions before any network sees them: a wall plane with an unpaired
    cell, a pawn plane that is not one-hot, or a wall count that contradicts the
    constant planes cannot be a real game.
    """
    src = np.asarray(src, dtype=np.float32)
    if src.shape != (LEGACY_PLANES, 9, 9):
        return None
    if int(src[0].sum()) != 1 or int(src[1].sum()) != 1:
        return None
    p0 = int(np.argmax(src[0]))
    p1 = int(np.argmax(src[1]))
    player = _constant_plane(src[2])
    walls0 = _walls_left(_constant_plane(src[3]))
    walls1 = _walls_left(_constant_plane(src[4]))
    if player is None or walls0 is None or walls1 is None:
        return None
    if player not in (0, 1):
        return None
    player = int(player)
    v_slots = _wall_slots(src[5], vertical=True)      # legacy plane 5 = vertical walls
    h_slots = _wall_slots(src[6], vertical=False)     # legacy plane 6 = horizontal walls
    if v_slots is None or h_slots is None:
        return None
    if len(v_slots) + len(h_slots) != 20 - walls0 - walls1:
        return None
    h = 0
    for i in h_slots:
        h |= 1 << i
    v = 0
    for i in v_slots:
        v |= 1 << i
    return State(p0, p1, walls0, walls1, h, v, player, 0)


def load_legacy(folder, limit=15000):
    """7-plane legacy replays -> (v1 tensor, pi, z) triples, bad planes refused."""
    kept = skipped = 0
    for p in sorted(Path(folder).rglob('*_legacy.pt')):
        d = load_checkpoint(p, map_location='cpu')
        for x, pi, z in d.get('replay', []):
            s = lift_sample(x)
            if s is None:
                skipped += 1
                continue
            yield (encode_v1(s), np.asarray(pi, dtype=np.float32), float(z))
            kept += 1
            if kept >= limit:
                return
    if kept == 0:
        raise FileNotFoundError(
            f'no usable legacy replay found in {folder}: {kept} kept, {skipped} '
            f'skipped')
    print(f'legacy replay: {kept} usable samples kept, {skipped} skipped '
          f'(unparseable wall planes)', flush=True)


def run(folder, output, channels=96, blocks=8, epochs=5, batch=512, seed=42):
    data = list(load_legacy(folder))
    non_zero_z = sum(1 for *_, z in data if z != 0.0)
    if non_zero_z == 0:
        # Regressing the value head to a constant 0.0 is not neutral - the net would
        # learn "every game is a draw" - so the value term is dropped and pretraining
        # tunes the policy only.
        print('WARN every legacy value target is 0.0; the value term is dropped and '
              'pretraining tunes the policy only', flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PolicyValueNet(channels, blocks, PLANES_BY_VERSION[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    rng = random.Random(seed)
    for ep in range(epochs):
        rng.shuffle(data)
        pl = 0.0
        n = 0
        for i in range(0, len(data), batch):
            b = data[i:i + batch]
            x = torch.from_numpy(np.stack([q[0] for q in b])).to(device)
            pi = torch.from_numpy(np.stack([q[1] for q in b])).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda',
                                dtype=torch.float16):
                logits, v = net(x)
                loss = -(pi * torch.log_softmax(logits, 1)).sum(1).mean()
                if non_zero_z:
                    z = torch.tensor([q[2] for q in b], dtype=torch.float32, device=device)
                    loss = loss + torch.nn.functional.mse_loss(v, z)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            scaler.step(opt)
            scaler.update()
            pl += loss.item()
            n += 1
        print(f'legacy epoch={ep} policy={pl / n:.4f}', flush=True)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': net.state_dict(), 'legacy_samples': len(data),
                'channels': channels, 'blocks': blocks, 'encoding': 1}, out)


def main():
    p = argparse.ArgumentParser(description='Pretrain on 7-plane legacy replays')
    p.add_argument('--legacy', default='legacy')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--channels', type=int, default=96)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    run(a.legacy, a.output, channels=a.channels, blocks=a.blocks,
        epochs=a.epochs, batch=a.batch, seed=a.seed)


if __name__ == '__main__':
    main()