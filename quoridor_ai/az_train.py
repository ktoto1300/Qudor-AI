"""AlphaZero training loop.

Design decisions that matter more than the hyperparameters:

  * The policy target is MCTS visit counts (see az_selfplay), not the network's own softmax.
  * The value target blends the game result z with the root search value q. Pure z is
    unbiased but carries the credit for a 150-ply game back to move 3, which is mostly
    noise; pure q is low-variance but only as good as the current net. KataGo's blend is
    the standard fix and it converges noticeably faster than either alone.
  * Every sample is also trained left-right flipped. That is an exact symmetry of Quoridor
    (the rules never distinguish left from right), so it is free supervision rather than a
    regulariser. The up-down symmetry is deliberately NOT used as augmentation: encoder v3
    already spends it on canonicalisation - it always frames the board so the mover's goal
    is row 0 - and flipping again would produce inputs the net never sees at play time.
  * Promotion is gated by a search-vs-search match. Accepting every candidate lets a
    regression poison the replay buffer for many iterations before it shows up.
  * An EMA of the weights is kept and gated alongside the raw net. EMA is usually the
    stronger player late in a run, but not always, so both compete rather than assuming.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .az_arena import compare
from .az_selfplay import selfplay
from .baseline import BOTS, play as play_baseline
from .core.encoding import FLIPLR, PLANES_BY_VERSION, BEST_VERSION
from .model import PolicyValueNet, arch_of, net_from_checkpoint
from .runtime import configure_threads, resolve_device
from .safe_loader import load_checkpoint

_FLR = np.asarray(FLIPLR)   # index array so augmentation is a gather, not a Python loop


def _augment(x, pi, flip):
    """Mirror a (state tensor, policy target) pair left-right."""
    if not flip:
        return x, pi
    return np.ascontiguousarray(x[:, :, ::-1]), pi[_FLR]


def _encoding_of(d):
    """Encoder version a checkpoint was written with, from the key or from the weights."""
    if 'encoding' in d:
        return int(d['encoding'])
    planes = arch_of(d['model'])[2]
    for v, n in PLANES_BY_VERSION.items():
        if n == planes:
            return v
    raise ValueError(f'checkpoint has {planes} input planes; no known encoder produces that')


def _lr_at(step, total, base, warmup, floor_frac=0.05):
    """Linear warmup then cosine decay to a small floor.

    `step` is a global optimiser-step count carried in the checkpoint, not `iteration *
    steps`. The difference matters when a run resumes under a config with a different
    `steps`: deriving the position from the iteration number would teleport the schedule -
    moving a run from a GPU config (160 steps) to a CPU one (24) would drop it back into
    warmup and undo the decay it had already earned.
    """
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return base * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * t)))


class EMA:
    """Exponential moving average of the float parameters and buffers."""

    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in net.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, net):
        d = self.decay
        for k, v in net.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def state_dict_for(self, net):
        """Full state_dict: EMA values where tracked, live values elsewhere (int buffers)."""
        return {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
                for k, v in net.state_dict().items()}

    def load(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.float())


def _save(path, payload):
    """Atomic save: a Colab disconnect mid-write must not destroy the checkpoint."""
    tmp = Path(str(path) + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run(config, output, resume=True, init=None, device=None):
    c = json.load(open(config, encoding='utf-8'))
    seed = int(c.get('seed', 42))
    if c.get('threads') is not None:
        configure_threads(c['threads'])
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
    enc = int(c.get('encoding', BEST_VERSION))
    if enc not in PLANES_BY_VERSION:
        raise ValueError(f'config encoding={enc} unknown; known: {sorted(PLANES_BY_VERSION)}')

    device = resolve_device(device or c.get('device'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    ck, best_ck = out / 'latest.pt', out / 'best.pt'

    d = None
    if resume and ck.exists():
        d = load_checkpoint(ck, map_location=device)
        found = _encoding_of(d)
        if found != enc:
            print(f'config encoding={enc} ignored; resuming checkpoint uses v{found}', flush=True)
            enc = found
    elif init and Path(init).exists():
        d = load_checkpoint(init, map_location=device)
        found = _encoding_of(d)
        if found != enc:
            raise ValueError(f'--init checkpoint uses encoding v{found} but config asks for v{enc}')

    net = PolicyValueNet(c['channels'], c['blocks'], PLANES_BY_VERSION[enc],
                         bool(c.get('se', True))).to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=float(c.get('weight_decay', 1e-4)))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    replay = deque(maxlen=int(c['replay']))
    ema = EMA(net, float(c.get('ema_decay', 0.999)))
    start, gen, gstep = 0, 0, 0

    if resume and ck.exists():
        net.load_state_dict(d['model'])
        opt.load_state_dict(d['optimizer'])
        sc = d.get('scaler')
        if sc and scaler.is_enabled():
            scaler.load_state_dict(sc)   # GradScaler rejects an empty dict.
        if d.get('ema'):
            ema.load(d['ema'])
        replay.extend(d.get('replay', []))
        start = int(d['iteration']) + 1
        gen = int(d.get('generation', 0))
        # Checkpoints written before global_step existed: reconstruct it from the config
        # they ran under, which is stored alongside them.
        gstep = int(d.get('global_step', start * int(d.get('config', c).get('steps', c['steps']))))
        # Re-seed off the resume point, otherwise the restart replays the identical games.
        random.seed(seed + start); np.random.seed((seed + start) % (2 ** 32)); torch.manual_seed(seed + start)
        print(f'resumed at iteration {start}, generation {gen}, replay {len(replay)}', flush=True)
        # The replay buffer inherited above was generated under the settings stored next to it,
        # so a config that changes the search is mixing two kinds of data in one buffer, and a
        # changed total_steps moves the learning-rate schedule under a run already on it.
        # Neither is fatal and neither is always wrong - but both belong in the first lines of
        # the log, not in a guess two hundred iterations later.
        was = d.get('config', {})
        for k in ('gumbel', 'gumbel_cap', 'sims', 'fast_sims', 'full_frac', 'total_steps'):
            if k in was and was[k] != c.get(k):
                print(f'  WARNING: {k} was {was[k]!r} in the checkpoint, now {c.get(k)!r}',
                      flush=True)
    elif d is not None:
        net.load_state_dict(d['model'])
        ema = EMA(net, float(c.get('ema_decay', 0.999)))
        print(f'initialized from {init}', flush=True)

    if not best_ck.exists():
        _save(best_ck, {'iteration': -1, 'model': net.state_dict(), 'config': c,
                        'encoding': enc, 'generation': 0})

    metrics = out / 'metrics.csv'
    cols = ('iteration,stage,generation,games,positions,replay,games_per_sec,positions_per_sec,'
            'avg_length,p0_wins,draws,policy_loss,value_loss,total_loss,lr,gate_win_rate,'
            'gate_elo,promoted,seconds,device,vram_mb')
    if not metrics.exists():
        metrics.write_text(cols + '\n')

    iterations = int(c['iterations'])
    # The cosine decays to zero at total_steps, so a run that resumes another config's
    # checkpoint has to declare that config's horizon or the learning rate jumps.
    total_steps = int(c.get('total_steps', iterations * int(c['steps'])))
    save_every = max(1, int(c.get('save_every', 1)))
    gate_every = int(c.get('gate_every', 5))
    gate_games = int(c.get('gate_games', 32))
    gate_sims = int(c.get('gate_sims', 100))
    gate_threshold = float(c.get('gate_threshold', 0.55))
    val_blend = float(c.get('value_blend_q', 0.4))
    # The promotion gate is necessarily relative: it only tells us whether a candidate
    # beat the preceding champion.  Fixed hand-written bots give the run an absolute
    # reference point, but are expensive enough that we measure only a newly promoted
    # champion, not every transient candidate.
    baseline_games = int(c.get('baseline_games', 0))
    baseline_sims = int(c.get('baseline_sims', gate_sims))
    baseline_bots = tuple(c.get('baseline_bots', ('rusher', 'greedy')))
    unknown_bots = set(baseline_bots) - BOTS.keys()
    if unknown_bots:
        raise ValueError(f'unknown baseline bots: {sorted(unknown_bots)}')
    baseline_file = out / 'baseline.csv'

    for it in range(start, iterations):
        t0 = time.time()
        net.eval()
        data, sp = selfplay(
            net, device, games=int(c['games']), encoding=enc,
            sims=int(c.get('sims', 200)), fast_sims=int(c.get('fast_sims', 50)),
            full_frac=float(c.get('full_frac', 0.25)), c_puct=float(c.get('c_puct', 1.6)),
            max_plies=int(c['max_plies']), temp_moves=int(c.get('temp_moves', 20)),
            noise_frac=float(c.get('noise_frac', 0.25)), resign_v=float(c.get('resign_v', -0.95)),
            gumbel=bool(c.get('gumbel', False)), gumbel_cap=int(c.get('gumbel_cap', 16)),
            seed=seed * 1000 + it)
        replay.extend(data)
        sp_sec = max(1e-6, time.time() - t0)

        net.train()
        steps = int(c['steps'])
        batch = int(c['batch'])
        pool = list(replay)
        pool_n = len(pool)
        pl = vl = 0.0
        lr_now = c['lr']
        for k in range(steps):
            lr_now = _lr_at(gstep, total_steps, c['lr'], int(c.get('warmup_steps', 200)))
            gstep += 1
            for g in opt.param_groups:
                g['lr'] = lr_now
            idx = random.sample(range(pool_n), min(batch, pool_n))
            xs, pis, zs = [], [], []
            for i in idx:
                x, pi, z, q = pool[i]
                x, pi = _augment(np.asarray(x), np.asarray(pi), random.random() < 0.5)
                xs.append(x); pis.append(pi)
                zs.append((1 - val_blend) * z + val_blend * q)
            x = torch.from_numpy(np.stack(xs)).float().to(device, memory_format=torch.channels_last)
            pi = torch.from_numpy(np.stack(pis)).float().to(device)
            z = torch.tensor(zs, dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.float16):
                logits, val = net(x)
                lp = -(pi * torch.log_softmax(logits, 1)).sum(1).mean()
                lv = torch.nn.functional.mse_loss(val, z)
                loss = lp + lv
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            pl += lp.item(); vl += lv.item()
        net.eval()

        # --- gating -------------------------------------------------------------
        wr = elo = float('nan')
        promoted = ''
        if gate_every > 0 and (it + 1) % gate_every == 0:
            best = net_from_checkpoint(load_checkpoint(best_ck, map_location=device), device)
            ema_net = copy.deepcopy(net)
            ema_net.load_state_dict(ema.state_dict_for(net)); ema_net.eval()
            trials = (('live', net), ('ema', ema_net))
            results = [(name, compare(cand, best, device, gate_games, gate_sims,
                                      temp=float(c.get('gate_temp', 0.6)),
                                      max_plies=int(c['max_plies']), seed=seed + it,
                                      gumbel=bool(c.get('gate_gumbel', c.get('gumbel', False))),
                                      gumbel_cap=int(c.get('gumbel_cap', 16))))
                       for name, cand in trials]
            name, r = max(results, key=lambda kv: kv[1]['win_rate'])
            wr, elo = r['win_rate'], r['elo_delta']
            for n2, r2 in results:
                print(f'  gate {n2}: win_rate={r2["win_rate"]:.3f} elo={r2["elo_delta"]:+.0f}', flush=True)
            if wr >= gate_threshold:
                gen += 1
                promoted = name
                sd = net.state_dict() if name == 'live' else ema.state_dict_for(net)
                _save(best_ck, {'iteration': it, 'model': sd, 'config': c, 'encoding': enc,
                                'generation': gen, 'gate': r})
                print(f'  promoted {name} -> generation {gen}', flush=True)
                if baseline_games > 0:
                    champion = net if name == 'live' else ema_net
                    new_file = not baseline_file.exists()
                    with baseline_file.open('a', newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=('iteration', 'generation', 'champion', 'bot',
                                           'games', 'wins', 'draws', 'losses', 'win_rate',
                                           'elo_delta', 'avg_plies', 'sims', 'temperature',
                                           'gumbel', 'seed'), extrasaction='ignore')
                        if new_file:
                            writer.writeheader()
                        for bot_i, bot_name in enumerate(baseline_bots):
                            br = play_baseline(
                                champion, BOTS[bot_name], device, games=baseline_games,
                                sims=baseline_sims, c_puct=float(c.get('c_puct', 1.6)),
                                temp=float(c.get('baseline_temp', c.get('gate_temp', 0.6))),
                                max_plies=int(c['max_plies']), seed=seed + it * 100 + bot_i,
                                gumbel=bool(c.get('baseline_gumbel', c.get('gumbel', False))),
                                gumbel_cap=int(c.get('gumbel_cap', 16)))
                            writer.writerow(dict(br, iteration=it, generation=gen,
                                                 champion=name, bot=bot_name))
                            print(f'  baseline {bot_name}: {br["win_rate"]:.3f} '
                                  f'({br["wins"]}W {br["draws"]}D {br["losses"]}L)', flush=True)
                if name == 'ema':
                    # The EMA won, so it becomes the trunk; otherwise the raw net keeps
                    # drifting away from the strongest weights we have.
                    net.load_state_dict(ema.state_dict_for(net))
            del best, ema_net
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        sec = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1048576 if device.type == 'cuda' else 0
        row = [it, 'alphazero', gen, sp['games'], sp['samples'], len(replay),
               sp['games'] / sp_sec, sp['samples'] / sp_sec, sp['avg_plies'],
               sp['p0_wins'], sp['draws'], pl / steps, vl / steps, (pl + vl) / steps,
               lr_now, wr, elo, promoted, sec, str(device), vram]
        with metrics.open('a', newline='') as f:
            csv.writer(f).writerow(row)
        # A CPU run's iterations are short and latest.pt is ~150 MB, so writing it every
        # time would push over a gigabyte an hour at Google Drive. save_every trades a
        # bounded amount of redone work for that bandwidth; the last iteration always saves.
        if (it + 1) % save_every == 0 or it == iterations - 1:
            _save(ck, {'iteration': it, 'model': net.state_dict(), 'optimizer': opt.state_dict(),
                       'scaler': scaler.state_dict(), 'ema': ema.shadow,
                       'replay': list(replay)[-int(c['checkpoint_replay']):],
                       'config': c, 'encoding': enc, 'generation': gen, 'global_step': gstep})
        (out / 'status.json').write_text(json.dumps(dict(zip(cols.split(','), row)),
                                                    indent=2, default=str))
        print(f'iteration={it} gen={gen} games={sp["games"]} samples={sp["samples"]} '
              f'plies={sp["avg_plies"]:.0f} loss={(pl + vl) / steps:.4f} lr={lr_now:.2e} '
              f'{sec:.0f}s vram={vram:.0f}MB', flush=True)


def main():
    p = argparse.ArgumentParser(description='AlphaZero training for Quoridor')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--init', help='checkpoint to start the weights from')
    p.add_argument('--no-resume', action='store_true')
    p.add_argument('--device', help="'cpu' or 'cuda'; overrides the config and autodetection")
    a = p.parse_args()
    run(a.config, a.output, not a.no_resume, a.init, a.device)


if __name__ == '__main__':
    main()
