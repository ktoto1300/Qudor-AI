# AlphaZero training in Google Colab

Two cells. The first one finds the code and gets rid of every stale copy of it; the second
one trains. Re-upload the desktop `Qudor` folder to Drive as usual, then run them in order.

## Cell 1 — mount Drive, keep exactly one copy of the code

A second copy of the folder is what makes an edited config look like it was ignored: the
launch line passes a *relative* `--config`, so which file gets read depends on which folder
the notebook happens to be sitting in. This cell removes the choice. Stale copies are moved
to `MyDrive/_Qudor_старое`, not erased, because the folder that looks stale by timestamp is
occasionally the one you wanted.

```python
import glob, os, shutil, time
from google.colab import drive
drive.mount('/content/drive')

ROOT, ATTIC = '/content/drive/MyDrive', '/content/drive/MyDrive/_Qudor_старое'

# A copy of the code is any folder holding quoridor_ai/az_train.py. Qudor_runs has no such
# file, so the checkpoints can never be caught by this.
found = set()
for depth in ('*', '*/*', '*/*/*'):
    for p in glob.glob(f'{ROOT}/{depth}/quoridor_ai/az_train.py'):
        d = os.path.dirname(os.path.dirname(p))
        if d != ATTIC and not d.startswith(ATTIC + '/'):
            found.add(d)

# Copies nest - MyDrive/Qudor/Qudor lives inside MyDrive/Qudor. Moving the outer one takes
# the inner one with it, so listing both means the second move looks for a folder that is no
# longer there. Only the outermost copies are entries; what is inside them travels along.
found = {d for d in found if not any(d.startswith(o + '/') for o in found)}

def stamp(d):
    fs = [f for s in ('', 'configs', 'quoridor_ai') for f in glob.glob(f'{d}/{s}/*')]
    return max((os.path.getmtime(f) for f in fs if os.path.isfile(f)), default=0)

copies = sorted(found, key=stamp, reverse=True)
assert copies, 'папка с кодом не найдена — залей Qudor в Мой диск'
for d in copies:
    print(f'  {time.strftime("%d.%m %H:%M", time.localtime(stamp(d)))}  {d}')

KEEP = copies[0]
for d in copies[1:]:
    dst = f'{ATTIC}/{os.path.basename(d)}-{int(stamp(d))}'
    os.makedirs(ATTIC, exist_ok=True)
    if os.path.exists(dst):        # a rerun must not merge two different folders into one
        dst += '-2'
    shutil.move(d, dst)
    print(f'убрал в {dst}')

os.chdir(KEEP)
print(f'\nработаю в {os.getcwd()}')

# The upload is the step most likely to be half-done, and a missing file here is a clear
# message now instead of an ImportError thirty seconds into cell 2.
missing = [f for f in ('train.py', 'configs/colab_az_gumbel.json', 'configs/colab_az_cpu.json')
           if not os.path.exists(f'{KEEP}/{f}')]
print('НЕ ХВАТАЕТ, перезалей папку:', missing) if missing else print('все файлы на месте')
```

`shutil.move` → `shutil.rmtree` on that line if you want them gone for good instead.

## Cell 2 — train

```python
!python "{KEEP}/train.py" --output /content/drive/MyDrive/Qudor_runs/az_15gb
```

`train.py` looks for a GPU and picks the config itself — `colab_az_gumbel.json` if it finds
one, `colab_az_cpu.json` if it does not. Both configs live next to `train.py` and are opened
by absolute path, so a relative path can no longer resolve into the wrong folder. It prints
the hardware and every setting it loaded before it does any work: if that printout disagrees
with what you edited, you know in the first second rather than two hundred iterations later.

Resuming is the default and needs no flag — the same command after any disconnect continues
`latest.pt` in the same directory, including across a GPU↔CPU switch. `az_train` also prints
a `WARNING` line for each search setting that differs from the one the checkpoint was made
with, because that mixes two kinds of data in one replay buffer.

Useful variants:

```bash
python train.py --dry-run      # print the plan, touch nothing
python train.py --force-cpu    # CPU profile on a machine that has a GPU
python train.py -- --config configs/colab_az_t4_fast.json --output .probe   # anything else
```

The last form skips detection entirely: an explicit `--config` is passed through untouched.

## When the GPU quota runs out

Colab's usage limits apply to the accelerator, not to the runtime — their FAQ recommends
switching to a standard runtime when you are not using the GPU. A CPU session still gets up
to 12 hours and costs no GPU quota, so `colab_az_cpu.json` keeps the *same* run moving:
same 128x10 network, same v3 encoder, same Gumbel search, only smaller batches.

Switch the runtime to **None**, then run the same command with the CPU config and the same
`--output`. It resumes `latest.pt` and writes back into the same directory; switching to a
GPU later is just running the GPU config again.

What it actually buys, measured on a 128x10 net at two threads:

| | self-play | optimiser step | one iteration |
|---|---|---|---|
| T4 | 41 000 states/s | ~0.05 s | ~4 min for 256 games |
| 2 vCPU (measured, scaled for Colab's slower cores) | ~130-190 states/s | ~4-6 s | ~60-80 s for 6 games |

That is roughly **2% of the GPU's sample rate** — about 70 000 samples over a 12-hour CPU
session against the GPU's ~3.5 M. Worth running because it is free and the run is otherwise
stopped, not because it competes.

Two things the CPU config deliberately does differently:

- `steps: 8` and `batch: 128`. An optimiser step on this network costs 4-6 s on two vCPUs,
  so training would otherwise eat most of the iteration; self-play is the useful work.
- `save_every: 10`. `latest.pt` is ~150 MB and CPU iterations are short, so saving every one
  would write over a gigabyte an hour into Drive and risk its I/O quota.

Do not run several CPU sessions in parallel to speed this up: `running distributed computing
workers` and `using multiple accounts to work around resource usage restrictions` are both
on Colab's disallowed list, and free-tier runtimes doing that are terminated without warning.

### Resuming across configs

`global_step` is stored in the checkpoint and drives the learning-rate schedule. Deriving it
from `iteration * steps` instead would break exactly this handover: the az_15gb run stopped
at iteration 3 with `steps: 160`, i.e. global step 640, and resuming under the CPU config's
`steps: 8` would have placed it at step 32 — back inside warmup, at a learning rate 19x too
small. Checkpoints written before this existed are reconstructed from the config stored
alongside them, so old runs resume correctly too.

For the same reason `total_steps` is pinned to `80000` (the GPU config's `500 x 160`) rather
than derived from the CPU config's own `iterations`: the cosine has to keep the original
horizon or it decays on a different curve.

## Which config

`colab_az_gumbel.json` is the one to use. Same network as `colab_az_15gb.json`, but the
search is Gumbel AlphaZero (Danihelka et al., ICLR 2022) instead of plain PUCT: the root
draws its candidates by Gumbel-Top-k, spends the budget by Sequential Halving, and trains on
the completed improved policy `softmax(logits + sigma(Q))` rather than on visit counts.

Averaged over a move that is **16 network evaluations instead of 78**, a 4.9x saving. Since
a Colab iteration is dominated by the number of evaluations, that is close to a linear
saving on quota.

The two configs write the same checkpoint format and use identical architecture, so an
existing `az_15gb` run can switch over: point the Gumbel config at the same `--output`
directory and it resumes from `latest.pt`.

`colab_az_15gb.json` is kept for a plain-PUCT baseline. Set `"gumbel": false` in any config
to get the old behaviour; `c_puct`, `noise_frac` and `temp_moves` only apply in that mode,
and `gumbel_cap` only in Gumbel mode.

### Why it is not simply "fewer simulations"

Lowering `sims` on plain PUCT would be strictly worse, because with ~40 legal actions per
Quoridor position a 24-simulation visit-count target reaches only a handful of them. Measured
against a 400-simulation reference search on the current trained network:

| search | picks the reference's move | actions with non-zero target | KL from the reference target |
|---|---|---|---|
| PUCT 24 sims | 66% | 7.2 of 40 | 3.56 |
| PUCT 48 sims | 70% | 11.3 of 40 | 2.61 |
| PUCT 128 sims | 84% | 19.2 of 40 | 0.87 |
| PUCT 192 sims | 88% | 22.6 of 40 | 0.58 |
| **Gumbel 24 sims** | **74%** | **39.7 of 40** | **1.00** |
| Gumbel 48 sims | 72% | 39.8 of 40 | 1.16 |

Gumbel at 24 simulations beats PUCT at 48 on both counts, and matches PUCT somewhere around
128 on target quality. Covering essentially every legal move is the point: one sample teaches
the network about all ~40 actions instead of about 7.

Two consequences worth knowing:

- Raising `sims` above ~24 buys little unless `gumbel_cap` rises with it. Extra simulations
  are spent re-visiting the same 16 candidates, which is why the 48-simulation row is not an
  improvement.
- Self-play and evaluation select moves differently on purpose. Self-play plays the Gumbel
  winner, whose noise replaces Dirichlet exploration; the arena and the UI take the argmax of
  the improved policy. Using the Gumbel winner to evaluate scores 54% instead of 74%, i.e.
  it would understate every candidate network during gating.

The `az_*` path is the current training pipeline. Do not use `quoridor_ai.train` or `configs/colab_t4_*.json` for new training: they are retained only to reproduce the earlier self-distillation experiments.

Checkpoints:

- `latest.pt` — current working network and replay snapshot;
- `best.pt` — last network promoted by MCTS-vs-MCTS arena gating;
- `metrics.csv` and `status.json` — live progress.
