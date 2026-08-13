# Qudor AI

An experimental AlphaZero-style AI for the board game Quoridor 9×9.

Qudor includes a custom bitboard engine, versioned state encoding, policy/value ResNet, MCTS self-play, Gumbel AlphaZero search, replay-buffer training, arena gating, baseline bots, and a local browser viewer.

> Status: research preview. The project is functional and tested, but the current model strength is not an official human or engine benchmark.

## Highlights

- 209-action Quoridor move space: 81 pawn moves, 64 horizontal walls, 64 vertical walls.
- Custom legality and path-preservation checks for wall placement.
- Three state encodings; v3 is the current encoding for new training.
- PUCT and Gumbel-Top-k / sequential-halving search paths.
- Candidate promotion through MCTS-vs-MCTS arena gating.
- Atomic checkpoint saves and restricted checkpoint loading.
- Local HTTP viewer with CSRF protection, Origin checks, and project-contained model paths.
- 93 automated tests covering the engine, encoder, model, search, self-play, arena, resume, baselines, minimax, and HTTP API.

## Quick start

Requirements: Python 3.10+, NumPy, PyTorch. Install the package and development tools with:

```bash
python -m pip install -e ".[dev]"
```

Run the viewer:

```bash
python app.py
```

Open `http://127.0.0.1:8765`.

Run a safe training preview:

```bash
python train.py --dry-run
```

Start training with automatic CPU/GPU configuration:

```bash
python train.py --output runs/Checkpoints
```

Run on CPU explicitly:

```bash
python train.py --force-cpu
```

The technical architecture and project status are described in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## Verification

```bash
python -m compileall -q app.py train.py quoridor_ai tests
python -m pytest -q
python scripts/benchmark.py
```

The last verified local test run passed 93 tests. The full suite takes roughly three minutes because the integration tests start the viewer and inspect available checkpoints.

The viewer's settings panel can switch the search between the per-visit reference and the
round-batched Gumbel variant (faster, slightly different search), enable int8
quantisation, and tune the per-move simulation budget.

## Checkpoints

Published checkpoints live in `checkpoints/` and are stored via Git LFS. They were
trained under an earlier restricted wall-placement variant (see below) with the v3 encoder
and a 128×10 SE ResNet; the engine has since switched to official wall-chaining rules.

| File | Generation | Iteration | Arena gate |
| --- | --- | --- | --- |
| `checkpoints/gen13_best.pt` | 13 | 439 | 85.2% (54/64) |
| `checkpoints/gen13_latest.pt` | 13 | 444 | — |
| `checkpoints/best.pt` | 12 | 239 | 90.6% (58/64) |
| `checkpoints/latest.pt` | 11 | 234 | — |

The viewer lists these in its model menu. The arena gate is the fraction of gated games
won against the previous best network at gate settings from the training configuration.

### Rules variant and search note

The engine implements official Quoridor rules: wall segments may touch end-to-end, and the
only placement constraint besides slot occupancy is that a wall must not completely block a
player's path to their goal row. Checkpoints published so far predate this switch — they
were trained under a restricted variant that forbade end-to-end wall chains. Training runs
started after the switch use the official rules.

The Gumbel implementation is an adaptation of the published method: its value mixture uses
prior-weighted candidate Q values rather than the paper's visit-count-weighted sum. This is
intentional and covered by the project's tests.

## Training history

Training runs are listed in wall-clock order. Generation counters were reset when the
training moved to a new hardware setup, so the same generation number may appear in more
than one campaign. `checkpoints/best.pt` and `checkpoints/latest.pt` come from the RTX 3060
Ti server campaign; the `gen13_*` files come from the earlier Colab campaign.

### Campaigns

| Campaign | Hardware | Period (UTC) | Duration | Generations |
| --- | --- | --- | --- | --- |
| v1 seed experiments | Google Colab | Aug 5–6 | not recorded | none (iteration counter only) |
| AlphaZero campaign | Google Colab, 15 GB GPU | Aug 7 20:18 → Aug 11 18:19+ | ≈ 94–121 h (est.) | 0 → 15 recorded, up to 17 per recollection |
| Server run (`rtx3060_24h`) | RTX 3060 Ti 8 GB | Aug 12 21:29 → Aug 13 12:35 | 15 h 06 m — gens 0 → 12 |
| Continuation | RTX 3060 Ti 8 GB | Aug 13 13:47 → Aug 13 17:39 | 3 h 52 m | 11 → 14 |
| Official rules (current) | RTX 3060 Ti 8 GB | Aug 13 17:40 → now | running | 14 → (continuing) |

How the project changed along the way:

- **v1 seed experiments (Aug 5–6, Colab):** first attempts — v1 11-plane encoder, a
  pre-ResBlock network, and seed-based runs (`seed_11`/`seed_22`/`seed_33`) with an
  iteration counter only, no generations.
- **AlphaZero campaign (Aug 7–11, Colab):** the modern pipeline took over — v3 16-plane
  encoder with canonical mirroring, 128×10 SE ResNet, PUCT/Gumbel search, replay-buffer
  training with EMA and arena gating, and the generation counter that reset to 0 whenever
  training moved to a new host. Reached gen 15 in saved checkpoints (up to ~17 per
  recollection) before the move to the dedicated GPU server.
- **Server run (Aug 12–13):** fresh start on the RTX 3060 Ti server, 256 games per
  iteration, gates every 20 iterations.
- **Continuation:** resumed from the server run's `latest.pt` with a fresh replay
  buffer; both EMA and live candidates were promoted across gates 12 → 14.
- **Official rules (current):** the engine switched to official wall-chaining rules;
  training restarted from the gen-14 champion with a fresh replay buffer.

### Arena gates

Every gate is a Gumbel-search match of the candidate against the previous champion at
48 simulations per move.

| Campaign | Generation | Iteration | Promoted | Games | Win rate | Elo Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Colab | 8 | 99 | — | 16 | 81.3% (13W 0D 3L) | +255 |
| Colab | 12 | 349 | — | 16 | 59.4% (9W 1D 6L) | +66 |
| Colab | 13 | 439 | — | 64 | 85.2% (54W 1D 9L) | +303 |
| Colab | 15 | 599 | — | 32 | 68.8% (22W 0D 10L) | +137 |
| Server | 12 | 238 | live | 64 | 90.6% (58W 0D 6L) | +394 |
| Continuation | 12 | 238 | ema | 64 | 67.2% (43W) | +124 |
| Continuation | 13 | 258 | live | 64 | 79.7% (51W) | +237 |
| Continuation | 14 | 278 | ema | 64 | 67.2% (43W) | +124 |

### Baseline evals

Bots are deterministic: greedy follows the shortest path, rusher sprints straight for the
goal. Settings: Gumbel search, temperature 0.6, 48 simulations per move (64 for the final
server-run eval).

| Campaign | Generation | Bot | Games | Win rate | Elo Δ |
| --- | --- | --- | --- | --- | --- |
| Server | 12 (iter 239) | greedy | 100 | 90.0% (90W 0D 10L) | +382 |
| Server | 12 (iter 239) | rusher | 100 | 100% (100W) | +1600 |
| Continuation | 12 | greedy | 64 | 96.9% (62W 0D 2L) | +597 |
| Continuation | 13 | greedy | 64 | 87.5% (56W 0D 8L) | +338 |
| Continuation | 14 | greedy | 64 | 85.9% (55W 0D 9L) | +314 |
| Continuation | 12–14 | rusher | 64 | 100% (64W) | +1600 |

## Training and evaluation

The main pipeline is implemented in:

- `quoridor_ai/az_selfplay.py` — self-play and MCTS targets;
- `quoridor_ai/az_train.py` — training, replay, EMA, checkpoints;
- `quoridor_ai/az_arena.py` — candidate/champion evaluation;
- `quoridor_ai/batched_mcts.py` — batched search;
- `quoridor_ai/baseline.py` — deterministic baseline bots.

The training history above summarises the recorded campaigns. They used different
hardware, configurations, and generation counters, so their numbers must not be combined
into a single leaderboard. Baseline evals are included for reference only; human play
observations are informal and are not used as a formal evaluation metric.

For a meaningful comparison, report the configuration, checkpoint, encoding version, MCTS simulations, number of games, player-color balancing, seed, and confidence interval.

## Repository layout

```text
quoridor_ai/       Core engine, model, search, training, and evaluation code
tests/             Automated tests
configs/           Training configurations
scripts/           Project benchmark scripts
bench/             Profiling and search diagnostics
tools/             Board/wall diagnostics
notebooks/         Training notebooks
legacy/            Historical experiments, excluded from the public package by default
```

Large checkpoints and local experiment outputs are intentionally excluded from Git. Cloud deployment/watch scripts are kept outside the publishable tree under the local-only `ops/` directory and are ignored by Git because they contain machine-specific infrastructure defaults.

## Publication status

This is a research/software preview, not a claim of superhuman play. The included code is intended to make the engine and training pipeline inspectable and reproducible. A future release should publish one selected checkpoint separately with its model card, SHA256 hash, training configuration, and baseline results.

## License

MIT. See [LICENSE](LICENSE).
