# Qudor AI

An experimental AlphaZero-style AI for the board game Quoridor 9×9.

Qudor includes a custom bitboard engine, versioned state encoding, policy/value ResNet, MCTS self-play, Gumbel AlphaZero search, replay-buffer training, arena gating, baseline bots, and a local browser viewer.

> Status: research preview. The project is functional and tested, but the current model strength is not an official human or engine benchmark.

## Highlights

- 209-action Quoridor move space: 81 pawn moves, 64 horizontal walls, 64 vertical walls.
- Custom legality and path-preservation checks for wall placement.
- Four state encodings; v3 is used by published checkpoints and v4 is experimental for future training.
- PUCT and Gumbel-Top-k / sequential-halving search paths.
- Candidate promotion through MCTS-vs-MCTS arena gating, with the EMA copy competing in the same race.
- Atomic checkpoint saves and restricted checkpoint loading (PyTorch `weights_only`).
- Disk-backed replay buffer (memmap) so 500k samples fit on an 8 GB RAM machine; self-play sharded across `selfplay_workers` processes with preserved logical RNG streams.
- Local HTTP viewer with CSRF protection, Origin checks, and project-contained model paths.
- Automated tests covering the engine, encoder, model, search, self-play, arena, resume, baselines, minimax, and HTTP API.

## Quick start

Requirements: Python 3.10+, NumPy, PyTorch.

For standard installation with development tools:

```bash
python -m pip install -e ".[dev]"
```

For a pinned local development environment (see [`MANIFEST.md`](MANIFEST.md) and [`reproducibility.json`](reproducibility.json)):

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
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

Training configurations live in `configs/`: `configs/local_cpu_az.json` for local CPU, `configs/colab_az_cpu.json` and `configs/colab_az_gumbel.json` for Colab, and `configs/rtx3060_official_batched.json` for the GPU server. The current Colab notebook is `notebooks/quoridor_az_training.ipynb`.

## Verification

```bash
python -m compileall -q app.py train.py quoridor_ai tests
python -m ruff check . --select F,E9,B
# Run fast unit tests only:
python -m pytest -m "not integration" -q
# Run full test suite including integration tests:
python -m pytest -q
python scripts/benchmark.py
```

The test suite is split into fast unit tests and slower integration tests using the `integration` marker:
- `python -m pytest -m "not integration"` executes fast unit tests (engine rules, bitboard invariants, encoding permutations, model architectures, data structures, and deterministic heuristics).
- `python -m pytest` executes the full suite including integration tests (HTTP web viewer server, full MCTS/minimax arena matches, multi-step training/resume runs, int8 quantization, and foreign-engine bridges).

The viewer's settings panel can switch the search between the per-visit reference and the
round-batched Gumbel variant (faster, slightly different search), enable int8
quantisation, and tune the per-move simulation budget.

CI (`.github/workflows/tests.yml`) runs compileall, ruff, and pytest on Python 3.11 for every push and pull request.


## Checkpoints

Published checkpoints live in `checkpoints/` and are stored via Git LFS. All of them use
the v3 16-plane encoder and the 128×10 SE ResNet architecture.

- **`gen69_*`**: Belongs to the `official_rules_v2` lineage (`RULES_VERSION=2`, first official-rules-valid campaign, iteration 1419/1429). The `gen69_best.pt` champion achieved 100% win rates across all decisive games against 4 external engines in the overnight tournament (450 games, 0 losses, 0 draws). Full SHA-256 integrity hashes, architecture details, and baseline benchmarks are documented in [`MODEL_CARD.md`](MODEL_CARD.md).
- **`gen85_best.pt`**: Published continuation champion under `official_rules_v2_batched`, evaluated in multi-engine matches (20 games per opponent at 64 sims, temperature 0.5; winning 20-0 against `berlioz10`, `marcobt15`, `dimitrijekaranfilovic`, `gorisanson`, `cryer`, and 11-9 against `sigma`). No `gen85_latest.pt` full training state is published.
- **`gen13_*`**: Legacy checkpoints from the earlier Colab campaign trained before the wall-overlap geometry fix.

| File | Generation | Iteration | Rules Version | Encoder | Architecture | Arena Gate / Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| [`checkpoints/gen85_best.pt`](checkpoints/gen85_best.pt) | 85 | 1759 | 2 | v3 (16 planes) | 128×10 SE ResNet | 68.75% (44/64), SHA-256 `2126e3c2937916972d633efee237f0176cc4e3b8bb23fb86de86189a8712e170` |
| [`checkpoints/gen69_best.pt`](checkpoints/gen69_best.pt) | 69 | 1419 | 2 | v3 (16 planes) | 128×10 SE ResNet | Tournament Champion ([`MODEL_CARD.md`](MODEL_CARD.md)) |
| [`checkpoints/gen69_latest.pt`](checkpoints/gen69_latest.pt) | 69 | 1429 | 2 | v3 (16 planes) | 128×10 SE ResNet | Full training state (optimizer, replay) |
| [`checkpoints/gen13_best.pt`](checkpoints/gen13_best.pt) | 13 | 439 | Legacy | v3 (16 planes) | 128×10 SE ResNet | 85.2% (54/64) (pre-v2 geometry) |
| [`checkpoints/gen13_latest.pt`](checkpoints/gen13_latest.pt) | 13 | 444 | Legacy | v3 (16 planes) | 128×10 SE ResNet | Legacy training state |

The viewer lists official-rules checkpoints first and separates the legacy `gen13_*`
checkpoints so they cannot be mistaken for official models. It automatically selects the
strongest published official checkpoint, currently `gen85_best.pt`. The arena gate is the fraction of gated games
won against the previous best network at gate settings from the training configuration.

### Rules variant and search note

The engine implements official wall geometry: two walls of the same orientation may touch
end-to-end when their slots are two positions apart, but adjacent slots are illegal because
the walls would overlap along one board edge. Perpendicular walls may not cross in the same
slot, and every placement must leave both players a path to their goal row.

Before this correction, the engine incorrectly allowed same-orientation walls in adjacent
slots. The campaign previously labelled "Official rules" was trained with that bug and its
checkpoints are not valid official-rules models. A new official-rules campaign must start
from a fresh replay buffer after this correction.

Checkpoints carry a `rules_version`. Resume refuses an incompatible or unversioned
checkpoint; `--init` transfers only model weights into a fresh replay buffer, optimiser,
learning-rate schedule, metrics and generation lineage.

Checkpoints are saved atomically through a temporary file and `os.replace`. Resumed
training uses `latest.pt`; the best verified candidate is kept in `best.pt`. User-supplied
checkpoints are loaded through PyTorch's restricted `weights_only` mode in
`quoridor_ai/safe_loader.py`.

The Gumbel implementation is an adaptation of the published method: its value mixture uses
prior-weighted candidate Q values rather than the paper's visit-count-weighted sum. This is
intentional and covered by the project's tests.

## Training history

Training runs are listed in wall-clock order. Generation counters were reset when the
training moved to a new hardware setup, so the same generation number may appear in more
than one campaign. The `gen69_*` files come from the `official_rules_v2` server campaign
(RULES_VERSION=2); the `gen13_*` files come from the earlier Colab campaign.

### Campaigns

| Campaign | Hardware | Period (UTC) | Duration | Generations |
| --- | --- | --- | --- | --- |
| v1 seed experiments | Google Colab | Aug 5–6 | not recorded | none (iteration counter only) |
| AlphaZero campaign | Google Colab, 15 GB GPU | Aug 7 20:18 → Aug 11 18:19+ | ≈ 94–121 h (est.) | 0 → 15 recorded, up to 17 per recollection |
| Server run (`rtx3060_24h`) | RTX 3060 Ti 8 GB | Aug 12 21:29 → Aug 13 12:35 | 15 h 06 m | 0 → 12 |
| Continuation | RTX 3060 Ti 8 GB | Aug 13 13:47 → Aug 13 17:39 | 3 h 52 m | 11 → 14 |
| Wall-overlap bug campaign (invalid) | RTX 3060 Ti 8 GB | Aug 13 17:40 → Aug 14 | stopped | 14 → 15 |

Other campaigns kept on the server alongside the current one: `rtx3060_24h`
(generations 0 → 12), `official_rules` (→ 15), `gen12_minimax_continuation` (→ 14),
`official_fixed_batched` (→ 17, stopped on signal; its checkpoints carry no
`rules_version`). Their numbers must not be mixed with the current lineage.

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
- **Wall-overlap bug campaign (invalid):** training restarted from the gen-14 champion,
  but the engine still allowed adjacent same-orientation walls to overlap. Its outputs must
  not be presented as official-rules checkpoints.

### Recorded campaign: `official_rules_v2_batched`

The recorded lineage under `RULES_VERSION=2`, started 14 August 11:08 UTC on the GPU
server, was initialised via `--init` from `runs/official_fixed_batched/latest.pt`
(i.e. with a fresh replay buffer, optimiser, LR schedule, metrics and generation
lineage).

Server hardware: RTX 3060 Ti 8 GB, 8 CPU cores, 7.9 GB RAM, torch 2.5.1+cu124,
driver 550.142. Campaign settings (as of 14 August): 256 games per iteration, 4 self-play
workers, ~2.2–2.5 games/s, ~80 positions/s, iteration takes 120–140 s, VRAM 816 MB of
8192, policy loss ≈ 0.82, value loss ≈ 0.18, LR 1.45e-3 on a cosine schedule, mean game
length growing from 63 to ~70–75 half-moves, replay buffer filled to the 500000-sample
cap.

Gates run every 20 iterations; all 11 so far ended in a promotion (threshold 0.6):

| Iteration | Generation | Winner | Win rate | Elo Δ |
| --- | --- | --- | --- | --- |
| 19 | 1 | live | 64.1% | +100 |
| 39 | 2 | ema | 73.4% | +177 |
| 59 | 3 | ema | 64.1% | +100 |
| 79 | 4 | live | 79.7% | +237 |
| 99 | 5 | ema | 67.2% | +124 |
| 119 | 6 | ema | 76.6% | +206 |
| 139 | 7 | ema | 78.1% | +221 |
| 159 | 8 | ema | 64.1% | +100 |
| 179 | 9 | live | 62.5% | +89 |
| 199 | 10 | ema | 85.9% | +314 |
| 219 | 11 | ema | 64.1% | +100 |

EMA won 8 of 11 gates, which confirms the point of the live-versus-EMA race.

Baseline evals of each new champion (64 games, Gumbel search, 48 simulations, temperature
0.6): against `rusher` — 100% from generation 6 on; against `greedy` — growing from
81.3% (generation 1) to 98.4% (generation 11). This is a relative benchmark, not a formal
strength metric.

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

### Foreign-engine arena

Unlike the baseline bots and the promotion gate — both written against this project's own
engine — these matches are against third-party Quoridor programs with independent board
representations, searches and authors, so they catch a systematic blind spot rather than a
regression. `tools/foreign_arena.py` plays them over a JSON-lines protocol through
dedicated bridge processes in `tools/bridges/`; the repository and bot directory paths are
set via the `QUDOR_REPO` and `BOTS_DIR` environment variables, and simulation budgets for
heavy engines are tuned per bridge. The repository contains smoke results and selected
summaries. Full overnight match records were produced on the server but their raw JSON
files are not published here; earlier 2-game smoke runs at 16 simulations are
statistically insignificant and kept for reference only.

#### gen69 champion (17 August 2026, 570 games)

The `gen69` champion (`official_rules_v2`, iteration 1419) played with Gumbel search at 64
simulations per move, temperature 0.5, colours alternating each game. Overnight run of
17 August 2026, 570 games total.

| Opponent | Type | Games | W | D | L | Win rate | Decisive | Avg plies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| berlioz10_quoridor-monte-carlo | MCTS | 150 | 150 | 0 | 0 | 100% | 150 | 22.5 |
| marcobt15_Quoridor_Reinforcement_Learning | RL | 120 | 120 | 0 | 0 | 100% | 120 | 25.8 |
| dimitrijekaranfilovic_quoridor | MCTS | 90 | 90 | 0 | 0 | 100% | 90 | 29.3 |
| gorisanson_quoridor-ai | MCTS + heuristics | 90 | 90 | 0 | 0 | 100% | 90 | 46.9 |
| cryer_AlphaZero_Quoridor | AlphaZero | 120 | 120 | 0 | 0 | 100% | 0 | 9.0 |

The net won every decisive game against the four engines that played to a finish: 450 games,
no losses, no draws. The `cryer` column is not evidence of strength — all 120 of its games
ended in an opponent forfeit (its bridge failed in the opening, average 9 plies), so its
`decisive` count is zero. A sixth engine, `v-ade-r_QuoridorAI-AlphaZero`, is excluded because
its bridge forfeits on the first move it must make as player 1.

#### gen85 champion (17–18 August 2026, 20 games per opponent)

The `gen85` champion (`official_rules_v2_batched`) played the same protocol at 64
simulations per move, temperature 0.5, 20 games per opponent, seeds fixed per round. The
run was interrupted several times by series of server reboots; giorgos and a replay of the
forfeited pavlosdais match did not complete.

| Opponent | Games | W | D | L | Decisive | Avg plies |
| --- | --- | --- | --- | --- | --- | --- |
| berlioz10_quoridor-monte-carlo | 20 | 20 | 0 | 0 | 20 | 24.1 |
| marcobt15_Quoridor_Reinforcement_Learning | 20 | 20 | 0 | 0 | 20 | 23.1 |
| dimitrijekaranfilovic_quoridor | 20 | 20 | 0 | 0 | 20 | 29.4 |
| gorisanson_quoridor-ai | 20 | 20 | 0 | 0 | 20 | 51.1 |
| cryer_AlphaZero_Quoridor | 20 | 20 | 0 | 0 | 20 | 26.5 |
| sigma (bartolomeo3000_SigmaQuoridor) | 20 | 11 | 0 | 9 | 20 | 64.2 |

`cryer` played to a finish this time (0 forfeits, average 26.5 plies), unlike in the gen69
run where its bridge forfeited in the opening. `sigma`, an AlphaZero-style net, is the only
opponent with real wins against the gen85 net: 9 wins out of 20, with long 64-ply games.
`pavlosdais` (C engine) delivered 20/20 wins but all 20 as opponent forfeits (average 19
plies) and its honest replay plus the giorgos match were lost to the server reboots.

These opponents are open-source hobby and student engines, not top-tier programs. The
result shows the net clearly outclasses this set; it is not a claim about the strongest
Quoridor engines or human play, which have not been tested.

## Training and evaluation

The main pipeline is implemented in:

- `quoridor_ai/az_selfplay.py` — self-play and MCTS targets;
- `quoridor_ai/az_train.py` — training, replay, EMA, checkpoints;
- `quoridor_ai/az_arena.py` — candidate/champion evaluation;
- `quoridor_ai/batched_mcts.py` — batched search;
- `quoridor_ai/baseline.py` — deterministic baseline bots.

With `replay_on_disk`, the replay buffer lives in memmap files (`quoridor_ai/replay.py`),
which lets 500k samples fit on an 8 GB RAM machine. Self-play is sharded across
`selfplay_workers` processes while preserving logical RNG streams.

The historical table above summarises the recorded campaigns. They used different
hardware, configurations, and generation counters, so their numbers must not be combined
into a single leaderboard. Baseline evals are included for reference only; human play
observations are informal and are not used as a formal evaluation metric.

For a meaningful comparison, report the configuration, checkpoint, encoding version, MCTS simulations, number of games, player-color balancing, seed, and confidence interval.

### Laptop foreign-engine workflow

This workflow uses `gen85_best.pt` against third-party engines only; it never adds the
project's `rusher` or `greedy` baselines. Start with dependency inspection, which does not
load the checkpoint or start a bridge:

```bash
python tools/overnight_arena.py --net checkpoints/gen85_best.pt --profile cpu-laptop --dry-run
```

After external bot repositories and weights are available, smoke-test selected bridges:

```bash
python tools/foreign_arena.py --opponent sigma --smoke
```

Then run the resumable serial arena. It persists after each game, excludes known-broken or
repeatedly failing bridges, and reports honest games separately from technical forfeits:

```bash
python tools/overnight_arena.py --net checkpoints/gen85_best.pt --profile cpu-laptop \
  --only berlioz,marcobt15,dimi,gorisanson,sigma --hours 8 --batch 10 \
  --outdir results/local_foreign_gen85
```

External bot repositories, native binaries and Sigma weights are not distributed in this
repository and cannot be restored automatically.

### Human matches and endgame diagnostics

`qudor-human-match` validates JSONL human-game records and emits deterministic JSON or
Markdown summaries. `python -m quoridor_ai.no_wall_solver --help` exposes the exact bounded
diagnostic solver for positions where both players have zero walls. Neither tool changes
the published checkpoint or training behavior.

## Deployment on the GPU server

A dedicated server instance (RTX 3060 Ti 8 GB, 8 cores, 7.9 GB RAM) ran the
`official_rules_v2_batched` campaign. Deployment and monitoring scripts are machine-specific
and intentionally kept outside this public repository. The server environment is documented
in [`requirements-server-cuda.txt`](requirements-server-cuda.txt) and [`MANIFEST.md`](MANIFEST.md).

## Repository layout

```text
quoridor_ai/       Core engine, model, search, training, and evaluation code
  core/            Game rules and encoding
tests/             Automated tests
configs/           Training configurations
scripts/           Project benchmark scripts
bench/             Profiling and search diagnostics
tools/             Wall diagnostics and the arena against external engines via bridges
results/           Summaries of evals and smoke runs
notebooks/         Experiment notebooks
legacy/            Historical experiments, excluded from the public package by default
```

Checkpoint files (`*.pt`, `*.pth`, `*.ckpt`), logs, `runs/`, local experiments and
`ops/` are excluded through `.gitignore`. Published checkpoints in `checkpoints/` are
stored via Git LFS. Dependencies and environments are pinned in [`requirements-lock.txt`](requirements-lock.txt) (Windows CPU development) and [`requirements-server-cuda.txt`](requirements-server-cuda.txt) (Linux CUDA server), with formal machine-readable manifests in [`MANIFEST.md`](MANIFEST.md) and [`reproducibility.json`](reproducibility.json).

## Known limitations

- Training is strongly limited by available CPU/GPU time and replay buffer size.
- The built-in HTTP server is meant for a local viewer, not a public production deployment.
- The legacy modules are kept for reproducibility of old experiments and are not the recommended new pipeline.
- Training results depend on the search configuration, the number of games, the random seed and the statistical power of the arena test.
- The legacy `gen13_*` checkpoints were trained before the wall-overlap correction and do not follow the official rules; the `gen69_*` checkpoints are the first published official-rules models (`RULES_VERSION=2`), and `gen85_best.pt` is the newest published champion. No gen85 latest/full-training-state checkpoint is published.
- The results against external engines come from open-source hobby/student engines and are not a basis for claims against commercial engines or human masters.
- Server deployment uses machine-specific infrastructure outside this public repository.

## Development priorities & Roadmap

### Completed Milestones
1. [x] **Lockfile & Reproducibility Manifest:** Pinned local CPU development in [`requirements-lock.txt`](requirements-lock.txt) and GPU server training in [`requirements-server-cuda.txt`](requirements-server-cuda.txt); documented in [`MANIFEST.md`](MANIFEST.md) and [`reproducibility.json`](reproducibility.json).
2. [x] **Fast / Integration Test Suite Splitting:** Configured `pyproject.toml` markers separating fast unit tests (`python -m pytest -m "not integration"`) from full integration suites (`python -m pytest`).
3. [x] **Experiment Artifacts & Metrics Manifest:** Tracked published runs via manifests and structured summaries in `results/` and [`MODEL_CARD.md`](MODEL_CARD.md).
4. [x] **Baseline Evaluation CLI:** Implemented `qudor-eval-baseline` (`quoridor_ai/baseline.py`) with Wilson score confidence intervals and Elo deltas against `rusher` and `greedy`.
5. [x] **Model Card Publication:** Published [`MODEL_CARD.md`](MODEL_CARD.md) detailing architecture parameters, training methodology, SHA-256 hashes, and baseline benchmarks for `gen69_best.pt`.
6. [x] **Multi-Engine Tournament Coordinator:** Implemented `tools/foreign_arena.py` with multi-process bridge adapters (`tools/bridges/`) testing against 6+ external Quoridor engines.

### v2.1 Functional Additions
1. [x] **Resign Diagnostics (A4):** Optional structured reason/Q/streak/ply logging; default threshold remains unchanged.
2. [x] **Policy-Target EMA (A5):** Optional early-target smoothing; disabled by default.
3. [x] **TD($\lambda$) Helper (B7):** Alternating-player target helper is tested but intentionally not production-wired.
4. [x] **Prioritized Replay Prototype (B8):** Standalone persistent sampler is tested but disabled in production training.
5. [x] **Opening Bank (B9):** Versioned, legal, mirrored and seeded opening lines; disabled by default.
6. [x] **Bounded No-Wall Solver (C10):** Exact diagnostic solver; not integrated into MCTS.
7. [x] **Gumbel Mixture Comparison:** Current adapted mode remains default; canonical visit-weighted mode is opt-in.

These additions are functionally tested, not strength-validated by a new training campaign.
SaberNet/SimpleGEMM and large tablebases remain future work.

## Publication status

This is a research/software preview, not a claim of superhuman play. The included code is intended to make the engine and training pipeline inspectable and reproducible. [`MODEL_CARD.md`](MODEL_CARD.md) records the published official-rules checkpoints, including the current `gen85_best.pt` champion (`RULES_VERSION=2`) and its SHA-256 integrity hash. No `gen85_latest.pt` is published. Future releases will continue to maintain model cards for subsequent training campaigns.

## License

MIT. See [LICENSE](LICENSE).
