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
- 88 automated tests covering the engine, encoder, model, search, self-play, arena, resume, baselines, minimax, and HTTP API.

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

The current Colab workflow is documented in [RUN_COLAB_AZ.md](RUN_COLAB_AZ.md). The technical architecture and project status are described in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

## Verification

```bash
python -m compileall -q app.py train.py quoridor_ai tests
python -m pytest -q
python scripts/benchmark.py
```

The last verified local test run passed 88 tests. The full test suite currently takes roughly one minute because integration tests start the viewer and inspect available checkpoints.

### Rules variant and search note

This research preview uses a deliberately restricted wall-placement variant: two wall
segments may not be placed end-to-end. Official Quoridor rules allow such wall chains, so
results from this project are not directly comparable with official-rule engines. Existing
checkpoints were trained under this variant.

The Gumbel implementation is an adaptation of the published method: its value mixture uses
prior-weighted candidate Q values rather than the paper's visit-count-weighted sum. This is
intentional and covered by the project's tests.

## Training and evaluation

The main pipeline is implemented in:

- `quoridor_ai/az_selfplay.py` — self-play and MCTS targets;
- `quoridor_ai/az_train.py` — training, replay, EMA, checkpoints;
- `quoridor_ai/az_arena.py` — candidate/champion evaluation;
- `quoridor_ai/batched_mcts.py` — batched search;
- `quoridor_ai/baseline.py` — deterministic baseline bots.

The repository contains local experiment logs, but they are not a controlled strength benchmark. Current retained logs represent 2,196 recorded self-play games across the available local runs; these runs use different configurations and must not be combined as a single leaderboard result. Human play observations are informal and are not used as a formal evaluation metric.

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
