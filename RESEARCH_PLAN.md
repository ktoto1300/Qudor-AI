# RESEARCH_PLAN — AlphaZero Stack Improvements & Implementation Status

Status & Tracking Document — Last Updated: 19.08.2026

---

## 1. Overview & Current Stack Status

- **GPU Configuration:** `configs/colab_az_gumbel.json` / `configs/rtx3060_official_batched.json` — 128×10 SE ResNet (`PolicyValueNet`), canonical state encoding (v3 16-plane / v4 18-plane support), Gumbel search (`gumbel_cap=16`, `sims=24`, `fast_sims=8`, `full_frac=0.5`), `max_plies=220`, `temp_moves=24`, `resign_v=-0.95`, `value_blend_q=0.4`, replay 500k, EMA net (`ema_decay=0.999`) with live vs. EMA gating.
- **Search Implementations:**
  - `quoridor_ai/batched_mcts.py`: Batched vectorized PUCT selection using NumPy array operations.
  - `quoridor_ai/az_selfplay.py`: Gumbel AlphaZero search with root visit compensation and tanh value transform.
- **Baseline & Tournament Infrastructure:**
  - `quoridor_ai/baseline.py`: Deterministic baseline evaluation (`rusher`, `greedy`) with CLI runner `qudor-eval-baseline`.
  - `tools/foreign_arena.py`: Multi-engine tournament coordinator with JSON-lines sub-process bridge protocol (`tools/bridges/`).

---

## 2. Research Items & Implementation Status

### A. Algorithmic & Search Enhancements

| Item | Title | Status | Implementation Details / Notes |
| :--- | :--- | :---: | :--- |
| **A1** | **BFS Distance Planes (v4 Encoding)** | **`COMPLETED`** | Implemented in `quoridor_ai/core/encoding.py` (`encode_v4`, `PLANES_BY_VERSION[4]=18`). Adds canonical BFS distance fields for active player and opponent (`planes 16..17`) to eliminate blind wall placements and accelerate early race tact navigation. |
| **A2** | **Tanh Value Transform** | **`COMPLETED`** | Implemented in Gumbel tree evaluation & target computation. Normalizes tree $Q$-values and $z$-targets via tanh transform ($\alpha=1$), stabilizing the $q$-blend trade-off across varying playout depths. |
| **A3** | **Root Visit Compensation** | **`COMPLETED`** | Implemented in Gumbel root selection (unbiased MCTS formulation). Scales exploration constant $c_{\text{puct}}$ by $\sqrt{N}$ at the root to correct candidate allocation under limited simulation budgets ($N \le 24$). |
| **A4** | **Resign Margin with Reason Logging** | **`IMPLEMENTED / FUNCTIONALLY TESTED`** | Optional structured records expose `reason`, `q`, `streak`, and `ply`; the existing default threshold remains `-0.95` and logging is off by default. Strength and training impact are not validated. |
| **A5** | **Visit-Averaged Policy-Target EMA** | **`IMPLEMENTED / FUNCTIONALLY TESTED`** | Optional `policy_target_ema` blends successive early policy targets; default `0.0` preserves existing targets. Strength and training impact are not validated. |

---

### B. Execution, Data & Replay Optimizations

| Item | Title | Status | Implementation Details / Notes |
| :--- | :--- | :---: | :--- |
| **B6** | **Vectorized PUCT Selection** | **`COMPLETED`** | Implemented in `quoridor_ai/batched_mcts.py`. Replaced the Python `max()` loop over node children with vectorized NumPy tensor operations ($Q + U(P, N)$ evaluations), yielding substantial CPU and GPU speedups without altering tie-breaking semantics. |
| **B7** | **$\text{TD}(\lambda)$ Target Bootstrapping** | **`IMPLEMENTED / FUNCTIONALLY TESTED`** | Pure alternating-player TD(lambda) targets are available in `quoridor_ai/td.py`; integration is intentionally disabled because the checkpoint replay contract remains the compatible four-tuple. Strength and training impact are not validated. |
| **B8** | **Prioritized Experience Replay (PER)** | **`EXPERIMENTAL`** | Self-contained, default-disabled `PrioritizedReplay` sidecar implemented in `quoridor_ai/replay.py`, with tests for updates, circular overwrite, seeded sampling, uniform fallback, export/import, and resume. It is not wired into production training and its playing-strength impact has not been validated. |
| **B9** | **Opening Bank Initialization** | **`IMPLEMENTED / FUNCTIONALLY TESTED`** | Versioned JSON loading, rules/legal-action validation, mirroring, and seeded selection are available; self-play injection is opt-in and default-off. Strength and training impact are not validated. |

---

### C. Advanced Solvers & Acceleration

| Item | Title | Status | Implementation Details / Notes |
| :--- | :--- | :---: | :--- |
| **C10** | **Bounded No-Wall Endgame Solver** | **`IMPLEMENTED / FUNCTIONALLY TESTED`** | Exact bounded diagnostic solver with cache, principal variation and node budget is available in `quoridor_ai/no_wall_solver.py`. It applies only when both players have zero walls and is intentionally not wired into MCTS. |
| **C11** | **SaberNet / SimpleGEMM Blocks** | **`PENDING`** | *Future Work:* Specialized GEMM kernel optimizations for small-batch network inference if batching becomes a throughput bottleneck on low-power devices. |

---

### D. Infrastructure, Evaluation & Quality Assurance

| Item | Title | Status | Implementation Details / Notes |
| :--- | :--- | :---: | :--- |
| **Infra-1** | **Baseline Evaluation CLI** | **`COMPLETED`** | Implemented in `quoridor_ai/baseline.py` and registered entry point `qudor-eval-baseline`. Outputs detailed statistical metrics, 95% Wilson confidence intervals, and Elo deltas in JSON/Markdown. |
| **Infra-2** | **Multi-Engine Tournament Coordinator** | **`COMPLETED`** | Implemented in `tools/foreign_arena.py` with protocol bridges in `tools/bridges/` (supporting `cryer`, `sigma`, `gorisanson`, `marcobt15`, `berlioz10`, `dimitrijekaranfilovic`). |
| **Infra-3** | **Fast/Integration Test Suite Separation** | **`COMPLETED`** | `pyproject.toml` marker configuration splitting unit tests (`pytest -m "not integration"`) from full integration suites. |
| **Infra-4** | **Reproducibility Manifest & Lockfiles** | **`COMPLETED`** | Published [`requirements-lock.txt`](requirements-lock.txt), [`requirements-server-cuda.txt`](requirements-server-cuda.txt), [`MANIFEST.md`](MANIFEST.md), and [`reproducibility.json`](reproducibility.json). |

---

## 3. Engineering Constraints & Invariants

- **Rules Versioning:** Maintain strict separation for `RULES_VERSION=2` (official Quoridor geometry with adjacent wall overlap prevention). Never mix legacy pre-v2 checkpoints with official campaigns.
- **Encoding Stability:** Do not alter existing plane definitions for v1, v2, or v3. New representations must increment version (v4, v5) with dedicated unit tests in `tests/test_encoder.py`.
- **Search Separation:** Keep PUCT and Gumbel configurations explicitly defined and validated at initialization (`train.py`).
- **External Bridges:** External bridge matches are evaluation-only; never pipe third-party gameplay directly into training replay buffers.

---

## 4. Verification & Acceptance Criteria

Every new feature or change must satisfy:
1. **Linting & Bytecode:** `python -m compileall -q .` and `python -m ruff check . --select F,E9,B` must pass with zero warnings/errors.
2. **Fast Test Suite:** `python -m pytest -m "not integration" -q` passes.
3. **Full Test Suite:** `python -m pytest -q` passes cleanly (260 tests as of v2.1).
4. **Independent Evaluation:** foreign-engine results must report honest games separately from technical forfeits; no strength threshold is claimed without a reproducible run.
