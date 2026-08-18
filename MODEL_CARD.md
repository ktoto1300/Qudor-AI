# Qudor AI — Model Card

## Model Details

- **Model Name:** Qudor AI (AlphaZero Quoridor)
- **Model Architecture:** 128×10 Squeeze-and-Excitation Residual Network (`PolicyValueNet`)
- **State Encoding:** Version 3 Canonical Encoding (`16` input planes)
- **Experimental State Encoding:** Version 4 Encoding (`18` input planes with BFS pawn distance fields)
- **Action Space:** `209` discrete actions (81 pawn moves + 64 horizontal walls + 64 vertical walls)
- **Rules Specification:** `RULES_VERSION = 2` (official Quoridor rules with strict wall-overlap prohibition, jump rules, and BFS goal path preservation)
- **Framework:** PyTorch (`>=2.2`)
- **License:** MIT

---

## Checkpoint Registry & Integrity

The table below catalogs all official and published checkpoints residing in [`checkpoints/`](checkpoints/). Checksums are computed using SHA-256 over raw binary contents.

> [!NOTE]
> **Published Checkpoints vs. Server Evaluation Runs (`gen69` vs. `gen85`):**
> - **Official Published Checkpoints:** The repository tracks and distributes `gen69_best.pt` and `gen69_latest.pt` under `RULES_VERSION = 2` in [`checkpoints/`](checkpoints/), alongside legacy `gen13_*` models.
> - **Active Server Evaluation Runs (`gen85`):** References to `gen85` in tournament records and benchmark summaries designate active server evaluation runs under `official_rules_v2_batched` rather than static checkpoint binaries published in the `checkpoints/` directory.

| Checkpoint File | Lineage / Description | Generation | Iteration | Input Planes | Size (Bytes) | SHA-256 Hash | Rules Version |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| [`checkpoints/gen69_best.pt`](checkpoints/gen69_best.pt) | Official Rules v2 Tournament Champion | 69 | 1419 | 16 (v3) | 13,022,994 | `4833101367d76b7d7a508d8d728b0ddd701d8330bbf5c50eda10acc08d26287c` | `2` |
| [`checkpoints/gen69_latest.pt`](checkpoints/gen69_latest.pt) | Official Rules v2 Full Training State | 69 | 1429 | 16 (v3) | 236,389,946 | `a4c569d8172c9f12824961ac3d474fe34a5dbf2ff403dc34976d992f99fb6d8c` | `2` |
| [`checkpoints/gen13_best.pt`](checkpoints/gen13_best.pt) | Legacy Colab Best (pre-v2 geometry) | 13 | 439 | 16 (v3) | 13,024,927 | `377355c77f8c5c5d1dc3dd31c3e0358e67d98cffe669754de61319ee30924c49` | Legacy |
| [`checkpoints/gen13_latest.pt`](checkpoints/gen13_latest.pt) | Legacy Colab Latest (pre-v2 geometry) | 13 | 444 | 16 (v3) | 235,507,723 | `f2e09decb0797ac3d7b120fe0b5106a5dc7e0fad4618fefe01a03762beed1823` | Legacy |
| [`checkpoints/best.pt.int8.pt`](checkpoints/best.pt.int8.pt) | Quantized INT8 Weights (Fast CPU Inference) | 12 | 239 | 16 (v3) | 3,445,215 | `d417de6d45f3121df456ca596f1788ed78076128d9103e3acc31fe46caf9fb96` | Legacy |
| [`checkpoints/gen13_best.pt.int8.pt`](checkpoints/gen13_best.pt.int8.pt) | Quantized INT8 Weights (Gen 13) | 13 | 439 | 16 (v3) | 3,447,687 | `290b09e0766410e84cc7c0686f4b3c1b62ba60129371bff06ba6522475e25608` | Legacy |

---

## Architecture Specifications

### 1. Canonical State Representation (v3 Encoder — 16 Planes)

Used by all standard and published checkpoints (`gen69_best.pt`, `gen69_latest.pt`, `gen13_best.pt`), as well as active server evaluation runs (`gen85`, etc.). The representation is framed from the perspective of the side to move (canonical perspective flip for player 1, mapping the player's goal row to row 0):

- **Input Dimensions:** `(16, 9, 9)`
- **Plane Layout:**
  - `Plane 0`: Current player pawn position (one-hot, canonical frame)
  - `Plane 1`: Opponent pawn position (one-hot, canonical frame)
  - `Plane 2`: Horizontal wall slot occupancy ($2\times 2$ block painted per wall on $9\times 9$ grid, canonical frame)
  - `Plane 3`: Vertical wall slot occupancy ($2\times 2$ block painted per wall on $9\times 9$ grid, canonical frame)
  - `Plane 4`: Combined wall slot occupancy ($h \lor v$, $2\times 2$ block painted per wall on $9\times 9$ grid, canonical frame)
  - `Plane 5`: Current player shortest-path BFS distance-to-goal field (normalized to $[0, 1]$, canonical frame)
  - `Plane 6`: Opponent shortest-path BFS distance-to-goal field (normalized to $[0, 1]$, canonical frame)
  - `Plane 7`: Current player remaining wall count normalized (`my_walls / 10.0`, scalar plane)
  - `Plane 8`: Opponent remaining wall count normalized (`their_walls / 10.0`, scalar plane)
  - `Plane 9`: Current player shortest distance to goal (`my_d`, normalized scalar plane)
  - `Plane 10`: Opponent shortest distance to goal (`their_d`, normalized scalar plane)
  - `Plane 11`: Race advantage metric ($\text{clip}(0.5 + 2 \cdot (d_{\text{opp}} - d_{\text{self}}), 0.0, 1.0)$, scalar plane)
  - `Plane 12`: Normalized game ply progress ($\min(1.0, \text{ply} / 220.0)$, scalar plane)
  - `Plane 13`: Constant bias plane filled with $1.0$ (allows convolutions to sense board boundaries and padding)
  - `Plane 14`: Canonical goal row indicator (one-hot row 0, `row 0 = 1`)
  - `Plane 15`: Wall placement eligibility / ability flag ($1.0$ if $\text{walls} > 0$ else $0.0$, scalar plane)

### 2. Experimental State Representation (v4 Encoder — 18 Planes)

Available for future training campaigns via `quoridor_ai.core.encoding.encode_v4`. Augments canonical v3 with full pawn-to-cell distance fields:

- **Input Dimensions:** `(18, 9, 9)`
- **Plane Layout:**
  - `Planes 0..15`: All 16 canonical planes from Encoding v3.
  - `Plane 16`: BFS distance field from current player pawn position to all 81 board cells (normalized to $[0, 1]$, canonical frame).
  - `Plane 17`: BFS distance field from opponent pawn position to all 81 board cells (normalized to $[0, 1]$, canonical frame).
- **Purpose:** Supplies dense, all-to-all topological reachability from each player's exact pawn position, assisting the convolutional trunk in assessing local corridor and maze routing without requiring deep receptive field propagation.

### 3. Network Topology (`128×10 SE ResNet`)

- **Total Parameters:** `3,235,366` (for 16-plane v3 input; `3,237,670` for 18-plane v4 input)
- **Stem:** Conv2d (`16 → 128`, kernel `3×3`, stride 1, padding 1, bias=False) + BatchNorm2d + SiLU
- **Body:** `10` Residual Blocks with Squeeze-and-Excitation (SE) gates:
  - Conv2d (`128 → 128`, `3×3`, padding 1, bias=False) + BatchNorm2d + SiLU + Conv2d (`128 → 128`, `3×3`, padding 1, bias=False) + BatchNorm2d
  - SE Channel Gate: Global Average Pooling → Linear (`128 → 32`) → SiLU → Linear (`32 → 128`) → Sigmoid
  - Residual connection + SiLU activation
- **Policy Head:** Conv2d (`128 → 8`, `1×1`) + SiLU + Flatten (`648`) + Linear (`648 → 209`) logits
- **Value Head:** Conv2d (`128 → 4`, `1×1`) + SiLU + Flatten (`324`) + Linear (`324 → 128`) + SiLU + Linear (`128 → 1`) + Tanh $\in [-1.0, +1.0]$

---

## Training Methodology & Hyperparameters

The model is trained tabula rasa via self-play reinforcement learning using Gumbel AlphaZero:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Search Algorithm** | Gumbel AlphaZero | Sequential Halving ($m=16$ considered actions, budget $N=24$) |
| **Fast Playout Search** | $N_{\text{fast}} = 8$ | Playout cap randomization (`full_frac = 0.5`) |
| **Optimizer** | AdamW | Initial learning rate `0.0015`, cosine annealing schedule |
| **Warmup Steps** | `800` steps | Linear learning rate warmup |
| **Weight Decay** | `1e-4` | L2 regularization |
| **Batch Size** | `256` (server) / `1024` (cloud) | Mini-batch sample size per optimization step |
| **Replay Buffer** | `500,000` transitions | Circular memory-mapped buffer on disk (`replay_on_disk=True`) |
| **Value Blending** | $\lambda = 0.4$ | $z_{\text{blend}} = 0.4 q + 0.6 z_{\text{game}}$ (TD/Monte Carlo mix) |
| **EMA Policy Decay** | `0.999` | Exponential Moving Average tracking |
| **Tournament Gating** | 64 games @ 48 sims | Requires $\ge 60\%$ score against reigning best to promote |

---

## Baseline Benchmarks & Strength

The model's strength is measured against deterministic, non-learning reference bots:
- **`rusher`**: Shortest-path BFS racer. Never places walls. Tests fundamental board navigation and race tactical awareness.
- **`greedy`**: 1-ply BFS search evaluating pawn steps and wall placements with race-differential scoring ($1.5 \cdot \Delta d_{\text{opp}} - \Delta d_{\text{self}} + 0.25 \cdot \Delta w$).

### Evaluation Results (100 Games, Alternating P0/P1, 64 Sims)

| Baseline Bot | Checkpoint | Win Rate (95% CI) | Elo Delta (95% CI) | P0 Win Rate | P1 Win Rate | Avg Game Length |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`rusher`** | `gen69_best.pt` | **97.0%** `[93.6%, 100.0%]` | **+603.9** `[+467.2, +1600.0]` | 94.0% | 100.0% | 18.8 plies |
| **`greedy`** | `gen69_best.pt` | **19.0%** `[11.3%, 26.7%]` | **-251.9** `[-357.7, -175.4]` | 14.0% | 24.0% | 61.6 plies |

---

## Usage Example

### Python API Inference & Evaluation

```python
import torch
from quoridor_ai.core.engine import State, legal_actions
from quoridor_ai.core.encoding import encode_batch, version_for_planes
from quoridor_ai.model import net_from_checkpoint
from quoridor_ai.safe_loader import load_checkpoint
from quoridor_ai.baseline import evaluate_checkpoint, format_markdown_summary

# 1. Load checkpoint safely with weights_only verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = load_checkpoint("checkpoints/gen69_best.pt", map_location=device)
net = net_from_checkpoint(checkpoint, device=device)
net.eval()

# 2. Forward pass on current game state
state = State()
encoding_version = version_for_planes(net.planes)
tensor = torch.from_numpy(encode_batch([state], version=encoding_version)).to(device)

with torch.no_grad():
    policy_logits, value = net(tensor)
    win_probability = (value.item() + 1.0) / 2.0
    print(f"Initial State Win Probability: {win_probability:.2%}")

# 3. Evaluate checkpoint against baseline bots
results = evaluate_checkpoint(
    net_path="checkpoints/gen69_best.pt",
    bot_names=["rusher", "greedy"],
    games=50,
    sims=32,
    device=device,
)
print(format_markdown_summary(results))
```

### CLI Command

```bash
# Run baseline evaluation via CLI entry point
qudor-eval-baseline --net checkpoints/gen69_best.pt --bot all --games 100 --json results.json --md results.md
```
