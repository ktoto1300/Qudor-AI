#!/usr/bin/env python3
r"""Automated Multi-Engine Tournament Coordinator for Qudor AI.

Coordinates Round-Robin and Swiss tournaments between:
1. Qudor AlphaZero checkpoints (with MCTS search)
2. Third-party foreign bot bridges in `tools/bridges/` (Berlioz, Cryer, Dimi,
   Giorgos, Gorisanson, Marcobt15, Pavlosdais, Sigma, Vader)
3. Hand-written baseline bots (Rusher, Greedy)
4. Custom external bridge scripts

Key Features:
- Round-Robin and Swiss tournament pairing formats.
- Alternating colors across games per matchup.
- Subprocess health monitoring, timeout management, illegal move logging, and forfeit capture.
- Complete tournament standings with Sonneborn-Berger and Buchholz tiebreak scoring.
- Pairwise head-to-head cross-table generation.
- Decisive games percentage and average plies tracking.
- Bayes-Elo / Bradley-Terry rating estimation with standard error and 95% confidence intervals.
- Structured JSON and Markdown output generation into `results/tournaments/`.
- Integration with `tools/metrics_manifest.py` (`results/MANIFEST.json` and `results/SUMMARY.md`).

Usage Examples:
    # Round-robin between a checkpoint and foreign engines:
    python tools/tournament.py --checkpoints checkpoints/gen13_best.pt \
        --bridges vader berlioz cryer --baselines rusher greedy \
        --format round-robin --games 4 --sims 64

    # Swiss tournament between multiple checkpoints and bridges:
    python tools/tournament.py --checkpoints checkpoints/*.pt \
        --bridges dimi sigma marcobt15 --format swiss --rounds 4 --games 2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from quoridor_ai.az_arena import _Duel, _search_round  # noqa: E402
from quoridor_ai.baseline import BOTS as BASELINE_BOTS  # noqa: E402
from quoridor_ai.core.encoding import version_for_planes  # noqa: E402
from quoridor_ai.core.engine import State, apply_unchecked, legal_actions  # noqa: E402
from quoridor_ai.model import net_from_checkpoint  # noqa: E402
from quoridor_ai.paths import repo_root  # noqa: E402
from quoridor_ai.runtime import configure_threads, resolve_device  # noqa: E402
from quoridor_ai.safe_loader import load_checkpoint  # noqa: E402

BRIDGES_DIR = Path(__file__).resolve().parent / "bridges"

# Standard foreign bridge definitions
BUILTIN_BRIDGES: dict[str, list[str]] = {
    "vader": [sys.executable, str(BRIDGES_DIR / "vader_bridge.py")],
    "berlioz": [sys.executable, str(BRIDGES_DIR / "berlioz_bridge.py")],
    "marcobt15": [sys.executable, str(BRIDGES_DIR / "marcobt15_bridge.py")],
    "cryer": [sys.executable, str(BRIDGES_DIR / "cryer_bridge.py")],
    "dimi": [sys.executable, str(BRIDGES_DIR / "dimi_bridge.py")],
    "gorisanson": ["node", str(BRIDGES_DIR / "gorisanson_bridge.js")],
    "sigma": [sys.executable, str(BRIDGES_DIR / "sigma_bridge.py")],
    "pavlosdais": [sys.executable, str(BRIDGES_DIR / "pavlosdais_bridge.py")],
    "giorgos": [sys.executable, str(BRIDGES_DIR / "giorgos_bridge.py")],
}

FORFEIT = object()
MOVE_TIMEOUT = 120.0
HANDSHAKE_TIMEOUT = 60.0


def _slots(mask: int) -> list[int]:
    """Set bit indices of a wall bitboard, low to high."""
    out = []
    while mask:
        b = mask & -mask
        out.append(b.bit_length() - 1)
        mask ^= b
    return out


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# -----------------------------------------------------------------------------
# Participant Abstraction
# -----------------------------------------------------------------------------

class Participant:
    """Base class for any competitor in a tournament."""

    def __init__(self, name: str, ptype: str):
        self.name = name
        self.ptype = ptype

    def start_session(self, seed: int) -> None:
        """Prepare or initialize resources for a match or session."""
        pass

    def get_action(self, state: State, rng: Any) -> tuple[int | object, str | None]:
        """Produce an action index (0..208) or (FORFEIT, reason)."""
        raise NotImplementedError

    def close_session(self) -> None:
        """Release any subprocess or background resources."""
        pass


class BaselineParticipant(Participant):
    """Hand-written baseline bot (e.g. rusher, greedy)."""

    def __init__(self, name: str, bot_fn: Callable[[State, Any], int]):
        super().__init__(name=name, ptype="baseline")
        self.bot_fn = bot_fn

    def get_action(self, state: State, rng: Any) -> tuple[int | object, str | None]:
        try:
            action = self.bot_fn(state, rng)
            if action in legal_actions(state):
                return action, None
            return FORFEIT, f"Baseline {self.name} returned illegal action {action}"
        except Exception as e:
            return FORFEIT, f"Baseline {self.name} exception: {e}"


class CallableParticipant(Participant):
    """Custom callable participant (useful for testing and in-memory bots)."""

    def __init__(self, name: str, fn: Callable[[State, Any], int | tuple[Any, str | None]], ptype: str = "custom"):
        super().__init__(name=name, ptype=ptype)
        self.fn = fn

    def get_action(self, state: State, rng: Any) -> tuple[int | object, str | None]:
        try:
            res = self.fn(state, rng)
            if isinstance(res, tuple):
                action, note = res
            else:
                action, note = res, None

            if action is FORFEIT:
                return FORFEIT, note or "Callable forfeit"
            if isinstance(action, int) and action in legal_actions(state):
                return action, note
            return FORFEIT, f"Illegal action {action} (note={note})"
        except Exception as e:
            return FORFEIT, f"Callable exception: {e}"


class CheckpointParticipant(Participant):
    """AlphaZero neural net checkpoint with MCTS search."""

    def __init__(
        self,
        name: str,
        checkpoint_path: str | Path,
        device: Any,
        sims: int = 64,
        c_puct: float = 1.6,
        temp: float = 0.6,
        gumbel: bool = True,
        gumbel_cap: int = 16,
        max_plies: int = 220,
    ):
        super().__init__(name=name, ptype="checkpoint")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.temp = temp
        self.gumbel = gumbel
        self.gumbel_cap = gumbel_cap
        self.max_plies = max_plies

        ck = load_checkpoint(str(self.checkpoint_path), map_location=device)
        self.net = net_from_checkpoint(ck, device)
        self.enc = version_for_planes(self.net.planes)
        self.generation = ck.get("generation")
        self.iteration = ck.get("iteration")

    def get_action(self, state: State, rng: Any) -> tuple[int | object, str | None]:
        try:
            seed_val = int(rng.integers(0, 1_000_000_000) if hasattr(rng, "integers") else random.randint(0, 1_000_000_000))
            d = _Duel(swap=False, seed=seed_val)
            d.s = state
            d.rng = rng
            _search_round(
                self.net,
                [d],
                self.device,
                self.enc,
                self.sims,
                self.c_puct,
                self.temp,
                self.max_plies,
                self.gumbel,
                self.gumbel_cap,
            )
            # Find which action led from state to d.s
            for act in legal_actions(state):
                if apply_unchecked(state, act) == d.s:
                    return act, None
            if d.root and getattr(d.root, "acts", None) is not None:
                return int(d.root.acts[0]), None
            return FORFEIT, "Checkpoint search failed to produce transition"
        except Exception as e:
            return FORFEIT, f"Checkpoint search error: {e}"


class BridgeParticipant(Participant):
    """External bot running as a separate process communicating via JSON-lines protocol."""

    def __init__(
        self,
        name: str,
        argv: list[str],
        logdir: Path | str,
        move_timeout: float = MOVE_TIMEOUT,
        handshake_timeout: float = HANDSHAKE_TIMEOUT,
    ):
        super().__init__(name=name, ptype="bridge")
        self.argv = list(argv)
        self.logdir = Path(logdir)
        self.logfile = self.logdir / f"{name}.log"
        self.dumpfile = self.logdir / "illegal_dump.jsonl"
        self.move_timeout = move_timeout
        self.handshake_timeout = handshake_timeout
        self.proc: subprocess.Popen | None = None
        self._err = None
        self._lock = threading.Lock()

    def start_session(self, seed: int) -> None:
        self.close_session()
        script = Path(self.argv[-1])
        if not script.is_file():
            raise RuntimeError(f"Bridge script not found: {script}")

        self.logdir.mkdir(parents=True, exist_ok=True)
        self.logfile.unlink(missing_ok=True)
        self._err = self.logfile.open("a", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                self.argv,
                cwd=str(script.resolve().parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._err,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            self._close_err()
            raise RuntimeError(f"Cannot start bridge {self.name}: {exc}") from exc

        try:
            reply = self._ask({"type": "hello", "seed": seed}, self.handshake_timeout)
            if not isinstance(reply, dict) or not reply.get("ok"):
                raise RuntimeError(f"Bridge {self.name} failed handshake: {reply!r}")
        except Exception:
            self.close_session()
            raise

    def get_action(self, state: State, rng: Any) -> tuple[int | object, str | None]:
        if self.proc is None:
            return FORFEIT, f"Bridge {self.name} is not running"

        req = {
            "type": "state",
            "p0": state.p0,
            "p1": state.p1,
            "w0": state.walls0,
            "w1": state.walls1,
            "h": _slots(state.h),
            "v": _slots(state.v),
            "player": state.player,
            "ply": state.ply,
        }

        try:
            reply = self._ask(req, self.move_timeout)
        except Exception as e:
            return FORFEIT, f"Bridge {self.name} communication error: {e}"

        if reply is None:
            return FORFEIT, f"Bridge {self.name} closed its stdout stream"
        if not isinstance(reply, dict):
            return FORFEIT, f"Bridge {self.name} sent invalid non-dict response: {type(reply).__name__}"
        if "forfeit" in reply:
            return FORFEIT, str(reply["forfeit"])
        if "a" not in reply:
            return FORFEIT, f"Bridge {self.name} sent message without action 'a': {reply!r}"

        action = reply["a"]
        if isinstance(action, bool) or not isinstance(action, int):
            return FORFEIT, f"Bridge {self.name} returned non-int action {action!r}"

        if action in legal_actions(state):
            return action, reply.get("illegal")

        self._dump_illegal(state, action, reply.get("illegal"))
        return FORFEIT, f"Illegal action {action} (self_reported={reply.get('illegal')})"

    def _ask(self, obj: dict, timeout: float) -> dict | None:
        if self.proc is None:
            raise RuntimeError(f"Bridge {self.name} is not running")
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Bridge {self.name} exited with code {self.proc.returncode}; see {self.logfile}")
            try:
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Bridge {self.name} closed stdin: {exc}") from exc

            line = self._readline(timeout)
            if not line:
                return None
            try:
                return json.loads(line)
            except ValueError as exc:
                raise RuntimeError(f"Bridge {self.name} sent non-JSON {line[:120]!r}: {exc}") from exc

    def _readline(self, timeout: float) -> str | None:
        result: dict[str, Any] = {}

        def _reader():
            try:
                result["line"] = self.proc.stdout.readline() if self.proc and self.proc.stdout else ""
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self._kill()
            raise TimeoutError(f"Bridge {self.name} did not answer within {timeout:.1f}s")
        if "error" in result:
            raise RuntimeError(f"Bridge {self.name} read error: {result['error']}")
        return result.get("line")

    def _dump_illegal(self, s: State, a: Any, note: Any) -> None:
        rec = {
            "timestamp": _now_utc(),
            "bot": self.name,
            "state": {
                "p0": s.p0,
                "p1": s.p1,
                "w0": s.walls0,
                "w1": s.walls1,
                "h": _slots(s.h),
                "v": _slots(s.v),
                "player": s.player,
                "ply": s.ply,
            },
            "action": a,
            "note": note,
        }
        try:
            self.dumpfile.parent.mkdir(parents=True, exist_ok=True)
            with self.dumpfile.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    def _close_err(self) -> None:
        if self._err is not None:
            try:
                self._err.close()
            except OSError:
                pass
            self._err = None

    def close_session(self) -> None:
        proc = self.proc
        if proc is None:
            self._close_err()
            return
        try:
            if proc.poll() is None:
                try:
                    with self._lock:
                        if proc.stdin:
                            proc.stdin.write(json.dumps({"type": "bye"}) + "\n")
                            proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._kill()
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        finally:
            self.proc = None
            self._close_err()


# -----------------------------------------------------------------------------
# Match & Game Orchestration
# -----------------------------------------------------------------------------

@dataclass
class GameResult:
    """Outcome of a single game between two participants."""
    game_idx: int
    p0_name: str
    p1_name: str
    score_p0: float
    score_p1: float
    plies: int
    winner_name: str | None
    forfeit_by: str | None = None
    forfeit_reason: str | None = None
    decisive: bool = False


@dataclass
class MatchResult:
    """Aggregate result of a series of games between two participants."""
    participant_a: str
    participant_b: str
    games: int
    score_a: float
    score_b: float
    wins_a: int
    draws: int
    losses_a: int
    forfeits_a: int
    forfeits_b: int
    decisive_games: int
    avg_plies: float
    win_rate_a_p0: float
    win_rate_a_p1: float
    game_details: list[GameResult] = field(default_factory=list)


def play_single_game(
    p0: Participant,
    p1: Participant,
    seed: int,
    game_idx: int = 0,
    max_plies: int = 220,
) -> GameResult:
    """Play one game with p0 moving first."""
    s = State()
    rng = random.Random(seed)
    players = [p0, p1]

    forfeit_by: str | None = None
    forfeit_reason: str | None = None

    while s.winner is None and s.ply < max_plies:
        curr_idx = s.player
        curr_p = players[curr_idx]

        action, note = curr_p.get_action(s, rng)
        if action is FORFEIT:
            forfeit_by = curr_p.name
            forfeit_reason = note or "Forfeited"
            break

        s = apply_unchecked(s, action)

    if forfeit_by is not None:
        if forfeit_by == p0.name:
            score_p0, score_p1 = 0.0, 1.0
            winner_name = p1.name
        else:
            score_p0, score_p1 = 1.0, 0.0
            winner_name = p0.name
        decisive = False
    elif s.winner is not None:
        if s.winner == 0:
            score_p0, score_p1 = 1.0, 0.0
            winner_name = p0.name
        else:
            score_p0, score_p1 = 0.0, 1.0
            winner_name = p1.name
        decisive = True
    else:
        score_p0, score_p1 = 0.5, 0.5
        winner_name = None
        decisive = False

    return GameResult(
        game_idx=game_idx,
        p0_name=p0.name,
        p1_name=p1.name,
        score_p0=score_p0,
        score_p1=score_p1,
        plies=s.ply,
        winner_name=winner_name,
        forfeit_by=forfeit_by,
        forfeit_reason=forfeit_reason,
        decisive=decisive,
    )


def play_matchup(
    pa: Participant,
    pb: Participant,
    games: int,
    base_seed: int = 0,
    max_plies: int = 220,
) -> MatchResult:
    """Play a matchup of N games between pa and pb with alternating colors."""
    if games <= 0:
        return MatchResult(
            participant_a=pa.name,
            participant_b=pb.name,
            games=0,
            score_a=0.0,
            score_b=0.0,
            wins_a=0,
            draws=0,
            losses_a=0,
            forfeits_a=0,
            forfeits_b=0,
            decisive_games=0,
            avg_plies=0.0,
            win_rate_a_p0=0.0,
            win_rate_a_p1=0.0,
        )

    try:
        pa.start_session(base_seed)
    except Exception as e:
        print(f"  [WARN] Failed to start participant {pa.name}: {e}", flush=True)

    try:
        pb.start_session(base_seed + 104729)
    except Exception as e:
        print(f"  [WARN] Failed to start participant {pb.name}: {e}", flush=True)

    game_results: list[GameResult] = []
    p0_scores_a: list[float] = []
    p1_scores_a: list[float] = []

    try:
        for g in range(games):
            pa_is_p0 = (g % 2 == 0)
            p0 = pa if pa_is_p0 else pb
            p1 = pb if pa_is_p0 else pa
            game_seed = base_seed * 1000 + g * 37 + 13

            res = play_single_game(p0, p1, seed=game_seed, game_idx=g, max_plies=max_plies)
            game_results.append(res)

            score_a = res.score_p0 if pa_is_p0 else res.score_p1
            if pa_is_p0:
                p0_scores_a.append(score_a)
            else:
                p1_scores_a.append(score_a)
    finally:
        pa.close_session()
        pb.close_session()

    total_score_a = sum(r.score_p0 if r.p0_name == pa.name else r.score_p1 for r in game_results)
    total_score_b = games - total_score_a
    wins_a = sum(1 for r in game_results if (r.winner_name == pa.name))
    losses_a = sum(1 for r in game_results if (r.winner_name == pb.name))
    draws = sum(1 for r in game_results if r.winner_name is None)

    forfeits_a = sum(1 for r in game_results if r.forfeit_by == pa.name)
    forfeits_b = sum(1 for r in game_results if r.forfeit_by == pb.name)
    decisive_cnt = sum(1 for r in game_results if r.decisive)
    avg_plies = sum(r.plies for r in game_results) / games if games > 0 else 0.0

    wr_p0 = sum(p0_scores_a) / len(p0_scores_a) if p0_scores_a else 0.0
    wr_p1 = sum(p1_scores_a) / len(p1_scores_a) if p1_scores_a else 0.0

    return MatchResult(
        participant_a=pa.name,
        participant_b=pb.name,
        games=games,
        score_a=total_score_a,
        score_b=total_score_b,
        wins_a=wins_a,
        draws=draws,
        losses_a=losses_a,
        forfeits_a=forfeits_a,
        forfeits_b=forfeits_b,
        decisive_games=decisive_cnt,
        avg_plies=round(avg_plies, 2),
        win_rate_a_p0=round(wr_p0, 4),
        win_rate_a_p1=round(wr_p1, 4),
        game_details=game_results,
    )


# -----------------------------------------------------------------------------
# Bayes-Elo / Bradley-Terry Rating Model
# -----------------------------------------------------------------------------

def compute_bradley_terry_ratings(
    participant_names: list[str],
    pairwise_scores: dict[tuple[str, str], float],
    pairwise_games: dict[tuple[str, str], int],
    anchor_name: str | None = None,
    anchor_rating: float = 1500.0,
    prior_lambda: float = 1.0,
    max_iters: int = 200,
    tol: float = 1e-7,
) -> dict[str, dict[str, Any]]:
    """Estimate Bradley-Terry Elo ratings using penalized Minorization-Maximization (MM)."""
    n = len(participant_names)
    if n == 0:
        return {}
    if n == 1:
        name = participant_names[0]
        return {
            name: {
                "rating": anchor_rating,
                "rating_se": 0.0,
                "rating_ci_95": [anchor_rating, anchor_rating],
            }
        }

    idx = {name: i for i, name in enumerate(participant_names)}
    S = [[0.0] * n for _ in range(n)]
    N = [[0] * n for _ in range(n)]

    for (p1, p2), s1 in pairwise_scores.items():
        if p1 in idx and p2 in idx:
            i, j = idx[p1], idx[p2]
            S[i][j] = s1
            N[i][j] = pairwise_games.get((p1, p2), 0)

    gamma = [1.0] * n
    gamma_0 = 1.0
    W = [sum(S[i][j] for j in range(n) if j != i) for i in range(n)]

    for _ in range(max_iters):
        new_gamma = [0.0] * n
        for i in range(n):
            denom = sum(
                N[i][j] / (gamma[i] + gamma[j])
                for j in range(n)
                if j != i and N[i][j] > 0
            ) + (prior_lambda / (gamma[i] + gamma_0))

            numer = W[i] + (prior_lambda * gamma_0 / (gamma[i] + gamma_0))
            new_gamma[i] = max(1e-8, numer / max(1e-12, denom))

        log_geo = sum(math.log(g) for g in new_gamma) / n
        new_gamma = [g / math.exp(log_geo) for g in new_gamma]

        delta = max(abs(new_gamma[i] - gamma[i]) for i in range(n))
        gamma = new_gamma
        if delta < tol:
            break

    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diag = 0.0
        for j in range(n):
            if i != j and N[i][j] > 0:
                term = N[i][j] * gamma[i] * gamma[j] / ((gamma[i] + gamma[j]) ** 2)
                H[i][j] = -term
                diag += term
        prior_term = prior_lambda * gamma[i] * gamma_0 / ((gamma[i] + gamma_0) ** 2)
        H[i][i] = diag + prior_term

    try:
        import numpy as np
        H_mat = np.array(H, dtype=np.float64)
        inv_H = np.linalg.pinv(H_mat)
        var_mu = [float(inv_H[i, i]) for i in range(n)]
    except Exception:
        var_mu = [1.0 / max(1e-4, H[i][i]) for i in range(n)]

    raw_elo = [400.0 * math.log10(max(1e-12, g)) for g in gamma]
    elo_se = [abs((400.0 / math.log(10.0)) * math.sqrt(max(0.0, v))) for v in var_mu]

    if anchor_name and anchor_name in idx:
        anchor_idx = idx[anchor_name]
        shift = anchor_rating - raw_elo[anchor_idx]
    else:
        shift = anchor_rating - (sum(raw_elo) / n)

    final_ratings: dict[str, dict[str, Any]] = {}
    for name, i in idx.items():
        rating = raw_elo[i] + shift
        se = elo_se[i]
        ci_low = rating - 1.95996 * se
        ci_high = rating + 1.95996 * se
        final_ratings[name] = {
            "rating": round(rating, 2),
            "rating_se": round(se, 2),
            "rating_ci_95": [round(ci_low, 2), round(ci_high, 2)],
        }

    return final_ratings


# -----------------------------------------------------------------------------
# Standings, Cross-Table, and Tiebreaks
# -----------------------------------------------------------------------------

def compute_standings_and_crosstable(
    participants: list[Participant],
    match_results: list[MatchResult],
    anchor_bot: str | None = None,
    anchor_rating: float = 1500.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute standings table, cross-table matrix, tiebreaks, and ratings."""
    names = [p.name for p in participants]
    ptypes = {p.name: p.ptype for p in participants}

    stats: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "ptype": ptypes[name],
            "matches_played": 0,
            "games_played": 0,
            "points": 0.0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "decisive_games": 0,
            "forfeits_conceded": 0,
            "forfeits_received": 0,
            "total_plies": 0,
            "p0_score": 0.0,
            "p0_games": 0,
            "p1_score": 0.0,
            "p1_games": 0,
            "opponents": set(),
        }
        for name in names
    }

    pairwise_scores: dict[tuple[str, str], float] = {}
    pairwise_games: dict[tuple[str, str], int] = {}
    pairwise_records: dict[tuple[str, str], dict[str, Any]] = {}

    for m in match_results:
        pa, pb = m.participant_a, m.participant_b
        if pa not in stats or pb not in stats:
            continue

        stats[pa]["matches_played"] += 1
        stats[pb]["matches_played"] += 1
        stats[pa]["games_played"] += m.games
        stats[pb]["games_played"] += m.games

        stats[pa]["points"] += m.score_a
        stats[pb]["points"] += m.score_b

        stats[pa]["wins"] += m.wins_a
        stats[pa]["losses"] += m.losses_a
        stats[pa]["draws"] += m.draws

        stats[pb]["wins"] += m.losses_a
        stats[pb]["losses"] += m.wins_a
        stats[pb]["draws"] += m.draws

        stats[pa]["decisive_games"] += m.decisive_games
        stats[pb]["decisive_games"] += m.decisive_games

        stats[pa]["forfeits_conceded"] += m.forfeits_a
        stats[pa]["forfeits_received"] += m.forfeits_b
        stats[pb]["forfeits_conceded"] += m.forfeits_b
        stats[pb]["forfeits_received"] += m.forfeits_a

        stats[pa]["total_plies"] += int(round(m.avg_plies * m.games))
        stats[pb]["total_plies"] += int(round(m.avg_plies * m.games))

        stats[pa]["opponents"].add(pb)
        stats[pb]["opponents"].add(pa)

        pairwise_scores[(pa, pb)] = m.score_a
        pairwise_scores[(pb, pa)] = m.score_b
        pairwise_games[(pa, pb)] = m.games
        pairwise_games[(pb, pa)] = m.games

        pairwise_records[(pa, pb)] = {
            "score_a": m.score_a,
            "score_b": m.score_b,
            "wins": m.wins_a,
            "draws": m.draws,
            "losses": m.losses_a,
            "forfeits_a": m.forfeits_a,
            "forfeits_b": m.forfeits_b,
        }
        pairwise_records[(pb, pa)] = {
            "score_a": m.score_b,
            "score_b": m.score_a,
            "wins": m.losses_a,
            "draws": m.draws,
            "losses": m.wins_a,
            "forfeits_a": m.forfeits_b,
            "forfeits_b": m.forfeits_a,
        }

    for name, st in stats.items():
        sb = 0.0
        buchholz = 0.0
        for opp in st["opponents"]:
            opp_points = stats[opp]["points"]
            h2h_score = pairwise_scores.get((name, opp), 0.0)
            sb += h2h_score * opp_points
            buchholz += opp_points
        st["sonneborn_berger"] = round(sb, 2)
        st["buchholz"] = round(buchholz, 2)

    ratings = compute_bradley_terry_ratings(
        participant_names=names,
        pairwise_scores=pairwise_scores,
        pairwise_games=pairwise_games,
        anchor_name=anchor_bot,
        anchor_rating=anchor_rating,
    )

    standings: list[dict[str, Any]] = []
    for name, st in stats.items():
        g = st["games_played"]
        pts = st["points"]
        win_rate = pts / g if g > 0 else 0.0
        decisive_pct = (st["decisive_games"] / g * 100.0) if g > 0 else 0.0
        avg_plies = (st["total_plies"] / g) if g > 0 else 0.0

        r_info = ratings.get(name, {"rating": anchor_rating, "rating_se": 0.0, "rating_ci_95": [anchor_rating, anchor_rating]})

        entry = {
            "name": name,
            "type": st["ptype"],
            "matches": st["matches_played"],
            "games": g,
            "points": pts,
            "points_pct": round(win_rate * 100.0, 2),
            "wins": st["wins"],
            "draws": st["draws"],
            "losses": st["losses"],
            "decisive_games": st["decisive_games"],
            "decisive_pct": round(decisive_pct, 2),
            "forfeits_conceded": st["forfeits_conceded"],
            "forfeits_received": st["forfeits_received"],
            "avg_plies": round(avg_plies, 2),
            "rating": r_info["rating"],
            "rating_se": r_info["rating_se"],
            "rating_ci_95": r_info["rating_ci_95"],
            "sonneborn_berger": st["sonneborn_berger"],
            "buchholz": st["buchholz"],
        }
        standings.append(entry)

    standings.sort(
        key=lambda x: (
            -x["points"],
            -x["sonneborn_berger"],
            -x["buchholz"],
            -x["decisive_games"],
            -x["wins"],
        )
    )

    for rank, item in enumerate(standings, start=1):
        item["rank"] = rank

    crosstable_rows = []
    for p1 in standings:
        row = {"name": p1["name"], "scores": {}}
        for p2 in standings:
            n1, n2 = p1["name"], p2["name"]
            if n1 == n2:
                row["scores"][n2] = {"self": True, "text": "-"}
            elif (n1, n2) in pairwise_records:
                rec = pairwise_records[(n1, n2)]
                row["scores"][n2] = {
                    "self": False,
                    "score": rec["score_a"],
                    "opp_score": rec["score_b"],
                    "wins": rec["wins"],
                    "draws": rec["draws"],
                    "losses": rec["losses"],
                    "text": f"{rec['score_a']:g}-{rec['score_b']:g}",
                }
            else:
                row["scores"][n2] = {"self": False, "unplayed": True, "text": "N/A"}
        crosstable_rows.append(row)

    cross_table = {
        "participants": [p["name"] for p in standings],
        "rows": crosstable_rows,
    }

    return standings, cross_table


# -----------------------------------------------------------------------------
# Tournament Coordinators (Round-Robin & Swiss)
# -----------------------------------------------------------------------------

def run_round_robin(
    participants: list[Participant],
    games_per_matchup: int,
    base_seed: int = 0,
    max_plies: int = 220,
    anchor_bot: str | None = None,
    anchor_rating: float = 1500.0,
    progress_cb: Callable[[int, int, str, str], None] | None = None,
) -> tuple[list[MatchResult], list[dict[str, Any]], dict[str, Any]]:
    """Run a complete Round-Robin tournament between all participant pairs."""
    match_results: list[MatchResult] = []
    n = len(participants)
    pairs = [(participants[i], participants[j]) for i in range(n) for j in range(i + 1, n)]
    total_matches = len(pairs)

    for idx, (pa, pb) in enumerate(pairs, start=1):
        if progress_cb:
            progress_cb(idx, total_matches, pa.name, pb.name)

        match_seed = base_seed + idx * 1009
        res = play_matchup(pa, pb, games=games_per_matchup, base_seed=match_seed, max_plies=max_plies)
        match_results.append(res)

    standings, crosstable = compute_standings_and_crosstable(
        participants, match_results, anchor_bot=anchor_bot, anchor_rating=anchor_rating
    )
    return match_results, standings, crosstable


def run_swiss_tournament(
    participants: list[Participant],
    rounds: int,
    games_per_matchup: int,
    base_seed: int = 0,
    max_plies: int = 220,
    anchor_bot: str | None = None,
    anchor_rating: float = 1500.0,
    progress_cb: Callable[[int, int, str, str], None] | None = None,
) -> tuple[list[MatchResult], list[dict[str, Any]], dict[str, Any]]:
    """Run a Swiss tournament across a specified number of rounds."""
    n = len(participants)
    if n < 2:
        standings, crosstable = compute_standings_and_crosstable(
            participants, [], anchor_bot=anchor_bot, anchor_rating=anchor_rating
        )
        return [], standings, crosstable

    p_map = {p.name: p for p in participants}
    played_pairs: set[frozenset[str]] = set()
    match_results: list[MatchResult] = []
    match_counter = 0
    total_planned = rounds * (n // 2)

    for r in range(rounds):
        curr_standings, _ = compute_standings_and_crosstable(
            participants, match_results, anchor_bot=anchor_bot, anchor_rating=anchor_rating
        )
        ordered_names = [st["name"] for st in curr_standings]

        paired_this_round: list[tuple[str, str]] = []
        unpaired = list(ordered_names)

        if len(unpaired) % 2 == 1:
            unpaired.pop()
            match_counter += 1

        while unpaired:
            p1 = unpaired.pop(0)
            candidate_idx = None
            for idx, cand in enumerate(unpaired):
                if frozenset({p1, cand}) not in played_pairs:
                    candidate_idx = idx
                    break

            if candidate_idx is not None:
                p2 = unpaired.pop(candidate_idx)
                paired_this_round.append((p1, p2))
                played_pairs.add(frozenset({p1, p2}))
            else:
                if unpaired:
                    p2 = unpaired.pop(0)
                    paired_this_round.append((p1, p2))
                    played_pairs.add(frozenset({p1, p2}))

        for p1_name, p2_name in paired_this_round:
            match_counter += 1
            pa = p_map[p1_name]
            pb = p_map[p2_name]

            if progress_cb:
                progress_cb(match_counter, total_planned, pa.name, pb.name)

            m_seed = base_seed + (r + 1) * 10007 + match_counter * 31
            res = play_matchup(pa, pb, games=games_per_matchup, base_seed=m_seed, max_plies=max_plies)
            match_results.append(res)

    final_standings, final_crosstable = compute_standings_and_crosstable(
        participants, match_results, anchor_bot=anchor_bot, anchor_rating=anchor_rating
    )
    return match_results, final_standings, final_crosstable


# -----------------------------------------------------------------------------
# Reporting & Output Generation
# -----------------------------------------------------------------------------

def format_markdown_report(
    tournament_data: dict[str, Any],
) -> str:
    """Render a comprehensive Markdown tournament report."""
    meta = tournament_data.get("metadata", {})
    cfg = tournament_data.get("config", {})
    standings = tournament_data.get("standings", [])
    crosstable = tournament_data.get("cross_table", {})
    matchups = tournament_data.get("matchups", [])

    t_id = meta.get("tournament_id", "tournament")
    t_name = meta.get("name", "Qudor Tournament")
    t_format = cfg.get("format", "round-robin").title()
    created_at = meta.get("created_at", "unknown")
    total_games = meta.get("total_games", 0)

    lines: list[str] = [
        f"# Qudor AI — {t_name}",
        "",
        f"**Tournament ID**: `{t_id}`  ",
        f"**Format**: {t_format}  ",
        f"**Generated**: {created_at}  ",
        f"**Total Competitors**: {len(standings)}  ",
        f"**Total Games Played**: {total_games}  ",
        f"**Matchup Games**: {cfg.get('games_per_matchup', 2)} per pair  ",
        "",
        "---",
        "",
        "## 1. Final Standings & Ratings",
        "",
        "| Rank | Participant | Type | Games | W / D / L | Points (Score) | Win Rate | Decisive % | Forfeits (C / R) | Rating (Bradley-Terry) | SB / Buchholz |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for st in standings:
        r = st.get("rank", "-")
        name = st.get("name", "unknown")
        ptype = st.get("type", "-")
        g = st.get("games", 0)
        wdl = f"{st.get('wins', 0)} / {st.get('draws', 0)} / {st.get('losses', 0)}"
        pts = f"{st.get('points', 0.0):g} / {g}"
        wr = f"{st.get('points_pct', 0.0):.1f}%"
        decisive = f"{st.get('decisive_pct', 0.0):.1f}%"
        forfeits = f"{st.get('forfeits_conceded', 0)} / {st.get('forfeits_received', 0)}"
        rating = st.get("rating", 1500.0)
        se = st.get("rating_se", 0.0)
        rating_str = f"**{rating:+.1f}** ± {se:.1f}"
        sb_buch = f"{st.get('sonneborn_berger', 0.0):.1f} / {st.get('buchholz', 0.0):.1f}"

        lines.append(
            f"| {r} | **{name}** | `{ptype}` | {g} | {wdl} | {pts} | {wr} | {decisive} | {forfeits} | {rating_str} | {sb_buch} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Head-to-Head Cross-Table",
        "",
    ])

    participants = crosstable.get("participants", [])
    if participants:
        header = "| # | Participant | " + " | ".join(f"**{i+1}**" for i in range(len(participants))) + " | Points |"
        sep = "| :---: | :--- | " + " | ".join(":---:" for _ in participants) + " | :---: |"
        lines.extend([header, sep])

        for i, row in enumerate(crosstable.get("rows", [])):
            name = row["name"]
            score_cells = []
            row_points = 0.0
            for opp in participants:
                cell_data = row["scores"].get(opp, {})
                if cell_data.get("self"):
                    score_cells.append("x")
                elif cell_data.get("unplayed"):
                    score_cells.append("-")
                else:
                    txt = cell_data.get("text", "-")
                    score_cells.append(f"`{txt}`")
                    row_points += float(cell_data.get("score", 0.0))

            lines.append(f"| **{i+1}** | **{name}** | " + " | ".join(score_cells) + f" | **{row_points:g}** |")
    else:
        lines.append("*No cross-table data available.*")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Matchup Breakdown",
        "",
        "| Matchup | Games | Outcome | W / D / L | Forfeits | Avg Plies | P0 / P1 Win Rate |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m in matchups:
        pa = m.get("participant_a", "A")
        pb = m.get("participant_b", "B")
        g = m.get("games", 0)
        sa = m.get("score_a", 0.0)
        sb = m.get("score_b", 0.0)
        wdl = f"{m.get('wins_a', 0)} / {m.get('draws', 0)} / {m.get('losses_a', 0)}"
        forfeits = f"{m.get('forfeits_a', 0)} / {m.get('forfeits_b', 0)}"
        plies = f"{m.get('avg_plies', 0.0):.1f}"
        p0_p1 = f"{m.get('win_rate_a_p0', 0.0):.2f} / {m.get('win_rate_a_p1', 0.0):.2f}"
        outcome = f"**{sa:g} - {sb:g}**"

        lines.append(f"| **{pa}** vs **{pb}** | {g} | {outcome} | {wdl} | {forfeits} | {plies} | {p0_p1} |")

    lines.append("")
    return "\n".join(lines)


def save_tournament_output(
    tournament_data: dict[str, Any],
    json_path: Path | str,
    md_path: Path | str | None = None,
) -> None:
    """Save structured JSON tournament results and optional Markdown report atomically."""
    j_path = Path(json_path).resolve()
    j_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_json = j_path.with_suffix(j_path.suffix + ".tmp")
    tmp_json.write_text(json.dumps(tournament_data, indent=2) + "\n", encoding="utf-8")
    tmp_json.replace(j_path)

    if md_path:
        m_path = Path(md_path).resolve()
        m_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = format_markdown_report(tournament_data)
        tmp_md = m_path.with_suffix(m_path.suffix + ".tmp")
        tmp_md.write_text(md_text, encoding="utf-8")
        tmp_md.replace(m_path)


# -----------------------------------------------------------------------------
# CLI and Coordinator Execution
# -----------------------------------------------------------------------------

def build_participants(args: argparse.Namespace, dev: Any) -> list[Participant]:
    """Instantiate participant objects from command line arguments."""
    participants: list[Participant] = []
    seen_names: set[str] = set()

    def _add(p: Participant):
        if p.name in seen_names:
            p.name = f"{p.name}_{len(seen_names)}"
        seen_names.add(p.name)
        participants.append(p)

    # 1. Baseline bots
    if args.baselines:
        selected_baselines = args.baselines
        if "all" in selected_baselines:
            selected_baselines = list(BASELINE_BOTS.keys())
        for b_name in selected_baselines:
            if b_name in BASELINE_BOTS:
                _add(BaselineParticipant(name=b_name, bot_fn=BASELINE_BOTS[b_name]))
            else:
                print(f"  [WARN] Unknown baseline bot: {b_name}", flush=True)

    # 2. Checkpoints
    if args.checkpoints:
        for ck_str in args.checkpoints:
            ck_path = Path(ck_str)
            if not ck_path.is_file():
                print(f"  [WARN] Checkpoint not found: {ck_path}", flush=True)
                continue
            name = ck_path.stem
            _add(
                CheckpointParticipant(
                    name=name,
                    checkpoint_path=ck_path,
                    device=dev,
                    sims=args.sims,
                    c_puct=args.c_puct,
                    temp=args.temp,
                    gumbel=not args.puct,
                    gumbel_cap=args.gumbel_cap,
                    max_plies=args.max_plies,
                )
            )

    # 3. Foreign Bridges
    bridge_names: list[str] = []
    if args.all_bridges:
        bridge_names = sorted(BUILTIN_BRIDGES.keys())
    elif args.bridges:
        bridge_names = args.bridges

    logdir = Path(args.logdir) if args.logdir else repo_root() / "runs" / "tournament_logs"
    for b_name in bridge_names:
        if b_name in BUILTIN_BRIDGES:
            _add(
                BridgeParticipant(
                    name=b_name,
                    argv=BUILTIN_BRIDGES[b_name],
                    logdir=logdir,
                    move_timeout=args.move_timeout,
                    handshake_timeout=args.handshake_timeout,
                )
            )
        else:
            print(f"  [WARN] Unknown foreign bridge: {b_name}", flush=True)

    # 4. Custom Bridges: name=cmd_or_script
    if args.custom_bridge:
        for spec in args.custom_bridge:
            if "=" not in spec:
                print(f"  [WARN] Invalid --custom-bridge format '{spec}'; expected NAME=PATH_OR_CMD", flush=True)
                continue
            name, cmd_str = spec.split("=", 1)
            argv = [sys.executable, cmd_str] if cmd_str.endswith(".py") else cmd_str.split()
            _add(
                BridgeParticipant(
                    name=name.strip(),
                    argv=argv,
                    logdir=logdir,
                    move_timeout=args.move_timeout,
                    handshake_timeout=args.handshake_timeout,
                )
            )

    return participants


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Automated Multi-Engine Tournament Coordinator for Qudor AI"
    )
    p.add_argument(
        "--format",
        choices=["round-robin", "swiss"],
        default="round-robin",
        help="Tournament pairing format (default: round-robin)",
    )
    p.add_argument(
        "--games",
        "--games-per-matchup",
        dest="games",
        type=int,
        default=2,
        help="Number of games per matchup (alternating colors, default: 2)",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of Swiss tournament rounds (only applicable when format=swiss, default: 3)",
    )
    p.add_argument(
        "--checkpoints",
        "--nets",
        nargs="+",
        default=[],
        help="One or more checkpoint paths (.pt files) to enter into the tournament",
    )
    p.add_argument(
        "--bridges",
        nargs="+",
        default=[],
        choices=sorted(BUILTIN_BRIDGES.keys()),
        help="Foreign engine bridges to enter (vader, berlioz, dimi, giorgos, gorisanson, marcobt15, pavlosdais, sigma, cryer)",
    )
    p.add_argument(
        "--all-bridges",
        action="store_true",
        help="Include all available third-party bot bridges in tools/bridges/",
    )
    p.add_argument(
        "--baselines",
        nargs="+",
        default=[],
        help="Baseline bots to enter (rusher, greedy, all)",
    )
    p.add_argument(
        "--custom-bridge",
        action="append",
        default=[],
        help="Custom external bridge specification as NAME=SCRIPT_OR_COMMAND",
    )
    p.add_argument(
        "--sims",
        type=int,
        default=64,
        help="MCTS simulations per move for checkpoints (default: 64)",
    )
    p.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="Sampling temperature for checkpoints (default: 0.6)",
    )
    p.add_argument(
        "--c-puct",
        type=float,
        default=1.6,
        help="PUCT exploration constant (default: 1.6)",
    )
    p.add_argument(
        "--puct",
        action="store_true",
        help="Use standard PUCT MCTS instead of default Gumbel MCTS",
    )
    p.add_argument(
        "--gumbel-cap",
        type=int,
        default=16,
        help="Gumbel top-k action consideration cap (default: 16)",
    )
    p.add_argument(
        "--max-plies",
        type=int,
        default=220,
        help="Max plies per game before declaring draw (default: 220)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    p.add_argument(
        "--move-timeout",
        type=float,
        default=MOVE_TIMEOUT,
        help=f"Per-move timeout in seconds for bridge subprocesses (default: {MOVE_TIMEOUT})",
    )
    p.add_argument(
        "--handshake-timeout",
        type=float,
        default=HANDSHAKE_TIMEOUT,
        help=f"Handshake timeout in seconds for bridge subprocesses (default: {HANDSHAKE_TIMEOUT})",
    )
    p.add_argument(
        "--device",
        type=str,
        help="'cpu' or 'cuda'; overrides auto-detection",
    )
    p.add_argument(
        "--threads",
        type=int,
        help="Intra-op threads for PyTorch CPU operations",
    )
    p.add_argument(
        "--logdir",
        type=str,
        help="Directory for bridge stdout/stderr logs and illegal dumps",
    )
    p.add_argument(
        "--anchor-bot",
        type=str,
        default=None,
        help="Participant name to anchor the Elo scale against (e.g. 'rusher')",
    )
    p.add_argument(
        "--anchor-rating",
        type=float,
        default=1500.0,
        help="Elo rating assigned to the anchor bot (default: 1500.0)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for structured JSON results (defaults to results/tournaments/tournament_<id>.json)",
    )
    p.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Output path for Markdown summary report (defaults to results/tournaments/tournament_<id>.md)",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip updating results/MANIFEST.json and results/SUMMARY.md",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose tournament progress logs",
    )
    args = p.parse_args(argv)

    if args.games <= 0:
        p.error("--games must be positive")

    dev = resolve_device(args.device)
    configure_threads(args.threads)

    participants = build_participants(args, dev)
    if len(participants) < 2:
        p.error(
            f"At least 2 participants are required to run a tournament; found {len(participants)}."
        )

    t_id = f"tournament_{_now_id()}"
    t_name = f"Qudor {args.format.title()} Tournament ({len(participants)} Engines)"

    if not args.quiet:
        print("=" * 70)
        print(f" {t_name}")
        print("=" * 70)
        print(f" Format:              {args.format}")
        print(f" Games per matchup:   {args.games}")
        print(f" Competitors ({len(participants)}):")
        for i, pt in enumerate(participants, 1):
            print(f"   {i:2d}. {pt.name:<20} ({pt.ptype})")
        print("=" * 70)

    start_time = time.time()

    def progress_callback(curr: int, total: int, pa: str, pb: str):
        if not args.quiet:
            print(f"  [{curr}/{total}] Matchup: {pa} vs {pb} ({args.games} games)...", flush=True)

    if args.format == "swiss":
        matches, standings, crosstable = run_swiss_tournament(
            participants=participants,
            rounds=args.rounds,
            games_per_matchup=args.games,
            base_seed=args.seed,
            max_plies=args.max_plies,
            anchor_bot=args.anchor_bot,
            anchor_rating=args.anchor_rating,
            progress_cb=progress_callback,
        )
    else:
        matches, standings, crosstable = run_round_robin(
            participants=participants,
            games_per_matchup=args.games,
            base_seed=args.seed,
            max_plies=args.max_plies,
            anchor_bot=args.anchor_bot,
            anchor_rating=args.anchor_rating,
            progress_cb=progress_callback,
        )

    elapsed = time.time() - start_time
    total_games = sum(m.games for m in matches)

    match_dicts = [asdict(m) for m in matches]

    tournament_data: dict[str, Any] = {
        "metadata": {
            "tournament_id": t_id,
            "name": t_name,
            "created_at": _now_utc(),
            "elapsed_seconds": round(elapsed, 2),
            "total_matches": len(matches),
            "total_games": total_games,
            "num_participants": len(participants),
            "winner": standings[0]["name"] if standings else None,
        },
        "config": {
            "format": args.format,
            "games_per_matchup": args.games,
            "rounds": args.rounds if args.format == "swiss" else None,
            "sims": args.sims,
            "temp": args.temp,
            "puct": args.puct,
            "gumbel": not args.puct,
            "seed": args.seed,
            "max_plies": args.max_plies,
            "anchor_bot": args.anchor_bot,
            "anchor_rating": args.anchor_rating,
        },
        "participants": [{"name": p.name, "type": p.ptype} for p in participants],
        "standings": standings,
        "cross_table": crosstable,
        "matchups": match_dicts,
    }

    r_dir = repo_root() / "results" / "tournaments"
    r_dir.mkdir(parents=True, exist_ok=True)

    json_file = Path(args.output) if args.output else (r_dir / f"{t_id}.json")
    md_file = Path(args.output_md) if args.output_md else (r_dir / f"{t_id}.md")

    save_tournament_output(tournament_data, json_file, md_file)

    if not args.quiet:
        print("\n" + "=" * 70)
        print(f" TOURNAMENT COMPLETED in {elapsed:.1f}s — STANDINGS")
        print("=" * 70)
        print(f"{'Rank':<5} {'Participant':<18} {'Points':<10} {'W/D/L':<12} {'Decisive%':<10} {'Elo Rating':<18}")
        print("-" * 70)
        for st in standings:
            r = st['rank']
            name = st['name']
            pts = f"{st['points']:g}/{st['games']}"
            wdl = f"{st['wins']}/{st['draws']}/{st['losses']}"
            dec = f"{st['decisive_pct']:.1f}%"
            elo = f"{st['rating']:+.1f} ± {st['rating_se']:.1f}"
            print(f"{r:<5} {name:<18} {pts:<10} {wdl:<12} {dec:<10} {elo:<18}")
        print("=" * 70)
        print(f" JSON results: {json_file}")
        print(f" Markdown report: {md_file}")

    # Synchronize metrics manifest
    if not args.no_manifest:
        try:
            from quoridor_ai.results_manifest import generate_manifest
            generate_manifest(repo_dir=repo_root(), write_files=True)
            if not args.quiet:
                print(" Updated results/MANIFEST.json and results/SUMMARY.md")
        except Exception as e:
            if not args.quiet:
                print(f" [WARN] Failed to update metrics manifest: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
