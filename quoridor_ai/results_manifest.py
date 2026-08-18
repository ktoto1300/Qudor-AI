"""Unified results index and metrics manifest generator for Qudor AI.

Scans the `results/` directory (and subdirectories), collects foreign arena benchmarks,
baseline bot evaluations, evaluation summaries, and training configs, and synthesises:
1. `results/MANIFEST.json`: Machine-readable structured database of all run outputs.
2. `results/SUMMARY.md`: Markdown summary tables for reports and documentation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quoridor_ai.paths import repo_root


SCHEMA_VERSION = "1.0.0"

# Directories and file patterns to ignore during scan
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "bridge_logs",
    "runs",
}
IGNORE_EXTENSIONS = {
    ".log",
    ".err",
    ".pid",
    ".tmp",
    ".jsonl",
    ".py",
    ".pyc",
}
IGNORE_FILENAMES = {
    "MANIFEST.json",
    "SUMMARY.md",
}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def get_git_metadata(repo_path: Path | None = None) -> dict[str, Any]:
    """Retrieve Git commit, branch, and dirty state if available."""
    target_repo = repo_path or repo_root()
    meta: dict[str, Any] = {
        "commit": "unknown",
        "branch": "unknown",
        "is_dirty": False,
    }
    try:
        res_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_commit.returncode == 0:
            meta["commit"] = res_commit.stdout.strip()

        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=target_repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_branch.returncode == 0:
            meta["branch"] = res_branch.stdout.strip()

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=target_repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_status.returncode == 0:
            meta["is_dirty"] = bool(res_status.stdout.strip())
    except Exception:
        pass
    return meta


def _format_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as UTC ISO string."""
    d = dt or datetime.now(timezone.utc)
    return d.strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_foreign_arena_entry(rel_path: str, data: dict[str, Any], sha256: str, mtime_iso: str) -> dict[str, Any]:
    """Parse and normalize a foreign arena match record."""
    opponent = str(data.get("opponent", "unknown"))
    games = _safe_int(data.get("games")) or 0
    wins = _safe_int(data.get("wins")) or 0
    draws = _safe_int(data.get("draws")) or 0
    losses = _safe_int(data.get("losses")) or 0
    opp_forfeits = _safe_int(data.get("opp_forfeits")) or 0

    win_rate = _safe_float(data.get("win_rate"))
    if win_rate is None and games > 0:
        win_rate = wins / games

    elo_delta = _safe_float(data.get("elo_delta"))
    avg_plies = _safe_float(data.get("avg_plies"))
    sims = _safe_int(data.get("sims"))
    temperature = _safe_float(data.get("temperature"))
    gumbel = data.get("gumbel")
    seed = _safe_int(data.get("seed"))

    net = str(data.get("net", ""))
    net_name = str(data.get("net_name", Path(net).name if net else ""))
    generation = _safe_int(data.get("generation"))
    iteration = _safe_int(data.get("iteration"))
    device = str(data.get("device", "unknown"))

    return {
        "file": rel_path,
        "opponent": opponent,
        "net": net,
        "net_name": net_name,
        "generation": generation,
        "iteration": iteration,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "opp_forfeits": opp_forfeits,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
        "elo_delta": round(elo_delta, 2) if elo_delta is not None else None,
        "win_rate_as_p0": _safe_float(data.get("win_rate_as_p0")),
        "win_rate_as_p1": _safe_float(data.get("win_rate_as_p1")),
        "avg_plies": round(avg_plies, 2) if avg_plies is not None else None,
        "sims": sims,
        "temperature": temperature,
        "gumbel": bool(gumbel) if gumbel is not None else None,
        "seed": seed,
        "device": device,
        "sha256": sha256,
        "mtime": mtime_iso,
    }


def _parse_baseline_entry(
    rel_path: str,
    data: dict[str, Any],
    sha256: str,
    mtime_iso: str,
    bot_override: str | None = None,
) -> dict[str, Any]:
    """Parse and normalize a baseline evaluation record."""
    bot = bot_override or str(data.get("bot", data.get("baseline", "unknown")))
    games = _safe_int(data.get("games")) or 0
    wins = _safe_int(data.get("wins")) or 0
    draws = _safe_int(data.get("draws")) or 0
    losses = _safe_int(data.get("losses")) or 0

    win_rate = _safe_float(data.get("win_rate"))
    if win_rate is None and games > 0:
        win_rate = wins / games

    elo_delta = _safe_float(data.get("elo_delta"))
    avg_plies = _safe_float(data.get("avg_plies"))
    sims = _safe_int(data.get("sims"))
    temperature = _safe_float(data.get("temperature"))
    gumbel = data.get("gumbel")
    seed = _safe_int(data.get("seed"))

    net = str(data.get("net", data.get("checkpoint", "")))
    generation = _safe_int(data.get("generation"))
    iteration = _safe_int(data.get("iteration"))
    device = str(data.get("device", "unknown"))

    return {
        "file": rel_path,
        "bot": bot,
        "net": net,
        "generation": generation,
        "iteration": iteration,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
        "elo_delta": round(elo_delta, 2) if elo_delta is not None else None,
        "win_rate_as_p0": _safe_float(data.get("win_rate_as_p0")),
        "win_rate_as_p1": _safe_float(data.get("win_rate_as_p1")),
        "avg_plies": round(avg_plies, 2) if avg_plies is not None else None,
        "sims": sims,
        "temperature": temperature,
        "gumbel": bool(gumbel) if gumbel is not None else None,
        "seed": seed,
        "device": device,
        "sha256": sha256,
        "mtime": mtime_iso,
    }


def _parse_training_config(rel_path: str, data: dict[str, Any], sha256: str, mtime_iso: str) -> dict[str, Any]:
    """Parse and normalize a training configuration record."""
    clean_params = {k: v for k, v in data.items() if not k.startswith("_")}
    return {
        "file": rel_path,
        "seed": _safe_int(data.get("seed")),
        "encoding": _safe_int(data.get("encoding")),
        "iterations": _safe_int(data.get("iterations")),
        "total_steps": _safe_int(data.get("total_steps")),
        "channels": _safe_int(data.get("channels")),
        "blocks": _safe_int(data.get("blocks")),
        "se": bool(data.get("se", False)),
        "gumbel": bool(data.get("gumbel", False)),
        "gumbel_cap": _safe_int(data.get("gumbel_cap")),
        "games": _safe_int(data.get("games")),
        "sims": _safe_int(data.get("sims")),
        "fast_sims": _safe_int(data.get("fast_sims")),
        "batch": _safe_int(data.get("batch")),
        "lr": _safe_float(data.get("lr")),
        "warmup_steps": _safe_int(data.get("warmup_steps")),
        "replay": _safe_int(data.get("replay")),
        "device": str(data.get("device", "unknown")),
        "threads": _safe_int(data.get("threads")),
        "parameters": clean_params,
        "sha256": sha256,
        "mtime": mtime_iso,
    }


def _parse_tournament_entry(rel_path: str, data: dict[str, Any], sha256: str, mtime_iso: str) -> dict[str, Any]:
    """Parse and normalize a multi-engine tournament record."""
    meta = data.get("metadata", {})
    cfg = data.get("config", {})
    standings = data.get("standings", [])
    crosstable = data.get("cross_table", {})
    matchups = data.get("matchups", [])

    t_id = str(meta.get("tournament_id", data.get("tournament_id", Path(rel_path).stem)))
    t_name = str(meta.get("name", data.get("name", "Tournament")))
    t_format = str(cfg.get("format", data.get("format", "round-robin")))
    total_games = _safe_int(meta.get("total_games", data.get("total_games")))
    if total_games is None:
        total_games = sum(_safe_int(m.get("games")) or 0 for m in matchups) if matchups else 0

    winner = meta.get("winner") or (standings[0].get("name") if standings else "unknown")
    num_participants = len(standings) if standings else len(data.get("participants", []))

    return {
        "file": rel_path,
        "tournament_id": t_id,
        "name": t_name,
        "format": t_format,
        "num_participants": num_participants,
        "total_games": total_games,
        "winner": winner,
        "standings": standings,
        "cross_table": crosstable,
        "matchups_count": len(matchups),
        "sha256": sha256,
        "mtime": mtime_iso,
    }


def scan_results_directory(results_dir: Path | str) -> dict[str, Any]:
    """Recursively scan `results_dir` and collect structured manifests."""
    base_path = Path(results_dir).resolve()
    if not base_path.exists() or not base_path.is_dir():
        return {
            "foreign_arena": [],
            "baseline_evaluations": [],
            "evaluation_summaries": [],
            "training_configs": [],
            "tournaments": [],
            "checksums": [],
            "scanned_files": 0,
            "errors": [f"Directory not found: {base_path}"],
        }

    foreign_arena: list[dict[str, Any]] = []
    baseline_evaluations: list[dict[str, Any]] = []
    evaluation_summaries: list[dict[str, Any]] = []
    training_configs: list[dict[str, Any]] = []
    tournaments: list[dict[str, Any]] = []
    checksums: list[dict[str, Any]] = []
    errors: list[str] = []
    nested_baseline_candidates: list[dict[str, Any]] = []
    scanned_count = 0

    for root_str, dirs, files in os.walk(base_path):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        root_path = Path(root_str)
        for filename in sorted(files):
            if filename in IGNORE_FILENAMES:
                continue

            file_path = root_path / filename
            ext = file_path.suffix.lower()
            if ext in IGNORE_EXTENSIONS:
                continue

            try:
                rel_path = file_path.relative_to(base_path).as_posix()
            except ValueError:
                rel_path = file_path.name

            scanned_count += 1
            sha256 = compute_file_sha256(file_path)
            try:
                mtime_iso = datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")
            except OSError:
                mtime_iso = "unknown"

            if ext == ".sha256":
                try:
                    content = file_path.read_text(encoding="utf-8").strip()
                    checksums.append({
                        "file": rel_path,
                        "content": content,
                        "sha256": sha256,
                        "mtime": mtime_iso,
                    })
                except OSError as e:
                    errors.append(f"Failed to read {rel_path}: {e}")
                continue

            if ext == ".json":
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception as e:
                    errors.append(f"JSON decode error in {rel_path}: {e}")
                    continue

                if not isinstance(data, dict):
                    continue

                # 1. Multi-Engine Tournament result
                if (
                    "standings" in data
                    or "cross_table" in data
                    or filename.startswith("tournament_")
                    or "tournament_id" in data
                    or "tournament_id" in data.get("metadata", {})
                ):
                    tournaments.append(
                        _parse_tournament_entry(rel_path, data, sha256, mtime_iso)
                    )
                # 2. Foreign Arena result
                elif "opponent" in data or filename.startswith("foreign_"):
                    foreign_arena.append(
                        _parse_foreign_arena_entry(rel_path, data, sha256, mtime_iso)
                    )
                # 3. Evaluation Summary containing nested bot evaluations
                elif "evaluation" in data and isinstance(data["evaluation"], dict):
                    summary_entry = {
                        "file": rel_path,
                        "checkpoint": str(data.get("checkpoint", "")),
                        "training_exit_code": data.get("training_exit_code"),
                        "raw": data,
                        "sha256": sha256,
                        "mtime": mtime_iso,
                    }
                    evaluation_summaries.append(summary_entry)
                    # Extract nested bot results
                    eval_dict = data["evaluation"]
                    for bot_key, bot_data in eval_dict.items():
                        if isinstance(bot_data, dict):
                            nested_baseline_candidates.append(
                                _parse_baseline_entry(
                                    f"{rel_path}#{bot_key}",
                                    bot_data,
                                    sha256,
                                    mtime_iso,
                                    bot_override=bot_key,
                                )
                            )
                # 4. Individual Baseline Evaluation
                elif "bot" in data or "baseline" in data or filename.startswith("eval_") or filename.startswith("baseline_"):
                    baseline_evaluations.append(
                        _parse_baseline_entry(rel_path, data, sha256, mtime_iso)
                    )
                # 5. Training configuration
                elif any(k in data for k in ("total_steps", "encoding", "channels", "blocks", "replay", "gumbel_cap", "warmup_steps")):
                    training_configs.append(
                        _parse_training_config(rel_path, data, sha256, mtime_iso)
                    )

    # Deduplicate nested baseline candidates against standalone baseline files
    existing_signatures = {
        (b.get("bot"), b.get("net"), b.get("games"), b.get("wins"), b.get("losses"))
        for b in baseline_evaluations
    }
    for candidate in nested_baseline_candidates:
        sig = (candidate.get("bot"), candidate.get("net"), candidate.get("games"), candidate.get("wins"), candidate.get("losses"))
        if sig not in existing_signatures:
            baseline_evaluations.append(candidate)
            existing_signatures.add(sig)

    # Sort outputs deterministically
    tournaments.sort(key=lambda x: (x.get("tournament_id") or "", x.get("file") or ""))
    foreign_arena.sort(key=lambda x: (x.get("opponent") or "", x.get("net") or "", x.get("file") or ""))
    baseline_evaluations.sort(key=lambda x: (x.get("bot") or "", x.get("net") or "", x.get("file") or ""))
    training_configs.sort(key=lambda x: x.get("file") or "")
    evaluation_summaries.sort(key=lambda x: x.get("file") or "")
    checksums.sort(key=lambda x: x.get("file") or "")

    return {
        "tournaments": tournaments,
        "foreign_arena": foreign_arena,
        "baseline_evaluations": baseline_evaluations,
        "evaluation_summaries": evaluation_summaries,
        "training_configs": training_configs,
        "checksums": checksums,
        "scanned_files": scanned_count,
        "errors": errors,
    }


def generate_markdown_summary(manifest_data: dict[str, Any]) -> str:
    """Render a clean Markdown summary document from manifest data."""
    meta = manifest_data.get("metadata", {})
    git = manifest_data.get("git", {})
    foreign = manifest_data.get("foreign_arena", [])
    baselines = manifest_data.get("baseline_evaluations", [])
    configs = manifest_data.get("training_configs", [])
    sums = manifest_data.get("checksums", [])

    lines: list[str] = [
        "# Qudor AI — Results & Metrics Manifest",
        "",
        "This document is automatically generated by `tools/metrics_manifest.py`. "
        "It catalogs foreign engine matches, baseline benchmarks, and training configurations.",
        "",
        "---",
        "",
        "## 1. Manifest Metadata",
        "",
        "| Attribute | Value |",
        "| :--- | :--- |",
        f"| **Generated At** | `{meta.get('generated_at', 'unknown')}` |",
        f"| **Schema Version** | `{meta.get('schema_version', SCHEMA_VERSION)}` |",
        f"| **Git Commit** | `{git.get('commit', 'unknown')}` |",
        f"| **Git Branch** | `{git.get('branch', 'unknown')}` |",
        f"| **Working Tree Dirty** | `{'Yes' if git.get('is_dirty') else 'Clean'}` |",
        f"| **Scanned Artifacts** | `{meta.get('total_scanned_files', 0)} files` |",
        "",
        "---",
        "",
        "## 2. Foreign Engine Arena Outcomes",
        "",
    ]

    if foreign:
        lines.extend([
            "| Opponent | Checkpoint / Net | Gen (Iter) | Games | W / D / L | Win Rate | Elo Δ | Avg Plies | Playouts | Device | Source File |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])
        for entry in foreign:
            opp = entry.get("opponent", "unknown")
            net = entry.get("net_name") or Path(entry.get("net", "")).name or "unknown"
            gen = entry.get("generation")
            it = entry.get("iteration")
            gen_str = f"g{gen}" if gen is not None else "-"
            if it is not None:
                gen_str += f" (i{it})"
            games = entry.get("games", 0)
            wdl = f"{entry.get('wins', 0)} / {entry.get('draws', 0)} / {entry.get('losses', 0)}"
            wr = f"{entry.get('win_rate_pct', 0.0):.1f}%" if entry.get("win_rate_pct") is not None else "-"
            elo = f"{entry.get('elo_delta', 0.0):+.1f}" if entry.get("elo_delta") is not None else "-"
            plies = f"{entry.get('avg_plies', 0.0):.1f}" if entry.get("avg_plies") is not None else "-"
            sims = entry.get("sims")
            gumbel = " (Gumbel)" if entry.get("gumbel") else ""
            playouts = f"{sims}{gumbel}" if sims else "-"
            dev = entry.get("device", "-")
            src = f"`{entry.get('file', '')}`"
            lines.append(
                f"| **{opp}** | `{net}` | {gen_str} | {games} | {wdl} | **{wr}** | {elo} | {plies} | {playouts} | {dev} | {src} |"
            )
    else:
        lines.append("*No foreign arena match records found.*")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Baseline Bot Evaluations",
        "",
    ])

    if baselines:
        lines.extend([
            "| Bot | Checkpoint / Net | Gen (Iter) | Games | W / D / L | Win Rate | Elo Δ | Avg Plies | Playouts | Device | Source File |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])
        for entry in baselines:
            bot = entry.get("bot", "unknown")
            net = Path(entry.get("net", "")).name or "unknown"
            gen = entry.get("generation")
            it = entry.get("iteration")
            gen_str = f"g{gen}" if gen is not None else "-"
            if it is not None:
                gen_str += f" (i{it})"
            games = entry.get("games", 0)
            wdl = f"{entry.get('wins', 0)} / {entry.get('draws', 0)} / {entry.get('losses', 0)}"
            wr = f"{entry.get('win_rate_pct', 0.0):.1f}%" if entry.get("win_rate_pct") is not None else "-"
            elo = f"{entry.get('elo_delta', 0.0):+.1f}" if entry.get("elo_delta") is not None else "-"
            plies = f"{entry.get('avg_plies', 0.0):.1f}" if entry.get("avg_plies") is not None else "-"
            sims = entry.get("sims")
            gumbel = " (Gumbel)" if entry.get("gumbel") else ""
            playouts = f"{sims}{gumbel}" if sims else "-"
            dev = entry.get("device", "-")
            src = f"`{entry.get('file', '')}`"
            lines.append(
                f"| **{bot}** | `{net}` | {gen_str} | {games} | {wdl} | **{wr}** | {elo} | {plies} | {playouts} | {dev} | {src} |"
            )
    else:
        lines.append("*No baseline evaluation records found.*")

    tourneys = manifest_data.get("tournaments", [])

    if tourneys:
        lines.extend([
            "",
            "---",
            "",
            "## 4. Multi-Engine Tournament Standings & Cross-Tables",
            "",
        ])
        for t in tourneys:
            t_id = t.get("tournament_id", "tournament")
            t_name = t.get("name", "Tournament")
            t_fmt = t.get("format", "round-robin")
            t_winner = t.get("winner", "unknown")
            t_games = t.get("total_games", 0)
            t_parts = t.get("num_participants", 0)
            t_src = f"`{t.get('file', '')}`"

            lines.extend([
                f"### {t_name} (`{t_id}`)",
                "",
                f"- **Format**: {t_fmt.title()} | **Competitors**: {t_parts} | **Total Games**: {t_games} | **Winner**: **{t_winner}** | **Source**: {t_src}",
                "",
            ])

            standings = t.get("standings", [])
            if standings:
                lines.extend([
                    "| Rank | Participant | Type | Games | W / D / L | Points | Win Rate | Decisive % | Forfeits (C/R) | Rating (BT) |",
                    "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ])
                for st in standings:
                    rk = st.get("rank", "-")
                    nm = st.get("name", "unknown")
                    tp = st.get("type", "-")
                    gm = st.get("games", 0)
                    wdl = f"{st.get('wins', 0)} / {st.get('draws', 0)} / {st.get('losses', 0)}"
                    pts = f"{st.get('points', 0.0):g}"
                    wr = f"{st.get('points_pct', 0.0):.1f}%"
                    dec = f"{st.get('decisive_pct', 0.0):.1f}%"
                    forf = f"{st.get('forfeits_conceded', 0)} / {st.get('forfeits_received', 0)}"
                    rat = st.get("rating", 1500.0)
                    se = st.get("rating_se", 0.0)
                    r_str = f"**{rat:+.1f}** ± {se:.1f}"
                    lines.append(
                        f"| {rk} | **{nm}** | `{tp}` | {gm} | {wdl} | {pts} | {wr} | {dec} | {forf} | {r_str} |"
                    )
                lines.append("")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Training Run Configurations",
        "",
    ])

    if configs:
        lines.extend([
            "| Config File | Architecture | Steps (Iters) | Batch / LR | Gumbel / Cap | Playouts | Device (Threads) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cfg in configs:
            src = f"`{cfg.get('file', '')}`"
            c = cfg.get("channels", "-")
            b = cfg.get("blocks", "-")
            se = "+SE" if cfg.get("se") else ""
            arch = f"{c}ch x {b}b{se}"
            steps = cfg.get("total_steps") or "-"
            iters = cfg.get("iterations") or "-"
            step_str = f"{steps} ({iters} iters)"
            batch = cfg.get("batch", "-")
            lr = cfg.get("lr", "-")
            batch_lr = f"{batch} / {lr}"
            gumbel = "Yes" if cfg.get("gumbel") else "No"
            cap = cfg.get("gumbel_cap")
            if cap:
                gumbel += f" (cap {cap})"
            games = cfg.get("games", "-")
            sims = cfg.get("sims", "-")
            fast = cfg.get("fast_sims")
            play_str = f"{games}g, {sims}s"
            if fast:
                play_str += f" ({fast} fast)"
            dev = cfg.get("device", "-")
            threads = cfg.get("threads")
            dev_str = f"{dev} ({threads}t)" if threads else dev
            lines.append(
                f"| {src} | {arch} | {step_str} | {batch_lr} | {gumbel} | {play_str} | {dev_str} |"
            )
    else:
        lines.append("*No training configurations found.*")

    if sums:
        lines.extend([
            "",
            "---",
            "",
            "## 6. Artifact Verification & Hashes",
            "",
            "| File | Recorded Value / SHA256 |",
            "| :--- | :--- |",
        ])
        for s in sums:
            f = f"`{s.get('file', '')}`"
            c = s.get("content", "").replace("\n", " ")
            lines.append(f"| {f} | `{c}` |")

    lines.append("")
    return "\n".join(lines)


def generate_manifest(
    results_dir: Path | str | None = None,
    output_json: Path | str | None = None,
    output_md: Path | str | None = None,
    repo_dir: Path | str | None = None,
    write_files: bool = True,
) -> dict[str, Any]:
    """Scan results directory and produce manifest dictionary + outputs."""
    root = Path(repo_dir).resolve() if repo_dir else repo_root()
    r_dir = Path(results_dir).resolve() if results_dir else (root / "results")

    scan_data = scan_results_directory(r_dir)
    git_meta = get_git_metadata(root)
    now_utc = _format_timestamp()

    manifest: dict[str, Any] = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_utc,
            "results_dir": str(r_dir),
            "total_scanned_files": scan_data["scanned_files"],
            "counts": {
                "tournaments": len(scan_data["tournaments"]),
                "foreign_arena": len(scan_data["foreign_arena"]),
                "baseline_evaluations": len(scan_data["baseline_evaluations"]),
                "evaluation_summaries": len(scan_data["evaluation_summaries"]),
                "training_configs": len(scan_data["training_configs"]),
                "checksums": len(scan_data["checksums"]),
            },
        },
        "git": git_meta,
        "tournaments": scan_data["tournaments"],
        "foreign_arena": scan_data["foreign_arena"],
        "baseline_evaluations": scan_data["baseline_evaluations"],
        "evaluation_summaries": scan_data["evaluation_summaries"],
        "training_configs": scan_data["training_configs"],
        "checksums": scan_data["checksums"],
        "errors": scan_data["errors"],
    }

    if write_files:
        json_target = Path(output_json).resolve() if output_json else (r_dir / "MANIFEST.json")
        md_target = Path(output_md).resolve() if output_md else (r_dir / "SUMMARY.md")

        json_target.parent.mkdir(parents=True, exist_ok=True)
        md_target.parent.mkdir(parents=True, exist_ok=True)

        json_str = json.dumps(manifest, indent=2) + "\n"
        # Atomic write
        tmp_json = json_target.with_suffix(json_target.suffix + ".tmp")
        tmp_json.write_text(json_str, encoding="utf-8")
        tmp_json.replace(json_target)

        md_content = generate_markdown_summary(manifest)
        tmp_md = md_target.with_suffix(md_target.suffix + ".tmp")
        tmp_md.write_text(md_content, encoding="utf-8")
        tmp_md.replace(md_target)

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan experiment results and generate structured MANIFEST.json and SUMMARY.md"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to results directory (defaults to <repo_root>/results)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output path for MANIFEST.json (defaults to <results_dir>/MANIFEST.json)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Output path for SUMMARY.md (defaults to <results_dir>/SUMMARY.md)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Perform dry run without writing files to disk",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args(argv)

    manifest = generate_manifest(
        results_dir=args.results_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        write_files=not args.no_write,
    )

    if not args.quiet:
        counts = manifest["metadata"]["counts"]
        print("=" * 60)
        print(" Qudor AI — Metrics Manifest Generated")
        print("=" * 60)
        print(f" Timestamp:           {manifest['metadata']['generated_at']}")
        print(f" Git Commit:          {manifest['git']['commit']}")
        print(f" Scanned Files:       {manifest['metadata']['total_scanned_files']}")
        print(f" Tournaments:         {counts['tournaments']}")
        print(f" Foreign Arena Runs:  {counts['foreign_arena']}")
        print(f" Baseline Evals:      {counts['baseline_evaluations']}")
        print(f" Training Configs:    {counts['training_configs']}")
        if manifest["errors"]:
            print(f" Warnings/Errors:     {len(manifest['errors'])}")
            for err in manifest["errors"]:
                print(f"   ! {err}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
