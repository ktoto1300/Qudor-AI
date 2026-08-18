"""Tests for the results index and metrics manifest generator."""
from __future__ import annotations

import json
from pathlib import Path

from quoridor_ai.results_manifest import (
    compute_file_sha256,
    generate_manifest,
    generate_markdown_summary,
    get_git_metadata,
    main,
    scan_results_directory,
)


def test_compute_file_sha256(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("qudor AI verification", encoding="utf-8")
    h = compute_file_sha256(f)
    assert isinstance(h, str)
    assert len(h) == 64

    missing = tmp_path / "does_not_exist.txt"
    assert compute_file_sha256(missing) == ""


def test_get_git_metadata(tmp_path: Path):
    meta = get_git_metadata(tmp_path)
    assert "commit" in meta
    assert "branch" in meta
    assert "is_dirty" in meta


def test_scan_empty_or_missing_directory(tmp_path: Path):
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir()
    res = scan_results_directory(empty_dir)
    assert res["scanned_files"] == 0
    assert len(res["foreign_arena"]) == 0
    assert len(res["baseline_evaluations"]) == 0
    assert len(res["training_configs"]) == 0
    assert len(res["errors"]) == 0

    missing_dir = tmp_path / "non_existent"
    res_missing = scan_results_directory(missing_dir)
    assert len(res_missing["errors"]) > 0


def test_scan_foreign_arena_results(tmp_path: Path):
    r_dir = tmp_path / "results"
    r_dir.mkdir()

    vader_json = {
        "games": 10,
        "wins": 9,
        "draws": 0,
        "losses": 1,
        "opp_forfeits": 1,
        "win_rate": 0.9,
        "elo_delta": 381.7,
        "win_rate_as_p0": 1.0,
        "win_rate_as_p1": 0.8,
        "avg_plies": 45.2,
        "sims": 32,
        "temperature": 0.6,
        "gumbel": True,
        "seed": 42,
        "opponent": "vader",
        "net": "checkpoints/gen13_best.pt",
        "generation": 13,
        "iteration": 439,
        "device": "cpu",
        "net_name": "gen13_best.pt",
    }
    (r_dir / "foreign_vader.json").write_text(json.dumps(vader_json), encoding="utf-8")

    scan = scan_results_directory(r_dir)
    assert scan["scanned_files"] == 1
    assert len(scan["foreign_arena"]) == 1
    entry = scan["foreign_arena"][0]
    assert entry["opponent"] == "vader"
    assert entry["wins"] == 9
    assert entry["win_rate_pct"] == 90.0
    assert entry["generation"] == 13
    assert entry["device"] == "cpu"
    assert len(entry["sha256"]) == 64


def test_scan_baseline_evaluations_and_summaries(tmp_path: Path):
    r_dir = tmp_path / "results"
    r_dir.mkdir()

    greedy_json = {
        "games": 50,
        "wins": 45,
        "draws": 0,
        "losses": 5,
        "win_rate": 0.9,
        "elo_delta": 380.0,
        "avg_plies": 50.0,
        "sims": 64,
        "temperature": 0.6,
        "gumbel": True,
        "seed": 123,
        "bot": "greedy",
        "net": "checkpoints/best.pt",
        "generation": 10,
        "iteration": 200,
        "device": "cuda",
    }
    (r_dir / "eval_greedy.json").write_text(json.dumps(greedy_json), encoding="utf-8")

    summary_json = {
        "checkpoint": "checkpoints/best.pt",
        "training_exit_code": 0,
        "evaluation": {
            "greedy": greedy_json,
            "rusher": {
                "games": 50,
                "wins": 50,
                "draws": 0,
                "losses": 0,
                "win_rate": 1.0,
                "elo_delta": 1600.0,
                "avg_plies": 30.0,
                "sims": 64,
                "temperature": 0.6,
                "gumbel": True,
                "seed": 124,
                "bot": "rusher",
                "net": "checkpoints/best.pt",
                "generation": 10,
                "iteration": 200,
                "device": "cuda",
            },
        },
    }
    (r_dir / "evaluation_summary.json").write_text(json.dumps(summary_json), encoding="utf-8")

    scan = scan_results_directory(r_dir)
    assert len(scan["evaluation_summaries"]) == 1
    # greedy is deduplicated, rusher from summary is included
    bots = {b["bot"] for b in scan["baseline_evaluations"]}
    assert bots == {"greedy", "rusher"}


def test_scan_training_configs_and_checksums(tmp_path: Path):
    r_dir = tmp_path / "results"
    r_dir.mkdir()

    cfg_json = {
        "_comment": "Test config",
        "seed": 42,
        "encoding": 3,
        "iterations": 1000,
        "total_steps": 50000,
        "channels": 128,
        "blocks": 8,
        "se": True,
        "gumbel": True,
        "gumbel_cap": 16,
        "games": 128,
        "sims": 32,
        "lr": 0.001,
        "batch": 128,
        "replay": 200000,
        "device": "cuda",
        "threads": 4,
    }
    (r_dir / "training_config.json").write_text(json.dumps(cfg_json), encoding="utf-8")
    (r_dir / "checkpoint.sha256").write_text("abc123sha256hash  checkpoints/best.pt\n", encoding="utf-8")

    scan = scan_results_directory(r_dir)
    assert len(scan["training_configs"]) == 1
    cfg = scan["training_configs"][0]
    assert cfg["channels"] == 128
    assert cfg["blocks"] == 8
    assert cfg["se"] is True
    assert "_comment" not in cfg["parameters"]

    assert len(scan["checksums"]) == 1
    assert "abc123sha256hash" in scan["checksums"][0]["content"]


def test_scan_ignores_transient_logs_and_directories(tmp_path: Path):
    r_dir = tmp_path / "results"
    r_dir.mkdir()

    # Transient outputs
    (r_dir / "run.log").write_text("stdout log", encoding="utf-8")
    (r_dir / "run.err").write_text("stderr log", encoding="utf-8")
    (r_dir / "run.pid").write_text("12345", encoding="utf-8")
    (r_dir / "file.tmp").write_text("atomic write in progress", encoding="utf-8")
    (r_dir / "trace.jsonl").write_text('{"step": 1}\n', encoding="utf-8")

    # Bridge logs dir
    b_dir = r_dir / "bridge_logs"
    b_dir.mkdir()
    (b_dir / "bridge.log").write_text("bridge output", encoding="utf-8")

    # Valid result file
    (r_dir / "foreign_test.json").write_text(
        json.dumps({"opponent": "dimi", "games": 2, "wins": 2, "losses": 0}), encoding="utf-8"
    )

    scan = scan_results_directory(r_dir)
    assert scan["scanned_files"] == 1
    assert len(scan["foreign_arena"]) == 1


def test_generate_manifest_and_markdown(tmp_path: Path):
    r_dir = tmp_path / "results"
    r_dir.mkdir()

    dimi_json = {
        "games": 2,
        "wins": 2,
        "draws": 0,
        "losses": 0,
        "opp_forfeits": 0,
        "win_rate": 1.0,
        "elo_delta": 1600.0,
        "avg_plies": 40.0,
        "sims": 16,
        "temperature": 0.6,
        "gumbel": True,
        "seed": 1,
        "opponent": "dimi",
        "net": "checkpoints/gen13_best.pt",
        "generation": 13,
        "iteration": 439,
        "device": "cpu",
    }
    (r_dir / "foreign_dimi.json").write_text(json.dumps(dimi_json), encoding="utf-8")

    json_path = r_dir / "MANIFEST.json"
    md_path = r_dir / "SUMMARY.md"

    manifest = generate_manifest(
        results_dir=r_dir,
        output_json=json_path,
        output_md=md_path,
        write_files=True,
    )

    assert json_path.exists()
    assert md_path.exists()
    assert manifest["metadata"]["counts"]["foreign_arena"] == 1

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_json["metadata"]["counts"]["foreign_arena"] == 1

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Qudor AI — Results & Metrics Manifest" in md_text
    assert "**dimi**" in md_text
    assert "100.0%" in md_text

    # Test direct markdown generation helper
    rendered = generate_markdown_summary(manifest)
    assert "Foreign Engine Arena Outcomes" in rendered


def test_manifest_cli_execution(tmp_path: Path):
    r_dir = tmp_path / "results"
    r_dir.mkdir()
    (r_dir / "eval_rusher.json").write_text(
        json.dumps({"bot": "rusher", "games": 10, "wins": 10, "losses": 0}), encoding="utf-8"
    )

    json_out = r_dir / "CUSTOM_MANIFEST.json"
    md_out = r_dir / "CUSTOM_SUMMARY.md"

    exit_code = main([
        "--results-dir", str(r_dir),
        "--output-json", str(json_out),
        "--output-md", str(md_out),
    ])
    assert exit_code == 0
    assert json_out.exists()
    assert md_out.exists()


def test_live_repository_manifest_generation():
    """Verify that generate_manifest() succeeds on the actual workspace results/ directory."""
    manifest = generate_manifest(write_files=True)
    assert manifest["metadata"]["counts"]["foreign_arena"] >= 5
    assert manifest["metadata"]["counts"]["baseline_evaluations"] >= 2
    assert manifest["metadata"]["counts"]["training_configs"] >= 1
    assert len(manifest["errors"]) == 0
