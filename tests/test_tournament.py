"""Comprehensive test suite for the Multi-Engine Tournament Coordinator."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from quoridor_ai.baseline import BOTS
from quoridor_ai.core.engine import legal_actions
from quoridor_ai.results_manifest import generate_manifest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "tournament", ROOT / "tools" / "tournament.py"
)
tournament = importlib.util.module_from_spec(SPEC)
sys.modules["tournament"] = tournament
SPEC.loader.exec_module(tournament)


# -----------------------------------------------------------------------------
# 1. Built-in Bridges Validation
# -----------------------------------------------------------------------------

def test_tournament_all_builtin_bridges_exist():
    """Verify that all configured foreign bridges have existing script files."""
    assert len(tournament.BUILTIN_BRIDGES) == 9
    for name, argv in tournament.BUILTIN_BRIDGES.items():
        script_path = Path(argv[-1])
        assert script_path.is_file(), f"Bridge script for {name} does not exist: {script_path}"


# -----------------------------------------------------------------------------
# 2. Bradley-Terry / Bayes-Elo Rating Estimator Tests
# -----------------------------------------------------------------------------

def test_bradley_terry_ratings_symmetric_and_anchor():
    """Test Bradley-Terry rating convergence on symmetric and transitive records."""
    names = ["bot_a", "bot_b", "bot_c"]
    scores = {
        ("bot_a", "bot_b"): 8.0,
        ("bot_b", "bot_a"): 2.0,
        ("bot_b", "bot_c"): 8.0,
        ("bot_c", "bot_b"): 2.0,
        ("bot_a", "bot_c"): 9.0,
        ("bot_c", "bot_a"): 1.0,
    }
    games = {
        ("bot_a", "bot_b"): 10,
        ("bot_b", "bot_a"): 10,
        ("bot_b", "bot_c"): 10,
        ("bot_c", "bot_b"): 10,
        ("bot_a", "bot_c"): 10,
        ("bot_c", "bot_a"): 10,
    }

    ratings = tournament.compute_bradley_terry_ratings(
        participant_names=names,
        pairwise_scores=scores,
        pairwise_games=games,
        anchor_name="bot_b",
        anchor_rating=1500.0,
    )

    assert "bot_a" in ratings and "bot_b" in ratings and "bot_c" in ratings
    assert ratings["bot_b"]["rating"] == pytest.approx(1500.0, abs=1.0)
    assert ratings["bot_a"]["rating"] > ratings["bot_b"]["rating"]
    assert ratings["bot_b"]["rating"] > ratings["bot_c"]["rating"]

    for r_data in ratings.values():
        assert r_data["rating_se"] > 0.0
        assert r_data["rating_ci_95"][0] < r_data["rating"] < r_data["rating_ci_95"][1]


def test_bradley_terry_ratings_handles_clean_sweep_and_winless():
    """Prior regularization must keep Elo finite even on 100% win or 0% win records."""
    names = ["invincible", "unlucky"]
    scores = {
        ("invincible", "unlucky"): 10.0,
        ("unlucky", "invincible"): 0.0,
    }
    games = {
        ("invincible", "unlucky"): 10,
        ("unlucky", "invincible"): 10,
    }

    ratings = tournament.compute_bradley_terry_ratings(
        participant_names=names,
        pairwise_scores=scores,
        pairwise_games=games,
        anchor_name=None,
        anchor_rating=1500.0,
    )

    assert not math.isinf(ratings["invincible"]["rating"])
    assert not math.isinf(ratings["unlucky"]["rating"])
    assert ratings["invincible"]["rating"] > ratings["unlucky"]["rating"]


def test_bradley_terry_single_participant():
    """Single participant returns anchor rating with zero standard error."""
    ratings = tournament.compute_bradley_terry_ratings(
        participant_names=["solo"],
        pairwise_scores={},
        pairwise_games={},
        anchor_rating=1600.0,
    )
    assert ratings["solo"]["rating"] == 1600.0
    assert ratings["solo"]["rating_se"] == 0.0


# -----------------------------------------------------------------------------
# 3. Game & Matchup Orchestration
# -----------------------------------------------------------------------------

def test_play_single_game_rusher_vs_greedy():
    """Verify single game execution between baseline bots."""
    p_rusher = tournament.BaselineParticipant("rusher", BOTS["rusher"])
    p_greedy = tournament.BaselineParticipant("greedy", BOTS["greedy"])

    res = tournament.play_single_game(p_rusher, p_greedy, seed=123, max_plies=220)
    assert res.score_p0 + res.score_p1 == 1.0
    assert res.plies > 0
    assert res.forfeit_by is None
    assert res.winner_name in ["rusher", "greedy"]


def test_play_matchup_alternating_colors():
    """Matchup of 4 games must alternate p0/p1 and aggregate scores accurately."""
    p_rusher = tournament.BaselineParticipant("rusher", BOTS["rusher"])
    p_greedy = tournament.BaselineParticipant("greedy", BOTS["greedy"])

    match = tournament.play_matchup(p_rusher, p_greedy, games=4, base_seed=42)
    assert match.games == 4
    assert match.score_a + match.score_b == 4.0
    assert match.wins_a + match.losses_a + match.draws == 4
    assert len(match.game_details) == 4

    # Check color alternation
    assert match.game_details[0].p0_name == "rusher"
    assert match.game_details[1].p0_name == "greedy"
    assert match.game_details[2].p0_name == "rusher"
    assert match.game_details[3].p0_name == "greedy"


# -----------------------------------------------------------------------------
# 4. Standings, Cross-Table, and Tiebreaks
# -----------------------------------------------------------------------------

def test_standings_and_crosstable_computation():
    p1 = tournament.CallableParticipant("A", lambda s, r: legal_actions(s)[0])
    p2 = tournament.CallableParticipant("B", lambda s, r: legal_actions(s)[0])
    p3 = tournament.CallableParticipant("C", lambda s, r: legal_actions(s)[0])

    # Construct mock match results
    m1 = tournament.MatchResult(
        participant_a="A",
        participant_b="B",
        games=2,
        score_a=2.0,
        score_b=0.0,
        wins_a=2,
        draws=0,
        losses_a=0,
        forfeits_a=0,
        forfeits_b=0,
        decisive_games=2,
        avg_plies=30.0,
        win_rate_a_p0=1.0,
        win_rate_a_p1=1.0,
    )
    m2 = tournament.MatchResult(
        participant_a="B",
        participant_b="C",
        games=2,
        score_a=1.5,
        score_b=0.5,
        wins_a=1,
        draws=1,
        losses_a=0,
        forfeits_a=0,
        forfeits_b=0,
        decisive_games=1,
        avg_plies=40.0,
        win_rate_a_p0=0.5,
        win_rate_a_p1=1.0,
    )
    m3 = tournament.MatchResult(
        participant_a="A",
        participant_b="C",
        games=2,
        score_a=2.0,
        score_b=0.0,
        wins_a=2,
        draws=0,
        losses_a=0,
        forfeits_a=0,
        forfeits_b=0,
        decisive_games=2,
        avg_plies=25.0,
        win_rate_a_p0=1.0,
        win_rate_a_p1=1.0,
    )

    standings, crosstable = tournament.compute_standings_and_crosstable(
        participants=[p1, p2, p3],
        match_results=[m1, m2, m3],
        anchor_bot="B",
        anchor_rating=1500.0,
    )

    assert len(standings) == 3
    assert standings[0]["name"] == "A"
    assert standings[0]["points"] == 4.0
    assert standings[0]["rank"] == 1
    assert standings[1]["name"] == "B"
    assert standings[1]["points"] == 1.5
    assert standings[2]["name"] == "C"
    assert standings[2]["points"] == 0.5

    # Check cross-table
    assert crosstable["participants"] == ["A", "B", "C"]
    rows = {r["name"]: r["scores"] for r in crosstable["rows"]}
    assert rows["A"]["A"]["self"] is True
    assert rows["A"]["B"]["text"] == "2-0"
    assert rows["B"]["A"]["text"] == "0-2"
    assert rows["B"]["C"]["text"] == "1.5-0.5"


# -----------------------------------------------------------------------------
# 5. Round-Robin & Swiss Tournament Runner Tests
# -----------------------------------------------------------------------------

def test_run_round_robin_tournament():
    """Verify complete round-robin tournament execution."""
    p_rusher = tournament.BaselineParticipant("rusher", BOTS["rusher"])
    p_greedy = tournament.BaselineParticipant("greedy", BOTS["greedy"])

    matches, standings, crosstable = tournament.run_round_robin(
        participants=[p_rusher, p_greedy],
        games_per_matchup=2,
        base_seed=100,
    )

    assert len(matches) == 1
    assert len(standings) == 2
    assert standings[0]["rank"] == 1
    assert standings[1]["rank"] == 2
    assert sum(s["games"] for s in standings) == 4


def test_run_swiss_tournament_even_participants():
    """Verify Swiss tournament pairing and scoring with 4 participants."""
    participants = [
        tournament.BaselineParticipant("greedy_1", BOTS["greedy"]),
        tournament.BaselineParticipant("rusher_1", BOTS["rusher"]),
        tournament.BaselineParticipant("greedy_2", BOTS["greedy"]),
        tournament.BaselineParticipant("rusher_2", BOTS["rusher"]),
    ]

    matches, standings, crosstable = tournament.run_swiss_tournament(
        participants=participants,
        rounds=2,
        games_per_matchup=1,
        base_seed=200,
    )

    assert len(matches) == 4
    assert len(standings) == 4


def test_run_swiss_tournament_odd_participants():
    """Verify Swiss tournament handles odd participant counts gracefully."""
    participants = [
        tournament.BaselineParticipant("greedy_1", BOTS["greedy"]),
        tournament.BaselineParticipant("rusher_1", BOTS["rusher"]),
        tournament.BaselineParticipant("rusher_2", BOTS["rusher"]),
    ]

    matches, standings, crosstable = tournament.run_swiss_tournament(
        participants=participants,
        rounds=2,
        games_per_matchup=1,
        base_seed=300,
    )

    assert len(standings) == 3
    assert all(st["games"] >= 0 for st in standings)


# -----------------------------------------------------------------------------
# 6. Forfeit & Subprocess Health Recovery Tests
# -----------------------------------------------------------------------------

def test_forfeit_recovery_from_illegal_move_bridge(tmp_path: Path):
    """Bridge returning illegal action must forfeit immediately without crashing."""
    script = tmp_path / "illegal_bridge.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if msg['type'] == 'hello':\n"
        "        print(json.dumps({'ok': True, 'name': 'illegal_bot'}), flush=True)\n"
        "    elif msg['type'] == 'state':\n"
        "        print(json.dumps({'a': 9999}), flush=True)\n"
        "    elif msg['type'] == 'bye':\n"
        "        break\n",
        encoding="utf-8",
    )

    p_illegal = tournament.BridgeParticipant(
        name="illegal_bot",
        argv=[sys.executable, str(script)],
        logdir=tmp_path / "logs",
        move_timeout=2.0,
    )
    p_rusher = tournament.BaselineParticipant("rusher", BOTS["rusher"])

    match = tournament.play_matchup(p_illegal, p_rusher, games=2, base_seed=400)
    assert match.games == 2
    assert match.score_a == 0.0
    assert match.score_b == 2.0
    assert match.forfeits_a == 2
    assert (tmp_path / "logs" / "illegal_dump.jsonl").is_file()


def test_forfeit_recovery_from_crashing_bridge(tmp_path: Path):
    """Bridge crashing midway must forfeit the game."""
    script = tmp_path / "crash_bridge.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if msg['type'] == 'hello':\n"
        "        print(json.dumps({'ok': True, 'name': 'crasher'}), flush=True)\n"
        "    elif msg['type'] == 'state':\n"
        "        sys.exit(42)\n",
        encoding="utf-8",
    )

    p_crash = tournament.BridgeParticipant(
        name="crash_bot",
        argv=[sys.executable, str(script)],
        logdir=tmp_path / "logs",
        move_timeout=2.0,
    )
    p_greedy = tournament.BaselineParticipant("greedy", BOTS["greedy"])

    match = tournament.play_matchup(p_crash, p_greedy, games=1, base_seed=500)
    assert match.games == 1
    assert match.score_a == 0.0
    assert match.score_b == 1.0
    assert match.forfeits_a == 1


def test_forfeit_recovery_from_explicit_forfeit_msg(tmp_path: Path):
    """Bridge self-reporting forfeit must be credited as forfeit."""
    script = tmp_path / "surrender_bridge.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if msg['type'] == 'hello':\n"
        "        print(json.dumps({'ok': True, 'name': 'surrender_bot'}), flush=True)\n"
        "    elif msg['type'] == 'state':\n"
        "        print(json.dumps({'forfeit': 'I resign'}), flush=True)\n"
        "    elif msg['type'] == 'bye':\n"
        "        break\n",
        encoding="utf-8",
    )

    p_surrender = tournament.BridgeParticipant(
        name="surrender_bot",
        argv=[sys.executable, str(script)],
        logdir=tmp_path / "logs",
        move_timeout=2.0,
    )
    p_rusher = tournament.BaselineParticipant("rusher", BOTS["rusher"])

    match = tournament.play_matchup(p_surrender, p_rusher, games=1, base_seed=600)
    assert match.score_a == 0.0
    assert match.score_b == 1.0
    assert match.forfeits_a == 1


# -----------------------------------------------------------------------------
# 7. Reporting & File Output
# -----------------------------------------------------------------------------

def test_save_tournament_output_and_markdown(tmp_path: Path):
    t_data = {
        "metadata": {
            "tournament_id": "test_t1",
            "name": "Unit Test Tournament",
            "created_at": "2026-08-18 12:00:00 UTC",
            "elapsed_seconds": 1.5,
            "total_matches": 1,
            "total_games": 2,
            "num_participants": 2,
            "winner": "rusher",
        },
        "config": {
            "format": "round-robin",
            "games_per_matchup": 2,
        },
        "participants": [
            {"name": "rusher", "type": "baseline"},
            {"name": "greedy", "type": "baseline"},
        ],
        "standings": [
            {
                "rank": 1,
                "name": "rusher",
                "type": "baseline",
                "matches": 1,
                "games": 2,
                "points": 2.0,
                "points_pct": 100.0,
                "wins": 2,
                "draws": 0,
                "losses": 0,
                "decisive_games": 2,
                "decisive_pct": 100.0,
                "forfeits_conceded": 0,
                "forfeits_received": 0,
                "avg_plies": 42.0,
                "rating": 1600.0,
                "rating_se": 15.0,
                "sonneborn_berger": 2.0,
                "buchholz": 0.0,
            },
            {
                "rank": 2,
                "name": "greedy",
                "type": "baseline",
                "matches": 1,
                "games": 2,
                "points": 0.0,
                "points_pct": 0.0,
                "wins": 0,
                "draws": 0,
                "losses": 2,
                "decisive_games": 2,
                "decisive_pct": 100.0,
                "forfeits_conceded": 0,
                "forfeits_received": 0,
                "avg_plies": 42.0,
                "rating": 1400.0,
                "rating_se": 15.0,
                "sonneborn_berger": 0.0,
                "buchholz": 2.0,
            },
        ],
        "cross_table": {
            "participants": ["rusher", "greedy"],
            "rows": [
                {
                    "name": "rusher",
                    "scores": {
                        "rusher": {"self": True, "text": "-"},
                        "greedy": {"self": False, "score": 2.0, "text": "2-0"},
                    },
                },
                {
                    "name": "greedy",
                    "scores": {
                        "rusher": {"self": False, "score": 0.0, "text": "0-2"},
                        "greedy": {"self": True, "text": "-"},
                    },
                },
            ],
        },
        "matchups": [
            {
                "participant_a": "rusher",
                "participant_b": "greedy",
                "games": 2,
                "score_a": 2.0,
                "score_b": 0.0,
                "wins_a": 2,
                "draws": 0,
                "losses_a": 0,
                "forfeits_a": 0,
                "forfeits_b": 0,
                "avg_plies": 42.0,
                "win_rate_a_p0": 1.0,
                "win_rate_a_p1": 1.0,
            }
        ],
    }

    json_path = tmp_path / "tournaments" / "test_t1.json"
    md_path = tmp_path / "tournaments" / "test_t1.md"

    tournament.save_tournament_output(t_data, json_path, md_path)

    assert json_path.is_file()
    assert md_path.is_file()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["tournament_id"] == "test_t1"

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Qudor AI — Unit Test Tournament" in md_content
    assert "Head-to-Head Cross-Table" in md_content
    assert "Final Standings & Ratings" in md_content


# -----------------------------------------------------------------------------
# 8. Metrics Manifest Integration Test
# -----------------------------------------------------------------------------

def test_metrics_manifest_scans_tournament_outputs(tmp_path: Path):
    """Verify that results_manifest.py indexes tournament results."""
    r_dir = tmp_path / "results"
    t_dir = r_dir / "tournaments"
    t_dir.mkdir(parents=True)

    t_data = {
        "metadata": {
            "tournament_id": "tournament_sample",
            "name": "Sample Tournament",
            "created_at": "2026-08-18 12:00:00 UTC",
            "total_games": 10,
            "winner": "bot_alpha",
        },
        "config": {
            "format": "round-robin",
            "games_per_matchup": 2,
        },
        "participants": [{"name": "bot_alpha"}, {"name": "bot_beta"}],
        "standings": [
            {"rank": 1, "name": "bot_alpha", "points": 8.0, "games": 10, "rating": 1650.0},
            {"rank": 2, "name": "bot_beta", "points": 2.0, "games": 10, "rating": 1350.0},
        ],
        "cross_table": {"participants": ["bot_alpha", "bot_beta"], "rows": []},
        "matchups": [],
    }

    (t_dir / "tournament_sample.json").write_text(json.dumps(t_data), encoding="utf-8")

    manifest = generate_manifest(
        results_dir=r_dir,
        output_json=r_dir / "MANIFEST.json",
        output_md=r_dir / "SUMMARY.md",
        repo_dir=tmp_path,
        write_files=True,
    )

    assert manifest["metadata"]["counts"]["tournaments"] == 1
    assert len(manifest["tournaments"]) == 1
    assert manifest["tournaments"][0]["tournament_id"] == "tournament_sample"
    assert manifest["tournaments"][0]["winner"] == "bot_alpha"

    summary_text = (r_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Multi-Engine Tournament Standings & Cross-Tables" in summary_text
    assert "Sample Tournament" in summary_text


# -----------------------------------------------------------------------------
# 9. CLI Invocations
# -----------------------------------------------------------------------------

def test_tournament_cli_minimal_execution(tmp_path: Path):
    """Test tournament tool execution via CLI arguments."""
    out_json = tmp_path / "t_cli.json"
    out_md = tmp_path / "t_cli.md"

    code = tournament.main([
        "--baselines", "rusher", "greedy",
        "--format", "round-robin",
        "--games", "1",
        "--output", str(out_json),
        "--output-md", str(out_md),
        "--no-manifest",
        "--quiet",
    ])

    assert code == 0
    assert out_json.is_file()
    assert out_md.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["metadata"]["total_games"] == 1
    assert len(data["standings"]) == 2
