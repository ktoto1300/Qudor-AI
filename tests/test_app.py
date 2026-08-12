"""HTTP API tests: CSRF, path containment, game rules and payload shape.

Each test gets a fresh server on an ephemeral port so ordering never matters.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import app as webapp
from quoridor_ai.core.engine import State, legal_actions


@pytest.fixture
def server():
    import time
    with webapp.LOCK:
        webapp.GAME.update(state=State(), model=None, model_path=None, mode="human",
                           paused=False, thinking=False, speed=1.0, history=[], probs=[],
                           error="", encoding=1)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler)
    port = httpd.server_address[1]
    webapp.ALLOWED_ORIGINS = webapp.origins_for("127.0.0.1", port)
    # main() primes this in the background at startup, and the first scan reads every
    # checkpoint under the repo - seconds, since a latest.pt carries a replay buffer.
    # Priming here too keeps request timeouts about the request, and keeps the promise in
    # this module's docstring that test order never matters.
    webapp.safe_models()
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    # Wait for the server to be ready to accept connections
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with urllib.request.urlopen(base + "/api/token", timeout=0.5) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    try:
        yield base
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(base, path, body, token=None, origin=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
    if token:
        req.add_header("X-Qudor-Token", token)
    if origin:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_and_readonly_endpoints(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        assert r.status == 200 and b"<html" in r.read().lower()
    with urllib.request.urlopen(server + "/english.js", timeout=10) as r:
        assert r.status == 200 and b"Local game" in r.read()
    assert get(server, "/api/token")[1]["token"]
    assert isinstance(get(server, "/api/models")[1], list)
    assert isinstance(get(server, "/api/metrics")[1], list)
    assert get(server, "/api/nope")[0] == 404


def test_mutations_require_the_csrf_token(server):
    for path, body in [("/api/reset", {}), ("/api/settings", {"mode": "ai"}),
                       ("/api/step", {}), ("/api/action", {"action": 67}),
                       ("/api/model", {"path": "x.pt"})]:
        assert post(server, path, body)[0] == 403, path
        assert post(server, path, body, token="wrong-token")[0] == 403, path


def test_foreign_origin_is_rejected_even_with_a_valid_token(server):
    token = get(server, "/api/token")[1]["token"]
    assert post(server, "/api/reset", {}, token, origin="http://evil.example")[0] == 403
    assert post(server, "/api/reset", {}, token, origin=sorted(webapp.ALLOWED_ORIGINS)[0])[0] == 200


def test_model_path_must_stay_inside_the_project(server):
    token = get(server, "/api/token")[1]["token"]
    for bad in ["../../../etc/passwd", "/etc/passwd", "C:\\Windows\\win.ini", "", "does_not_exist.pt"]:
        status, body = post(server, "/api/model", {"path": bad}, token)
        assert status == 400 and "error" in body, bad


def test_state_payload_shape_and_no_raw_bitboards(server):
    _, s = get(server, "/api/state")
    for key in ["p0", "p1", "hc", "vc", "hcOwner", "vcOwner", "walls0", "walls1", "player",
                "ply", "winner", "mode", "draw", "maxPlies", "history", "historyTotal",
                "probs", "error", "legal", "legalWallsH", "legalWallsV"]:
        assert key in s, key
    assert "h" not in s and "v" not in s          # raw bitboards must not leak
    assert len(s["hcOwner"]) == len(s["hc"])
    assert len(s["vcOwner"]) == len(s["vc"])
    assert s["maxPlies"] == webapp.MAX_PLIES
    assert s["draw"] is False and s["winner"] is None
    assert set(s["legal"]) == set(legal_actions(State()))


def test_legal_action_is_applied_and_illegal_is_refused(server):
    token = get(server, "/api/token")[1]["token"]
    with webapp.LOCK:
        webapp.GAME["human_player"] = 0
    legal = get(server, "/api/state")[1]["legal"]
    move = next(a for a in legal if a < 81)
    assert post(server, "/api/action", {"action": move}, token)[0] == 200
    after = get(server, "/api/state")[1]
    assert after["ply"] == 1 and after["p0"] == move
    assert after["historyTotal"] == 1 and after["history"][-1]["action"] == move
    # An off-board action and a currently-illegal one are both refused.
    assert post(server, "/api/action", {"action": 9999}, token)[0] == 400
    assert post(server, "/api/action", {"action": -1}, token)[0] == 400


def test_reset_clears_history_and_state(server):
    token = get(server, "/api/token")[1]["token"]
    move = next(a for a in get(server, "/api/state")[1]["legal"] if a < 81)
    post(server, "/api/action", {"action": move}, token)
    assert post(server, "/api/reset", {}, token)[0] == 200
    s = get(server, "/api/state")[1]
    assert s["ply"] == 0 and s["historyTotal"] == 0 and s["p0"] == State().p0


def test_settings_clamp_speed(server):
    token = get(server, "/api/token")[1]["token"]
    post(server, "/api/settings", {"speed": 99.0}, token)
    assert webapp.GAME["speed"] == 3.0
    post(server, "/api/settings", {"speed": -5.0}, token)
    assert webapp.GAME["speed"] == 0.25


def test_settings_clamp_simulations(server):
    token = get(server, "/api/token")[1]["token"]
    assert post(server, "/api/settings", {"sims": 250}, token)[0] == 200
    assert webapp.GAME["sims"] == 250
    post(server, "/api/settings", {"sims": 0}, token)
    assert webapp.GAME["sims"] == 1
    post(server, "/api/settings", {"sims": 9999}, token)
    assert webapp.GAME["sims"] == webapp.SIMS_MAX


def test_state_probabilities_include_display_score(server):
    with webapp.LOCK:
        webapp.GAME["probs"] = [{"action": 0, "prob": 1.0, "score": 0.0}]
    _, state = get(server, "/api/state")
    assert state["probs"][0]["score"] == 0.0


def test_history_is_capped_but_total_is_reported(server):
    """A 220-ply game must not ship its whole log on every poll."""
    with webapp.LOCK:
        webapp.GAME["history"] = [{"ply": i, "player": i % 2, "action": 40} for i in range(500)]
    s = get(server, "/api/state")[1]
    assert len(s["history"]) == webapp.HISTORY_LIMIT
    assert s["historyTotal"] == 500
    assert s["history"][-1]["ply"] == 499


def test_finished_game_reports_the_outcome_and_refuses_moves(server):
    token = get(server, "/api/token")[1]["token"]
    with webapp.LOCK:
        webapp.GAME["state"] = State(p0=4, p1=40, player=0, ply=30)   # player 0 has won
    s = get(server, "/api/state")[1]
    assert s["winner"] == 0 and s["legal"] == [] and s["draw"] is False
    assert post(server, "/api/action", {"action": 13}, token)[0] == 400


def test_ply_cap_is_reported_as_a_draw(server):
    with webapp.LOCK:
        webapp.GAME["state"] = State(ply=webapp.MAX_PLIES)
    s = get(server, "/api/state")[1]
    assert s["winner"] is None and s["draw"] is True and s["legal"] == []


def test_oversized_request_is_rejected(server):
    """A declared body over the cap is refused without reading it.

    The body is never drained, so the server closes on a socket that still holds data;
    on Windows that surfaces to the client as a reset rather than the 413 itself. Both
    outcomes mean the same thing here — the request did not reach the handler.
    """
    token = get(server, "/api/token")[1]["token"]
    req = urllib.request.Request(server + "/api/reset", data=b"{}", method="POST")
    req.add_header("X-Qudor-Token", token)
    req.add_header("Content-Length", "2000000")
    before = webapp.GAME["state"].ply
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 413
    except urllib.error.HTTPError as e:
        assert e.code == 413
    except (ConnectionError, urllib.error.URLError):
        pass  # connection reset — the request was still refused
    assert webapp.GAME["state"].ply == before


def test_origins_for_covers_loopback_aliases():
    assert webapp.origins_for("127.0.0.1", 8765) == {
        "http://127.0.0.1:8765", "http://localhost:8765"}
    assert webapp.origins_for("0.0.0.0", 80) == {
        "http://0.0.0.0:80", "http://127.0.0.1:80", "http://localhost:80"}
    assert webapp.origins_for("192.168.1.5", 9000) == {"http://192.168.1.5:9000"}
