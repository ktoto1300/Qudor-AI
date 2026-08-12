"""Safe local Quoridor viewer. Static UI lives in index.html."""
from __future__ import annotations
import argparse, csv, json, math, os, secrets, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from quoridor_ai.az_selfplay import search
from quoridor_ai.core.encoding import version_for_planes
from quoridor_ai.core.engine import State, ACTION_SIZE, apply_unchecked, legal_actions
from quoridor_ai.model import arch_of, net_from_checkpoint
from quoridor_ai.runtime import resolve_device
from quoridor_ai.safe_loader import load_checkpoint

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.RLock()
# Matches the training loop's cap (train.py config max_plies); a game past it is a draw.
MAX_PLIES = 220
HISTORY_LIMIT = 200
MODEL_CACHE_TTL = 20.0
DEVICE = resolve_device()
# Search budget per move. The network's raw policy head is a prior, not a player: the
# arena gates on policy *plus* search for exactly this reason (see az_arena's header).
# 64 keeps a move around a second on a CPU while playing far above the bare argmax.
SIMS_DEFAULT = 64
SIMS_MAX = 800
GAME = {"state": State(player=secrets.randbelow(2)), "model": None, "model_path": None, "mode": "human",
        "human_player": 0,
        "paused": False, "thinking": False, "speed": 1.0, "history": [], "probs": [], "error": "",
        "encoding": 1, "sims": SIMS_DEFAULT, "value": None, "searchMs": 0}
MODEL_CACHE: list[dict] | None = None
MODEL_CACHE_AT = 0.0
# Separate from LOCK: a cold scan reads every checkpoint under ROOT and must not block the
# game state, but two scans running at once would each pay that cost for nothing.
SCAN_LOCK = threading.Lock()
# rel path -> ((mtime, size), description | None). Keeps the expensive read off the rescan.
MODEL_META: dict[str, tuple[tuple[float, int], dict | None]] = {}
# Same map, persisted, so the cost is paid once ever rather than once per server start.
MODEL_META_FILE = ROOT / ".model_index.json"
MODEL_META_VERSION = 1
# Filled in by main() from the actual --host/--port so the allowlist matches the running server.
ALLOWED_ORIGINS: set[str] = set()


def load_meta_cache() -> None:
    """Seed MODEL_META from disk, ignoring anything that does not look like we wrote it.

    Every field is re-validated instead of trusted: this file sits in the working directory
    and a malformed or hand-edited one must degrade to a slow scan, never to a bad entry in
    the list. The path is always taken from the map key, so the cache can describe a
    checkpoint but never name one - discovery stays with the directory walk.
    """
    try:
        blob = json.loads(MODEL_META_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(blob, dict) or blob.get("version") != MODEL_META_VERSION:
        return
    for rel, entry in (blob.get("entries") or {}).items():
        try:
            stamp = (float(entry["mtime"]), int(entry["size"]))
            desc = entry["desc"]
            if desc is not None:
                desc = {"path": rel, "label": str(desc["label"]),
                        "generation": int(desc["generation"]), "iteration": int(desc["iteration"]),
                        "params": int(desc["params"]),
                        "winRate": None if desc["winRate"] is None else float(desc["winRate"])}
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        MODEL_META[str(rel)] = (stamp, desc)


def save_meta_cache() -> None:
    """Write MODEL_META back. Best effort: a read-only directory just means slow starts."""
    blob = {"version": MODEL_META_VERSION,
            "entries": {rel: {"mtime": stamp[0], "size": stamp[1], "desc": desc}
                        for rel, (stamp, desc) in MODEL_META.items()}}
    tmp = MODEL_META_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, MODEL_META_FILE)   # atomic, so a kill cannot leave a half-written index
    except OSError:
        tmp.unlink(missing_ok=True)


def describe_checkpoint(rel: str, path: Path) -> dict | None:
    """Identity of one checkpoint, or None when the viewer cannot play it.

    Actually building the network is the only honest compatibility test - the pre-ResBlock
    checkpoints under legacy/ raise IncompatibleCheckpoint, and re-implementing that
    condition here would let the two drift apart and put dead entries back in the list.
    """
    try:
        data = load_checkpoint(path)
        net = net_from_checkpoint(data, "cpu")
    except Exception:
        return None
    gate = data.get("gate") or {}
    run = Path(rel).parent.as_posix() or "."
    gen, it = data.get("generation"), data.get("iteration")
    win = gate.get("win_rate")
    channels, blocks, _planes, _se = arch_of(data["model"])
    # Counted off the built network rather than derived from channels x blocks: it is the
    # figure the ordering below actually compares, so it should not be an estimate of itself.
    params = sum(p.numel() for p in net.parameters())
    # A promoted checkpoint is the only one with evidence behind it, so say so; the rest
    # are described by how far training got, which is all they have.
    if gen:
        note = f"поколение {gen}" + (f", {win * 100:.0f}% в турнире" if win is not None else "")
    elif it is not None and it >= 0:
        note = f"порция {it}, турнир не проходила"
    else:
        note = "стартовая сеть, не обучена"
    return {"path": rel, "label": f"{run} · {Path(rel).stem} — сеть {channels}×{blocks}, {note}",
            "generation": gen or 0, "iteration": it if isinstance(it, int) else -1,
            "winRate": win, "params": params}


def safe_models(force: bool = False) -> list[dict]:
    """Playable checkpoints under ROOT, strongest first, re-scanning periodically.

    A long-running viewer must notice checkpoints written by a concurrent training run,
    so the cache expires instead of being computed once per process. Per-file descriptions
    are keyed on (mtime, size) and persisted, because reading one checkpoint costs seconds -
    a latest.pt carries a replay buffer - and the whole tree here is ~380 MB. Neither an
    unchanged file nor a restart should pay that again.
    """
    global MODEL_CACHE, MODEL_CACHE_AT
    if MODEL_CACHE is not None and not force and time.monotonic() - MODEL_CACHE_AT <= MODEL_CACHE_TTL:
        return MODEL_CACHE
    with SCAN_LOCK:
        # Re-check: while this caller waited for the lock, whoever held it may have finished
        # the very scan it is about to start. Without this, every request that piles up
        # behind a cold scan runs its own afterwards.
        if MODEL_CACHE is not None and not force and time.monotonic() - MODEL_CACHE_AT <= MODEL_CACHE_TTL:
            return MODEL_CACHE
        return _scan_models()


def _scan_models() -> list[dict]:
    """The scan itself. Call through safe_models() - this assumes SCAN_LOCK is held."""
    global MODEL_CACHE, MODEL_CACHE_AT
    now = time.monotonic()
    if MODEL_CACHE is None and not MODEL_META:
        load_meta_cache()
    before = dict(MODEL_META)
    found, seen = [], set()
    for p in ROOT.rglob("*.pt"):
        rp = p.resolve()
        if not (rp.is_file() and ROOT in rp.parents):
            continue
        rel = rp.relative_to(ROOT).as_posix()
        try:
            st = rp.stat()
        except OSError:
            continue              # vanished mid-scan, e.g. a training run replacing it
        seen.add(rel)
        stamp = (st.st_mtime, st.st_size)
        hit = MODEL_META.get(rel)
        if hit is None or hit[0] != stamp:
            hit = (stamp, describe_checkpoint(rel, rp))
            MODEL_META[rel] = hit
        if hit[1] is not None:
            found.append(dict(hit[1], mtime=st.st_mtime))
    # Evict only what is gone from disk. Dropping the None entries too would make every
    # rescan re-read the checkpoints that cannot be played - the slowest files in the tree.
    for rel in MODEL_META.keys() - seen:
        MODEL_META.pop(rel, None)
    # Strongest first, since the client loads the first entry on startup. Capacity outranks
    # generation because generations are only comparable inside one run: a 48x4 smoke net
    # reaches generation 12 while the real 128x10 run is still at 8. At equal generation a
    # gated checkpoint outranks an ungated one - best.pt won a tournament, whereas latest.pt
    # has only trained further, which is not by itself evidence of being stronger.
    found.sort(key=lambda m: (-m["params"], -m["generation"], m["winRate"] is None,
                              -m["iteration"], -m["mtime"], m["path"]))
    MODEL_CACHE = [{k: v for k, v in m.items() if k != "mtime"} for m in found]
    MODEL_CACHE_AT = now
    if MODEL_META != before:
        save_meta_cache()
    return MODEL_CACHE


def model_paths() -> set[str]:
    return {m["path"] for m in safe_models(force=True)}


def is_over(s: State) -> bool:
    """True when no further move should be played: someone won or the ply cap hit."""
    return s.winner is not None or s.ply >= MAX_PLIES


def load_model(rel: str) -> None:
    if rel not in model_paths():
        raise ValueError("checkpoint отсутствует в разрешённом каталоге")
    path = (ROOT / Path(rel)).resolve()
    if ROOT not in path.parents:
        raise ValueError("недопустимый путь")
    data = load_checkpoint(path)
    model = net_from_checkpoint(data, DEVICE)
    model.eval()
    version = version_for_planes(model.planes)
    with LOCK:
        GAME["model"], GAME["model_path"], GAME["error"] = model, rel, ""
        GAME["encoding"] = version
        # Stale search output would otherwise be read as the new network's opinion.
        GAME["probs"], GAME["value"] = [], None
        should_start = (GAME["mode"] == "human" and GAME["state"].player != GAME["human_player"]
                        and not GAME["paused"] and not is_over(GAME["state"]))
    # If the random opening belongs to the AI, loading a model must start that opening
    # move immediately; otherwise the board remains stuck at ply 0 until another action.
    if should_start:
        threading.Thread(target=ai_move, daemon=True).start()


def wall_indices(s: State) -> tuple[list[int], list[int]]:
    return ([i for i in range(64) if (s.h >> i) & 1],
            [i for i in range(64) if (s.v >> i) & 1])


def state_payload() -> dict:
    with LOCK:
        s = GAME["state"]
        over = is_over(s)
        acts = legal_actions(s) if not over else []
        hc, vc = wall_indices(s)
        # Wall ownership is derived from the full history here, not on the client,
        # because the history sent below is truncated.
        owner = {m["action"]: m["player"] for m in GAME["history"] if m["action"] >= 81}
        return {"p0": s.p0, "p1": s.p1, "hc": hc, "vc": vc,
                "hcOwner": [owner.get(81 + i, 1) for i in hc],
                "vcOwner": [owner.get(145 + i, 1) for i in vc],
                "walls0": s.walls0, "walls1": s.walls1, "player": s.player,
                "ply": s.ply, "winner": s.winner, "mode": GAME["mode"],
                "humanPlayer": GAME["human_player"],
                "draw": s.winner is None and s.ply >= MAX_PLIES,
                "maxPlies": MAX_PLIES,
                "model": GAME["model_path"], "paused": GAME["paused"],
                "thinking": GAME["thinking"],
                # Cap the payload: a 220-ply game would otherwise ship the full log 1.25×/s.
                "history": GAME["history"][-HISTORY_LIMIT:],
                "historyTotal": len(GAME["history"]),
                "probs": GAME["probs"], "error": GAME["error"],
                "sims": GAME["sims"], "value": GAME["value"], "searchMs": GAME["searchMs"],
                "legal": acts,
                "legalWallsH": [a - 81 for a in acts if 81 <= a < 145],
                "legalWallsV": [a - 145 for a in acts if 145 <= a < ACTION_SIZE]}


def ai_move() -> None:
    with LOCK:
        s, model = GAME["state"], GAME["model"]
        if (GAME["thinking"] or GAME["paused"] or is_over(s)
                or (GAME["mode"] == "human" and s.player == GAME["human_player"])):
            return
        if model is None:
            # Returning quietly here leaves the board on the AI's turn forever, with nothing
            # on screen to say why. It is the AI's move and it cannot make one, so say so.
            GAME["error"] = "модель не загружена — выберите её в списке сверху"
            return
        GAME["thinking"] = True
        version = s.ply
        speed, sims, encoding = GAME["speed"], GAME["sims"], GAME["encoding"]
    time.sleep(0.6 / max(0.25, speed))
    try:
        with LOCK:
            # Re-check after the delay: reset, pause, mode switch, or a completed game wins.
            s, model = GAME["state"], GAME["model"]
            if (s.ply != version or model is None or GAME["paused"] or is_over(s)
                    or (GAME["mode"] == "human" and s.player == GAME["human_player"])):
                return
            if not legal_actions(s):
                return
        # Deliberately outside the lock: a 64-simulation search is ~65 forward passes and
        # takes about a second on a CPU. Holding LOCK across it would stall every poll of
        # /api/state, so the board would freeze while the AI thinks.
        t0 = time.monotonic()
        actions, probs, value = search(model, s, DEVICE, encoding=encoding, sims=sims,
                                       max_plies=MAX_PLIES)
        elapsed = int((time.monotonic() - t0) * 1000)
        if not actions:
            return
        order = probs.argsort()[::-1]
        action = int(actions[int(order[0])])
        with LOCK:
            # The board may have moved on while the lock was released - a reset, a pause,
            # or a mode switch. Applying a move searched against a stale board would
            # corrupt the game, so the result is dropped instead.
            if GAME["state"].ply != version or GAME["state"] is not s:
                return
            GAME["probs"] = [{"action": int(actions[int(i)]), "prob": float(probs[int(i)]),
                               "score": math.log(max(float(probs[int(i)]), 1e-9))}
                             for i in order[:5]]
            GAME["value"], GAME["searchMs"] = float(value), elapsed
            GAME["state"] = apply_unchecked(s, action)
            GAME["history"].append({"ply": s.ply + 1, "player": s.player, "action": action})
    finally:
        with LOCK:
            GAME["thinking"] = False
            # Chain the next move only when this one was actually played and the game
            # still wants one. Recomputing this after every early return - a pause, a
            # reset, a stale board - is what used to spawn threads for moves that never
            # happened.
            nxt = GAME["state"]
            again = (nxt.ply == version + 1 and not is_over(nxt) and not GAME["paused"]
                     and (GAME["mode"] == "ai" or nxt.player != GAME["human_player"]))
    if again:
        threading.Thread(target=ai_move, daemon=True).start()


def metrics() -> list[dict]:
    rows = []
    for f in sorted(ROOT.rglob("metrics.csv")):
        try:
            with f.open(encoding="utf-8", newline="") as stream:
                rows.extend(dict(row, source=f.relative_to(ROOT).as_posix()) for row in csv.DictReader(stream))
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows[-300:]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def json(self, data: dict | list, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        """Require the CSRF token on every mutating request.

        Origin is checked when the browser sends one, but it is absent on non-browser
        clients, so it cannot be the only gate — the token is mandatory either way.
        """
        origin = self.headers.get("Origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return False
        return secrets.compare_digest(self.headers.get("X-Qudor-Token", ""), TOKEN)

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/":
                body = (ROOT / "index.html").read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path == "/english.js":
                body = (ROOT / "english.js").read_bytes()
                self.send_response(200); self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path == "/api/token": return self.json({"token": TOKEN})
            if path == "/api/models": return self.json(safe_models())
            if path == "/api/state": return self.json(state_payload())
            if path == "/api/metrics": return self.json(metrics())
            return self.json({"error": "not found"}, 404)
        except Exception as exc:
            return self.json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                # The body is never read, so the socket still holds it; closing without
                # draining makes Windows reset the connection and the client sees a
                # network error instead of this status.
                self.close_connection = True
                return self.json({"error": "request too large"}, 413)
            # Drain the body before any early return: an unread request body plus a close
            # is an aborted connection on Windows, so a 403 would never reach the client.
            raw = self.rfile.read(length)
            if not self.authorized(): return self.json({"error": "forbidden"}, 403)
            body = json.loads(raw or b"{}")
            path = urlparse(self.path).path
            if path == "/api/model":
                load_model(str(body.get("path", ""))); return self.json({"ok": True})
            if path == "/api/reset":
                with LOCK:
                    # The human is always blue (player 0), while the opening turn is random.
                    GAME.update(state=State(player=secrets.randbelow(2)), human_player=0, history=[],
                                probs=[], paused=False, thinking=False, error="", value=None,
                                searchMs=0)
                    should_start = GAME["mode"] == "human" and GAME["state"].player != GAME["human_player"]
                if should_start: threading.Thread(target=ai_move, daemon=True).start()
                return self.json({"ok": True})
            if path == "/api/settings":
                with LOCK:
                    GAME["mode"] = body.get("mode", GAME["mode"])
                    GAME["speed"] = max(.25, min(3.0, float(body.get("speed", GAME["speed"]))))
                    if "sims" in body:
                        GAME["sims"] = max(1, min(SIMS_MAX, int(body["sims"])))
                    if "paused" in body: GAME["paused"] = bool(body["paused"])
                    should_start = GAME["mode"] == "ai" and not GAME["paused"]
                if should_start: threading.Thread(target=ai_move, daemon=True).start()
                return self.json({"ok": True})
            if path == "/api/step":
                with LOCK: GAME["paused"] = False
                threading.Thread(target=ai_move, daemon=True).start(); return self.json({"ok": True})
            if path == "/api/action":
                action = int(body["action"])
                with LOCK:
                    s = GAME["state"]
                    if (GAME["thinking"] or GAME["mode"] != "human" or s.player != GAME["human_player"]
                            or is_over(s) or action not in legal_actions(s)):
                        return self.json({"error": "illegal action"}, 400)
                    GAME["state"] = apply_unchecked(s, action); GAME["history"].append({"ply": s.ply + 1, "player": s.player, "action": action})
                threading.Thread(target=ai_move, daemon=True).start(); return self.json({"ok": True})
            return self.json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self.json({"error": str(exc)}, 400)
        except Exception as exc:
            return self.json({"error": str(exc)}, 500)


def origins_for(host: str, port: int) -> set[str]:
    """Origins a browser may legitimately use to reach this server."""
    hosts = {host, "127.0.0.1", "localhost"} if host in {"127.0.0.1", "localhost", "0.0.0.0"} else {host}
    return {f"http://{h}:{port}" for h in hosts}


ALLOWED_ORIGINS = origins_for("127.0.0.1", 8765)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    global ALLOWED_ORIGINS
    ALLOWED_ORIGINS = origins_for(args.host, args.port)
    print(f"Qudor viewer: http://{args.host}:{args.port}", flush=True)
    # Describing every checkpoint means reading it, and a latest.pt carries a replay buffer -
    # several seconds for the tree here. Starting that now overlaps it with the user opening
    # the browser instead of making their first page load wait for all of it.
    threading.Thread(target=safe_models, daemon=True).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__": main()
