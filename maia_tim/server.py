# maia-tim-play -- GPL-3.0
"""Local web player for maia-tim. Run via play.bat / play.sh."""

import argparse
import datetime as dt
import http.server
import json
import os
import pathlib
import sys
import threading
import webbrowser

if "--gpu" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import botcore  # noqa: E402
import engine  # noqa: E402

GAMES = HERE.parent / "games"
META = json.loads((HERE / "meta.json").read_text(encoding="utf-8"))
engine.CKPT = HERE / "weights.pt"


def configure(eng, cfg, opp_rating):
    eng.new_game()
    eng.rating = int(cfg.get("rating", eng.rating))
    eng.opp_rating = int(opp_rating or 0) or None
    eng.temperature = float(cfg.get("temperature", 100)) / 100.0
    eng.mimic_time = bool(cfg.get("mimic", True))


class Handler(http.server.BaseHTTPRequestHandler):
    engines = {}
    info = {}
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "play.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self._json(self.info)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/new":
            b = self._body()
            with self.lock:
                for s, other in (("w", "b"), ("b", "w")):
                    if b.get(s) is not None:
                        opp = (b.get(other) or {}).get("rating") or b.get("human_rating")
                        configure(self.engines[s], b[s], opp)
            self._json({"ok": True})
        elif self.path == "/api/move":
            b = self._body()
            eng = self.engines[b.get("side", "w")]
            with self.lock:
                eng.set_position(None, b.get("moves", []))
                mv = botcore.think(eng, engine.N, int(b["wtime"]), int(b["btime"]), 0, 0,
                                   temperature=eng.temperature, mimic=eng.mimic_time)
            self._json({"move": mv.uci()})
        elif self.path == "/api/save":
            b = self._body()
            GAMES.mkdir(parents=True, exist_ok=True)
            name = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".pgn"
            (GAMES / name).write_text(b.get("pgn", ""), encoding="utf-8")
            self._json({"saved": f"games/{name}"})
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser(description="Play maia-tim in your browser.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--gpu", action="store_true", help="use a CUDA GPU (default: CPU)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    Handler.engines = {"w": engine.Engine(), "b": engine.Engine()}
    for e in Handler.engines.values():
        e.rating = META["default_rating"]
    Handler.info = {
        "version": META["version"],
        "device": "gpu" if args.gpu else "cpu",
        "default_rating": META["default_rating"],
        "rating_buckets": [{"rating": r} for r in META["eras"]],
        "bucket": META["step"],
    }
    url = f"http://localhost:{args.port}/"
    print(f"maia-tim {META['version']} on {Handler.info['device']}\n"
          f"play at {url}   (Ctrl+C to stop)", flush=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
