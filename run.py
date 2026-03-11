"""Launch the recommendation API server and the Gradio GUI together.

1. Checks for pre-computed embeddings — runs embed step if missing.
2. Starts the Gradio GUI immediately (browsing works without the API).
3. Starts the FastAPI API in the background (search/recommendations
   become available once it finishes loading the model).
4. Starts a lightweight bridge server on port 7861 for JS interactivity.
"""

import os
import subprocess
import sys
import threading
import time

import yaml


BRIDGE_PORT = 7861


def _load_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _ensure_embeddings(cfg):
    """Run the embedding step if the .npz file is missing."""
    emb_path = cfg["data"]["embeddings"]
    if os.path.exists(emb_path):
        print(f"[run] Embeddings found at {emb_path}")
        return

    print(f"[run] {emb_path} not found — computing embeddings (one-time step) ...")
    print("[run] This downloads the model on first run and encodes all books.")
    print("[run] Progress bar will appear below.\n")
    result = subprocess.run(
        [sys.executable, "-m", "recommender.embed"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print("[run] ERROR: Embedding step failed. Fix the issue and rerun.")
        sys.exit(1)
    print(f"\n[run] Embeddings saved to {emb_path}\n")


def _start_api(cfg):
    """Start the FastAPI server in a daemon thread via uvicorn."""
    port = cfg["api"]["port"]

    def _run():
        print(f"[run] Loading recommendation model (this takes 30-60 s) ...")
        import uvicorn
        from recommender.api import app
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return port


def _wait_for_api(port, timeout=120):
    """Poll the API /health endpoint until it responds or timeout expires."""
    import httpx

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200:
                print(f"[run] API ready on http://localhost:{port}")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"[run] WARNING: API did not respond within {timeout}s")
    return False


def _start_bridge():
    """Start the lightweight bridge server for JS ↔ Python communication."""

    def _run():
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from gui.app import _handle_bookflix

        bridge = FastAPI()
        bridge.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class _Req(BaseModel, extra="allow"):
            isbn: str = ""
            user_list: list[int] = []

        @bridge.post("/bookflix/{path:path}")
        def handle(path: str, req: _Req):
            result = _handle_bookflix(f"/{path}", req.model_dump())
            if result is None:
                return {"error": "unknown path"}
            return result

        uvicorn.run(bridge, host="0.0.0.0", port=BRIDGE_PORT, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _kill_port(port):
    """Best-effort cleanup of a port occupied by a previous run."""
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in out.split():
            os.kill(int(pid), 9)
        time.sleep(0.3)
    except Exception:
        pass


def main():
    cfg = _load_cfg()
    gui_port = cfg.get("gui", {}).get("port", 7860)
    api_port = cfg["api"]["port"]

    _ensure_embeddings(cfg)

    for p in (api_port, BRIDGE_PORT, gui_port):
        _kill_port(p)

    print("[run] Starting API server in background ...")
    api_port = _start_api(cfg)

    print(f"[run] Starting bridge server on port {BRIDGE_PORT} ...")
    _start_bridge()
    time.sleep(1)

    threading.Thread(
        target=_wait_for_api, args=(api_port,), daemon=True,
    ).start()

    print(f"[run] Launching GUI on http://localhost:{gui_port}")
    print("[run] Browse immediately — search becomes available once the API loads.\n")
    from gui.app import app, CSS
    app.launch(
        server_name="0.0.0.0",
        server_port=gui_port,
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
