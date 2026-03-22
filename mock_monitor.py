"""
Mock monitor server for local UI testing.
Serves the dashboard HTML and pushes fake training metrics via WebSocket.
Run with:  python mock_monitor.py
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
from pathlib import Path

try:
    from aiohttp import web
    import aiohttp
except ImportError:
    raise SystemExit("Run: pip install aiohttp")

PORT = 6006

# ── Build HTML ────────────────────────────────────────────────────────────────

def build_html() -> str:
    base = Path(__file__).parent
    html = (base / "index.html").read_text(encoding="utf-8")
    js   = (base / "app.js").read_text(encoding="utf-8")
    html = html.replace("/* __APP_JS_PLACEHOLDER__ */", js)
    html = html.replace("Neko.rs Training Monitor (TDD)", "RinGo-DLLM Training Monitor")
    html = html.replace("Neko.rs Web Dashboard",          "RinGo-DLLM Training Monitor")
    return html

HTML = build_html()

# ── Fake training state ───────────────────────────────────────────────────────

TOTAL_STEPS = 1_000
clients:  set  = set()
history:  list = []
paused       = False

state = {
    "progress": {
        "epoch": 1, "total_epochs": 10,
        "step": 0,  "steps_per_epoch": 100, "total_steps": TOTAL_STEPS,
        "loss": 6.5, "avg_loss": 6.5,
        "learning_rate": 1e-5, "eta_sec": 3600,
    },
    "hardware": {
        "gpu_utilization": 0.0,
        "vram_used_mb": 0.0, "vram_total_mb": 8192.0,
        "ram_used_mb":  0.0, "ram_total_mb":  24576.0,
        "pcie_tx_mb": 0.0,   "pcie_rx_mb": 0.0,
    },
    "process": {
        "cpu_usage_percent": 0.0,
        "vram_used_mb": 0.0,
        "ram_used_mb":  0.0,
    },
    "dataset": {
        "total_samples": 13_300_000,
        "total_tokens":  1_702_400_000,
        "valid_tokens":  1_702_400_000,
        "max_seq_len":   128,
    },
    "config": {
        "vocab_size": 32768, "hidden_dim": 512,
        "num_layers": 12,    "num_heads":  8,
        "ffn_dim":    2048,  "max_seq_len": 128,
        "T": 25,             "mask_schedule":    "cosine",
        "mask_loss_weight":  5.0,
        "num_epochs":  10,   "batch_size":  128,
        "learning_rate": 5e-4, "lr_min": 1e-5,
        "lr_schedule": "cosine", "warmup_steps": 4000,
        "weight_decay": 0.05,    "grad_clip": 1.0,
        "dataset_name":   "wikipedia-ja",
        "tokenizer_name": "tohoku-nlp/bert-base-japanese-v3",
        "checkpoint_dir": "checkpoints_ja_v2",
    },
    "module_resources": [],
    "controls": {
        "log_level": "info", "log_filters": [],
        "available_log_filters": ["training", "model", "optimiser", "data"],
        "breakpoint_events": [], "paused": False,
    },
    "logs": [
        "[mock] RinGo-DLLM Training Monitor — mock mode",
        "[mock] config: ja-large  |  batch=128  |  epochs=10",
        "[mock] device: CUDA (NVIDIA GeForce RTX 3070 Ti)",
        "[mock] Trainable parameters: 55,032,832",
    ],
}


def snapshot() -> dict:
    s = {k: v for k, v in state.items() if k != "logs"}
    s["logs"] = list(state["logs"])
    return s


async def broadcast(msg: dict) -> None:
    if not clients:
        return
    data = json.dumps(msg, ensure_ascii=False)
    dead = set()
    for ws in list(clients):
        try:
            await ws.send_str(data)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ── HTTP handler ──────────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=HTML, content_type="text/html", charset="utf-8")


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)

    try:
        await ws.send_str(json.dumps({"type": "init",    "state":  snapshot()}))
        await ws.send_str(json.dumps({"type": "history", "points": history}))

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                    if cmd.get("type") == "set_paused":
                        state["controls"]["paused"] = cmd.get("paused", False)
                        await broadcast({"type": "update", "state": snapshot()})
                except Exception:
                    pass
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        clients.discard(ws)

    return ws


# ── Training simulation loop ──────────────────────────────────────────────────

async def sim_loop() -> None:
    step       = 0
    start_time = time.time()

    while True:
        await asyncio.sleep(1.0)
        if state["controls"]["paused"]:
            continue

        step      += 1
        epoch      = (step - 1) // 100 + 1
        step_local = (step - 1) % 100 + 1

        loss     = 6.5 * math.exp(-step / 400) + random.gauss(0, 0.05)
        avg_loss = 6.5 * math.exp(-step / 450)
        lr       = 5e-4 * min(1.0, step / 4000) * (0.5 + 0.5 * math.cos(math.pi * step / 1000))

        gpu_util  = 90 + random.gauss(0, 3)
        vram_used = 4800 + random.gauss(0, 50)
        ram_proc  = 8200 + random.gauss(0, 100)
        cpu_proc  = 15   + random.gauss(0, 2)
        pcie_tx   = 1800 + random.gauss(0, 200)   # KB/s
        pcie_rx   = 400  + random.gauss(0, 50)
        tok_s     = 128 * 128 + random.gauss(0, 500)
        elapsed   = time.time() - start_time
        remaining = (TOTAL_STEPS - step) / max(step / max(elapsed, 1), 1)

        state["progress"].update({
            "epoch": min(epoch, 10), "step": step_local,
            "loss": round(loss, 4),  "avg_loss": round(avg_loss, 4),
            "learning_rate": lr,     "eta_sec": remaining,
        })
        state["hardware"].update({
            "gpu_utilization": round(gpu_util, 1),
            "vram_used_mb":    round(vram_used, 0),
            "ram_used_mb":     round(ram_proc + 2000, 0),
            "pcie_tx_mb":      round(pcie_tx / 1024, 2),
            "pcie_rx_mb":      round(pcie_rx / 1024, 2),
        })
        state["process"].update({
            "cpu_usage_percent": round(cpu_proc, 1),
            "vram_used_mb":      round(vram_used * 0.85, 0),
            "ram_used_mb":       round(ram_proc, 0),
        })

        if step % 10 == 0:
            log = (f"  step {step:5d} | loss {loss:.4f} "
                   f"| avg {avg_loss:.4f} | lr {lr:.2e}")
            state["logs"].append(log)
            if len(state["logs"]) > 200:
                state["logs"] = state["logs"][-200:]

        point = {
            "global_step": step, "t": elapsed,
            "loss": round(loss, 4), "avg_loss": round(avg_loss, 4),
            "tok_s": round(tok_s, 0),
            "gpu_sys":   round(gpu_util, 1),
            "vram_proc": round(vram_used * 0.85, 0),
            "cpu_proc":  round(cpu_proc, 1),
            "ram_proc":  round(ram_proc, 0),
            "pcie_tx":   round(pcie_tx / 1024, 2),
            "pcie_rx":   round(pcie_rx / 1024, 2),
        }
        history.append(point)
        if len(history) > 5000:
            history.pop(0)

        await broadcast({"type": "history_append", "point": point})
        await broadcast({"type": "update",         "state": snapshot()})

        if step >= TOTAL_STEPS:
            step = 0
            history.clear()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    app = web.Application()
    app.router.add_get("/",          handle_index)
    app.router.add_get("/index.html", handle_index)
    app.router.add_get("/ws",        handle_ws)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"Mock monitor: http://localhost:{PORT}")
    await sim_loop()


if __name__ == "__main__":
    asyncio.run(main())
