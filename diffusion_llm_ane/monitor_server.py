"""
RinGo-DLLM Training Monitor Server
====================================
Starts a combined HTTP + WebSocket server in a background daemon thread so
the training loop can push live metrics to a browser dashboard.

WebSocket protocol (matches the Neko.rs dashboard frontend):
  Server → Client:
    {"type": "init",           "state": {...}}         – full snapshot on connect
    {"type": "update",         "state": {...}}         – incremental update
    {"type": "history",        "points": [...]}        – bulk chart history on connect
    {"type": "history_append", "point":  {...}}        – single chart point
    {"type": "logs",           "lines":  [...]}        – log ring-buffer

  Client → Server:
    {"type": "set_log_level",   "level":   "info"}
    {"type": "set_log_filters", "filters": [...]}
    {"type": "set_breakpoints", "events":  [...]}
    {"type": "set_paused",      "paused":  bool}       – pause / resume training

Usage in train.py::

    from monitor_server import TrainingMonitor

    monitor = TrainingMonitor(port=6006)
    monitor.configure(config, steps_per_epoch, total_steps, dataset_info)
    monitor.start()

    for epoch in ...:
        for step, batch in ...:
            ...
            monitor.push_step(epoch, step, global_step, loss, avg_loss, lr)
            monitor.push_log(f"  step {global_step} | loss {loss:.4f} | lr {lr:.2e}")
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

# ── Optional dependencies (graceful degradation when missing) ─────────────────

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    _psutil = None  # type: ignore[assignment]

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
    _pynvml = None  # type: ignore[assignment]

try:
    from aiohttp import web as _web
    import aiohttp as _aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False
    _web = None      # type: ignore[assignment]
    _aiohttp = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────

class TrainingMonitor:
    """Thread-safe bridge between the training loop and the WebSocket dashboard."""

    _MAX_LOG_LINES = 500
    _MAX_HISTORY   = 5_000

    def __init__(self, host: str = "0.0.0.0", port: int = 6006) -> None:
        if not _HAS_AIOHTTP:
            raise ImportError(
                "aiohttp is required.  Install it with:  pip install aiohttp"
            )

        self._host = host
        self._port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set = set()
        self._lock = threading.Lock()

        # ── Shared state ──────────────────────────────────────────────────────
        self._state: dict[str, Any] = {
            "progress": {
                "epoch": 0, "total_epochs": 0,
                "step":  0, "steps_per_epoch": 0, "total_steps": 0,
                "loss": 0.0, "avg_loss": 0.0,
                "learning_rate": 0.0, "eta_sec": 0.0,
            },
            "hardware": {
                "gpu_utilization": 0.0,
                "vram_used_mb": 0.0, "vram_total_mb": 0.0,
                "ram_used_mb":  0.0, "ram_total_mb":  0.0,
                "pcie_tx_mb": 0.0,   "pcie_rx_mb": 0.0,
            },
            "process": {
                "cpu_usage_percent": 0.0,
                "vram_used_mb": 0.0,
                "ram_used_mb":  0.0,
            },
            "dataset": {
                "total_samples": 0, "total_tokens": 0,
                "valid_tokens":  0, "max_seq_len":  0,
            },
            "config": {},
            "module_resources": [],
            "controls": {
                "log_level":   "info",
                "log_filters": [],
                "available_log_filters": ["training", "model", "optimiser", "data"],
                "breakpoint_events": [],
                "paused": False,
            },
        }
        self._logs:    deque[str] = deque(maxlen=self._MAX_LOG_LINES)
        self._history: list[dict] = []

        # threading.Event: set = running, cleared = paused
        self._paused_event = threading.Event()
        self._paused_event.set()

        self._start_time: float        = time.time()
        self._step_times: deque[float] = deque(maxlen=50)

        # psutil process handle
        self._proc = _psutil.Process(os.getpid()) if _HAS_PSUTIL else None

        # NVML device handle (GPU 0)
        self._nvml_dev = None
        if _HAS_NVML:
            try:
                self._nvml_dev = _pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

        # Cached HTML (built once in start())
        self._html: str = ""

    # ── Public API (call from the training loop, main thread) ─────────────────

    def configure(
        self,
        config: Any,
        steps_per_epoch: int,
        total_steps: int,
        dataset_info: dict | None = None,
    ) -> None:
        """Set static model / training config.  Call once before the loop."""
        cfg = {
            "vocab_size":       getattr(config, "vocab_size",       0),
            "hidden_dim":       getattr(config, "hidden_dim",       0),
            "num_layers":       getattr(config, "num_layers",       0),
            "num_heads":        getattr(config, "num_heads",        0),
            "ffn_dim":          getattr(config, "ffn_dim",          0),
            "max_seq_len":      getattr(config, "max_seq_len",      128),
            "T":                getattr(config, "T",                25),
            "mask_schedule":    getattr(config, "mask_schedule",    "cosine"),
            "mask_loss_weight": getattr(config, "mask_loss_weight", 5.0),
            "num_epochs":       getattr(config, "num_epochs",       0),
            "batch_size":       getattr(config, "batch_size",       0),
            "learning_rate":    getattr(config, "learning_rate",    0.0),
            "lr_min":           getattr(config, "lr_min",           0.0),
            "lr_schedule":      getattr(config, "lr_schedule",      "cosine"),
            "warmup_steps":     getattr(config, "warmup_steps",     0),
            "weight_decay":     getattr(config, "weight_decay",     0.0),
            "grad_clip":        getattr(config, "grad_clip",        1.0),
            "dataset_name":     getattr(config, "dataset_name",     ""),
            "tokenizer_name":   getattr(config, "tokenizer_name",   "bert-base-uncased"),
            "checkpoint_dir":   getattr(config, "checkpoint_dir",   ""),
        }
        with self._lock:
            self._state["config"] = cfg
            self._state["progress"].update({
                "total_epochs":    getattr(config, "num_epochs", 0),
                "steps_per_epoch": steps_per_epoch,
                "total_steps":     total_steps,
            })
            if dataset_info:
                self._state["dataset"].update(dataset_info)

    def push_step(
        self,
        epoch:       int,
        step:        int,
        global_step: int,
        loss:        float,
        avg_loss:    float,
        lr:          float,
    ) -> None:
        """Push one training step.  Blocks while the client has paused training."""
        now = time.time()
        self._step_times.append(now)

        # ── tok/s and ETA from rolling window ────────────────────────────────
        tok_s   = 0.0
        eta_sec = 0.0
        bs      = self._state["config"].get("batch_size",  0)
        seq     = self._state["config"].get("max_seq_len", 128)
        n_times = len(self._step_times)
        if n_times >= 2:
            elapsed = self._step_times[-1] - self._step_times[0]
            n_steps = n_times - 1
            if elapsed > 0 and n_steps > 0:
                tok_s     = n_steps * bs * seq / elapsed
                sps       = n_steps / elapsed
                remaining = self._state["progress"]["total_steps"] - global_step
                eta_sec   = remaining / sps if sps > 0 else 0.0

        hw = self._sample_hardware()

        with self._lock:
            self._state["progress"].update({
                "epoch":         epoch,
                "step":          step,
                "loss":          loss,
                "avg_loss":      avg_loss,
                "learning_rate": lr,
                "eta_sec":       eta_sec,
            })
            self._state["hardware"].update(hw["hardware"])
            self._state["process"].update(hw["process"])

        point: dict = {
            "global_step": global_step,
            "t":           now - self._start_time,
            "loss":        loss,
            "avg_loss":    avg_loss,
            "tok_s":       tok_s,
            "gpu_sys":     hw["hardware"]["gpu_utilization"],
            "vram_proc":   hw["process"]["vram_used_mb"],
            "cpu_proc":    hw["process"]["cpu_usage_percent"],
            "ram_proc":    hw["process"]["ram_used_mb"],
            "pcie_tx":     hw["hardware"]["pcie_tx_mb"],
            "pcie_rx":     hw["hardware"]["pcie_rx_mb"],
        }
        with self._lock:
            self._history.append(point)
            if len(self._history) > self._MAX_HISTORY:
                self._history = self._history[-self._MAX_HISTORY:]

        self._schedule(self._broadcast({"type": "history_append", "point": point}))
        self._schedule(self._broadcast_update())

        # Block here if the user clicked Pause in the dashboard
        self._paused_event.wait()

    def push_log(self, line: str) -> None:
        """Append a log line and push to all connected clients."""
        with self._lock:
            self._logs.append(line)
            lines = list(self._logs)
        self._schedule(self._broadcast({"type": "logs", "lines": lines}))

    def start(self) -> None:
        """Start the monitor in a background daemon thread and return immediately."""
        self._html = self._build_html()
        t = threading.Thread(
            target=self._run_loop, daemon=True, name="RinGoMonitor"
        )
        t.start()
        time.sleep(0.4)  # let the server bind before printing
        print(f"Training monitor: http://localhost:{self._port}")

    # ── Server internals ──────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._run_server())

    async def _run_server(self) -> None:
        html = self._html

        async def handle_index(request):
            return _web.Response(text=html, content_type="text/html", charset="utf-8")

        async def handle_ws(request):
            ws = _web.WebSocketResponse()
            await ws.prepare(request)
            self._clients.add(ws)
            try:
                await ws.send_str(json.dumps(
                    {"type": "init",    "state":  self._snapshot()},
                    ensure_ascii=False,
                ))
                with self._lock:
                    history_copy = list(self._history)
                await ws.send_str(json.dumps(
                    {"type": "history", "points": history_copy},
                    ensure_ascii=False,
                ))
                async for msg in ws:
                    if msg.type == _aiohttp.WSMsgType.TEXT:
                        try:
                            cmd = json.loads(msg.data)
                            await self._handle_command(cmd, ws)
                        except Exception:
                            pass
                    elif msg.type in (
                        _aiohttp.WSMsgType.ERROR,
                        _aiohttp.WSMsgType.CLOSE,
                    ):
                        break
            finally:
                self._clients.discard(ws)
            return ws

        app = _web.Application()
        app.router.add_get("/",           handle_index)
        app.router.add_get("/index.html", handle_index)
        app.router.add_get("/ws",         handle_ws)

        runner = _web.AppRunner(app)
        await runner.setup()
        site = _web.TCPSite(runner, self._host, self._port)
        await site.start()

        await asyncio.get_event_loop().create_future()  # run forever

    async def _handle_command(self, msg: dict, _ws=None) -> None:
        kind = msg.get("type", "")
        with self._lock:
            if kind == "set_log_level":
                self._state["controls"]["log_level"] = msg.get("level", "info")
            elif kind == "set_log_filters":
                self._state["controls"]["log_filters"] = msg.get("filters", [])
            elif kind == "set_breakpoints":
                self._state["controls"]["breakpoint_events"] = msg.get("events", [])
            elif kind == "set_paused":
                paused = bool(msg.get("paused", False))
                self._state["controls"]["paused"] = paused
                if paused:
                    self._paused_event.clear()   # block push_step
                else:
                    self._paused_event.set()     # unblock push_step
        await self._broadcast_update()

    # ── Broadcast helpers ─────────────────────────────────────────────────────

    def _schedule(self, coro) -> None:
        """Schedule a coroutine on the monitor event loop from any thread."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _broadcast_update(self) -> None:
        await self._broadcast({"type": "update", "state": self._snapshot()})

    async def _broadcast(self, msg: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(msg, ensure_ascii=False)
        dead: set = set()
        for ws in list(self._clients):
            try:
                await ws.send_str(data)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ── State snapshot ────────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        with self._lock:
            logs = list(self._logs)
            snap = {
                "progress":         dict(self._state["progress"]),
                "hardware":         dict(self._state["hardware"]),
                "process":          dict(self._state["process"]),
                "dataset":          dict(self._state["dataset"]),
                "config":           dict(self._state["config"]),
                "module_resources": list(self._state["module_resources"]),
                "controls": {
                    **self._state["controls"],
                    "log_filters":           list(self._state["controls"]["log_filters"]),
                    "breakpoint_events":     list(self._state["controls"]["breakpoint_events"]),
                    "available_log_filters": list(self._state["controls"]["available_log_filters"]),
                },
            }
        snap["logs"] = logs
        return snap

    # ── Hardware sampling ─────────────────────────────────────────────────────

    def _sample_hardware(self) -> dict:
        hw: dict = {
            "gpu_utilization": 0.0,
            "vram_used_mb": 0.0, "vram_total_mb": 0.0,
            "ram_used_mb":  0.0, "ram_total_mb":  0.0,
            "pcie_tx_mb": 0.0,   "pcie_rx_mb": 0.0,
        }
        proc: dict = {
            "cpu_usage_percent": 0.0,
            "vram_used_mb": 0.0,
            "ram_used_mb":  0.0,
        }

        # ── CPU / RAM ─────────────────────────────────────────────────────────
        if _HAS_PSUTIL:
            vm = _psutil.virtual_memory()
            hw["ram_used_mb"]  = vm.used  / 1_048_576
            hw["ram_total_mb"] = vm.total / 1_048_576
            if self._proc:
                try:
                    mi = self._proc.memory_info()
                    proc["ram_used_mb"]       = mi.rss / 1_048_576
                    proc["cpu_usage_percent"] = self._proc.cpu_percent(interval=None)
                except Exception:
                    pass

        # ── GPU / VRAM / PCIe ─────────────────────────────────────────────────
        if _HAS_NVML and self._nvml_dev is not None:
            try:
                util = _pynvml.nvmlDeviceGetUtilizationRates(self._nvml_dev)
                hw["gpu_utilization"] = float(util.gpu)

                mem = _pynvml.nvmlDeviceGetMemoryInfo(self._nvml_dev)
                hw["vram_used_mb"]  = mem.used  / 1_048_576
                hw["vram_total_mb"] = mem.total / 1_048_576

                # Per-process VRAM (Linux / WSL2)
                try:
                    pid = os.getpid()
                    for p in _pynvml.nvmlDeviceGetComputeRunningProcesses(self._nvml_dev):
                        if p.pid == pid and p.usedGpuMemory:
                            proc["vram_used_mb"] = p.usedGpuMemory / 1_048_576
                            break
                except Exception:
                    pass

                # PCIe throughput — returns KB/s on Linux / WSL2
                try:
                    tx_kb = _pynvml.nvmlDeviceGetPcieThroughput(
                        self._nvml_dev, _pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    rx_kb = _pynvml.nvmlDeviceGetPcieThroughput(
                        self._nvml_dev, _pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                    hw["pcie_tx_mb"] = tx_kb / 1_024
                    hw["pcie_rx_mb"] = rx_kb / 1_024
                except Exception:
                    pass

            except Exception:
                pass

        return {"hardware": hw, "process": proc}

    # ── HTML builder ──────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        base_dir  = Path(__file__).parent.parent  # repo root
        html_path = base_dir / "index.html"
        js_path   = base_dir / "app.js"

        html = html_path.read_text(encoding="utf-8")
        js   = js_path.read_text(encoding="utf-8")

        html = html.replace("/* __APP_JS_PLACEHOLDER__ */", js)
        html = html.replace("Neko.rs Training Monitor (TDD)", "RinGo-DLLM Training Monitor")
        html = html.replace("Neko.rs Web Dashboard",          "RinGo-DLLM Training Monitor")

        return html
