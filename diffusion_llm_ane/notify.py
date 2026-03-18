"""
Discord Webhook notification utility.

Usage:
    from notify import Notifier
    notifier = Notifier("https://discord.com/api/webhooks/...")
    notifier.send("学習開始しました")

Webhook URL は環境変数 DISCORD_WEBHOOK_URL でも指定可能。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request
import urllib.error
from datetime import datetime

# .env ファイルから DISCORD_WEBHOOK_URL を読み込む（存在する場合）
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class Notifier:
    """Thin Discord Webhook client (stdlib only — no extra dependencies)."""

    def __init__(self, webhook_url: str | None = None) -> None:
        self.webhook_url = (
            webhook_url
            or os.environ.get("DISCORD_WEBHOOK_URL", "")
        )

    def send(self, content: str, *, embed: dict | None = None) -> bool:
        """
        POST a message to the Discord webhook.

        Args:
            content: Plain text message
            embed:   Optional Discord embed dict

        Returns:
            True on success, False on failure (errors are printed, not raised)
        """
        if not self.webhook_url:
            return False

        payload: dict = {"content": content}
        if embed:
            payload["embeds"] = [embed]

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except urllib.error.URLError as exc:
            print(f"[notify] Discord 送信失敗: {exc}")
            return False

    # ── Convenience helpers ───────────────────────────────────────────────

    def training_start(self, n_params: int, n_epochs: int, device: str) -> None:
        self.send(
            "",
            embed={
                "title": "🚀 学習開始",
                "color": 0x5865F2,
                "fields": [
                    {"name": "デバイス",     "value": device,          "inline": True},
                    {"name": "パラメータ数", "value": f"{n_params:,}", "inline": True},
                    {"name": "エポック数",   "value": str(n_epochs),   "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def epoch_update(
        self,
        epoch: int,
        n_epochs: int,
        train_loss: float,
        val_loss: float,
        elapsed_s: float,
        is_best: bool,
    ) -> None:
        title = f"{'✨ Best ' if is_best else ''}Epoch {epoch}/{n_epochs}"
        colour = 0x57F287 if is_best else 0xFEE75C
        self.send(
            "",
            embed={
                "title": title,
                "color": colour,
                "fields": [
                    {"name": "Train Loss", "value": f"{train_loss:.4f}", "inline": True},
                    {"name": "Val Loss",   "value": f"{val_loss:.4f}",   "inline": True},
                    {"name": "時間",        "value": f"{elapsed_s:.0f}s", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def training_complete(self, best_val_loss: float, n_epochs: int) -> None:
        self.send(
            "",
            embed={
                "title": "🎉 学習完了",
                "color": 0x57F287,
                "fields": [
                    {"name": "最良 Val Loss", "value": f"{best_val_loss:.4f}", "inline": True},
                    {"name": "総エポック数",  "value": str(n_epochs),          "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def error(self, message: str) -> None:
        self.send(
            "",
            embed={
                "title": "❌ エラー発生",
                "description": f"```\n{message[:1000]}\n```",
                "color": 0xED4245,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
