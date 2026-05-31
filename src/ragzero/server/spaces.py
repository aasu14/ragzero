"""Published Space — turn the host's tested setup into a public, ask-only assistant.

A Space is a server-side, persistent snapshot of the host's session (providers +
ingested sources + strategy) plus a public display/access config. Visitors hit a
locked-down /public API that runs queries against the Space without ever seeing
data, models, keys, or config.

v1 is single-space: one published assistant per server. State persists to
~/.ragzero/space.json so it survives restarts. The live pipeline is rebuilt by
re-ingesting the snapshot's sources (the server owns that — see main.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

SPACE_DIR = Path(os.environ.get("RAGZERO_HOME", str(Path.home() / ".ragzero")))
SPACE_PATH = SPACE_DIR / "space.json"

ALL_MODES = ["simple", "graph", "agentic", "multilingual"]


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


@dataclass
class SpaceConfig:
    """Everything about a published assistant — display, access, and the
    snapshot of the host's pipeline needed to answer."""
    slug: str = "assistant"
    name: str = "Knowledge Assistant"
    title: str = "Ask me anything"
    welcome: str = "Ask a question about the knowledge base and I'll answer with citations."
    accent: str = "#d4ff00"
    footer: str = ""
    suggested_questions: list[str] = field(default_factory=list)

    allowed_modes: list[str] = field(default_factory=lambda: ["simple"])
    default_mode: str = "simple"
    show_citations: bool = True
    show_confidence: bool = True
    streaming: bool = True  # client-side typewriter reveal of the final answer

    access_code_hash: str | None = None
    rate_limit_per_min: int = 12
    daily_cap: int = 500

    published: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Snapshot of the host session (built into a live pipeline by the server).
    snapshot: dict[str, Any] = field(default_factory=dict)

    # ---- public-facing view (no secrets, no snapshot) ----
    def public_meta(self) -> dict[str, Any]:
        return {
            "published": self.published,
            "slug": self.slug,
            "name": self.name,
            "title": self.title,
            "welcome": self.welcome,
            "accent": self.accent,
            "footer": self.footer,
            "suggested": self.suggested_questions,
            "modes": self.allowed_modes,
            "default_mode": self.default_mode,
            "show_citations": self.show_citations,
            "show_confidence": self.show_confidence,
            "streaming": self.streaming,
            "access_required": bool(self.access_code_hash),
        }

    # ---- admin-facing view (settings + readiness; never secrets) ----
    def admin_view(self) -> dict[str, Any]:
        snap = self.snapshot or {}
        return {
            "slug": self.slug,
            "name": self.name,
            "title": self.title,
            "welcome": self.welcome,
            "accent": self.accent,
            "footer": self.footer,
            "suggested_questions": self.suggested_questions,
            "allowed_modes": self.allowed_modes,
            "default_mode": self.default_mode,
            "show_citations": self.show_citations,
            "show_confidence": self.show_confidence,
            "streaming": self.streaming,
            "access_required": bool(self.access_code_hash),
            "rate_limit_per_min": self.rate_limit_per_min,
            "daily_cap": self.daily_cap,
            "published": self.published,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "snapshot_info": {
                "llm": (snap.get("llm") or {}).get("provider"),
                "embedder": (snap.get("embedder") or {}).get("provider"),
                "vector_store": (snap.get("vector_store") or {}).get("provider"),
                "n_sources": len(snap.get("sources") or []),
            },
        }


class SpaceManager:
    """Single-space store: persistence + access-code + rate limiting.

    The live pipeline/SessionState is owned by the server (main.py); this class
    holds only serializable config and the abuse-control counters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config: SpaceConfig | None = None
        # rate limiting (in-memory, per client IP)
        self._hits: dict[str, deque] = defaultdict(deque)         # minute window
        self._daily: dict[str, int] = defaultdict(int)            # per-day counter
        self._daily_date: str = date.today().isoformat()
        self.load()

    # ---- persistence ----
    def load(self) -> None:
        try:
            if SPACE_PATH.exists():
                data = json.loads(SPACE_PATH.read_text("utf-8"))
                self.config = SpaceConfig(**data)
        except Exception:
            # Corrupt/incompatible file shouldn't crash the server.
            self.config = None

    def save(self) -> None:
        if self.config is None:
            return
        SPACE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SPACE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self.config), indent=2), "utf-8")
        os.replace(tmp, SPACE_PATH)
        try:
            os.chmod(SPACE_PATH, 0o600)  # snapshot may contain provider keys
        except OSError:
            pass

    # ---- mutations ----
    def apply_settings(self, cfg: SpaceConfig, settings: dict[str, Any]) -> None:
        """Apply host-editable display/access fields onto a config in place."""
        for k in (
            "name", "title", "welcome", "accent", "footer", "suggested_questions",
            "allowed_modes", "default_mode", "show_citations", "show_confidence",
            "streaming", "rate_limit_per_min", "daily_cap",
        ):
            if k in settings and settings[k] is not None:
                setattr(cfg, k, settings[k])
        # normalize modes
        cfg.allowed_modes = [m for m in (cfg.allowed_modes or ["simple"]) if m in ALL_MODES] or ["simple"]
        if cfg.default_mode not in cfg.allowed_modes:
            cfg.default_mode = cfg.allowed_modes[0]
        # access code: "" clears, None leaves unchanged, otherwise sets (hashed)
        if "access_code" in settings:
            code = settings["access_code"]
            if code == "":
                cfg.access_code_hash = None
            elif code is not None:
                cfg.access_code_hash = _hash_code(str(code))
        cfg.updated_at = time.time()

    def check_access(self, code: str | None) -> bool:
        cfg = self.config
        if cfg is None or not cfg.access_code_hash:
            return True
        return bool(code) and _hash_code(code) == cfg.access_code_hash

    # ---- rate limiting ----
    def check_rate(self, ip: str) -> tuple[bool, str]:
        cfg = self.config
        if cfg is None:
            return False, "not published"
        now = time.time()
        with self._lock:
            today = date.today().isoformat()
            if today != self._daily_date:
                self._daily.clear()
                self._daily_date = today
            # per-minute sliding window
            dq = self._hits[ip]
            while dq and now - dq[0] > 60:
                dq.popleft()
            if len(dq) >= max(1, cfg.rate_limit_per_min):
                return False, "Rate limit reached — please wait a moment."
            if self._daily[ip] >= max(1, cfg.daily_cap):
                return False, "Daily question limit reached for this assistant."
            dq.append(now)
            self._daily[ip] += 1
            return True, ""
