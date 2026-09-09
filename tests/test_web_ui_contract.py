"""Deterministic contract tests for the shared browser control center."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "web" / "index.html"
SERVER = ROOT / "health_server.py"


def test_control_center_has_core_ux_contract():
    html = UI.read_text(encoding="utf-8")
    for marker in (
        "HFAPI Control Center",
        "Refresh now",
        "/api/status",
        "Telegram + Web",
        "System checks",
        "prefers-reduced-motion",
    ):
        assert marker in html


def test_shared_server_exposes_ui_and_safe_status():
    source = SERVER.read_text(encoding="utf-8")
    assert 'add_get("/", self.root_endpoint)' in source
    assert 'add_get("/api/status", self.api_status_endpoint)' in source
    assert '"checks": {' in source
    assert '"Cache-Control": "no-store"' in source


def test_ui_does_not_render_known_secret_names():
    html = UI.read_text(encoding="utf-8").lower()
    for secret_name in ("telegram_bot_token", "hf_token", "encryption_seed", "mongodb_uri", "supabase_key"):
        assert secret_name not in html
