"""Deterministic checks for the shared web UI contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UI_FILE = ROOT / "web" / "index.html"


def test_shared_ui_exists_and_contains_operational_endpoints():
    html = UI_FILE.read_text(encoding="utf-8")
    assert "HFAPI Control Center" in html
    assert "/api/status" in html
    assert "/health/json" in html


def test_health_server_exposes_ui_and_status_routes():
    source = (ROOT / "health_server.py").read_text(encoding="utf-8")
    assert 'add_get("/", self.root_endpoint)' in source
    assert 'add_get("/api/status", self.api_status_endpoint)' in source
    assert 'add_get("/health", self.health_endpoint)' in source
    assert 'add_get("/health/json", self.health_json_endpoint)' in source
