#!/usr/bin/env python3
"""Shared web server for HFAPI health endpoints and browser control center."""

import asyncio
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response

from health_check import health_checker

logger = logging.getLogger(__name__)
UI_FILE = Path(__file__).resolve().parent / "web" / "index.html"
DEFAULT_PORT = 8080
MIN_PORT = 1
MAX_PORT = 65535


class HealthServer:
    """Serve the HFAPI UI and health endpoints from the bot's shared HTTP process."""

    def __init__(self, port: Optional[int] = None):
        self.port = port if port is not None else self._configured_port()
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.actual_port: Optional[int] = None

    @staticmethod
    def _validate_port(value: int) -> int:
        """Validate a TCP port before attempting to bind it."""
        if not isinstance(value, int) or isinstance(value, bool) or not MIN_PORT <= value <= MAX_PORT:
            raise ValueError(f"Port must be an integer between {MIN_PORT} and {MAX_PORT}")
        return value

    @classmethod
    def _configured_port(cls) -> int:
        """Read PORT safely and fail with a useful error instead of a raw ValueError."""
        raw_port = os.getenv("PORT")
        if raw_port is None or not raw_port.strip():
            return DEFAULT_PORT
        try:
            return cls._validate_port(int(raw_port.strip()))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid PORT environment variable: {raw_port!r}") from exc

    @staticmethod
    def _response_headers() -> dict[str, str]:
        """Return conservative browser security headers for the public control center."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }

    async def health_endpoint(self, request: Request) -> Response:
        """Return a lightweight health response suitable for platform probes."""
        try:
            health_status = await health_checker.get_health_status()
            status = health_status.get("status", "unknown")
            status_code = 200 if status in {"healthy", "degraded"} else 503
            text = status.upper()
            if status == "degraded":
                text += " - Core functionality operational"
            if status not in {"healthy", "degraded"}:
                text += " - " + health_status.get("message", "Critical functionality impaired")
            return web.Response(
                text=text,
                status=status_code,
                headers={
                    **self._response_headers(),
                    "X-Health-Status": status,
                    "X-Health-Message": health_status.get("message", ""),
                },
            )
        except Exception:
            logger.exception("Health check endpoint error")
            return web.Response(
                text="ERROR - Health check system failure",
                status=500,
                headers=self._response_headers(),
            )

    async def health_json_endpoint(self, request: Request) -> Response:
        """Return detailed machine-readable health information."""
        try:
            health_status = await health_checker.get_health_status()
            status = health_status.get("status", "unknown")
            return web.json_response(
                health_status,
                status=200 if status in {"healthy", "degraded"} else 503,
                headers={
                    **self._response_headers(),
                    "X-Health-Status": status,
                    "X-Health-Message": health_status.get("message", ""),
                },
            )
        except Exception:
            logger.exception("Health JSON endpoint error")
            return web.json_response(
                {
                    "status": "error",
                    "healthy": False,
                    "degraded": False,
                    "message": "Health check system failure",
                    "timestamp": datetime.utcnow().isoformat(),
                },
                status=500,
                headers=self._response_headers(),
            )

    async def api_status_endpoint(self, request: Request) -> Response:
        """Expose only safe status data to the browser UI; never expose secrets."""
        try:
            health_status = await health_checker.get_health_status()
            safe = {
                "status": health_status.get("status", "unknown"),
                "message": health_status.get("message", ""),
                "uptime_seconds": round(float(health_status.get("uptime", 0)), 1),
                "timestamp": health_status.get("timestamp"),
                "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "development",
                "checks": {
                    name: {"healthy": bool(value.get("healthy", False))}
                    for name, value in health_status.get("checks", {}).items()
                    if isinstance(value, dict)
                },
            }
            return web.json_response(
                safe,
                headers={**self._response_headers(), "Cache-Control": "no-store"},
            )
        except Exception:
            logger.exception("Status API endpoint error")
            return web.json_response(
                {
                    "status": "error",
                    "message": "Status temporarily unavailable",
                    "uptime_seconds": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment": "unknown",
                    "checks": {},
                },
                status=503,
                headers={**self._response_headers(), "Cache-Control": "no-store"},
            )

    async def root_endpoint(self, request: Request) -> Response:
        """Serve the browser control center from the same HTTP server as health probes."""
        if not UI_FILE.is_file():
            return web.json_response(
                {"service": "HFAPI", "status": "running", "ui": "not installed"},
                status=200,
                headers=self._response_headers(),
            )
        response = web.FileResponse(UI_FILE, headers=self._response_headers())
        response.headers["Cache-Control"] = "no-store"
        return response

    def setup_routes(self) -> None:
        assert self.app is not None
        self.app.router.add_get("/", self.root_endpoint)
        self.app.router.add_get("/api/status", self.api_status_endpoint)
        self.app.router.add_get("/health", self.health_endpoint)
        self.app.router.add_get("/health/json", self.health_json_endpoint)
        self.app.router.add_get("/healthcheck", self.health_endpoint)
        self.app.router.add_get("/status", self.health_endpoint)

    def _is_port_available(self, port: int) -> bool:
        self._validate_port(port)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
                return True
        except OSError:
            return False

    def _find_available_port(self, preferred_port: int, max_attempts: int = 10) -> Optional[int]:
        """Choose a bindable port, preserving the platform-assigned PORT when provided."""
        preferred_port = self._validate_port(preferred_port)
        if "PORT" in os.environ:
            configured_port = self._configured_port()
            if self._is_port_available(configured_port):
                return configured_port
            # Never silently move away from a platform-assigned port: the reverse proxy
            # routes traffic to that port, so an alternate port would make the service
            # appear healthy locally but unreachable externally.
            raise OSError(f"Configured PORT {configured_port} is not available")
        if self._is_port_available(preferred_port):
            return preferred_port
        alternatives = [8080, 8000, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088][:max_attempts]
        for port in alternatives:
            if port != preferred_port and self._is_port_available(port):
                return port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("0.0.0.0", 0))
                return sock.getsockname()[1]
        except OSError:
            return None

    async def start(self) -> None:
        logger.info("Starting shared HFAPI web server (preferred port: %s)", self.port)
        available_port = self._find_available_port(self.port)
        if available_port is None:
            raise RuntimeError("Could not find any available port for health server")
        self.actual_port = available_port
        self.app = web.Application()
        self.setup_routes()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.actual_port)
        await self.site.start()
        if self.actual_port != self.port:
            os.environ["HEALTH_SERVER_PORT"] = str(self.actual_port)
        logger.info("Shared web server started on port %s", self.actual_port)

    async def stop(self) -> None:
        try:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
        finally:
            self.site = None
            self.runner = None
            self.app = None


health_server = HealthServer()


async def main() -> None:
    await health_server.start()
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await health_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
