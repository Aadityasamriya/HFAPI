#!/usr/bin/env python3
"""Shared web server for HFAPI health endpoints and browser control center."""

import asyncio
import json
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


class HealthServer:
    """Serve the HFAPI web UI and operational endpoints on the same PORT as the bot process."""

    def __init__(self, port: Optional[int] = None):
        self.port = port if port is not None else int(os.getenv("PORT", "8080"))
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.actual_port: Optional[int] = None

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
                    "X-Health-Status": status,
                    "X-Health-Message": health_status.get("message", ""),
                },
            )
        except Exception:
            logger.exception("Health check endpoint error")
            return web.Response(text="ERROR - Health check system failure", status=500)

    async def health_json_endpoint(self, request: Request) -> Response:
        """Return detailed machine-readable health information."""
        try:
            health_status = await health_checker.get_health_status()
            status = health_status.get("status", "unknown")
            return web.json_response(
                health_status,
                status=200 if status in {"healthy", "degraded"} else 503,
                headers={
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
            )

    async def api_status_endpoint(self, request: Request) -> Response:
        """Expose only safe status data to the browser UI; never expose secrets."""
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
        return web.json_response(safe, headers={"Cache-Control": "no-store"})

    async def root_endpoint(self, request: Request) -> Response:
        """Serve the browser control center from the same HTTP server as health probes."""
        if not UI_FILE.is_file():
            return web.json_response(
                {"service": "HFAPI", "status": "running", "ui": "not installed"},
                status=200,
            )
        return web.FileResponse(UI_FILE)

    def setup_routes(self) -> None:
        assert self.app is not None
        self.app.router.add_get("/", self.root_endpoint)
        self.app.router.add_get("/api/status", self.api_status_endpoint)
        self.app.router.add_get("/health", self.health_endpoint)
        self.app.router.add_get("/health/json", self.health_json_endpoint)
        self.app.router.add_get("/healthcheck", self.health_endpoint)
        self.app.router.add_get("/status", self.health_endpoint)

    def _is_port_available(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
                return True
        except OSError:
            return False

    def _find_available_port(self, preferred_port: int, max_attempts: int = 10) -> Optional[int]:
        """Find a bindable port, preferring the platform-assigned PORT."""
        if "PORT" in os.environ:
            railway_port = int(os.environ["PORT"])
            if self._is_port_available(railway_port):
                return railway_port
            logger.warning("Assigned PORT %s is not available", railway_port)
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
