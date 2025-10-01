#!/usr/bin/env python3
"""
Health Check Web Server for Railway.com Deployment
Provides /health endpoint for Railway's modern Railpack infrastructure alongside the Telegram bot
Optimized for Railway's 2025 deployment system with dynamic PORT configuration
"""

import asyncio
import logging
import os
import socket
from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from typing import Optional, List
import json
from datetime import datetime

# Import the existing health checker
from health_check import health_checker

logger = logging.getLogger(__name__)

class HealthServer:
    """
    Simple web server providing health check endpoints for Railway.com
    Runs alongside the Telegram bot for deployment monitoring
    """
    
    def __init__(self, port: Optional[int] = None):
        # Use PORT environment variable when available (Railway.com compatibility)
        self.port = port if port is not None else int(os.getenv('PORT', '8080'))
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.actual_port: Optional[int] = None  # Track the actual port being used
    
    async def health_endpoint(self, request: Request) -> Response:
        """Main health check endpoint for Railway.com with degraded status support"""
        try:
            health_status = await health_checker.get_health_status()
            status = health_status.get('status', 'unknown')
            
            # Railway health checks: return 200 for both healthy and degraded states
            # Only return 503 when critical functionality is impaired
            if status in ['healthy', 'degraded']:
                response_text = f"{status.upper()}"
                if status == 'degraded':
                    response_text += " - Core functionality operational"
                    
                return web.Response(
                    text=response_text,
                    status=200,
                    headers={
                        'Content-Type': 'text/plain',
                        'X-Health-Status': status,
                        'X-Health-Message': health_status.get('message', '')
                    }
                )
            else:
                return web.Response(
                    text=f"UNHEALTHY - {health_status.get('message', 'Critical functionality impaired')}",
                    status=503,
                    headers={
                        'Content-Type': 'text/plain',
                        'X-Health-Status': status,
                        'X-Health-Message': health_status.get('message', '')
                    }
                )
        except Exception as e:
            logger.error(f"Health check endpoint error: {e}")
            return web.Response(
                text="ERROR - Health check system failure",
                status=500,
                headers={
                    'Content-Type': 'text/plain',
                    'X-Health-Status': 'error'
                }
            )
    
    async def health_json_endpoint(self, request: Request) -> Response:
        """Detailed JSON health check endpoint with degraded status support"""
        try:
            health_status = await health_checker.get_health_status()
            status = health_status.get('status', 'unknown')
            
            # Return 200 for both healthy and degraded states (Railway compatible)
            # Return 503 only when critical functionality is impaired
            if status in ['healthy', 'degraded']:
                status_code = 200
            else:
                status_code = 503
            
            return web.Response(
                text=json.dumps(health_status, indent=2),
                status=status_code,
                headers={
                    'Content-Type': 'application/json',
                    'X-Health-Status': status,
                    'X-Health-Message': health_status.get('message', '')
                }
            )
        except Exception as e:
            logger.error(f"Health JSON endpoint error: {e}")
            error_response = {
                'status': 'error',
                'healthy': False,
                'degraded': False,
                'message': 'Health check system failure',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            return web.Response(
                text=json.dumps(error_response, indent=2),
                status=500,
                headers={
                    'Content-Type': 'application/json',
                    'X-Health-Status': 'error'
                }
            )
    
    async def root_endpoint(self, request: Request) -> Response:
        """Root endpoint with basic info"""
        info = {
            'service': 'Hugging Face By AadityaLabs AI Bot',
            'status': 'running',
            'port': self.actual_port or self.port,
            'environment': 'railway' if 'RAILWAY_ENVIRONMENT' in os.environ else 'development',
            'endpoints': {
                'health_simple': '/health',
                'health_detailed': '/health/json',
                'health_check': '/healthcheck',
                'status': '/status'
            }
        }
        return web.Response(
            text=json.dumps(info, indent=2),
            status=200,
            headers={'Content-Type': 'application/json'}
        )
    
    def setup_routes(self):
        """Setup all web routes"""
        assert self.app is not None, "Application not initialized"
        self.app.router.add_get('/', self.root_endpoint)
        self.app.router.add_get('/health', self.health_endpoint)
        self.app.router.add_get('/health/json', self.health_json_endpoint)
        # Alternative endpoints for compatibility
        self.app.router.add_get('/healthcheck', self.health_endpoint)
        self.app.router.add_get('/status', self.health_endpoint)
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False
    
    def _find_available_port(self, preferred_port: int, max_attempts: int = 10) -> Optional[int]:
        """Find an available port, starting with preferred_port"""
        # For Railway, always prioritize the PORT environment variable
        if 'PORT' in os.environ:
            railway_port = int(os.environ['PORT'])
            if self._is_port_available(railway_port):
                return railway_port
            logger.warning(f"⚠️  Railway assigned port {railway_port} is not available")
        
        # Try the preferred port first
        if self._is_port_available(preferred_port):
            return preferred_port
        
        # Try alternative ports in the range
        alternative_ports = [
            8080, 8000, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088
        ]
        
        for port in alternative_ports:
            if port != preferred_port and self._is_port_available(port):
                logger.info(f"🔄 Using alternative port {port} (preferred {preferred_port} unavailable)")
                return port
        
        # Last resort: let the system assign a random port
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('0.0.0.0', 0))
                random_port = sock.getsockname()[1]
                logger.warning(f"⚠️  Using system-assigned port {random_port}")
                return random_port
        except OSError:
            return None
    
    async def start(self):
        """Start the health check web server with port conflict handling"""
        try:
            logger.info(f"🌐 Starting health check web server (preferred port: {self.port})...")
            
            # Find an available port
            available_port = self._find_available_port(self.port)
            if available_port is None:
                raise RuntimeError("Could not find any available port for health server")
            
            self.actual_port = available_port
            
            logger.info(f"🔌 Using port {self.actual_port} (from {'Railway PORT env' if 'PORT' in os.environ and self.actual_port == int(os.environ['PORT']) else 'alternative selection' if self.actual_port != self.port else 'preferred'})")
            
            self.app = web.Application()
            self.setup_routes()
            
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(
                self.runner, 
                '0.0.0.0',  # Bind to all interfaces for Railway.com
                self.actual_port
            )
            await self.site.start()
            
            logger.info(f"✅ Health check web server started successfully")
            logger.info(f"🔗 Health endpoint: http://0.0.0.0:{self.actual_port}/health")
            logger.info(f"📊 Port selection: {'Railway dynamic assignment' if 'PORT' in os.environ else 'Local development'}")
            
            # Update environment for other components that might need the actual port
            if self.actual_port != self.port:
                os.environ['HEALTH_SERVER_PORT'] = str(self.actual_port)
            
        except Exception as e:
            logger.error(f"❌ Failed to start health check web server: {e}")
            raise
    
    async def stop(self):
        """Stop the health check web server"""
        try:
            logger.info("🛑 Stopping health check web server...")
            
            if self.site:
                await self.site.stop()
                self.site = None
            
            if self.runner:
                await self.runner.cleanup()
                self.runner = None
                
            self.app = None
            logger.info("✅ Health check web server stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping health check web server: {e}")

# Global health server instance (will use PORT env var if available)
health_server = HealthServer()

async def main():
    """Test the health server standalone"""
    try:
        await health_server.start()
        logger.info("Health server running... Press Ctrl+C to stop")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down health server...")
    finally:
        await health_server.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())