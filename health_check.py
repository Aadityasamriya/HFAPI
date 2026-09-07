#!/usr/bin/env python3
"""
Health Check Endpoints for Railway.com Deployment
Provides health monitoring for production deployment
"""

import asyncio
import json
import time
from typing import Dict, Any
from datetime import datetime
import logging
import os
from pathlib import Path

try:
    from bot.config import Config
    from bot.storage_manager import storage_manager
    from bot.admin.system import admin_system
    BOT_IMPORTS_AVAILABLE = True
except ImportError:
    Config = None
    storage_manager = None
    admin_system = None
    BOT_IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health check system for production deployment."""

    def __init__(self):
        self.start_time = time.time()
        self.last_check = None
        self.health_cache = None
        self.cache_duration = 30

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        now = time.time()
        if (self.health_cache and self.last_check and
                (now - self.last_check) < self.cache_duration):
            return self.health_cache

        health_status = {
            'status': 'unknown',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': now - self.start_time,
            'checks': {}
        }
        overall_healthy = True
        is_degraded = False

        try:
            health_status['checks']['bot_imports'] = await self._check_bot_imports()
            if not health_status['checks']['bot_imports']['healthy']:
                overall_healthy = False

            health_status['checks']['database'] = await self._check_database()
            if not health_status['checks']['database']['healthy']:
                overall_healthy = False

            health_status['checks']['environment'] = await self._check_environment()
            if not health_status['checks']['environment']['healthy']:
                overall_healthy = False

            health_status['checks']['huggingface_api'] = await self._check_huggingface_api()
            if not health_status['checks']['huggingface_api']['healthy']:
                is_degraded = True
                logger.info("Hugging Face API unavailable; system is operating in degraded mode")

            health_status['checks']['external_databases'] = await self._check_external_databases()
            db_result = health_status['checks']['external_databases']
            if not db_result.get('mongodb_healthy', False):
                overall_healthy = False
            elif db_result.get('mongodb_healthy', False) and not db_result.get('supabase_healthy', True):
                is_degraded = True
                logger.info("System degraded: MongoDB operational, Supabase unavailable")

            health_status['checks']['admin_system'] = await self._check_admin_system()
            health_status['checks']['file_system'] = await self._check_file_system()
            if not health_status['checks']['file_system']['healthy']:
                overall_healthy = False

            health_status['checks']['railway_environment'] = await self._check_railway_environment()
        except Exception:
            logger.exception("Health check error")
            health_status['checks']['general_error'] = {
                'healthy': False,
                'error': 'Health check execution failed',
                'timestamp': datetime.utcnow().isoformat()
            }
            overall_healthy = False

        if overall_healthy:
            if is_degraded:
                health_status['status'] = 'degraded'
                health_status['healthy'] = True
                health_status['degraded'] = True
                health_status['message'] = 'Core functionality operational, some features may be limited'
            else:
                health_status['status'] = 'healthy'
                health_status['healthy'] = True
                health_status['degraded'] = False
                health_status['message'] = 'All systems operational'
        else:
            health_status['status'] = 'unhealthy'
            health_status['healthy'] = False
            health_status['degraded'] = False
            health_status['message'] = 'Critical functionality impaired'

        self.health_cache = health_status
        self.last_check = now
        return health_status

    async def _check_bot_imports(self) -> Dict[str, Any]:
        """Check if bot modules can be imported."""
        result = {'healthy': False, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        try:
            if BOT_IMPORTS_AVAILABLE:
                result['healthy'] = True
                result['details'] = {
                    'imports': 'available',
                    'config': 'loaded',
                    'storage_manager': 'imported',
                    'admin_system': 'imported'
                }
            else:
                result['details'] = {'imports': 'failed', 'error': 'Bot modules not available'}
        except Exception:
            result['details'] = {'error': 'Bot import check failed', 'imports': 'failed'}
        return result

    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        result = {'healthy': False, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        try:
            if not BOT_IMPORTS_AVAILABLE:
                result['details'] = {'error': 'Bot imports not available'}
                return result
            has_db_config = Config and (Config.has_mongodb_config() or Config.has_supabase_config())
            if not has_db_config:
                result['details'] = {
                    'error': 'No database configuration found',
                    'mongodb': Config.has_mongodb_config() if Config else False,
                    'supabase': Config.has_supabase_config() if Config else False
                }
                return result
            if storage_manager and not storage_manager.initialized:
                await storage_manager.initialize()
            if storage_manager:
                storage = await storage_manager.ensure_connected()
                if storage is not None:
                    result['healthy'] = True
                    result['details'] = {
                        'connected': True,
                        'provider': type(storage).__name__,
                        'initialized': storage_manager.initialized
                    }
                else:
                    result['details'] = {'connected': False, 'error': 'Storage connection failed'}
            else:
                result['details'] = {'connected': False, 'error': 'Storage manager not available'}
        except Exception:
            result['details'] = {'error': 'Database health check failed', 'connected': False}
        return result

    async def _check_environment(self) -> Dict[str, Any]:
        """Check required environment configuration without exposing secrets."""
        result = {'healthy': False, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        try:
            if not BOT_IMPORTS_AVAILABLE:
                result['details'] = {'error': 'Bot imports not available'}
                return result
            required_vars = ['TELEGRAM_BOT_TOKEN', 'ENCRYPTION_SEED']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            db_configured = Config and (Config.has_mongodb_config() or Config.has_supabase_config())
            if not missing_vars and db_configured:
                result['healthy'] = True
                result['details'] = {
                    'required_vars': 'present',
                    'database': 'configured',
                    'environment_type': Config.get_environment_type() if Config else 'unknown'
                }
            else:
                result['details'] = {
                    'missing_vars': missing_vars,
                    'database_configured': bool(db_configured),
                    'environment_type': Config.get_environment_type() if Config else 'unknown'
                }
        except Exception:
            result['details'] = {'error': 'Environment health check failed'}
        return result

    async def _check_admin_system(self) -> Dict[str, Any]:
        """Check admin system status."""
        result = {'healthy': False, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        try:
            if not BOT_IMPORTS_AVAILABLE:
                result['details'] = {'error': 'Bot imports not available'}
                return result
            if admin_system and not admin_system._initialized:
                await admin_system.initialize()
            if admin_system:
                bootstrap_completed = admin_system.is_bootstrap_completed()
                result['healthy'] = True
                result['details'] = {
                    'initialized': admin_system._initialized,
                    'bootstrap_completed': bootstrap_completed,
                    'admin_count': len(admin_system._admin_users),
                    'status': 'ready' if bootstrap_completed else 'needs_bootstrap'
                }
            else:
                result['details'] = {'error': 'Admin system not available', 'initialized': False}
        except Exception:
            result['details'] = {'error': 'Admin system health check failed', 'initialized': False}
        return result

    async def _check_file_system(self) -> Dict[str, Any]:
        """Check temporary file read/write access."""
        result = {'healthy': False, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        test_file = None
        try:
            import tempfile
            test_file = Path(tempfile.gettempdir()) / f'health_check_{os.getpid()}.tmp'
            test_content = f'health-check-{time.time_ns()}'
            test_file.write_text(test_content, encoding='utf-8')
            read_content = test_file.read_text(encoding='utf-8')
            result['healthy'] = read_content == test_content
            result['details'] = {
                'write_access': True,
                'read_access': True,
                'test_passed': result['healthy']
            }
        except Exception:
            result['details'] = {'error': 'File system health check failed', 'write_access': False, 'read_access': False}
        finally:
            if test_file:
                try:
                    test_file.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Unable to clean up health-check temporary file")
        return result

    async def _check_huggingface_api(self) -> Dict[str, Any]:
        """Verify Hugging Face authentication and API reachability with a bounded request."""
        result = {'healthy': False, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HUGGING_FACE_TOKEN')
        if not token:
            result['details'] = {
                'token_configured': False,
                'check_type': 'authenticated_api_request',
                'error': 'Hugging Face API token is not configured'
            }
            return result

        try:
            import httpx
            timeout = httpx.Timeout(5.0, connect=3.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(
                    'https://huggingface.co/api/whoami-v2',
                    headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
                )
            if response.status_code == 200:
                result['healthy'] = True
                result['details'] = {
                    'token_configured': True,
                    'api_reachable': True,
                    'authenticated': True,
                    'status_code': response.status_code,
                    'check_type': 'authenticated_api_request'
                }
            elif response.status_code in (401, 403):
                result['details'] = {
                    'token_configured': True,
                    'api_reachable': True,
                    'authenticated': False,
                    'status_code': response.status_code,
                    'check_type': 'authenticated_api_request',
                    'error': 'Hugging Face authentication failed'
                }
            else:
                result['details'] = {
                    'token_configured': True,
                    'api_reachable': True,
                    'authenticated': None,
                    'status_code': response.status_code,
                    'check_type': 'authenticated_api_request',
                    'error': 'Hugging Face API returned an unexpected status'
                }
        except Exception as exc:
            logger.warning("Hugging Face health check failed: %s", type(exc).__name__)
            result['details'] = {
                'token_configured': True,
                'api_reachable': False,
                'check_type': 'authenticated_api_request',
                'error': 'Hugging Face API request failed'
            }
        return result

    async def _check_external_databases(self) -> Dict[str, Any]:
        """Check external database dependencies with graceful degradation support."""
        result = {
            'healthy': False,
            'details': {},
            'timestamp': datetime.utcnow().isoformat(),
            'mongodb_healthy': False,
            'supabase_healthy': False
        }
        try:
            if not BOT_IMPORTS_AVAILABLE:
                result['details'] = {'error': 'Bot imports not available'}
                return result
            mongodb_status = {'configured': False, 'accessible': False}
            supabase_status = {'configured': False, 'accessible': False}

            mongodb_uri = Config.MONGODB_URI if Config else None
            if mongodb_uri and mongodb_uri.startswith(('mongodb://', 'mongodb+srv://')):
                mongodb_status['configured'] = True
                try:
                    import pymongo
                    client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
                    await asyncio.get_event_loop().run_in_executor(None, lambda: client.admin.command('ping'))
                    mongodb_status['accessible'] = True
                    result['mongodb_healthy'] = True
                    client.close()
                except Exception as exc:
                    mongodb_status['error'] = type(exc).__name__
                    logger.warning("MongoDB connectivity failed: %s", type(exc).__name__)
            else:
                logger.warning("MongoDB not configured - core functionality may be limited")

            supabase_url = Config.SUPABASE_MGMT_URL if Config else None
            if supabase_url and supabase_url.startswith(('postgresql://', 'postgres://')):
                supabase_status['configured'] = True
                try:
                    import asyncpg
                    conn = await asyncpg.connect(supabase_url, command_timeout=5)
                    await conn.execute('SELECT 1')
                    await conn.close()
                    supabase_status['accessible'] = True
                    result['supabase_healthy'] = True
                except Exception as exc:
                    supabase_status['error'] = type(exc).__name__
                    logger.warning("Supabase connectivity failed: %s", type(exc).__name__)
            else:
                result['supabase_healthy'] = True

            mongodb_operational = mongodb_status['configured'] and mongodb_status['accessible']
            supabase_operational = not supabase_status['configured'] or supabase_status['accessible']
            result['healthy'] = mongodb_operational
            if mongodb_operational and not supabase_operational:
                result['degraded_reason'] = 'Supabase unavailable, operating in MongoDB-only fallback mode'
            result['details'] = {
                'mongodb': mongodb_status,
                'supabase': supabase_status,
                'mongodb_operational': mongodb_operational,
                'supabase_operational': supabase_operational,
                'fallback_mode': mongodb_operational and not supabase_operational and supabase_status['configured'],
                'core_functionality': 'operational' if mongodb_operational else 'impaired'
            }
        except Exception:
            result['details'] = {'error': 'External database health check failed', 'database_check_failed': True}
            logger.exception("External database health check failed")
        return result

    async def _check_railway_environment(self) -> Dict[str, Any]:
        """Check Railway-specific environment and production safety settings."""
        result = {'healthy': True, 'details': {}, 'timestamp': datetime.utcnow().isoformat()}
        try:
            railway_vars = {
                'RAILWAY_ENVIRONMENT': os.getenv('RAILWAY_ENVIRONMENT'),
                'RAILWAY_SERVICE_NAME': os.getenv('RAILWAY_SERVICE_NAME'),
                'RAILWAY_PROJECT_ID': os.getenv('RAILWAY_PROJECT_ID'),
                'PORT': os.getenv('PORT')
            }
            is_railway = any(railway_vars.values())
            result['details'] = {
                'is_railway_environment': is_railway,
                'railway_vars': {k: bool(v) for k, v in railway_vars.items()},
                'port_assignment': 'dynamic' if railway_vars['PORT'] else 'static',
                'environment_type': railway_vars['RAILWAY_ENVIRONMENT'] or 'unknown'
            }
            if is_railway:
                port_value = railway_vars['PORT']
                result['details']['port_valid'] = bool(port_value and port_value.isdigit() and 1000 <= int(port_value) <= 65535)
                if port_value and port_value.isdigit():
                    result['details']['assigned_port'] = int(port_value)
                if railway_vars['RAILWAY_ENVIRONMENT'] == 'production':
                    encryption_seed = os.getenv('ENCRYPTION_SEED')
                    test_mode = os.getenv('TEST_MODE', 'false').lower()
                    result['details']['production_security'] = {
                        'encryption_seed_set': bool(encryption_seed),
                        'test_mode_disabled': test_mode not in ('true', '1', 'yes', 'on')
                    }
        except Exception:
            result['details'] = {'error': 'Railway environment check failed', 'railway_check_failed': True}
        return result

    async def get_simple_health(self) -> str:
        """Get simple health status for basic endpoints."""
        try:
            health = await self.get_health_status()
            return "OK" if health['healthy'] else "ERROR"
        except Exception:
            return "ERROR"

    async def get_readiness(self) -> Dict[str, Any]:
        """Get readiness status for Railway deployment."""
        health = await self.get_health_status()
        critical_checks = ['bot_imports', 'database', 'environment', 'file_system']
        ready = all(health['checks'].get(check, {}).get('healthy', False) for check in critical_checks)
        return {
            'ready': ready,
            'status': 'ready' if ready else 'not_ready',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': health['uptime']
        }


health_checker = HealthChecker()


async def main():
    """CLI health check for testing."""
    print("Running Health Check...")
    health = await health_checker.get_health_status()
    print(json.dumps(health, indent=2))
    print(f"\nOverall Status: {'HEALTHY' if health['healthy'] else 'UNHEALTHY'}")
    print(f"Uptime: {health['uptime']:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())