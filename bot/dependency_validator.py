"""
Dependency Validation Module for Hugging Face By AadityaLabs AI Bot
P0 CRITICAL: Prevents runtime import failures and ensures deployment reliability

This module validates all critical dependencies and environment variables at startup,
providing fail-fast behavior with clear error messages for missing or incompatible dependencies.
"""

import sys
import os
import logging
from typing import Dict, List, Tuple, Optional
from importlib import import_module
import subprocess

logger = logging.getLogger(__name__)

class DependencyValidationError(Exception):
    """Raised when critical dependencies are missing or incompatible"""
    pass

class EnvironmentValidationError(Exception):
    """Raised when required environment variables are missing"""
    pass

class DependencyValidator:
    """
    Comprehensive dependency and environment validation system
    
    Features:
    - Import validation with version checking
    - Environment variable validation with fail-fast behavior
    - Clear error messages with actionable guidance
    - Graceful degraded mode handling for optional dependencies
    """
    
    # Critical dependencies that MUST be present for bot to function
    CRITICAL_DEPENDENCIES = {
        'telegram': {
            'package': 'python-telegram-bot',
            'min_version': '22.4',
            'import_names': ['telegram', 'telegram.ext'],
            'validation_code': 'from telegram import Update; from telegram.ext import Application'
        },
        'aiohttp': {
            'package': 'aiohttp',
            'min_version': '3.12.15',
            'import_names': ['aiohttp'],
            'validation_code': 'import aiohttp'
        },
        'cryptography': {
            'package': 'cryptography',
            'min_version': '46.0.1',
            'import_names': ['cryptography'],
            'validation_code': 'import cryptography'
        },
        'pymongo': {
            'package': 'pymongo',
            'min_version': '4.15.1',
            'import_names': ['pymongo'],
            'validation_code': 'import pymongo'
        },
        'motor': {
            'package': 'motor',
            'min_version': '3.7.1',
            'import_names': ['motor'],
            'validation_code': 'import motor'
        },
        'dotenv': {
            'package': 'python-dotenv',
            'min_version': None,
            'import_names': ['dotenv'],
            'validation_code': 'from dotenv import load_dotenv'
        },
        'PIL': {
            'package': 'Pillow',
            'min_version': None,
            'import_names': ['PIL'],
            'validation_code': 'from PIL import Image'
        },
        'pydantic': {
            'package': 'pydantic',
            'min_version': None,
            'import_names': ['pydantic'],
            'validation_code': 'import pydantic'
        }
    }
    
    # Optional dependencies that enable enhanced features
    OPTIONAL_DEPENDENCIES = {
        'httpx': {
            'package': 'httpx',
            'min_version': '0.28.1',
            'import_names': ['httpx'],
            'validation_code': 'import httpx',
            'feature': 'Enhanced HTTP client capabilities'
        },
        'uvloop': {
            'package': 'uvloop',
            'min_version': None,
            'import_names': ['uvloop'],
            'validation_code': 'import uvloop',
            'feature': 'High-performance async event loop (Linux only)',
            'platform_specific': 'linux'
        },
        'orjson': {
            'package': 'orjson',
            'min_version': None,
            'import_names': ['orjson'],
            'validation_code': 'import orjson',
            'feature': 'Fast JSON serialization'
        }
    }
    
    # Required environment variables
    REQUIRED_ENV_VARS = {
        'TELEGRAM_BOT_TOKEN': {
            'description': 'Telegram Bot Token from @BotFather',
            'validation': lambda x: x and len(x) > 10 and ':' in x,
            'error_message': 'TELEGRAM_BOT_TOKEN must be obtained from @BotFather on Telegram'
        },
        'MONGODB_URI': {
            'description': 'MongoDB connection string',
            'validation': lambda x: x and ('mongodb://' in x or 'mongodb+srv://' in x),
            'error_message': 'MONGODB_URI must be a valid MongoDB connection string'
        }
    }
    
    # Railway-specific environment validation for production deployment
    RAILWAY_REQUIRED_ENV_VARS = {
        'RAILWAY_ENVIRONMENT': {
            'description': 'Railway environment identifier',
            'validation': lambda x: x in ['production', 'staging', 'development'],
            'error_message': 'RAILWAY_ENVIRONMENT must be one of: production, staging, development'
        },
        'PORT': {
            'description': 'Railway-assigned dynamic port',
            'validation': lambda x: x and x.isdigit() and 1000 <= int(x) <= 65535,
            'error_message': 'PORT must be a valid port number between 1000-65535 assigned by Railway'
        }
    }
    
    # Railway-compatible environment variables (Railway may set these)
    RAILWAY_OPTIONAL_ENV_VARS = {
        'RAILWAY_SERVICE_NAME': {
            'description': 'Railway service name for identification',
            'validation': lambda x: x and len(x) > 0,
            'degraded_message': 'Service identification may be limited without RAILWAY_SERVICE_NAME'
        },
        'RAILWAY_PROJECT_ID': {
            'description': 'Railway project identifier',
            'validation': lambda x: x and len(x) > 0,
            'degraded_message': 'Project identification may be limited without RAILWAY_PROJECT_ID'
        }
    }
    
    # Optional environment variables with degraded mode handling
    OPTIONAL_ENV_VARS = {
        'HF_TOKEN': {
            'description': 'Hugging Face API Token for AI features (checks HF_TOKEN, HUGGINGFACE_API_KEY, HUGGING_FACE_TOKEN)',
            'validation': lambda x: x and len(x) > 10,
            'degraded_message': 'AI features will be limited without HF_TOKEN. Get one at https://huggingface.co/settings/tokens',
            'multi_check': ['HF_TOKEN', 'HUGGINGFACE_API_KEY', 'HUGGING_FACE_TOKEN']  # Check multiple variants
        },
        'OWNER_ID': {
            'description': 'Telegram user ID for admin features',
            'validation': lambda x: x and x.isdigit() and int(x) > 0,
            'degraded_message': 'Admin features will be unavailable without OWNER_ID'
        }
    }
    
    def __init__(self):
        self.validation_results = {
            'dependencies': {},
            'environment': {},
            'warnings': [],
            'errors': []
        }
    
    def validate_dependency_import(self, name: str, config: Dict) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a single dependency import and version
        
        Returns: (success, message, version)
        """
        try:
            # Test import
            exec(config['validation_code'])
            
            # Get version if possible
            version = None
            try:
                module = import_module(config['import_names'][0])
                if hasattr(module, '__version__'):
                    version = module.__version__
                elif hasattr(module, 'version'):
                    version = module.version
                elif hasattr(module, 'VERSION'):
                    version = module.VERSION
            except:
                pass
            
            # Version checking
            if config.get('min_version') and version:
                try:
                    from packaging import version as version_parser
                    if version_parser.parse(version) < version_parser.parse(config['min_version']):
                        return False, f"Version {version} < required {config['min_version']}", version
                except ImportError:
                    # Fallback to string comparison if packaging not available
                    if version < config['min_version']:
                        return False, f"Version {version} < required {config['min_version']}", version
            
            return True, f"✅ Successfully imported", version
            
        except ImportError as e:
            return False, f"❌ Import failed: {e}", None
        except Exception as e:
            return False, f"❌ Validation failed: {e}", None
    
    def validate_environment_variable(self, var_name: str, config: Dict) -> Tuple[bool, str]:
        """
        Validate a single environment variable (supports multi-check for alternatives)
        
        Returns: (success, message)
        """
        # Check if this config supports multiple environment variable names
        if 'multi_check' in config:
            for check_var in config['multi_check']:
                value = os.getenv(check_var)
                if value and config['validation'](value):
                    return True, f"✅ {var_name} is valid (found as {check_var})"
            return False, f"❌ {var_name} is not set (checked: {', '.join(config['multi_check'])})"
        else:
            # Original single-variable check
            value = os.getenv(var_name)
            
            if not value:
                return False, f"❌ {var_name} is not set"
            
            if not config['validation'](value):
                return False, f"❌ {var_name} is invalid: {config['error_message']}"
            
            return True, f"✅ {var_name} is valid"
    
    def validate_critical_dependencies(self) -> bool:
        """
        Validate all critical dependencies
        
        Returns: True if all critical dependencies are valid
        """
        logger.info("🔍 Validating critical dependencies...")
        
        all_valid = True
        
        for name, config in self.CRITICAL_DEPENDENCIES.items():
            success, message, version = self.validate_dependency_import(name, config)
            
            self.validation_results['dependencies'][name] = {
                'success': success,
                'message': message,
                'version': version,
                'package': config['package'],
                'critical': True
            }
            
            if success:
                version_info = f" (v{version})" if version else ""
                logger.info(f"✅ {name}{version_info}: {message}")
            else:
                logger.error(f"❌ {name}: {message}")
                self.validation_results['errors'].append(
                    f"Critical dependency '{name}' failed: {message}. "
                    f"Install with: pip install {config['package']}"
                )
                all_valid = False
        
        return all_valid
    
    def validate_optional_dependencies(self) -> None:
        """
        Validate optional dependencies and note missing features
        """
        logger.info("🔍 Validating optional dependencies...")
        
        for name, config in self.OPTIONAL_DEPENDENCIES.items():
            # Skip platform-specific dependencies
            if config.get('platform_specific'):
                if config['platform_specific'] == 'linux' and sys.platform != 'linux':
                    logger.info(f"⏭️ Skipping {name} (Linux-only dependency)")
                    continue
            
            success, message, version = self.validate_dependency_import(name, config)
            
            self.validation_results['dependencies'][name] = {
                'success': success,
                'message': message,
                'version': version,
                'package': config['package'],
                'critical': False,
                'feature': config['feature']
            }
            
            if success:
                version_info = f" (v{version})" if version else ""
                logger.info(f"✅ {name}{version_info}: {message}")
            else:
                logger.warning(f"⚠️ {name}: {message}")
                self.validation_results['warnings'].append(
                    f"Optional dependency '{name}' missing: {config['feature']} will be unavailable. "
                    f"Install with: pip install {config['package']}"
                )
    
    def validate_environment_variables(self) -> bool:
        """
        Validate all required environment variables with enhanced Config integration
        Enhanced with comprehensive Config class validation for Railway deployment
        
        Returns: True if all required variables are valid
        """
        logger.info("🔍 Validating environment variables with enhanced Railway support...")
        
        all_valid = True
        
        # ENHANCED: Use Config class comprehensive validation first
        try:
            from bot.config import Config
            
            logger.info("🚀 Running comprehensive Config class validation...")
            
            # Run master validation method that includes all security checks
            try:
                Config.validate_all_environment_variables()
                logger.info("✅ Config comprehensive validation passed")
                
                # Add success to validation results
                self.validation_results['environment']['config_comprehensive_validation'] = {
                    'success': True,
                    'message': "✅ All Config validations passed",
                    'required': True,
                    'enhanced': True
                }
                
            except ValueError as e:
                # Config validation failed - this is critical
                logger.error(f"❌ Config comprehensive validation failed: {e}")
                self.validation_results['errors'].append(f"CRITICAL CONFIG VALIDATION FAILED: {str(e)}")
                self.validation_results['environment']['config_comprehensive_validation'] = {
                    'success': False,
                    'message': f"❌ Config validation failed: {str(e)}",
                    'required': True,
                    'enhanced': True
                }
                all_valid = False
                
            except Exception as e:
                # Unexpected error in config validation
                logger.error(f"❌ Unexpected Config validation error: {e}")
                self.validation_results['errors'].append(f"UNEXPECTED CONFIG ERROR: {str(e)}")
                self.validation_results['environment']['config_comprehensive_validation'] = {
                    'success': False,
                    'message': f"❌ Unexpected error: {str(e)}",
                    'required': True,
                    'enhanced': True
                }
                all_valid = False
            
            # Run production security check
            try:
                Config.prevent_development_defaults_in_production()
                logger.info("✅ Production security validation passed")
                
                self.validation_results['environment']['production_security_validation'] = {
                    'success': True,
                    'message': "✅ Production security checks passed",
                    'required': True,
                    'enhanced': True
                }
                
            except ValueError as e:
                logger.error(f"❌ Production security violation: {e}")
                self.validation_results['errors'].append(f"PRODUCTION SECURITY VIOLATION: {str(e)}")
                self.validation_results['environment']['production_security_validation'] = {
                    'success': False,
                    'message': f"❌ Production security violation: {str(e)}",
                    'required': True,
                    'enhanced': True
                }
                all_valid = False
                
        except ImportError as e:
            logger.error(f"❌ Cannot import Config class: {e}")
            self.validation_results['errors'].append(f"Config import failed: {e}")
            all_valid = False
        
        # Legacy validation for backward compatibility
        logger.info("🔍 Running legacy environment variable validation for completeness...")
        
        # Validate required variables using legacy method
        for var_name, config in self.REQUIRED_ENV_VARS.items():
            success, message = self.validate_environment_variable(var_name, config)
            
            # Use legacy prefix to distinguish from enhanced validation
            legacy_key = f"legacy_{var_name}"
            self.validation_results['environment'][legacy_key] = {
                'success': success,
                'message': message,
                'required': True,
                'legacy': True
            }
            
            if success:
                logger.info(f"✅ Legacy {var_name}: {message}")
            else:
                # Only treat as error if Config validation didn't catch it
                if not any('TELEGRAM_BOT_TOKEN' in error or 'MONGODB_URI' in error for error in self.validation_results['errors']):
                    logger.warning(f"⚠️ Legacy {var_name}: {message}")
                    self.validation_results['warnings'].append(f"Legacy validation failed for {var_name}: {config['error_message']}")
        
        # Validate optional variables and note degraded features
        for var_name, config in self.OPTIONAL_ENV_VARS.items():
            success, message = self.validate_environment_variable(var_name, config)
            
            legacy_key = f"legacy_{var_name}"
            self.validation_results['environment'][legacy_key] = {
                'success': success,
                'message': message,
                'required': False,
                'legacy': True
            }
            
            if success:
                logger.info(f"✅ Legacy {var_name}: {message}")
            else:
                logger.info(f"ℹ️ Legacy {var_name}: {message}")
                # Don't add to warnings if Config validation already handled it
                if not any('HF_TOKEN' in warn for warn in self.validation_results['warnings']):
                    self.validation_results['warnings'].append(f"Legacy: {config['degraded_message']}")
        
        return all_valid
    
    def is_railway_environment(self) -> bool:
        """Check if running in Railway environment"""
        return ('RAILWAY_ENVIRONMENT' in os.environ or 
                'RAILWAY_SERVICE_NAME' in os.environ or
                'RAILWAY_PROJECT_ID' in os.environ)
    
    def validate_railway_environment(self) -> bool:
        """Validate Railway-specific environment variables"""
        if not self.is_railway_environment():
            return True  # Skip Railway validation if not in Railway
        
        logger.info("🚂 Validating Railway environment variables...")
        
        all_valid = True
        
        # Validate Railway required variables
        for var_name, config in self.RAILWAY_REQUIRED_ENV_VARS.items():
            success, message = self.validate_environment_variable(var_name, config)
            
            self.validation_results['environment'][f'railway_{var_name}'] = {
                'success': success,
                'message': message,
                'required': True,
                'railway_specific': True
            }
            
            if success:
                logger.info(f"✅ Railway {var_name}: {message}")
            else:
                logger.error(f"❌ Railway {var_name}: {message}")
                self.validation_results['errors'].append(
                    f"Railway required environment variable '{var_name}' failed: {config['error_message']}"
                )
                all_valid = False
        
        # Validate Railway optional variables
        for var_name, config in self.RAILWAY_OPTIONAL_ENV_VARS.items():
            success, message = self.validate_environment_variable(var_name, config)
            
            self.validation_results['environment'][f'railway_optional_{var_name}'] = {
                'success': success,
                'message': message,
                'required': False,
                'railway_specific': True
            }
            
            if success:
                logger.info(f"✅ Railway {var_name}: {message}")
            else:
                logger.warning(f"⚠️ Railway {var_name}: {message}")
                self.validation_results['warnings'].append(
                    f"Railway optional variable '{var_name}' missing: {config['degraded_message']}"
                )
        
        return all_valid
    
    def validate_railway_deployment_readiness(self) -> bool:
        """
        Comprehensive Railway deployment readiness check
        
        Returns: True if ready for Railway deployment
        """
        logger.info("🚂 Validating Railway deployment readiness...")
        
        deployment_ready = True
        
        # Check critical dependencies
        if not self.validate_critical_dependencies():
            deployment_ready = False
            logger.error("❌ Critical dependencies missing - deployment will fail")
        
        # Check environment variables
        if not self.validate_environment_variables():
            deployment_ready = False
            logger.error("❌ Required environment variables missing - deployment will fail")
        
        # Check Railway-specific environment if in Railway
        railway_valid = self.validate_railway_environment()
        deployment_ready = deployment_ready and railway_valid
        
        # Check production security requirements
        if self.is_railway_environment():
            env_type = os.getenv('RAILWAY_ENVIRONMENT', 'unknown')
            if env_type == 'production':
                encryption_seed = os.getenv('ENCRYPTION_SEED')
                if not encryption_seed or len(encryption_seed) < 32:
                    deployment_ready = False
                    logger.error("❌ ENCRYPTION_SEED must be at least 32 characters for production")
                    self.validation_results['errors'].append(
                        "ENCRYPTION_SEED must be at least 32 characters for production deployment"
                    )
                
                # Check that TEST_MODE is disabled in production
                test_mode = os.getenv('TEST_MODE', 'false').lower()
                if test_mode in ('true', '1', 'yes', 'on'):
                    deployment_ready = False
                    logger.error("❌ TEST_MODE must be disabled in production")
                    self.validation_results['errors'].append(
                        "TEST_MODE must be disabled for production deployment"
                    )
        
        if deployment_ready:
            logger.info("✅ Railway deployment readiness check passed")
        else:
            logger.error("❌ Railway deployment readiness check failed")
        
        return deployment_ready
    
    def print_validation_summary(self) -> None:
        """
        Print a comprehensive validation summary
        """
        logger.info("📊 Dependency Validation Summary")
        logger.info("=" * 50)
        
        # Critical dependencies summary
        critical_deps = {k: v for k, v in self.validation_results['dependencies'].items() if v['critical']}
        critical_success = sum(1 for dep in critical_deps.values() if dep['success'])
        logger.info(f"Critical Dependencies: {critical_success}/{len(critical_deps)} ✅")
        
        for name, info in critical_deps.items():
            status = "✅" if info['success'] else "❌"
            version = f" (v{info['version']})" if info['version'] else ""
            logger.info(f"  {status} {name}{version}")
        
        # Optional dependencies summary
        optional_deps = {k: v for k, v in self.validation_results['dependencies'].items() if not v['critical']}
        optional_success = sum(1 for dep in optional_deps.values() if dep['success'])
        logger.info(f"Optional Dependencies: {optional_success}/{len(optional_deps)} ✅")
        
        for name, info in optional_deps.items():
            status = "✅" if info['success'] else "⚠️"
            version = f" (v{info['version']})" if info['version'] else ""
            logger.info(f"  {status} {name}{version}")
        
        # Environment variables summary
        required_vars = {k: v for k, v in self.validation_results['environment'].items() if v['required']}
        required_success = sum(1 for var in required_vars.values() if var['success'])
        logger.info(f"Required Environment: {required_success}/{len(required_vars)} ✅")
        
        optional_vars = {k: v for k, v in self.validation_results['environment'].items() if not v['required']}
        optional_env_success = sum(1 for var in optional_vars.values() if var['success'])
        logger.info(f"Optional Environment: {optional_env_success}/{len(optional_vars)} ✅")
        
        # Warnings and errors
        if self.validation_results['warnings']:
            logger.warning("⚠️ Warnings:")
            for warning in self.validation_results['warnings']:
                logger.warning(f"  • {warning}")
        
        if self.validation_results['errors']:
            logger.error("❌ Critical Errors:")
            for error in self.validation_results['errors']:
                logger.error(f"  • {error}")
        
        logger.info("=" * 50)
    
    def validate_all(self) -> bool:
        """
        Run complete dependency and environment validation
        
        Returns: True if all critical validations pass, False otherwise
        Raises: DependencyValidationError or EnvironmentValidationError on critical failures
        """
        logger.info("🚀 Starting comprehensive dependency validation...")
        
        try:
            # Validate critical dependencies
            deps_valid = self.validate_critical_dependencies()
            
            # Validate optional dependencies (warnings only)
            self.validate_optional_dependencies()
            
            # Validate environment variables
            env_valid = self.validate_environment_variables()
            
            # Print summary
            self.print_validation_summary()
            
            # Determine overall success
            all_critical_valid = deps_valid and env_valid
            
            if all_critical_valid:
                logger.info("🎉 All critical validations passed! Bot is ready to start.")
                return True
            else:
                # Prepare detailed error message
                error_msg = "Critical validation failures detected:\n"
                for error in self.validation_results['errors']:
                    error_msg += f"  • {error}\n"
                
                error_msg += "\nPlease fix these issues before starting the bot."
                
                logger.error(f"💥 Critical validation failed: {len(self.validation_results['errors'])} errors")
                raise DependencyValidationError(error_msg)
                
        except Exception as e:
            if isinstance(e, (DependencyValidationError, EnvironmentValidationError)):
                raise
            else:
                logger.error(f"💥 Unexpected error during validation: {e}")
                raise DependencyValidationError(f"Validation system error: {e}")

def validate_startup_dependencies() -> bool:
    """
    Convenience function to run startup dependency validation
    
    Returns: True if validation passes
    Raises: DependencyValidationError or EnvironmentValidationError on failures
    """
    validator = DependencyValidator()
    return validator.validate_all()

# For direct script execution
if __name__ == "__main__":
    try:
        validate_startup_dependencies()
        print("✅ All validations passed!")
        sys.exit(0)
    except (DependencyValidationError, EnvironmentValidationError) as e:
        print(f"❌ Validation failed:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)