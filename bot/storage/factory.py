"""
Storage Provider Factory
Supports MongoDB and Supabase User Provider with automatic detection
"""

import os
import logging
from typing import Optional
from .base import StorageProvider
from bot.config import Config

logger = logging.getLogger(__name__)

class StorageProviderFactory:
    """Enhanced factory supporting multiple storage providers"""
    
    _providers = {}
    
    @classmethod
    def create_provider(cls, provider_name: Optional[str] = None) -> StorageProvider:
        """
        Create storage provider instance with automatic detection
        CRITICAL FIX: Added early validation for all provider creation
        
        Args:
            provider_name (Optional[str]): Specific provider ('mongodb', 'supabase', 'hybrid') or auto-detect if None
            
        Returns:
            StorageProvider: Storage provider instance
            
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If configuration is missing
        """
        # Auto-detect provider if not specified
        if not provider_name:
            provider_name = cls._detect_provider()
        
        provider_name = provider_name.lower()
        
        # CRITICAL FIX: For any provider creation, ensure early validation
        # This prevents misleading "validated" log messages before actual validation
        if provider_name in ['hybrid', 'supabase', 'mongodb']:
            from bot.config import Config
            logger.info(f"🔍 Performing early configuration validation for {provider_name} provider...")
            
            # For hybrid provider, use strict validation
            if provider_name == 'hybrid':
                try:
                    Config.validate_hybrid_config_early()
                    logger.info("✅ Early hybrid validation passed")
                except ValueError as e:
                    logger.error(f"❌ Early hybrid validation failed: {e}")
                    raise
        
        # Create provider based on name
        if provider_name == 'supabase':
            return cls._create_supabase_provider()
        elif provider_name == 'mongodb':
            return cls._create_mongodb_provider()
        elif provider_name == 'hybrid':
            return cls._create_hybrid_provider()
        elif provider_name == 'resilient_hybrid':
            return cls._create_resilient_hybrid_provider()
        else:
            raise ValueError(f"Unknown storage provider: {provider_name}")
    
    @classmethod
    def _detect_provider(cls) -> str:
        """
        Auto-detect which storage provider to use based on environment variables
        ENHANCED: Now supports graceful fallback to MongoDB-only mode when Supabase fails
        
        Returns:
            str: Provider name ('hybrid', 'mongodb', or 'resilient_hybrid')
            
        Raises:
            ValueError: If MongoDB configuration is missing (MongoDB is required for core functionality)
        """
        # Import Config for strict validation
        from bot.config import Config
        
        # MongoDB is REQUIRED for core functionality (API keys, admin data)
        has_mongodb = Config.has_strict_mongodb_config()
        if not has_mongodb:
            error_msg = (
                "CRITICAL STORAGE CONFIGURATION ERROR: MongoDB configuration is missing.\n"
                "MongoDB is required for core bot functionality (API keys, admin data, telegram IDs).\n"
                "Please set MONGODB_URI or MONGO_URI environment variable."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check Supabase availability for user data storage (Railway.com compatible)
        has_supabase = Config.has_strict_supabase_config()
        
        # RAILWAY OPTIMIZATION: Log Railway detection for better debugging
        if Config._is_railway_environment():
            railway_env = os.getenv('RAILWAY_ENVIRONMENT', 'unknown')
            logger.info(f"🚂 Railway.com deployment detected (environment: {railway_env})")
            
            # On Railway, check both SUPABASE_MGMT_URL and DATABASE_URL
            # Use Config methods for Railway compatibility
            has_supabase_config = Config.has_supabase_config()
            
            if has_supabase_config:
                logger.info("   Using Supabase configuration for PostgreSQL connection")
            else:
                logger.warning("   No PostgreSQL database detected on Railway")
        
        if has_mongodb and has_supabase:
            logger.info("🔍 Auto-detected hybrid configuration - using ResilientHybridProvider (with fallback capability)")
            return 'resilient_hybrid'
        elif has_mongodb and not has_supabase:
            if Config._is_railway_environment():
                logger.warning("⚠️ Railway deployment with MongoDB-only configuration detected")
                logger.warning("   Consider adding a PostgreSQL database service on Railway for enhanced user data storage")
                logger.warning("   The bot will operate in fallback mode with limited user data features")
            else:
                logger.warning("⚠️ MongoDB-only configuration detected - Supabase unavailable")
                logger.warning("   User data storage will be limited. Consider adding SUPABASE_MGMT_URL for full functionality.")
            return 'mongodb'
        else:
            # This shouldn't happen given the MongoDB check above, but included for completeness
            platform_info = "Railway" if Config._is_railway_environment() else "local"
            error_msg = (
                "UNKNOWN STORAGE CONFIGURATION ERROR: Unexpected configuration state.\n"
                "Please verify MONGODB_URI and optionally SUPABASE_MGMT_URL environment variables."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @classmethod
    def _create_supabase_provider(cls) -> StorageProvider:
        """
        Create SupabaseUserProvider instance
        
        Returns:
            StorageProvider: SupabaseUserProvider instance
            
        Raises:
            ImportError: If Supabase dependencies are not available
        """
        try:
            from .supabase_user_provider import SupabaseUserProvider
            logger.info("🚀 Creating SupabaseUserProvider instance")
            return SupabaseUserProvider()
        except ImportError as e:
            raise ImportError(
                f"Supabase dependencies not available: {e}\n"
                "Install with: pip install asyncpg sqlalchemy[asyncio] psycopg2-binary"
            )
    
    @classmethod
    def _create_mongodb_provider(cls) -> StorageProvider:
        """
        Create MongoDBProvider instance
        
        Returns:
            StorageProvider: MongoDBProvider instance
            
        Raises:
            ImportError: If MongoDB dependencies are not available
        """
        try:
            from .mongodb_provider import MongoDBProvider
            logger.info("🚀 Creating MongoDBProvider instance")
            return MongoDBProvider()
        except ImportError as e:
            raise ImportError(
                f"MongoDB dependencies not available: {e}\n"
                "Install with: pip install motor pymongo"
            )
    
    @classmethod
    def _create_hybrid_provider(cls) -> StorageProvider:
        """
        Create HybridProvider instance
        CRITICAL FIX: Added early validation before provider creation
        
        Returns:
            StorageProvider: HybridProvider instance
            
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If required configurations are missing
        """
        # CRITICAL FIX: Perform early validation before creating provider
        from bot.config import Config
        
        logger.info("🔍 Performing early validation before HybridProvider creation...")
        try:
            Config.validate_hybrid_config_early()
            logger.info("✅ Early validation passed - proceeding with HybridProvider creation")
        except ValueError as e:
            logger.error(f"❌ Early validation failed - aborting HybridProvider creation: {e}")
            raise
        
        try:
            from .hybrid_provider import HybridProvider
            logger.info("🚀 Creating HybridProvider instance (validation confirmed)")
            return HybridProvider()
        except ImportError as e:
            raise ImportError(
                f"Hybrid provider dependencies not available: {e}\n"
                "Install with: pip install motor pymongo asyncpg sqlalchemy[asyncio] psycopg2-binary"
            )
    
    @classmethod
    def _create_resilient_hybrid_provider(cls) -> StorageProvider:
        """
        Create ResilientHybridProvider instance with graceful fallback capability
        
        Returns:
            StorageProvider: ResilientHybridProvider instance
            
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If MongoDB configuration is missing (Supabase is optional)
        """
        # Validate MongoDB is available (required)
        from bot.config import Config
        
        logger.info("🔍 Creating ResilientHybridProvider with fallback capabilities...")
        
        if not Config.has_strict_mongodb_config():
            raise ValueError(
                "ResilientHybridProvider requires MongoDB configuration. "
                "Please set MONGODB_URI or MONGO_URI environment variable."
            )
        
        # Supabase is optional for resilient provider
        has_supabase = Config.has_strict_supabase_config()
        if has_supabase:
            logger.info("✅ Both MongoDB and Supabase configured - full hybrid mode available")
        else:
            logger.warning("⚠️ Only MongoDB configured - will operate in fallback mode")
        
        try:
            from .resilient_hybrid_provider import ResilientHybridProvider
            logger.info("🚀 Creating ResilientHybridProvider instance")
            return ResilientHybridProvider()
        except ImportError as e:
            raise ImportError(
                f"ResilientHybridProvider dependencies not available: {e}\n"
                "Install with: pip install motor pymongo asyncpg sqlalchemy[asyncio] psycopg2-binary"
            )
    
    @classmethod
    def list_available_providers(cls) -> dict:
        """
        Get list of available providers and their status
        
        Returns:
            dict: Provider availability status
        """
        providers = {}
        
        # Check MongoDB availability
        try:
            from .mongodb_provider import MongoDBProvider
            providers['mongodb'] = {
                'available': True,
                'configured': Config.has_mongodb_config(),
                'description': 'MongoDB storage provider with Motor async driver'
            }
        except ImportError:
            providers['mongodb'] = {
                'available': False,
                'configured': False,
                'description': 'MongoDB storage provider - dependencies missing'
            }
        
        # Check Supabase availability
        try:
            from .supabase_user_provider import SupabaseUserProvider
            providers['supabase'] = {
                'available': True,
                'configured': Config.has_supabase_config(),
                'description': 'Supabase storage provider with individual user database isolation'
            }
        except ImportError:
            providers['supabase'] = {
                'available': False,
                'configured': False,
                'description': 'Supabase storage provider - dependencies missing'
            }
        
        # Check Hybrid availability
        mongodb_available = providers.get('mongodb', {}).get('available', False)
        supabase_available = providers.get('supabase', {}).get('available', False)
        mongodb_configured = providers.get('mongodb', {}).get('configured', False)
        supabase_configured = providers.get('supabase', {}).get('configured', False)
        
        try:
            from .hybrid_provider import HybridProvider
            providers['hybrid'] = {
                'available': mongodb_available and supabase_available,
                'configured': mongodb_configured and supabase_configured,
                'description': 'Hybrid storage provider routing between MongoDB and Supabase'
            }
        except ImportError:
            providers['hybrid'] = {
                'available': False,
                'configured': False,
                'description': 'Hybrid storage provider - dependencies missing'
            }
        
        return providers


def create_storage_provider(provider_name: Optional[str] = None) -> StorageProvider:
    """
    Convenience function to create storage provider with automatic detection
    
    Args:
        provider_name (Optional[str]): Specific provider ('mongodb', 'supabase', 'hybrid') or auto-detect if None
        
    Returns:
        StorageProvider: Storage provider instance
        
    Examples:
        # Auto-detect based on environment variables
        provider = create_storage_provider()
        
        # Force specific provider
        provider = create_storage_provider('supabase')
        provider = create_storage_provider('mongodb')
        provider = create_storage_provider('hybrid')
    """
    return StorageProviderFactory.create_provider(provider_name)


def get_provider_info() -> dict:
    """
    Get information about available storage providers
    
    Returns:
        dict: Provider availability and configuration status
    """
    return StorageProviderFactory.list_available_providers()