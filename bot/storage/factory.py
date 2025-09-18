"""
Storage Provider Factory
Supports MongoDB and Supabase User Provider with automatic detection
"""

import os
import logging
from typing import Optional
from .base import StorageProvider

logger = logging.getLogger(__name__)

class StorageProviderFactory:
    """Enhanced factory supporting multiple storage providers"""
    
    _providers = {}
    
    @classmethod
    def create_provider(cls, provider_name: Optional[str] = None) -> StorageProvider:
        """
        Create storage provider instance with automatic detection
        
        Args:
            provider_name (Optional[str]): Specific provider ('mongodb', 'supabase') or auto-detect if None
            
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
        
        # Create provider based on name
        if provider_name == 'supabase':
            return cls._create_supabase_provider()
        elif provider_name == 'mongodb':
            return cls._create_mongodb_provider()
        else:
            raise ValueError(f"Unknown storage provider: {provider_name}")
    
    @classmethod
    def _detect_provider(cls) -> str:
        """
        Auto-detect which storage provider to use based on environment variables
        
        Returns:
            str: Provider name ('supabase' or 'mongodb')
        """
        # Check for Supabase configuration
        if os.getenv('SUPABASE_MGMT_URL') or os.getenv('DATABASE_URL'):
            logger.info("🔍 Auto-detected Supabase configuration - using SupabaseUserProvider")
            return 'supabase'
        
        # Check for MongoDB configuration  
        elif os.getenv('MONGODB_URI') or os.getenv('MONGO_URI'):
            logger.info("🔍 Auto-detected MongoDB configuration - using MongoDBProvider")
            return 'mongodb'
        
        # Default to Supabase if no specific config found (requires manual setup)
        else:
            logger.warning("⚠️ No database configuration detected - defaulting to SupabaseUserProvider")
            logger.warning("📝 Please configure SUPABASE_MGMT_URL or MONGODB_URI environment variables")
            return 'supabase'
    
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
                'configured': bool(os.getenv('MONGODB_URI') or os.getenv('MONGO_URI')),
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
                'configured': bool(os.getenv('SUPABASE_MGMT_URL') or os.getenv('DATABASE_URL')),
                'description': 'Supabase storage provider with individual user database isolation'
            }
        except ImportError:
            providers['supabase'] = {
                'available': False,
                'configured': False,
                'description': 'Supabase storage provider - dependencies missing'
            }
        
        return providers


def create_storage_provider(provider_name: Optional[str] = None) -> StorageProvider:
    """
    Convenience function to create storage provider with automatic detection
    
    Args:
        provider_name (Optional[str]): Specific provider ('mongodb', 'supabase') or auto-detect if None
        
    Returns:
        StorageProvider: Storage provider instance
        
    Examples:
        # Auto-detect based on environment variables
        provider = create_storage_provider()
        
        # Force specific provider
        provider = create_storage_provider('supabase')
        provider = create_storage_provider('mongodb')
    """
    return StorageProviderFactory.create_provider(provider_name)


def get_provider_info() -> dict:
    """
    Get information about available storage providers
    
    Returns:
        dict: Provider availability and configuration status
    """
    return StorageProviderFactory.list_available_providers()