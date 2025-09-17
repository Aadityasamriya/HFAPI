"""
Storage Provider Factory
Automatically detects and creates the appropriate storage provider based on configuration
"""

import os
import logging
from typing import Optional
from .base import StorageProvider

logger = logging.getLogger(__name__)

class StorageProviderFactory:
    """Factory for creating storage provider instances"""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """
        Register a storage provider class
        
        Args:
            name (str): Provider name (e.g., 'mongodb', 'supabase')
            provider_class: Provider class that implements StorageProvider
        """
        cls._providers[name.lower()] = provider_class
        logger.info(f"✅ Registered storage provider: {name}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available provider names
        
        Returns:
            list: List of registered provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def detect_provider(cls) -> str:
        """
        Auto-detect the best available storage provider based on environment variables
        
        Returns:
            str: Provider name to use
        """
        # Check for explicit provider configuration
        provider = os.getenv('STORAGE_PROVIDER', '').lower()
        if provider and provider in cls._providers:
            logger.info(f"🎯 Using explicitly configured storage provider: {provider}")
            return provider
        
        # Auto-detection based on available environment variables
        detection_rules = [
            ('mongodb', ['MONGO_URI', 'MONGODB_URI']),
            ('supabase', ['SUPABASE_URL', 'SUPABASE_ANON_KEY']),
        ]
        
        for provider_name, env_vars in detection_rules:
            if provider_name in cls._providers:
                # Check if all required environment variables are present
                if all(os.getenv(var) for var in env_vars):
                    logger.info(f"🔍 Auto-detected storage provider: {provider_name}")
                    return provider_name
        
        # Default fallback to MongoDB for backward compatibility
        default_provider = 'mongodb'
        if default_provider in cls._providers:
            logger.info(f"⚠️ No storage provider detected, using default: {default_provider}")
            return default_provider
        
        raise ValueError("No storage provider available. Please configure MONGO_URI or SUPABASE_URL.")
    
    @classmethod
    def create_provider(cls, provider_name: Optional[str] = None) -> StorageProvider:
        """
        Create a storage provider instance
        
        Args:
            provider_name (Optional[str]): Specific provider name, or auto-detect if None
            
        Returns:
            StorageProvider: Configured storage provider instance
            
        Raises:
            ValueError: If provider is not available
        """
        # Auto-detect if not specified
        if provider_name is None:
            provider_name = cls.detect_provider()
        
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            available = ', '.join(cls.get_available_providers())
            raise ValueError(f"Storage provider '{provider_name}' not available. Available: {available}")
        
        provider_class = cls._providers[provider_name]
        logger.info(f"🚀 Creating {provider_name} storage provider instance")
        
        return provider_class()


def create_storage_provider(provider_name: Optional[str] = None) -> StorageProvider:
    """
    Convenience function to create a storage provider
    
    Args:
        provider_name (Optional[str]): Specific provider name, or auto-detect if None
        
    Returns:
        StorageProvider: Configured storage provider instance
    """
    return StorageProviderFactory.create_provider(provider_name)


# Auto-register available providers when module is imported
def _auto_register_providers():
    """Auto-register available providers based on installed dependencies"""
    
    # Try to register MongoDB provider
    try:
        from .mongodb_provider import MongoDBProvider
        StorageProviderFactory.register_provider('mongodb', MongoDBProvider)
    except ImportError as e:
        logger.debug(f"MongoDB provider not available: {e}")
    
    # Try to register Supabase provider
    try:
        from .supabase_provider import SupabaseProvider
        StorageProviderFactory.register_provider('supabase', SupabaseProvider)
    except ImportError as e:
        logger.debug(f"Supabase provider not available: {e}")


# Auto-register providers on module import
_auto_register_providers()