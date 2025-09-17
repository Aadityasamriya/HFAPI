"""
Storage Provider Factory
Simplified to only support MongoDB
"""

import os
import logging
from typing import Optional
from .base import StorageProvider

logger = logging.getLogger(__name__)

class StorageProviderFactory:
    """Simplified factory for MongoDB storage provider only"""
    
    _mongodb_provider = None
    
    @classmethod
    def create_provider(cls, provider_name: Optional[str] = None) -> StorageProvider:
        """
        Create a MongoDB storage provider instance
        
        Args:
            provider_name (Optional[str]): Ignored, only MongoDB is supported
            
        Returns:
            StorageProvider: MongoDB storage provider instance
            
        Raises:
            ImportError: If MongoDB dependencies are not available
            ValueError: If MongoDB configuration is missing
        """
        try:
            from .mongodb_provider import MongoDBProvider
            logger.info("🚀 Creating MongoDB storage provider instance")
            return MongoDBProvider()
        except ImportError as e:
            raise ImportError(
                f"MongoDB dependencies not available: {e}\n"
                "Install with: pip install motor pymongo"
            )


def create_storage_provider(provider_name: Optional[str] = None) -> StorageProvider:
    """
    Convenience function to create a MongoDB storage provider
    
    Args:
        provider_name (Optional[str]): Ignored, only MongoDB is supported
        
    Returns:
        StorageProvider: MongoDB storage provider instance
    """
    return StorageProviderFactory.create_provider(provider_name)