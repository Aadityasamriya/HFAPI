"""
Global Storage Manager for Hugging Face By AadityaLabs AI
Provides singleton access to the storage provider across all bot modules
"""

import asyncio
import logging
import threading
from typing import Optional
from .storage import create_storage_provider, StorageProvider

# Export list for proper imports - CRITICAL for backward compatibility
__all__ = [
    'StorageManager', 
    'storage_manager', 
    'get_storage', 
    'init_storage', 
    'close_storage', 
    'LegacyDatabaseWrapper', 
    'db'
]

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Global storage manager providing singleton access to the storage provider
    
    This ensures all bot modules use the same storage instance and handles
    automatic initialization and connection management.
    """
    
    _instance: Optional['StorageManager'] = None
    _storage: Optional[StorageProvider] = None
    _initialized: bool = False
    _connection_lock = asyncio.Lock()
    
    # CRITICAL FIX: Proper thread-safe singleton with double-checked locking
    _creation_lock = threading.Lock()
    
    def __new__(cls):
        # CRITICAL FIX: Thread-safe singleton with double-checked locking pattern
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def storage(self) -> Optional[StorageProvider]:
        """Get the storage provider instance"""
        return self._storage
    
    @property
    def connected(self) -> bool:
        """Check if storage is connected"""
        return self._storage is not None and self._storage.connected
    
    @property
    def initialized(self) -> bool:
        """Check if storage manager is initialized"""
        return self._initialized
    
    async def initialize(self, provider_name: Optional[str] = None) -> StorageProvider:
        """
        Initialize the storage provider with auto-detection and connection
        
        Args:
            provider_name (Optional[str]): Specific provider to use, or auto-detect if None
            
        Returns:
            StorageProvider: Connected and initialized storage provider
            
        Raises:
            RuntimeError: If initialization fails
        """
        async with self._connection_lock:
            if self._storage and self._storage.connected:
                logger.info("✅ Storage already initialized and connected")
                return self._storage
            
            try:
                logger.info("🚀 Initializing global storage manager...")
                
                # CRITICAL: Ensure ENCRYPTION_SEED is available before storage initialization
                from bot.config import Config
                try:
                    Config.ensure_encryption_seed()
                    logger.info("🔐 ENCRYPTION_SEED verified before storage initialization")
                except ValueError as e:
                    logger.error(f"❌ ENCRYPTION_SEED validation failed: {e}")
                    raise RuntimeError(f"ENCRYPTION_SEED validation failed: {e}")
                
                # CRITICAL: Verify HF token availability for AI functionality
                hf_token = Config.get_hf_token()
                if hf_token:
                    # Safe logging - only indicates presence, never the actual token
                    logger.info("✅ HF token found and ready for AI functionality")
                    logger.debug(f"   Token length: {len(hf_token)} characters")
                else:
                    logger.warning("⚠️ HF token not found - AI features may be limited")
                    logger.warning("   Set one of: HF_TOKEN, HUGGINGFACE_API_KEY, or HUGGING_FACE_TOKEN")
                
                # Create MongoDB storage provider instance
                self._storage = create_storage_provider()
                logger.info(f"📦 Created MongoDB storage provider: {type(self._storage).__name__}")
                
                # Connect to storage backend
                await self._storage.connect()
                logger.info("🔗 Connected to storage backend")
                
                # Initialize storage backend (create tables, indexes, etc.)
                await self._storage.initialize()
                logger.info("⚙️ Storage backend initialized")
                
                self._initialized = True
                logger.info("✅ Global storage manager initialization completed successfully")
                
                return self._storage
                
            except Exception as e:
                logger.error(f"❌ CRITICAL: Storage manager initialization failed: {e}")
                self._storage = None
                self._initialized = False
                raise RuntimeError(f"Storage initialization failed: {e}")
    
    async def disconnect(self) -> None:
        """
        Disconnect from storage backend gracefully
        """
        async with self._connection_lock:
            if self._storage:
                try:
                    await self._storage.disconnect()
                    logger.info("✅ Storage backend disconnected successfully")
                except Exception as e:
                    logger.error(f"❌ Error disconnecting storage: {e}")
                finally:
                    self._storage = None
                    self._initialized = False
    
    async def health_check(self) -> bool:
        """
        Perform health check on the storage backend
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self._storage:
                return False
            
            return await self._storage.health_check()
            
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")
            return False
    
    async def ensure_connected(self) -> StorageProvider:
        """
        Ensure storage is connected, initializing if needed
        
        Returns:
            StorageProvider: Connected storage provider
            
        Raises:
            RuntimeError: If connection cannot be established
        """
        if not self._storage or not self._storage.connected:
            logger.info("🔄 Storage not connected, initializing...")
            await self.initialize()
        
        if not self._storage:
            raise RuntimeError("Storage provider initialization failed")
        
        return self._storage
    
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """Save user's API key"""
        storage = await self.ensure_connected()
        return await storage.save_user_api_key(user_id, api_key)
    
    async def store_api_key(self, user_id: int, api_key: str) -> bool:
        """Save user's API key (alias for backward compatibility)"""
        return await self.save_user_api_key(user_id, api_key)
    
    async def save_api_key(self, user_id: int, api_key: str) -> bool:
        """Save user's API key (alias for backward compatibility)"""
        return await self.save_user_api_key(user_id, api_key)
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """Get user's API key"""
        storage = await self.ensure_connected()
        return await storage.get_user_api_key(user_id)
    
    async def get_api_key(self, user_id: int) -> Optional[str]:
        """Get user's API key (alias for backward compatibility)"""
        return await self.get_user_api_key(user_id)
    
    async def set_user_hf_api_key(self, user_id: int, api_key: str) -> bool:
        """Set user's Hugging Face API key (alias for test compatibility)"""
        return await self.save_user_api_key(user_id, api_key)
    
    async def get_user_hf_api_key(self, user_id: int) -> Optional[str]:
        """Get user's Hugging Face API key (alias for test compatibility)"""
        return await self.get_user_api_key(user_id)


# Global storage manager instance
storage_manager = StorageManager()


# Backward compatibility functions for easy migration
async def get_storage() -> StorageProvider:
    """
    Get the global storage provider instance
    
    Returns:
        StorageProvider: Connected storage provider
        
    Raises:
        RuntimeError: If storage is not initialized
    """
    await storage_manager.ensure_connected()
    
    if storage_manager.storage is None:
        raise RuntimeError("Storage provider is not initialized despite ensure_connected() call")
    
    return storage_manager.storage


async def init_storage(provider_name: Optional[str] = None) -> StorageProvider:
    """
    Initialize the global storage provider
    
    Args:
        provider_name (Optional[str]): Specific provider to use, or auto-detect if None
        
    Returns:
        StorageProvider: Connected and initialized storage provider
    """
    return await storage_manager.initialize(provider_name)


async def close_storage() -> None:
    """
    Close the global storage provider connection
    """
    await storage_manager.disconnect()


# Legacy compatibility - create a db-like object for existing code
class LegacyDatabaseWrapper:
    """
    Legacy wrapper that mimics the old Database interface for backward compatibility
    
    This allows existing code using `db.method()` to work without changes
    while transparently using the new StorageProvider system underneath.
    """
    
    @property
    def connected(self) -> bool:
        """Check if storage is connected"""
        return storage_manager.connected
    
    async def connect(self) -> None:
        """Connect to storage backend"""
        await storage_manager.ensure_connected()
    
    async def disconnect(self) -> None:
        """Disconnect from storage backend"""
        await storage_manager.disconnect()
    
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """Save user's API key"""
        storage = await get_storage()
        return await storage.save_user_api_key(user_id, api_key)
    
    async def store_api_key(self, user_id: int, api_key: str) -> bool:
        """Save user's API key (alias for backward compatibility)"""
        return await self.save_user_api_key(user_id, api_key)
    
    async def save_api_key(self, user_id: int, api_key: str) -> bool:
        """Save user's API key (alias for backward compatibility)"""
        return await self.save_user_api_key(user_id, api_key)
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """Get user's API key"""
        storage = await get_storage()
        return await storage.get_user_api_key(user_id)
    
    async def get_api_key(self, user_id: int) -> Optional[str]:
        """Get user's API key (alias for backward compatibility)"""
        return await self.get_user_api_key(user_id)
    
    async def reset_user_database(self, user_id: int) -> bool:
        """Reset user's data"""
        storage = await get_storage()
        return await storage.reset_user_database(user_id)
    
    async def save_conversation(self, user_id: int, conversation_data: dict) -> bool:
        """Save conversation"""
        storage = await get_storage()
        return await storage.save_conversation(user_id, conversation_data)
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> list:
        """Get user conversations"""
        storage = await get_storage()
        return await storage.get_user_conversations(user_id, limit, skip)
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[dict]:
        """Get conversation details"""
        storage = await get_storage()
        return await storage.get_conversation_details(user_id, conversation_id)
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """Delete conversation"""
        storage = await get_storage()
        return await storage.delete_conversation(user_id, conversation_id)
    
    async def clear_user_history(self, user_id: int) -> bool:
        """Clear user history"""
        storage = await get_storage()
        return await storage.clear_user_history(user_id)
    
    async def get_conversation_count(self, user_id: int) -> int:
        """Get conversation count"""
        storage = await get_storage()
        return await storage.get_conversation_count(user_id)


# Create global db instance for backward compatibility
db = LegacyDatabaseWrapper()