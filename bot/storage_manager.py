"""
Global Storage Manager for Hugging Face By AadityaLabs AI
Provides singleton access to the storage provider across all bot modules
"""

import asyncio
import logging
import threading
from typing import Optional
from .storage import create_storage_provider, StorageProvider

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
    """Global storage manager providing singleton access to one storage provider."""

    _instance: Optional['StorageManager'] = None
    _storage: Optional[StorageProvider] = None
    _initialized: bool = False
    _connection_lock = asyncio.Lock()
    _creation_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def storage(self) -> Optional[StorageProvider]:
        return self._storage

    @property
    def connected(self) -> bool:
        return self._storage is not None and self._storage.connected

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self, provider_name: Optional[str] = None) -> StorageProvider:
        """
        Initialize the configured storage provider.

        provider_name is intentionally forwarded to the factory. Previously this
        argument was accepted but silently ignored, making init_storage('mongodb')
        and other explicit-provider calls ineffective.
        """
        async with self._connection_lock:
            if self._storage and self._storage.connected:
                if provider_name is None:
                    logger.info("✅ Storage already initialized and connected")
                    return self._storage
                current_name = type(self._storage).__name__.lower()
                requested_name = provider_name.lower()
                if requested_name in current_name:
                    logger.info("✅ Requested storage provider is already initialized")
                    return self._storage
                logger.warning(
                    "⚠️ A different storage provider was requested (%s); reconnecting.",
                    requested_name,
                )
                await self._storage.disconnect()
                self._storage = None
                self._initialized = False

            try:
                logger.info("🚀 Initializing global storage manager...")

                from bot.config import Config
                from bot.crypto_utils import initialize_crypto

                try:
                    Config.ensure_encryption_seed()
                    encryption_seed = Config.ENCRYPTION_SEED
                    if not encryption_seed:
                        raise ValueError("ENCRYPTION_SEED is None despite validation")
                    initialize_crypto(encryption_seed)
                    logger.info("🔒 Global crypto system initialized successfully")
                except ValueError as e:
                    logger.error("❌ ENCRYPTION_SEED validation failed: %s", e)
                    raise RuntimeError(f"ENCRYPTION_SEED validation failed: {e}") from e

                hf_token = Config.get_hf_token()
                if hf_token:
                    logger.info("✅ HF token found and ready for AI functionality")
                else:
                    logger.warning("⚠️ HF token not found - AI features may be limited")

                # IMPORTANT: Forward the caller's explicit provider selection.
                self._storage = create_storage_provider(provider_name)
                logger.info("📦 Created storage provider: %s", type(self._storage).__name__)

                await self._storage.connect()
                logger.info("🔗 Connected to storage backend")
                await self._storage.initialize()
                logger.info("⚙️ Storage backend initialized")

                self._initialized = True
                logger.info("✅ Global storage manager initialization completed successfully")
                return self._storage

            except Exception as e:
                logger.error("❌ CRITICAL: Storage manager initialization failed: %s", e)
                self._storage = None
                self._initialized = False
                raise RuntimeError(f"Storage initialization failed: {e}") from e

    async def disconnect(self) -> None:
        async with self._connection_lock:
            if self._storage:
                try:
                    await self._storage.disconnect()
                    logger.info("✅ Storage backend disconnected successfully")
                except Exception as e:
                    logger.error("❌ Error disconnecting storage: %s", e)
                finally:
                    self._storage = None
                    self._initialized = False

    async def health_check(self) -> bool:
        try:
            if not self._storage:
                return False
            return await self._storage.health_check()
        except Exception as e:
            logger.warning("Storage health check failed: %s", e)
            return False

    async def ensure_connected(self) -> StorageProvider:
        if not self._storage or not self._storage.connected:
            await self.initialize()
        if not self._storage:
            raise RuntimeError("Storage provider initialization failed")
        return self._storage

    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        storage = await self.ensure_connected()
        return await storage.save_user_api_key(user_id, api_key)

    async def store_api_key(self, user_id: int, api_key: str) -> bool:
        return await self.save_user_api_key(user_id, api_key)

    async def save_api_key(self, user_id: int, api_key: str) -> bool:
        return await self.save_user_api_key(user_id, api_key)

    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        storage = await self.ensure_connected()
        return await storage.get_user_api_key(user_id)

    async def get_api_key(self, user_id: int) -> Optional[str]:
        return await self.get_user_api_key(user_id)

    async def set_user_hf_api_key(self, user_id: int, api_key: str) -> bool:
        return await self.save_user_api_key(user_id, api_key)

    async def get_user_hf_api_key(self, user_id: int) -> Optional[str]:
        return await self.get_user_api_key(user_id)


storage_manager = StorageManager()


async def get_storage() -> StorageProvider:
    await storage_manager.ensure_connected()
    if storage_manager.storage is None:
        raise RuntimeError("Storage provider is not initialized despite ensure_connected() call")
    return storage_manager.storage


async def init_storage(provider_name: Optional[str] = None) -> StorageProvider:
    return await storage_manager.initialize(provider_name)


async def close_storage() -> None:
    await storage_manager.disconnect()


class LegacyDatabaseWrapper:
    @property
    def connected(self) -> bool:
        return storage_manager.connected

    async def connect(self) -> None:
        await storage_manager.ensure_connected()

    async def disconnect(self) -> None:
        await storage_manager.disconnect()

    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        storage = await get_storage()
        return await storage.save_user_api_key(user_id, api_key)

    async def store_api_key(self, user_id: int, api_key: str) -> bool:
        return await self.save_user_api_key(user_id, api_key)

    async def save_api_key(self, user_id: int, api_key: str) -> bool:
        return await self.save_user_api_key(user_id, api_key)

    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        storage = await get_storage()
        return await storage.get_user_api_key(user_id)

    async def get_api_key(self, user_id: int) -> Optional[str]:
        return await self.get_user_api_key(user_id)

    async def reset_user_database(self, user_id: int) -> bool:
        storage = await get_storage()
        return await storage.reset_user_database(user_id)

    async def save_conversation(self, user_id: int, conversation_data: dict) -> bool:
        storage = await get_storage()
        return await storage.save_conversation(user_id, conversation_data)

    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> list:
        storage = await get_storage()
        return await storage.get_user_conversations(user_id, limit, skip)

    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[dict]:
        storage = await get_storage()
        return await storage.get_conversation_details(user_id, conversation_id)

    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        storage = await get_storage()
        return await storage.delete_conversation(user_id, conversation_id)

    async def clear_user_history(self, user_id: int) -> bool:
        storage = await get_storage()
        return await storage.clear_user_history(user_id)

    async def get_conversation_count(self, user_id: int) -> int:
        storage = await get_storage()
        return await storage.get_conversation_count(user_id)


db = LegacyDatabaseWrapper()
