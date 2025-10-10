"""
Hybrid Storage Provider Implementation
Routes operations appropriately between MongoDB and Supabase based on data type:
- MongoDB: API keys, telegram IDs, developer database, admin data, usage logs
- Supabase: User data storage (conversations, preferences, files, user-specific data)
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import logging
from .base import StorageProvider
from .mongodb_provider import MongoDBProvider
from .supabase_user_provider import SupabaseUserProvider
from bot.config import Config
from bot.security_utils import SecureLogger

logger = logging.getLogger(__name__)

class HybridProvider(StorageProvider):
    """
    Hybrid storage provider that routes operations between MongoDB and Supabase
    
    Routing Strategy:
    - MongoDB: API keys, telegram IDs, developer's database, admin data, usage logs, main operational data
    - Supabase: User data storage (conversations, preferences, files, user-specific data)
    
    This provider maintains both database connections and routes operations transparently
    while ensuring data consistency and proper error handling.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize both providers
        self.mongodb_provider = None
        self.supabase_provider = None
        
        # Connection status tracking
        self.mongodb_connected = False
        self.supabase_connected = False
        
        # Logging
        self.secure_logger = SecureLogger(logger)
        
        # Validate configuration
        self._validate_hybrid_config()
    
    def _validate_hybrid_config(self) -> None:
        """
        Validate that both MongoDB and Supabase configurations are available
        CRITICAL FIX: Uses strict validation that matches actual provider requirements
        
        Raises:
            ValueError: If required configurations are missing
        """
        # CRITICAL FIX: Use strict validation instead of fallback-accepting methods
        # This ensures validation matches what providers actually require
        try:
            Config.validate_hybrid_config_early()
            logger.info("✅ Hybrid storage configuration validated - both MongoDB and Supabase properly configured")
        except ValueError as e:
            # Re-raise with hybrid provider context
            logger.error(f"❌ HybridProvider configuration validation failed: {e}")
            raise ValueError(f"HybridProvider initialization failed: {e}")
        
        # Additional validation: Ensure we can access the specific environment variables
        mongodb_uri = Config.get_mongodb_uri()
        supabase_url = Config.get_supabase_mgmt_url()
        
        if not mongodb_uri:
            raise ValueError(
                "HybridProvider validation failed: MongoDB URI not accessible despite validation. "
                "Please verify MONGODB_URI or MONGO_URI environment variable is properly set."
            )
        
        if not supabase_url:
            raise ValueError(
                "HybridProvider validation failed: Supabase management URL not accessible despite validation. "
                "Please verify SUPABASE_MGMT_URL environment variable is properly set "
                "(DATABASE_URL fallback is not sufficient for actual provider requirements)."
            )
    
    # Connection Management
    async def connect(self) -> None:
        """
        Establish connections to both MongoDB and Supabase providers
        
        Raises:
            ConnectionError: If either database connection fails
            ValueError: If configuration is invalid
        """
        try:
            logger.info("🔗 Establishing hybrid storage connections...")
            
            # Initialize providers if not already done
            if self.mongodb_provider is None:
                try:
                    self.mongodb_provider = MongoDBProvider()
                    logger.info("📦 MongoDB provider initialized")
                except Exception as e:
                    raise ConnectionError(f"Failed to initialize MongoDB provider: {e}")
            
            if self.supabase_provider is None:
                try:
                    self.supabase_provider = SupabaseUserProvider()
                    logger.info("📦 Supabase provider initialized")
                except Exception as e:
                    raise ConnectionError(f"Failed to initialize Supabase provider: {e}")
            
            # Connect to both databases in parallel for faster startup
            connection_tasks = []
            
            # MongoDB connection
            async def connect_mongodb():
                try:
                    if self.mongodb_provider is not None:
                        await self.mongodb_provider.connect()
                        self.mongodb_connected = True
                        logger.info("✅ MongoDB connection established")
                except Exception as e:
                    logger.error(f"❌ MongoDB connection failed: {e}")
                    raise ConnectionError(f"MongoDB connection failed: {e}")
            
            # Supabase connection
            async def connect_supabase():
                try:
                    if self.supabase_provider is not None:
                        await self.supabase_provider.connect()
                        self.supabase_connected = True
                        logger.info("✅ Supabase connection established")
                except Exception as e:
                    logger.error(f"❌ Supabase connection failed: {e}")
                    raise ConnectionError(f"Supabase connection failed: {e}")
            
            connection_tasks.append(connect_mongodb())
            connection_tasks.append(connect_supabase())
            
            # Execute connections concurrently
            await asyncio.gather(*connection_tasks)
            
            self.connected = True
            logger.info("🎉 Hybrid storage provider connected successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect hybrid storage provider: {e}")
            self.connected = False
            # Clean up partial connections
            await self._cleanup_connections()
            raise
    
    async def disconnect(self) -> None:
        """
        Close connections to both MongoDB and Supabase providers gracefully
        """
        try:
            logger.info("🔌 Disconnecting hybrid storage provider...")
            
            # Disconnect from both databases in parallel for faster shutdown
            disconnect_tasks = []
            
            if self.mongodb_provider and self.mongodb_connected:
                async def disconnect_mongodb():
                    try:
                        if self.mongodb_provider is not None:
                            await self.mongodb_provider.disconnect()
                            self.mongodb_connected = False
                            logger.info("✅ MongoDB disconnected")
                    except Exception as e:
                        logger.warning(f"⚠️ Error disconnecting MongoDB: {e}")
                
                disconnect_tasks.append(disconnect_mongodb())
            
            if self.supabase_provider and self.supabase_connected:
                async def disconnect_supabase():
                    try:
                        if self.supabase_provider is not None:
                            await self.supabase_provider.disconnect()
                            self.supabase_connected = False
                            logger.info("✅ Supabase disconnected")
                    except Exception as e:
                        logger.warning(f"⚠️ Error disconnecting Supabase: {e}")
                
                disconnect_tasks.append(disconnect_supabase())
            
            # Execute disconnections concurrently if there are any
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            
            self.connected = False
            logger.info("✅ Hybrid storage provider disconnected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during hybrid storage disconnect: {e}")
        finally:
            self.connected = False
            self.mongodb_connected = False
            self.supabase_connected = False
    
    async def _cleanup_connections(self) -> None:
        """
        Clean up partial connections in case of connection failures
        """
        if self.mongodb_provider:
            try:
                await self.mongodb_provider.disconnect()
            except Exception:
                pass
        
        if self.supabase_provider:
            try:
                await self.supabase_provider.disconnect()
            except Exception:
                pass
        
        self.mongodb_connected = False
        self.supabase_connected = False
    
    async def initialize(self) -> None:
        """
        Initialize both MongoDB and Supabase providers with security hooks
        
        This should be called after successful connection to set up
        indexes, encryption, and other required components.
        
        SECURITY FIX: Ensures crypto system is initialized before any encryption operations
        """
        if not self.connected:
            raise ValueError("Must be connected before initialization")
        
        try:
            logger.info("🔧 Initializing hybrid storage provider...")
            
            # SECURITY FIX: Ensure crypto system is initialized
            from bot.config import Config
            from bot.crypto_utils import initialize_crypto, get_crypto
            
            encryption_seed = Config.ENCRYPTION_SEED
            if not encryption_seed:
                raise ValueError("ENCRYPTION_SEED not available - crypto initialization failed")
            
            # Initialize crypto if not already done
            try:
                get_crypto()
                logger.info("🔒 Crypto system already initialized")
            except RuntimeError:
                # Crypto not initialized yet, initialize it now
                initialize_crypto(encryption_seed)
                logger.info("🔒 Crypto system initialized successfully")
            
            # Initialize both providers in parallel for faster setup
            initialization_tasks = []
            
            if self.mongodb_provider and self.mongodb_connected:
                async def init_mongodb():
                    try:
                        if self.mongodb_provider is not None:
                            await self.mongodb_provider.initialize()
                            logger.info("✅ MongoDB provider initialized")
                    except Exception as e:
                        logger.error(f"❌ MongoDB initialization failed: {e}")
                        raise
                
                initialization_tasks.append(init_mongodb())
            
            if self.supabase_provider and self.supabase_connected:
                async def init_supabase():
                    try:
                        if self.supabase_provider is not None:
                            await self.supabase_provider.initialize()
                            logger.info("✅ Supabase provider initialized")
                    except Exception as e:
                        logger.error(f"❌ Supabase initialization failed: {e}")
                        raise
                
                initialization_tasks.append(init_supabase())
            
            # Execute initializations concurrently
            await asyncio.gather(*initialization_tasks)
            
            self._encryption_initialized = True
            logger.info("🎉 Hybrid storage provider initialized successfully with security hooks")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize hybrid storage provider: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Perform health check on both MongoDB and Supabase providers
        
        Returns:
            bool: True if both providers are healthy, False otherwise
        """
        try:
            if not self.connected:
                return False
            
            # Check health of both providers concurrently
            health_tasks = []
            
            if self.mongodb_provider and self.mongodb_connected:
                health_tasks.append(self.mongodb_provider.health_check())
            else:
                # MongoDB not connected - health check fails
                return False
            
            if self.supabase_provider and self.supabase_connected:
                health_tasks.append(self.supabase_provider.health_check())
            else:
                # Supabase not connected - health check fails
                return False
            
            # Execute health checks concurrently
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            # Check if all health checks passed
            mongodb_healthy = bool(health_results[0]) if not isinstance(health_results[0], Exception) else False
            supabase_healthy = bool(health_results[1]) if not isinstance(health_results[1], Exception) else False
            
            overall_health = mongodb_healthy and supabase_healthy
            
            if overall_health:
                logger.debug("✅ Hybrid storage provider health check passed")
            else:
                logger.warning(f"⚠️ Hybrid storage provider health check failed - MongoDB: {mongodb_healthy}, Supabase: {supabase_healthy}")
            
            return overall_health
            
        except Exception as e:
            logger.warning(f"⚠️ Hybrid storage provider health check failed: {e}")
            return False
    
    # MongoDB Routed Operations
    # These operations are routed to MongoDB as per the routing strategy
    
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save user's API key to MongoDB with encryption verification (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            api_key (str): Hugging Face API key (will be encrypted before storage)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("save_user_api_key", "MongoDB", user_id)
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for API key storage")
            return False
        
        # SECURITY FIX: Verify crypto is initialized before encryption
        from bot.crypto_utils import get_crypto
        try:
            get_crypto()
        except RuntimeError:
            logger.error("❌ CRITICAL: Crypto not initialized - cannot encrypt API key")
            return False
        
        try:
            result = await self.mongodb_provider.save_user_api_key(user_id, api_key)
            if result:
                logger.info(f"🔒 API key encrypted and saved successfully for user {user_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Error saving API key via MongoDB: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        Retrieve user's API key from MongoDB with tamper detection (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Optional[str]: Decrypted API key or None if not found
        """
        self._log_routing_decision("get_user_api_key", "MongoDB", user_id)
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for API key retrieval")
            return None
        
        # SECURITY FIX: Verify crypto is initialized before decryption
        from bot.crypto_utils import get_crypto, TamperDetectionError
        try:
            get_crypto()
        except RuntimeError:
            logger.error("❌ CRITICAL: Crypto not initialized - cannot decrypt API key")
            return None
        
        try:
            api_key = await self.mongodb_provider.get_user_api_key(user_id)
            if api_key:
                logger.info(f"🔓 API key retrieved and verified for user {user_id}")
            return api_key
        except TamperDetectionError:
            # SECURITY CRITICAL: Data tampering detected
            logger.error(f"🚨 SECURITY ALERT: Tamper detection triggered for user {user_id} API key")
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving API key via MongoDB: {e}")
            return None
    
    # Admin Operations (routed to MongoDB)
    async def get_admin_data(self) -> Optional[Dict[str, Any]]:
        """
        Get admin system configuration data from MongoDB (routed operation)
        
        Returns:
            Optional[Dict[str, Any]]: Admin data including user list, settings, etc.
        """
        self._log_routing_decision("get_admin_data", "MongoDB")
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for admin data retrieval")
            return None
        
        try:
            return await self.mongodb_provider.get_admin_data()
        except Exception as e:
            logger.error(f"❌ Error retrieving admin data via MongoDB: {e}")
            return None
    
    async def save_admin_data(self, admin_data: Dict[str, Any]) -> bool:
        """
        Save admin system configuration data to MongoDB (routed operation)
        
        Args:
            admin_data (Dict[str, Any]): Admin data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("save_admin_data", "MongoDB")
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for admin data storage")
            return False
        
        try:
            return await self.mongodb_provider.save_admin_data(admin_data)
        except Exception as e:
            logger.error(f"❌ Error saving admin data via MongoDB: {e}")
            return False
    
    async def log_admin_action(self, admin_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log admin action for audit trail to MongoDB (routed operation)
        
        Args:
            admin_id (int): Admin user ID
            action (str): Action performed
            details (Dict[str, Any]): Additional action details
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("log_admin_action", "MongoDB", admin_id)
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for admin action logging")
            return False
        
        try:
            return await self.mongodb_provider.log_admin_action(admin_id, action, details)
        except Exception as e:
            logger.error(f"❌ Error logging admin action via MongoDB: {e}")
            return False
    
    async def get_admin_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get admin action logs from MongoDB (routed operation)
        
        Args:
            limit (int): Maximum number of logs to return
            skip (int): Number of logs to skip (for pagination)
            
        Returns:
            List[Dict[str, Any]]: List of admin action logs
        """
        self._log_routing_decision("get_admin_logs", "MongoDB")
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for admin logs retrieval")
            return []
        
        try:
            return await self.mongodb_provider.get_admin_logs(limit, skip)
        except Exception as e:
            logger.error(f"❌ Error retrieving admin logs via MongoDB: {e}")
            return []
    
    # Usage Analytics Operations (routed to MongoDB)
    async def log_usage(self, user_id: int, action: str, model_used: str, tokens_used: int = 0) -> bool:
        """
        Log usage metrics for analytics to MongoDB (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            action (str): Action performed (text_generation, image_generation, etc.)
            model_used (str): AI model used
            tokens_used (int): Number of tokens consumed
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("log_usage", "MongoDB", user_id)
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for usage logging")
            return False
        
        try:
            return await self.mongodb_provider.log_usage(user_id, action, model_used, tokens_used)
        except Exception as e:
            logger.error(f"❌ Error logging usage via MongoDB: {e}")
            return False
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get usage statistics for a user from MongoDB (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            days (int): Number of days to look back
            
        Returns:
            Dict[str, Any]: Usage statistics
        """
        self._log_routing_decision("get_usage_stats", "MongoDB", user_id)
        
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for usage stats retrieval")
            return {}
        
        try:
            return await self.mongodb_provider.get_usage_stats(user_id, days)
        except Exception as e:
            logger.error(f"❌ Error retrieving usage stats via MongoDB: {e}")
            return {}
    
    # Supabase Routed Operations
    # These operations are routed to Supabase as per the routing strategy
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Get user preferences from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Dict[str, Any]: User preferences
        """
        self._log_routing_decision("get_user_preferences", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for user preferences retrieval")
            return {}
        
        try:
            return await self.supabase_provider.get_user_preferences(user_id)
        except Exception as e:
            logger.error(f"❌ Error retrieving user preferences via Supabase: {e}")
            return {}
    
    async def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences to Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            preferences (Dict[str, Any]): User preferences to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("save_user_preferences", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for user preferences storage")
            return False
        
        try:
            return await self.supabase_provider.save_user_preferences(user_id, preferences)
        except Exception as e:
            logger.error(f"❌ Error saving user preferences via Supabase: {e}")
            return False
    
    async def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """Get a specific user preference value by key from Supabase (routed operation)"""
        self._log_routing_decision("get_user_preference", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for user preference retrieval")
            return None
        
        try:
            return await self.supabase_provider.get_user_preference(user_id, key)
        except Exception as e:
            logger.error(f"❌ Error retrieving user preference via Supabase: {e}")
            return None
    
    async def save_user_preference(self, user_id: int, key: str, value: str) -> bool:
        """Save a specific user preference value by key to Supabase (routed operation)"""
        self._log_routing_decision("save_user_preference", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for user preference storage")
            return False
        
        try:
            return await self.supabase_provider.save_user_preference(user_id, key, value)
        except Exception as e:
            logger.error(f"❌ Error saving user preference via Supabase: {e}")
            return False
    
    # Conversation Storage (routed to Supabase)
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """
        Save conversation history to Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            conversation_data (Dict[str, Any]): Conversation data including messages, summary, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("save_conversation", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for conversation storage")
            return False
        
        try:
            return await self.supabase_provider.save_conversation(user_id, conversation_data)
        except Exception as e:
            logger.error(f"❌ Error saving conversation via Supabase: {e}")
            return False
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversation summaries from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            limit (int): Maximum number of conversations to return
            skip (int): Number of conversations to skip (for pagination)
        
        Returns:
            List[Dict[str, Any]]: List of conversation summaries
        """
        self._log_routing_decision("get_user_conversations", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for conversations retrieval")
            return []
        
        try:
            return await self.supabase_provider.get_user_conversations(user_id, limit, skip)
        except Exception as e:
            logger.error(f"❌ Error retrieving conversations via Supabase: {e}")
            return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed conversation data from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): Conversation identifier
            
        Returns:
            Optional[Dict[str, Any]]: Full conversation data or None if not found
        """
        self._log_routing_decision("get_conversation_details", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for conversation details retrieval")
            return None
        
        try:
            return await self.supabase_provider.get_conversation_details(user_id, conversation_id)
        except Exception as e:
            logger.error(f"❌ Error retrieving conversation details via Supabase: {e}")
            return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """
        Delete a specific conversation from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): Conversation identifier
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        self._log_routing_decision("delete_conversation", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for conversation deletion")
            return False
        
        try:
            return await self.supabase_provider.delete_conversation(user_id, conversation_id)
        except Exception as e:
            logger.error(f"❌ Error deleting conversation via Supabase: {e}")
            return False
    
    async def clear_user_history(self, user_id: int) -> bool:
        """
        Clear all conversation history for a user from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("clear_user_history", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for user history clearing")
            return False
        
        try:
            return await self.supabase_provider.clear_user_history(user_id)
        except Exception as e:
            logger.error(f"❌ Error clearing user history via Supabase: {e}")
            return False
    
    async def get_conversation_count(self, user_id: int) -> int:
        """
        Get total number of conversations for a user from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            int: Number of conversations (0 if error)
        """
        self._log_routing_decision("get_conversation_count", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for conversation count")
            return 0
        
        try:
            return await self.supabase_provider.get_conversation_count(user_id)
        except Exception as e:
            logger.error(f"❌ Error getting conversation count via Supabase: {e}")
            return 0
    
    # File Storage (routed to Supabase)
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """
        Save file data to Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            file_data (bytes): File content
            metadata (Dict[str, Any]): File metadata (name, type, size, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("save_file", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for file storage")
            return False
        
        try:
            return await self.supabase_provider.save_file(user_id, file_id, file_data, metadata)
        except Exception as e:
            logger.error(f"❌ Error saving file via Supabase: {e}")
            return False
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file data from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            
        Returns:
            Optional[Dict[str, Any]]: File data and metadata or None if not found
        """
        self._log_routing_decision("get_file", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for file retrieval")
            return None
        
        try:
            return await self.supabase_provider.get_file(user_id, file_id)
        except Exception as e:
            logger.error(f"❌ Error retrieving file via Supabase: {e}")
            return None
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """
        Delete file data from Supabase (routed operation)
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("delete_file", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for file deletion")
            return False
        
        try:
            return await self.supabase_provider.delete_file(user_id, file_id)
        except Exception as e:
            logger.error(f"❌ Error deleting file via Supabase: {e}")
            return False
    
    # Special Hybrid Operations
    # These operations need to coordinate between both databases
    
    async def reset_user_database(self, user_id: int) -> bool:
        """
        Reset all user data from both MongoDB and Supabase (hybrid operation)
        
        This operation coordinates between both databases to ensure complete
        user data removal while maintaining data consistency.
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("reset_user_database", "Both (MongoDB + Supabase)", user_id)
        
        # Validate both providers are available
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.error("❌ MongoDB not available for user database reset")
            return False
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for user database reset")
            return False
        
        try:
            # Reset user data in both databases concurrently
            # We use gather to ensure both operations complete
            mongodb_task = self.mongodb_provider.reset_user_database(user_id)
            supabase_task = self.supabase_provider.reset_user_database(user_id)
            
            mongodb_result, supabase_result = await asyncio.gather(
                mongodb_task, supabase_task, return_exceptions=True
            )
            
            # Check results
            mongodb_success = mongodb_result if not isinstance(mongodb_result, Exception) else False
            supabase_success = supabase_result if not isinstance(supabase_result, Exception) else False
            
            if isinstance(mongodb_result, Exception):
                logger.error(f"❌ MongoDB reset failed for user {user_id}: {mongodb_result}")
            
            if isinstance(supabase_result, Exception):
                logger.error(f"❌ Supabase reset failed for user {user_id}: {supabase_result}")
            
            # Overall success requires both to succeed
            overall_success = bool(mongodb_success and supabase_success)
            
            if overall_success:
                logger.info(f"✅ Successfully reset user database for user {user_id} in both MongoDB and Supabase")
            else:
                logger.error(f"❌ Partial failure in user database reset for user {user_id} - MongoDB: {mongodb_success}, Supabase: {supabase_success}")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"❌ Error resetting user database for user {user_id}: {e}")
            return False
    
    # Utility Methods
    def _log_routing_decision(self, operation: str, destination: str, user_id: Optional[int] = None) -> None:
        """
        Log routing decisions for transparency and debugging
        
        Args:
            operation (str): Name of the operation being routed
            destination (str): Target database(s)
            user_id (Optional[int]): User ID if applicable
        """
        user_info = f" for user {user_id}" if user_id else ""
        logger.debug(f"🔀 Routing {operation}{user_info} to {destination}")
    
    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get information about the current routing configuration and status
        
        Returns:
            Dict[str, Any]: Routing information and status
        """
        return {
            "provider_type": "hybrid",
            "mongodb_connected": self.mongodb_connected,
            "supabase_connected": self.supabase_connected,
            "overall_connected": self.connected,
            "routing_strategy": {
                "mongodb": [
                    "API keys (save_user_api_key, get_user_api_key)",
                    "Admin data (get_admin_data, save_admin_data, log_admin_action, get_admin_logs)",
                    "Usage analytics (log_usage, get_usage_stats)",
                    "Telegram IDs and developer database"
                ],
                "supabase": [
                    "User preferences (get_user_preferences, save_user_preferences)",
                    "Conversations (save_conversation, get_user_conversations, etc.)",
                    "File storage (save_file, get_file, delete_file)",
                    "User-specific data"
                ],
                "hybrid": [
                    "User database reset (coordinates both databases)"
                ]
            },
            "configuration": {
                "mongodb_configured": Config.has_mongodb_config(),
                "supabase_configured": Config.has_supabase_config(),
                "hybrid_config_valid": Config.has_hybrid_config()
            }
        }
    
    # Generic User Data Management (required abstract methods)
    async def save_user_data(self, user_id: int, data_key: str, data_value: Any) -> bool:
        """
        Save generic user data (routed to Supabase)
        
        Args:
            user_id (int): Telegram user ID
            data_key (str): Data key identifier
            data_value (Any): Data value to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("save_user_data", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for saving user data")
            return False
        
        try:
            return await self.supabase_provider.save_user_data(user_id, data_key, data_value)
        except Exception as e:
            logger.error(f"❌ Error saving user data via Supabase: {e}")
            return False
    
    async def get_user_data(self, user_id: int, data_key: str) -> Optional[Any]:
        """
        Get generic user data (routed to Supabase)
        
        Args:
            user_id (int): Telegram user ID
            data_key (str): Data key identifier
            
        Returns:
            Optional[Any]: Data value if found, None otherwise
        """
        self._log_routing_decision("get_user_data", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for retrieving user data")
            return None
        
        try:
            return await self.supabase_provider.get_user_data(user_id, data_key)
        except Exception as e:
            logger.error(f"❌ Error retrieving user data via Supabase: {e}")
            return None
    
    async def delete_user_data(self, user_id: int, data_key: str) -> bool:
        """
        Delete generic user data (routed to Supabase)
        
        Args:
            user_id (int): Telegram user ID
            data_key (str): Data key identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._log_routing_decision("delete_user_data", "Supabase", user_id)
        
        if not self.supabase_provider or not self.supabase_connected:
            logger.error("❌ Supabase not available for deleting user data")
            return False
        
        try:
            return await self.supabase_provider.delete_user_data(user_id, data_key)
        except Exception as e:
            logger.error(f"❌ Error deleting user data via Supabase: {e}")
            return False
    
    async def get_provider_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from both providers for monitoring and debugging
        
        Returns:
            Dict[str, Any]: Combined statistics from both providers
        """
        stats = {
            "hybrid_provider": {
                "connected": self.connected,
                "mongodb_connected": self.mongodb_connected,
                "supabase_connected": self.supabase_connected,
            },
            "mongodb_stats": {},
            "supabase_stats": {},
            "health_status": {}
        }
        
        # Get MongoDB stats if available
        if self.mongodb_provider and self.mongodb_connected:
            try:
                mongodb_health = await self.mongodb_provider.health_check()
                stats["mongodb_stats"]["healthy"] = mongodb_health
                stats["health_status"]["mongodb"] = "healthy" if mongodb_health else "unhealthy"
            except Exception as e:
                stats["mongodb_stats"]["error"] = str(e)
                stats["health_status"]["mongodb"] = "error"
        else:
            stats["health_status"]["mongodb"] = "disconnected"
        
        # Get Supabase stats if available
        if self.supabase_provider and self.supabase_connected:
            try:
                supabase_health = await self.supabase_provider.health_check()
                stats["supabase_stats"]["healthy"] = supabase_health
                stats["health_status"]["supabase"] = "healthy" if supabase_health else "unhealthy"
            except Exception as e:
                stats["supabase_stats"]["error"] = str(e)
                stats["health_status"]["supabase"] = "error"
        else:
            stats["health_status"]["supabase"] = "disconnected"
        
        return stats