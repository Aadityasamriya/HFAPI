"""
Resilient Hybrid Storage Provider Implementation
Routes operations between MongoDB and Supabase with graceful fallback to MongoDB-only mode
when Supabase connection fails. Ensures bot remains functional even with database issues.
"""

import asyncio
import os
import socket
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urlparse

import logging
from .base import StorageProvider
from .mongodb_provider import MongoDBProvider
from .supabase_user_provider import SupabaseUserProvider
from bot.config import Config
from bot.security_utils import SecureLogger

logger = logging.getLogger(__name__)

class ResilientHybridProvider(StorageProvider):
    """
    Resilient hybrid storage provider with graceful fallback capabilities
    
    Routing Strategy (when both databases are available):
    - MongoDB: API keys, telegram IDs, developer's database, admin data, usage logs, main operational data
    - Supabase: User data storage (conversations, preferences, files, user-specific data)
    
    Fallback Strategy (when Supabase fails):
    - MongoDB handles ALL data including user data
    - Bot remains fully functional
    - Automatic retry of Supabase connection in background
    
    This provider ensures the bot never fails to start due to database connection issues.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize providers
        self.mongodb_provider = None
        self.supabase_provider = None
        
        # Connection status tracking
        self.mongodb_connected = False
        self.supabase_connected = False
        self.fallback_mode = False
        
        # Logging
        self.secure_logger = SecureLogger(logger)
        
        # Supabase retry mechanism
        self._supabase_retry_count = 0
        self._max_supabase_retries = 3
        self._supabase_retry_delay = 5.0
        self._last_supabase_retry = 0
        
        # Validate configuration
        self._validate_resilient_config()
    
    def _validate_resilient_config(self) -> None:
        """
        Validate configuration for resilient mode (Railway.com optimized)
        MongoDB is required, Supabase is optional
        
        Raises:
            ValueError: If MongoDB configuration is missing
        """
        try:
            # RAILWAY OPTIMIZATION: Enhanced Railway detection and logging
            is_railway = Config._is_railway_environment()
            if is_railway:
                railway_env = os.getenv('RAILWAY_ENVIRONMENT', 'unknown')
                logger.info(f"🚂 Railway.com platform detected (environment: {railway_env})")
            
            # MongoDB is REQUIRED
            mongodb_uri = Config.get_mongodb_uri()
            if not mongodb_uri:
                platform_hint = " (Note: Railway provides managed databases)" if is_railway else ""
                raise ValueError(
                    f"ResilientHybridProvider requires MongoDB configuration{platform_hint}. "
                    "Please set MONGODB_URI or MONGO_URI environment variable."
                )
            
            # Supabase is OPTIONAL (Railway.com compatible)
            supabase_url = Config.get_supabase_mgmt_url()
            if supabase_url:
                if is_railway:
                    # Check Railway deployment configuration
                    logger.info("✅ Railway deployment with Supabase configuration - full hybrid mode available")
                else:
                    logger.info("✅ Both MongoDB and Supabase configured - full hybrid mode available")
            else:
                if is_railway:
                    logger.warning("⚠️ Railway deployment with MongoDB-only configuration")
                    logger.warning("   Consider adding a PostgreSQL database service on Railway for enhanced user data storage")
                    logger.warning("   Bot will operate in resilient fallback mode with limited user data features")
                else:
                    logger.warning("⚠️ Only MongoDB configured - will operate in fallback mode from start")
                self.fallback_mode = True
            
        except Exception as e:
            logger.error(f"❌ ResilientHybridProvider configuration validation failed: {e}")
            raise ValueError(f"ResilientHybridProvider initialization failed: {e}")
    
    def _is_connection_error(self, error: Exception) -> bool:
        """Check if an error is a connection-related error"""
        connection_error_types = (
            ConnectionError,
            socket.gaierror,  # DNS resolution errors
            socket.timeout,
            OSError,
            asyncio.TimeoutError,
        )
        
        # Also check for string patterns in error messages
        connection_error_patterns = [
            'connection',
            'timeout',
            'network',
            'dns',
            'resolve',
            'unreachable',
            'refused',
            'failed to establish',
            'name resolution',
            'nodename nor servname provided',
        ]
        
        if isinstance(error, connection_error_types):
            return True
        
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in connection_error_patterns)
    
    def _is_invalid_url_error(self, error: Exception) -> bool:
        """Check if an error is due to invalid URL configuration"""
        invalid_url_patterns = [
            'invalid url',
            'invalid connection string',
            'malformed url',
            'invalid database url',
            'could not parse',
            'invalid scheme',
        ]
        
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in invalid_url_patterns)
    
    # Connection Management
    async def connect(self) -> None:
        """
        Establish connections with graceful fallback
        MongoDB connection failure causes hard fail
        Supabase connection failure triggers fallback mode
        """
        try:
            logger.info("🔗 Establishing resilient hybrid storage connections...")
            
            # Initialize MongoDB provider (REQUIRED)
            if self.mongodb_provider is None:
                try:
                    self.mongodb_provider = MongoDBProvider()
                    logger.info("📦 MongoDB provider initialized")
                except Exception as e:
                    logger.error(f"❌ CRITICAL: Failed to initialize MongoDB provider: {e}")
                    raise ConnectionError(f"MongoDB provider initialization failed: {e}")
            
            # Connect to MongoDB (REQUIRED - hard fail if this fails)
            try:
                await self.mongodb_provider.connect()
                self.mongodb_connected = True
                logger.info("✅ MongoDB connection established (REQUIRED)")
            except Exception as e:
                logger.error(f"❌ CRITICAL: MongoDB connection failed: {e}")
                raise ConnectionError(f"MongoDB connection failed - bot cannot start: {e}")
            
            # Try to connect to Supabase (OPTIONAL - graceful fallback if this fails)
            if not self.fallback_mode:
                await self._attempt_supabase_connection()
            else:
                logger.info("🔄 Starting in fallback mode - Supabase will be retried in background")
            
            self.connected = True
            
            if self.supabase_connected:
                logger.info("🎉 Resilient hybrid storage provider connected successfully (full hybrid mode)")
            else:
                logger.info("🎉 Resilient hybrid storage provider connected successfully (fallback mode - MongoDB only)")
                logger.info("   Bot is fully functional. Supabase will be retried automatically.")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect resilient hybrid storage provider: {e}")
            self.connected = False
            # Clean up partial connections
            await self._cleanup_connections()
            raise
    
    async def _attempt_supabase_connection(self) -> None:
        """
        Attempt to connect to Supabase with comprehensive error handling
        """
        try:
            logger.info("🔗 Attempting Supabase connection...")
            
            # Initialize Supabase provider if needed
            if self.supabase_provider is None:
                try:
                    self.supabase_provider = SupabaseUserProvider()
                    logger.info("📦 Supabase provider initialized")
                except ImportError as e:
                    logger.warning(f"⚠️ Supabase dependencies not available: {e}")
                    logger.warning("   Operating in MongoDB-only fallback mode")
                    self.fallback_mode = True
                    return
                except Exception as e:
                    logger.warning(f"⚠️ Supabase provider initialization failed: {e}")
                    logger.warning("   Operating in MongoDB-only fallback mode")
                    self.fallback_mode = True
                    return
            
            # Attempt connection with timeout
            try:
                # Set a reasonable timeout for connection attempts
                await asyncio.wait_for(
                    self.supabase_provider.connect(),
                    timeout=30.0  # 30 second timeout
                )
                self.supabase_connected = True
                self._supabase_retry_count = 0  # Reset retry count on success
                logger.info("✅ Supabase connection established")
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ Supabase connection timed out after 30 seconds")
                self._handle_supabase_connection_failure("Connection timeout")
                
            except Exception as e:
                if self._is_connection_error(e):
                    logger.warning(f"⚠️ Supabase connection failed (network/DNS issue): {e}")
                    self._handle_supabase_connection_failure(f"Network error: {e}")
                elif self._is_invalid_url_error(e):
                    logger.warning(f"⚠️ Supabase URL configuration invalid: {e}")
                    logger.warning("   Please check SUPABASE_MGMT_URL environment variable")
                    self._handle_supabase_connection_failure(f"Invalid URL: {e}")
                else:
                    logger.warning(f"⚠️ Supabase connection failed: {e}")
                    self._handle_supabase_connection_failure(f"Connection error: {e}")
            
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error during Supabase connection attempt: {e}")
            self._handle_supabase_connection_failure(f"Unexpected error: {e}")
    
    def _handle_supabase_connection_failure(self, reason: str) -> None:
        """
        Handle Supabase connection failure gracefully
        """
        self.supabase_connected = False
        self.fallback_mode = True
        self._supabase_retry_count += 1
        
        if self._supabase_retry_count <= self._max_supabase_retries:
            logger.warning(f"   Will retry Supabase connection later (attempt {self._supabase_retry_count}/{self._max_supabase_retries})")
        else:
            logger.warning(f"   Max Supabase retry attempts reached. Operating in MongoDB-only mode.")
            logger.warning(f"   User data storage will be handled by MongoDB.")
        
        logger.warning(f"   Reason: {reason}")
        logger.info("✅ Bot will continue operating with MongoDB-only storage")
    
    async def disconnect(self) -> None:
        """
        Close connections to both providers gracefully
        """
        try:
            logger.info("🔌 Disconnecting resilient hybrid storage provider...")
            
            # Disconnect from both databases in parallel for faster shutdown
            disconnect_tasks = []
            
            if self.mongodb_provider and self.mongodb_connected:
                async def disconnect_mongodb():
                    try:
                        if self.mongodb_provider is not None:
                            await self.mongodb_provider.disconnect()
                            logger.info("✅ MongoDB disconnected")
                    except Exception as e:
                        logger.warning(f"Error disconnecting MongoDB: {e}")
                
                disconnect_tasks.append(disconnect_mongodb())
            
            if self.supabase_provider and self.supabase_connected:
                async def disconnect_supabase():
                    try:
                        if self.supabase_provider is not None:
                            await self.supabase_provider.disconnect()
                            logger.info("✅ Supabase disconnected")
                    except Exception as e:
                        logger.warning(f"Error disconnecting Supabase: {e}")
                
                disconnect_tasks.append(disconnect_supabase())
            
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            
            logger.info("✅ Resilient hybrid storage provider disconnected")
            
        except Exception as e:
            logger.warning(f"Error disconnecting resilient hybrid storage provider: {e}")
        finally:
            self.mongodb_connected = False
            self.supabase_connected = False
            self.connected = False
    
    async def _cleanup_connections(self) -> None:
        """Clean up partial connections on failure"""
        try:
            if self.mongodb_provider:
                await self.mongodb_provider.disconnect()
        except:
            pass
        
        try:
            if self.supabase_provider:
                await self.supabase_provider.disconnect()
        except:
            pass
        
        self.mongodb_connected = False
        self.supabase_connected = False
    
    async def initialize(self) -> None:
        """
        Initialize both storage backends with security hooks (MongoDB required, Supabase optional)
        
        SECURITY FIX: Ensures crypto system is initialized before any encryption operations
        """
        if not self.connected:
            raise ValueError("Must be connected before initialization")
        
        try:
            logger.info("⚙️ Initializing resilient hybrid storage backends...")
            
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
            
            # Initialize MongoDB (REQUIRED)
            if self.mongodb_provider and self.mongodb_connected:
                try:
                    await self.mongodb_provider.initialize()
                    logger.info("✅ MongoDB backend initialized")
                except Exception as e:
                    logger.error(f"❌ CRITICAL: MongoDB initialization failed: {e}")
                    raise RuntimeError(f"MongoDB initialization failed: {e}")
            
            # Initialize Supabase (OPTIONAL)
            if self.supabase_provider and self.supabase_connected:
                try:
                    await self.supabase_provider.initialize()
                    logger.info("✅ Supabase backend initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Supabase initialization failed: {e}")
                    logger.warning("   Disabling Supabase and falling back to MongoDB-only mode")
                    self.supabase_connected = False
                    self.fallback_mode = True
                    try:
                        await self.supabase_provider.disconnect()
                    except:
                        pass
            
            if self.fallback_mode:
                logger.info("ℹ️ Operating in MongoDB-only fallback mode")
                logger.info("   All data (including user data) will be stored in MongoDB")
            else:
                logger.info("ℹ️ Operating in full hybrid mode")
                logger.info("   MongoDB: API keys, admin data, core functionality")
                logger.info("   Supabase: User conversations, preferences, files")
            
            self._encryption_initialized = True
            logger.info("🎉 Resilient hybrid storage initialization completed with security hooks")
            
        except Exception as e:
            logger.error(f"❌ Storage initialization failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Perform health check on storage backends
        Returns True if MongoDB is healthy (Supabase is optional)
        """
        try:
            # MongoDB health check (REQUIRED)
            mongodb_healthy = False
            if self.mongodb_provider and self.mongodb_connected:
                mongodb_healthy = await self.mongodb_provider.health_check()
            
            # Supabase health check (OPTIONAL)
            supabase_healthy = True  # Default to True since it's optional
            if self.supabase_provider and self.supabase_connected:
                supabase_healthy = await self.supabase_provider.health_check()
                if not supabase_healthy:
                    logger.warning("⚠️ Supabase health check failed - may need to enable fallback mode")
            
            # Overall health: MongoDB must be healthy, Supabase is optional
            overall_healthy = mongodb_healthy
            
            if overall_healthy:
                if self.supabase_connected and supabase_healthy:
                    logger.debug("✅ All storage backends healthy (full hybrid mode)")
                else:
                    logger.debug("✅ MongoDB healthy, operating in fallback mode")
            else:
                logger.warning("❌ MongoDB health check failed - critical issue")
            
            return overall_healthy
            
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")
            return False
    
    # API Key Management (MongoDB only - critical functionality)
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save user's API key with encryption verification (MongoDB only - critical functionality)
        
        SECURITY FIX: Verifies crypto is initialized before encryption
        """
        if not self.mongodb_provider or not self.mongodb_connected:
            raise RuntimeError("MongoDB not connected - cannot save API key")
        
        # SECURITY FIX: Verify crypto is initialized before encryption
        from bot.crypto_utils import get_crypto
        try:
            get_crypto()
        except RuntimeError:
            logger.error("❌ CRITICAL: Crypto not initialized - cannot encrypt API key")
            raise RuntimeError("Crypto not initialized - cannot encrypt API key")
        
        try:
            result = await self.mongodb_provider.save_user_api_key(user_id, api_key)
            if result:
                logger.info(f"🔒 API key encrypted and saved successfully for user {user_id}")
            return result
        except Exception as e:
            logger.error(f"❌ Error saving API key: {e}")
            raise
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        Get user's API key with tamper detection (MongoDB only - critical functionality)
        
        SECURITY FIX: Includes tamper detection and crypto verification
        """
        if not self.mongodb_provider or not self.mongodb_connected:
            raise RuntimeError("MongoDB not connected - cannot retrieve API key")
        
        # SECURITY FIX: Verify crypto is initialized before decryption
        from bot.crypto_utils import get_crypto, TamperDetectionError
        try:
            get_crypto()
        except RuntimeError:
            logger.error("❌ CRITICAL: Crypto not initialized - cannot decrypt API key")
            raise RuntimeError("Crypto not initialized - cannot decrypt API key")
        
        try:
            api_key = await self.mongodb_provider.get_user_api_key(user_id)
            if api_key:
                logger.info(f"🔓 API key retrieved and verified for user {user_id}")
            return api_key
        except TamperDetectionError:
            # SECURITY CRITICAL: Data tampering detected
            logger.error(f"🚨 SECURITY ALERT: Tamper detection triggered for user {user_id} API key")
            raise
        except Exception as e:
            logger.error(f"❌ Error retrieving API key: {e}")
            raise
    
    # User Data Management (required by base class)
    async def reset_user_database(self, user_id: int) -> bool:
        """Delete all user data including API keys and conversations"""
        success = True
        
        # Try both providers to ensure complete cleanup
        if self.supabase_provider and self.supabase_connected:
            try:
                supabase_success = await self.supabase_provider.reset_user_database(user_id)
                if not supabase_success:
                    success = False
            except Exception as e:
                logger.warning(f"Supabase user reset failed: {e}")
                success = False
        
        if self.mongodb_provider and self.mongodb_connected:
            try:
                mongodb_success = await self.mongodb_provider.reset_user_database(user_id)
                if not mongodb_success:
                    success = False
            except Exception as e:
                logger.warning(f"MongoDB user reset failed: {e}")
                success = False
        
        return success
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user preferences (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.get_user_preferences(user_id)
            except Exception as e:
                logger.warning(f"Supabase preferences retrieval failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            prefs = await self.mongodb_provider.get_user_preferences(user_id)
            return prefs if prefs is not None else {}
        
        return {}  # Return empty dict if no backend available
    
    async def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Save user preferences (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.save_user_preferences(user_id, preferences)
            except Exception as e:
                logger.warning(f"Supabase preferences save failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.save_user_preferences(user_id, preferences)
        
        raise RuntimeError("No storage backend available for preferences save")
    
    async def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """Get a specific user preference value by key (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.get_user_preference(user_id, key)
            except Exception as e:
                logger.warning(f"Supabase preference retrieval failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.get_user_preference(user_id, key)
        
        return None  # Return None if no backend available
    
    async def save_user_preference(self, user_id: int, key: str, value: str) -> bool:
        """Save a specific user preference value by key (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.save_user_preference(user_id, key, value)
            except Exception as e:
                logger.warning(f"Supabase preference save failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.save_user_preference(user_id, key, value)
        
        raise RuntimeError("No storage backend available for preference save")
    
    # User Data Isolation Support - SECURITY FIX: Added missing methods
    async def save_user_data(self, user_id: int, data_key: str, data: Any, encrypt: bool = True) -> bool:
        """Save user-specific data with encryption and isolation (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                # Check if Supabase provider has save_user_data method
                if hasattr(self.supabase_provider, 'save_user_data'):
                    return await self.supabase_provider.save_user_data(user_id, data_key, data, encrypt)
                else:
                    logger.warning("Supabase provider doesn't support save_user_data, falling back to MongoDB")
                    self._trigger_fallback_mode()
            except Exception as e:
                logger.warning(f"Supabase user data save failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.save_user_data(user_id, data_key, data, encrypt)
        
        raise RuntimeError("No storage backend available for user data save")
    
    async def get_user_data(self, user_id: int, data_key: str) -> Any:
        """Get user-specific data with decryption and isolation (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                # Check if Supabase provider has get_user_data method
                if hasattr(self.supabase_provider, 'get_user_data'):
                    return await self.supabase_provider.get_user_data(user_id, data_key)
                else:
                    logger.warning("Supabase provider doesn't support get_user_data, falling back to MongoDB")
                    self._trigger_fallback_mode()
            except Exception as e:
                logger.warning(f"Supabase user data retrieval failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.get_user_data(user_id, data_key)
        
        return None  # Return None if data not found or no backend available
    
    # User Management (MongoDB only - simplified for core functionality)
    async def create_user(self, user_id: int, username: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create user record (MongoDB only for simplicity)"""
        # Simplified: Use MongoDB only for user creation to avoid complexity
        if self.mongodb_provider and self.mongodb_connected:
            # MongoDB doesn't have create_user, so we'll implement basic user creation
            try:
                # Save basic user info as preferences
                user_data = {
                    "username": username,
                    "created_at": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                return await self.mongodb_provider.save_user_preferences(user_id, user_data)
            except Exception as e:
                logger.error(f"User creation failed: {e}")
                return False
        
        raise RuntimeError("MongoDB not available for user creation")
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user record (MongoDB preferences as fallback)"""
        # Simplified: Use MongoDB preferences as user data
        if self.mongodb_provider and self.mongodb_connected:
            try:
                return await self.mongodb_provider.get_user_preferences(user_id)
            except Exception as e:
                logger.warning(f"User retrieval failed: {e}")
                return None
        
        return None
    
    async def update_user(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """Update user record (MongoDB preferences)"""
        # Simplified: Update MongoDB preferences
        if self.mongodb_provider and self.mongodb_connected:
            try:
                # Get current preferences and merge updates
                current_prefs = await self.mongodb_provider.get_user_preferences(user_id)
                current_prefs.update(updates)
                return await self.mongodb_provider.save_user_preferences(user_id, current_prefs)
            except Exception as e:
                logger.error(f"User update failed: {e}")
                return False
        
        raise RuntimeError("MongoDB not available for user update")
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user record (use reset_user_database)"""
        # Simplified: Use reset_user_database which handles full cleanup
        return await self.reset_user_database(user_id)
    
    def _trigger_fallback_mode(self) -> None:
        """Trigger fallback mode due to Supabase failure"""
        if not self.fallback_mode:
            logger.warning("🔄 Supabase failure detected - enabling fallback mode")
            logger.warning("   All subsequent operations will use MongoDB only")
            self.fallback_mode = True
            self.supabase_connected = False
    
    # Conversation Management (required by base class) - Supabase preferred, MongoDB fallback
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
        """Get conversation summaries (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.get_user_conversations(user_id, limit, skip)
            except Exception as e:
                logger.warning(f"Supabase conversation list failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.get_user_conversations(user_id, limit, skip)
        
        return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed conversation data (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.get_conversation_details(user_id, conversation_id)
            except Exception as e:
                logger.warning(f"Supabase conversation details failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.get_conversation_details(user_id, conversation_id)
        
        return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """Delete conversation (both backends for complete removal)"""
        success = True
        
        # Try both backends
        if self.supabase_provider and self.supabase_connected:
            try:
                supabase_success = await self.supabase_provider.delete_conversation(user_id, conversation_id)
                if not supabase_success:
                    success = False
            except Exception as e:
                logger.warning(f"Supabase conversation deletion failed: {e}")
                success = False
        
        if self.mongodb_provider and self.mongodb_connected:
            try:
                mongodb_success = await self.mongodb_provider.delete_conversation(user_id, conversation_id)
                if not mongodb_success:
                    success = False
            except Exception as e:
                logger.warning(f"MongoDB conversation deletion failed: {e}")
                success = False
        
        return success
    
    async def clear_user_history(self, user_id: int) -> bool:
        """Clear all conversation history (both backends)"""
        success = True
        
        # Try both backends
        if self.supabase_provider and self.supabase_connected:
            try:
                supabase_success = await self.supabase_provider.clear_user_history(user_id)
                if not supabase_success:
                    success = False
            except Exception as e:
                logger.warning(f"Supabase history clear failed: {e}")
                success = False
        
        if self.mongodb_provider and self.mongodb_connected:
            try:
                mongodb_success = await self.mongodb_provider.clear_user_history(user_id)
                if not mongodb_success:
                    success = False
            except Exception as e:
                logger.warning(f"MongoDB history clear failed: {e}")
                success = False
        
        return success
    
    async def get_conversation_count(self, user_id: int) -> int:
        """Get conversation count (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.get_conversation_count(user_id)
            except Exception as e:
                logger.warning(f"Supabase conversation count failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            try:
                return await self.mongodb_provider.get_conversation_count(user_id)
            except Exception as e:
                logger.warning(f"MongoDB conversation count failed: {e}")
                return 0
        
        return 0
    
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """Save conversation (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.save_conversation(user_id, conversation_data)
            except Exception as e:
                logger.warning(f"Supabase conversation save failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.save_conversation(user_id, conversation_data)
        
        raise RuntimeError("No storage backend available for conversation save")
    
    # Legacy method for backward compatibility
    async def get_conversation_history(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history - legacy method, use get_user_conversations instead"""
        return await self.get_user_conversations(user_id, limit=limit, skip=0)
    
    # File Management (required by base class) - Supabase preferred, MongoDB fallback
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """Delete file (both backends for complete removal)"""
        success = True
        
        # Try both backends
        if self.supabase_provider and self.supabase_connected:
            try:
                supabase_success = await self.supabase_provider.delete_file(user_id, file_id)
                if not supabase_success:
                    success = False
            except Exception as e:
                logger.warning(f"Supabase file deletion failed: {e}")
                success = False
        
        if self.mongodb_provider and self.mongodb_connected:
            try:
                mongodb_success = await self.mongodb_provider.delete_file(user_id, file_id)
                if not mongodb_success:
                    success = False
            except Exception as e:
                logger.warning(f"MongoDB file deletion failed: {e}")
                success = False
        
        return success
    
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save file (Supabase preferred, MongoDB fallback)"""
        if metadata is None:
            metadata = {}
            
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.save_file(user_id, file_id, file_data, metadata)
            except Exception as e:
                logger.warning(f"Supabase file save failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.save_file(user_id, file_id, file_data, metadata)
        
        raise RuntimeError("No storage backend available for file save")
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file (Supabase preferred, MongoDB fallback)"""
        if self.supabase_provider and self.supabase_connected:
            try:
                return await self.supabase_provider.get_file(user_id, file_id)
            except Exception as e:
                logger.warning(f"Supabase file retrieval failed, falling back to MongoDB: {e}")
                self._trigger_fallback_mode()
        
        # Fallback to MongoDB
        if self.mongodb_provider and self.mongodb_connected:
            return await self.mongodb_provider.get_file(user_id, file_id)
        
        return None
    
    # Admin Data Management (required by base class) - MongoDB only
    async def get_admin_data(self) -> Optional[Dict[str, Any]]:
        """Get admin system configuration (MongoDB only)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot retrieve admin data")
            return None
        return await self.mongodb_provider.get_admin_data()
    
    async def save_admin_data(self, admin_data: Dict[str, Any]) -> bool:
        """Save admin system configuration (MongoDB only)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            raise RuntimeError("MongoDB not connected - cannot save admin data")
        return await self.mongodb_provider.save_admin_data(admin_data)
    
    async def log_admin_action(self, admin_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Log admin action (MongoDB only)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot log admin action")
            return False
        return await self.mongodb_provider.log_admin_action(admin_id, action, details)
    
    async def get_admin_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get admin action logs (MongoDB only)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot retrieve admin logs")
            return []
        return await self.mongodb_provider.get_admin_logs(limit, skip)
    
    async def log_usage(self, user_id: int, action: str, model_used: Optional[str] = None, 
                       tokens_used: Optional[int] = None, cost: Optional[float] = None, 
                       success: bool = True, details: Optional[str] = None) -> bool:
        """Log usage metrics (MongoDB only)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot log usage")
            return False
        return await self.mongodb_provider.log_usage(user_id, action, model_used or "unknown", tokens_used or 0)
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics (MongoDB only)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot retrieve usage stats")
            return {}
        return await self.mongodb_provider.get_usage_stats(user_id, days)
    
    # Additional admin functions (not in base interface but useful)
    async def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users (MongoDB only - admin function)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot retrieve user list")
            return []
        # Use get_user_preferences to get user data since get_all_users may not exist
        try:
            admin_data = await self.mongodb_provider.get_admin_data()
            if isinstance(admin_data, dict) and "users" in admin_data:
                return admin_data["users"]
            return []
        except Exception as e:
            logger.warning(f"Could not retrieve all users: {e}")
            return []
    
    async def get_user_count(self) -> int:
        """Get user count (MongoDB only - admin function)"""
        if not self.mongodb_provider or not self.mongodb_connected:
            logger.warning("MongoDB not connected - cannot retrieve user count")
            return 0
        try:
            all_users = await self.get_all_users()
            return len(all_users)
        except Exception as e:
            logger.warning(f"Could not get user count: {e}")
            return 0
    
    async def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Cleanup old data (both backends) - additional utility method"""
        results = {"mongodb": 0, "supabase": 0}
        
        # MongoDB cleanup
        if self.mongodb_provider and self.mongodb_connected:
            try:
                # Since cleanup_old_data may not exist, we'll use a simple approach
                # This could be implemented as clearing old conversations, etc.
                logger.info(f"MongoDB cleanup requested for data older than {days} days")
                results["mongodb"] = 0  # Placeholder
            except Exception as e:
                logger.warning(f"MongoDB cleanup failed: {e}")
        
        # Supabase cleanup  
        if self.supabase_provider and self.supabase_connected:
            try:
                logger.info(f"Supabase cleanup requested for data older than {days} days")
                results["supabase"] = 0  # Placeholder
            except Exception as e:
                logger.warning(f"Supabase cleanup failed: {e}")
        
        return results