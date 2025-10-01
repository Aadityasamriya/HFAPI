"""
PostgreSQL Storage Provider Implementation
Provides automatic database setup with zero configuration required
"""

import asyncio
import os
import json
import uuid
import base64
import secrets
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union

# PostgreSQL imports with proper type checking support
if TYPE_CHECKING:
    from asyncpg import Connection, Pool
    import asyncpg
    POSTGRESQL_AVAILABLE = True  # For type checking, assume available
else:
    try:
        import asyncpg
        from asyncpg import Connection, Pool
        POSTGRESQL_AVAILABLE = True
    except ImportError:
        # Create placeholder types to avoid unbound variable errors
        asyncpg = None  # type: ignore
        Connection = None  # type: ignore
        Pool = None  # type: ignore
        POSTGRESQL_AVAILABLE = False

# Encryption imports
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import logging
from .base import StorageProvider
from bot.config import Config
from bot.security_utils import SecureLogger

logger = logging.getLogger(__name__)

class PostgreSQLProvider(StorageProvider):
    """
    PostgreSQL storage provider with automatic database setup
    
    Provides zero-configuration database experience using Replit's automatic PostgreSQL
    """
    
    def __init__(self):
        super().__init__()
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL dependencies not available. Install with: pip install asyncpg")
        
        self.pool: Optional["Pool"] = None
        self._encryption_key = None
        self._aesgcm = None
        self._global_seed = None
        self.secure_logger = SecureLogger(logger)
    
    # Connection Management
    async def connect(self) -> None:
        """Establish connection to PostgreSQL with automatic configuration"""
        try:
            # Get database URL from configuration (Railway.com compatible)
            database_url = Config.get_supabase_mgmt_url() or os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL not found - PostgreSQL database not provisioned")
            
            # Create connection pool with optimal settings
            if asyncpg is None:
                raise RuntimeError("asyncpg not available")
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={
                    'jit': 'off',  # Disable JIT for better performance with small queries
                    'application_name': 'huggingface_ai_bot'
                }
            )
            
            # Test connection
            assert self.pool is not None, "Pool should not be None at this point"
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            self.connected = True
            logger.info("✅ Successfully connected to PostgreSQL with automatic configuration")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.connected = False
            self.pool = None
            raise
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool gracefully"""
        if self.pool:
            try:
                await self.pool.close()
                logger.info("✅ PostgreSQL connection pool closed gracefully")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL pool: {e}")
            finally:
                self.pool = None
                self.connected = False
    
    async def initialize(self) -> None:
        """Initialize PostgreSQL database (create tables, setup encryption, etc.)"""
        if not self.connected or not self.pool:
            raise ValueError("Must be connected before initialization")
        
        async with self.pool.acquire() as conn:
            # Create database schema
            await self._create_tables(conn)
            
            # Initialize encryption
            await self._initialize_encryption(conn)
            
            # Create indexes for performance
            await self._create_indexes(conn)
        
        self._encryption_initialized = True
        logger.info("✅ PostgreSQL provider initialized successfully")
    
    async def health_check(self) -> bool:
        """Perform health check on PostgreSQL"""
        try:
            if not self.connected or not self.pool:
                return False
            
            async with self.pool.acquire() as conn:
                await conn.execute('SELECT 1')
                return True
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
            return False
    
    async def _create_tables(self, conn: "Connection") -> None:
        """Create necessary database tables"""
        
        # Users table for API keys and user data
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                encrypted_api_key BYTEA,
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Conversations table for chat history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id BIGINT NOT NULL,
                summary TEXT NOT NULL,
                messages JSONB NOT NULL,
                started_at TIMESTAMP WITH TIME ZONE NOT NULL,
                last_message_at TIMESTAMP WITH TIME ZONE NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        
        # Files table for document storage
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id BIGINT NOT NULL,
                file_id TEXT NOT NULL,
                file_data BYTEA NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, file_id)
            )
        """)
        
        # Admin system table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_system (
                key TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Admin logs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                admin_id BIGINT NOT NULL,
                action TEXT NOT NULL,
                details JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Usage analytics table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_analytics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id BIGINT NOT NULL,
                action TEXT NOT NULL,
                model_used TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        
        # Encryption keys table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS encryption_keys (
                key_name TEXT PRIMARY KEY,
                encrypted_seed BYTEA NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        logger.info("✅ PostgreSQL database tables created successfully")
    
    async def _create_indexes(self, conn: "Connection") -> None:
        """Create database indexes for performance"""
        
        # Conversations indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_last_message ON conversations(last_message_at DESC)")
        
        # Files indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_files_user_id ON files(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at DESC)")
        
        # Admin logs indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_admin_logs_admin_id ON admin_logs(admin_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_admin_logs_created_at ON admin_logs(created_at DESC)")
        
        # Usage analytics indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage_analytics(user_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_created_at ON usage_analytics(created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_action ON usage_analytics(action)")
        
        logger.info("✅ PostgreSQL database indexes created successfully")
    
    async def _initialize_encryption(self, conn: "Connection") -> None:
        """Initialize encryption with seed from centralized Config only (SECURITY HARDENED)"""
        try:
            # SECURITY FIX: Get seed only from centralized Config - no storage layer mutations
            encryption_seed = Config.ENCRYPTION_SEED
            
            if not encryption_seed:
                raise ValueError(
                    "CRITICAL: Config.validate_config() must be called before storage initialization. "
                    "Seed management is centralized in Config only for security."
                )
            
            # Convert string seed to bytes for PBKDF2
            self._global_seed = encryption_seed.encode('utf-8')
            
            # Derive encryption key from seed
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'huggingface_ai_bot_salt',
                iterations=100000,
            )
            self._encryption_key = kdf.derive(self._global_seed)
            
            # Initialize AES-GCM cipher
            self._aesgcm = AESGCM(self._encryption_key)
            
            logger.info("✅ Encryption system initialized with seed from centralized Config (security hardened)")
            logger.info("🔒 Storage layer encryption ready - seed sourced from Config.validate_config() only")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt string data using AES-256-GCM"""
        if not self._aesgcm:
            raise ValueError("Encryption not initialized")
        
        try:
            # Generate random nonce for each encryption
            nonce = secrets.token_bytes(12)
            
            # Encrypt the data
            encrypted_data = self._aesgcm.encrypt(nonce, data.encode('utf-8'), None)
            
            # Combine nonce + encrypted data
            return nonce + encrypted_data
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt data using AES-256-GCM"""
        if not self._aesgcm:
            raise ValueError("Encryption not initialized")
        
        try:
            # Extract nonce (first 12 bytes) and ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Decrypt the data
            decrypted_data = self._aesgcm.decrypt(nonce, ciphertext, None)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    # API Key Management
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """Save or update user's Hugging Face API key with encryption"""
        try:
            if not self.connected or not self.pool:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("api_key must be a non-empty string")
            
            # Encrypt the API key
            encrypted_key = self._encrypt_data(api_key.strip())
            
            async with self.pool.acquire() as conn:
                # Insert or update user record
                await conn.execute("""
                    INSERT INTO users (user_id, encrypted_api_key, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (user_id) 
                    DO UPDATE SET encrypted_api_key = $2, updated_at = NOW()
                """, user_id, encrypted_key)
            
            self.secure_logger.info(f"✅ API key saved successfully for user {user_id}")
            return True
            
        except Exception as e:
            self.secure_logger.error(f"Failed to save API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """Retrieve and decrypt user's Hugging Face API key"""
        try:
            if not self.connected or not self.pool:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT encrypted_api_key FROM users WHERE user_id = $1",
                    user_id
                )
            
            if not result or not result['encrypted_api_key']:
                return None
            
            # Decrypt the API key
            decrypted_key = self._decrypt_data(result['encrypted_api_key'])
            
            self.secure_logger.info(f"✅ API key retrieved successfully for user {user_id}")
            return decrypted_key
            
        except Exception as e:
            self.secure_logger.error(f"Failed to get API key for user {user_id}: {e}")
            return None
    
    # User Data Management
    async def reset_user_database(self, user_id: int) -> bool:
        """Delete all user data including API keys and conversations"""
        try:
            if not self.connected or not self.pool:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Delete all user data (CASCADE will handle related tables)
                    await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
            
            logger.info(f"✅ User database reset completed for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset user database for user {user_id}: {e}")
            return False
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user preferences and settings"""
        try:
            if not self.connected or not self.pool:
                return {}
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT preferences FROM users WHERE user_id = $1",
                    user_id
                )
            
            if result and result['preferences']:
                return dict(result['preferences'])
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get user preferences for user {user_id}: {e}")
            return {}
    
    async def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Save user preferences and settings"""
        try:
            if not self.connected or not self.pool:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO users (user_id, preferences, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (user_id) 
                    DO UPDATE SET preferences = $2, updated_at = NOW()
                """, user_id, json.dumps(preferences))
            
            logger.info(f"✅ User preferences saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user preferences for user {user_id}: {e}")
            return False
    
    async def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """Get a specific user preference value by key"""
        try:
            if not self.connected or not self.pool:
                return None
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT preferences FROM users WHERE user_id = $1",
                    user_id
                )
            
            if result and result['preferences']:
                preferences = dict(result['preferences'])
                value = preferences.get(key)
                # Convert to string if not None, to match the interface signature
                return str(value) if value is not None else None
            
            return None
                
        except Exception as e:
            logger.error(f"Failed to get user preference '{key}' for user {user_id}: {e}")
            return None
    
    async def save_user_preference(self, user_id: int, key: str, value: str) -> bool:
        """Save a specific user preference value by key"""
        try:
            if not self.connected or not self.pool:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Get current preferences or use empty dict
            current_preferences = await self.get_user_preferences(user_id)
            
            # Update the specific key
            current_preferences[key] = value
            
            # Save the updated preferences
            return await self.save_user_preferences(user_id, current_preferences)
                
        except Exception as e:
            logger.error(f"Failed to save user preference '{key}' for user {user_id}: {e}")
            return False
    
    # Conversation Storage
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """Save conversation history with metadata"""
        try:
            if not self.connected or not self.pool:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            self._validate_conversation_data(conversation_data)
            
            async with self.pool.acquire() as conn:
                # Ensure user exists
                await conn.execute("""
                    INSERT INTO users (user_id) VALUES ($1)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                
                # Insert conversation
                await conn.execute("""
                    INSERT INTO conversations (user_id, summary, messages, started_at, last_message_at, message_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                    user_id,
                    conversation_data['summary'],
                    json.dumps(conversation_data['messages']),
                    conversation_data['started_at'],
                    conversation_data['last_message_at'],
                    conversation_data['message_count']
                )
            
            logger.info(f"✅ Conversation saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation for user {user_id}: {e}")
            return False
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
        """Get conversation summaries for history browsing with pagination"""
        try:
            if not self.connected or not self.pool:
                return []
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, summary, started_at, last_message_at, message_count
                    FROM conversations 
                    WHERE user_id = $1
                    ORDER BY last_message_at DESC
                    LIMIT $2 OFFSET $3
                """, user_id, limit, skip)
            
            conversations = []
            for row in rows:
                conversations.append({
                    'id': str(row['id']),
                    'summary': row['summary'],
                    'started_at': row['started_at'],
                    'last_message_at': row['last_message_at'],
                    'message_count': row['message_count']
                })
            
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed conversation data by ID"""
        try:
            if not self.connected or not self.pool:
                return None
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM conversations 
                    WHERE user_id = $1 AND id = $2
                """, user_id, uuid.UUID(conversation_id))
            
            if not row:
                return None
            
            return {
                'id': str(row['id']),
                'summary': row['summary'],
                'messages': json.loads(row['messages']),
                'started_at': row['started_at'],
                'last_message_at': row['last_message_at'],
                'message_count': row['message_count'],
                'created_at': row['created_at']
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation details for user {user_id}: {e}")
            return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """Delete a specific conversation by ID"""
        try:
            if not self.connected or not self.pool:
                return False
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM conversations 
                    WHERE user_id = $1 AND id = $2
                """, user_id, uuid.UUID(conversation_id))
            
            return result == "DELETE 1"
            
        except Exception as e:
            logger.error(f"Failed to delete conversation for user {user_id}: {e}")
            return False
    
    async def clear_user_history(self, user_id: int) -> bool:
        """Clear all conversation history for a user"""
        try:
            if not self.connected or not self.pool:
                return False
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM conversations WHERE user_id = $1",
                    user_id
                )
            
            logger.info(f"✅ Conversation history cleared for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear conversation history for user {user_id}: {e}")
            return False
    
    async def get_conversation_count(self, user_id: int) -> int:
        """Get total number of conversations for a user"""
        try:
            if not self.connected or not self.pool:
                return 0
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = $1",
                    user_id
                )
            
            return result or 0
            
        except Exception as e:
            logger.error(f"Failed to get conversation count for user {user_id}: {e}")
            return 0
    
    # File Storage
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """Save file data with metadata"""
        try:
            if not self.connected or not self.pool:
                return False
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                # Ensure user exists
                await conn.execute("""
                    INSERT INTO users (user_id) VALUES ($1)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                
                # Insert file
                await conn.execute("""
                    INSERT INTO files (user_id, file_id, file_data, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (user_id, file_id) 
                    DO UPDATE SET file_data = $3, metadata = $4
                """, user_id, file_id, file_data, json.dumps(metadata))
            
            logger.info(f"✅ File saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save file for user {user_id}: {e}")
            return False
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file data and metadata"""
        try:
            if not self.connected or not self.pool:
                return None
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT file_data, metadata FROM files
                    WHERE user_id = $1 AND file_id = $2
                """, user_id, file_id)
            
            if not row:
                return None
            
            return {
                'file_data': row['file_data'],
                'metadata': json.loads(row['metadata'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get file for user {user_id}: {e}")
            return None
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """Delete file data"""
        try:
            if not self.connected or not self.pool:
                return False
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM files 
                    WHERE user_id = $1 AND file_id = $2
                """, user_id, file_id)
            
            return result == "DELETE 1"
            
        except Exception as e:
            logger.error(f"Failed to delete file for user {user_id}: {e}")
            return False
    
    # Admin Data Management
    async def get_admin_data(self) -> Optional[Dict[str, Any]]:
        """Get admin system configuration data"""
        try:
            if not self.connected or not self.pool:
                return None
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT data FROM admin_system WHERE key = 'admin_config'
                """)
            
            if row and row['data']:
                return dict(row['data'])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get admin data: {e}")
            return None
    
    async def save_admin_data(self, admin_data: Dict[str, Any]) -> bool:
        """Save admin system configuration data"""
        try:
            if not self.connected or not self.pool:
                return False
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO admin_system (key, data, updated_at)
                    VALUES ('admin_config', $1, NOW())
                    ON CONFLICT (key) 
                    DO UPDATE SET data = $1, updated_at = NOW()
                """, json.dumps(admin_data))
            
            logger.info("✅ Admin data saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save admin data: {e}")
            return False
    
    async def log_admin_action(self, admin_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Log admin action for audit trail"""
        try:
            if not self.connected or not self.pool:
                return False
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO admin_logs (admin_id, action, details)
                    VALUES ($1, $2, $3)
                """, admin_id, action, json.dumps(details) if details else None)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log admin action: {e}")
            return False
    
    async def get_admin_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get admin action logs for audit trail"""
        try:
            if not self.connected or not self.pool:
                return []
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM admin_logs
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """, limit, skip)
            
            logs = []
            for row in rows:
                logs.append({
                    'id': str(row['id']),
                    'admin_id': row['admin_id'],
                    'action': row['action'],
                    'details': json.loads(row['details']) if row['details'] else None,
                    'created_at': row['created_at']
                })
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get admin logs: {e}")
            return []
    
    # Analytics and Metrics
    async def log_usage(self, user_id: int, action: str, model_used: str, tokens_used: int = 0) -> bool:
        """Log usage metrics for analytics"""
        try:
            if not self.connected or not self.pool:
                return False
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                # Ensure user exists
                await conn.execute("""
                    INSERT INTO users (user_id) VALUES ($1)
                    ON CONFLICT (user_id) DO NOTHING
                """, user_id)
                
                # Log usage
                await conn.execute("""
                    INSERT INTO usage_analytics (user_id, action, model_used, tokens_used)
                    VALUES ($1, $2, $3, $4)
                """, user_id, action, model_used, tokens_used)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log usage for user {user_id}: {e}")
            return False
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        try:
            if not self.connected or not self.pool:
                return {}
            
            user_id = self._validate_user_id(user_id)
            
            async with self.pool.acquire() as conn:
                # Get usage stats for the last N days
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_actions,
                        SUM(tokens_used) as total_tokens,
                        COUNT(DISTINCT action) as unique_actions,
                        COUNT(DISTINCT model_used) as unique_models
                    FROM usage_analytics 
                    WHERE user_id = $1 
                    AND created_at >= NOW() - INTERVAL '%s days'
                """, user_id, days)
                
                # Get action breakdown
                action_stats = await conn.fetch("""
                    SELECT action, COUNT(*) as count
                    FROM usage_analytics 
                    WHERE user_id = $1 
                    AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY action
                    ORDER BY count DESC
                """, user_id, days)
            
            return {
                'total_actions': stats['total_actions'] or 0,
                'total_tokens': stats['total_tokens'] or 0,
                'unique_actions': stats['unique_actions'] or 0,
                'unique_models': stats['unique_models'] or 0,
                'action_breakdown': {row['action']: row['count'] for row in action_stats}
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage stats for user {user_id}: {e}")
            return {}