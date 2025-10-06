"""
Supabase User Provider Implementation
Provides individual user database isolation using PostgreSQL schemas within Supabase
"""

import asyncio
import os
import base64
import secrets
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Union

# Supabase/PostgreSQL imports with proper typing
try:
    import asyncpg
    from sqlalchemy import create_engine, text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    SUPABASE_AVAILABLE = True
except ImportError:
    # Define stub types for when Supabase is not available
    asyncpg = None
    text = None
    create_engine = None
    create_async_engine = None
    AsyncSession = None
    sessionmaker = None
    SUPABASE_AVAILABLE = False

# Encryption imports
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import logging
from .base import StorageProvider
from bot.config import Config
from bot.security_utils import SecureLogger

logger = logging.getLogger(__name__)

class SupabaseUserProvider(StorageProvider):
    """
    Supabase storage provider with individual user database isolation
    
    This provider creates separate PostgreSQL schemas for each user within
    a Supabase instance, providing complete data isolation while maintaining
    scalability and manageability.
    
    Architecture:
    - Management database: Stores user-to-schema mappings and account info
    - User schemas: Individual PostgreSQL schemas (user_123456) for each user
    - Row Level Security: Additional security layer for complete isolation
    - Automatic provisioning: New schemas created automatically for new users
    """
    
    def __init__(self):
        super().__init__()
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase dependencies not available. Install with: pip install asyncpg sqlalchemy[asyncio]")
        
        # Validate that imports are available
        if text is None or create_async_engine is None:
            raise ImportError("Required SQLAlchemy components not available")
        
        # Management database connection (for user mappings)
        self.mgmt_connection = None
        self.mgmt_engine = None
        
        # User database connections pool
        self.user_connections = {}  # user_id -> connection
        self.user_engines = {}      # user_id -> engine
        
        # Configuration
        self.mgmt_database_url = None
        self.base_user_db_url = None
        
        # Encryption and security
        self._encryption_key = None
        self._aesgcm = None
        self._global_seed = None
        self.secure_logger = SecureLogger(logger)
        
        # Schema management
        self.schema_prefix = "user_"
        
    # Connection Management
    async def connect(self) -> None:
        """Establish connection to Supabase management database with comprehensive error handling"""
        import socket
        from urllib.parse import urlparse
        
        # Type assertions for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert create_async_engine is not None, "SQLAlchemy create_async_engine function not available"
        
        try:
            # Get Supabase URLs from configuration (Railway.com compatible)
            original_mgmt_url = Config.get_supabase_mgmt_url()
            original_user_url = Config.get_supabase_user_base_url()
            
            if not original_mgmt_url:
                raise ValueError("SUPABASE_MGMT_URL is not configured")
            
            logger.info(f"🔍 Using SUPABASE_MGMT_URL: {original_mgmt_url[:50]}...")
            
            # Validate URL format before attempting connection
            try:
                parsed_url = urlparse(original_mgmt_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    raise ValueError(f"Invalid SUPABASE_MGMT_URL format: {original_mgmt_url}")
                
                # Check for DNS resolution before attempting connection
                hostname = parsed_url.hostname
                if hostname:
                    try:
                        socket.gethostbyname(hostname)
                        logger.debug(f"✅ DNS resolution successful for {hostname}")
                    except socket.gaierror as dns_error:
                        raise ConnectionError(
                            f"DNS resolution failed for hostname '{hostname}': {dns_error}. "
                            "Please check your internet connection and verify the Supabase URL is correct."
                        )
                    except Exception as dns_error:
                        raise ConnectionError(f"DNS lookup error for '{hostname}': {dns_error}")
                        
            except ValueError as url_error:
                raise ValueError(f"Invalid Supabase URL configuration: {url_error}")
            
            # Convert URLs to use asyncpg driver explicitly
            def convert_to_asyncpg_url(url: str) -> str:
                """Convert PostgreSQL URL to use asyncpg driver explicitly"""
                if url.startswith('postgresql://'):
                    return url.replace('postgresql://', 'postgresql+asyncpg://', 1)
                elif url.startswith('postgres://'):
                    return url.replace('postgres://', 'postgresql+asyncpg://', 1)
                elif url.startswith('postgresql+asyncpg://'):
                    return url  # Already correct
                else:
                    # Assume it's a PostgreSQL URL without explicit scheme
                    return f'postgresql+asyncpg://{url}'
            
            self.mgmt_database_url = convert_to_asyncpg_url(original_mgmt_url)
            
            if not original_user_url:
                # Use management URL as base if not specified
                self.base_user_db_url = self.mgmt_database_url
                logger.info("Using management database as base user database")
            else:
                self.base_user_db_url = convert_to_asyncpg_url(original_user_url)
            
            logger.info("🔧 Using asyncpg driver for async operations")
            
            # Create management database connection with timeout
            try:
                self.mgmt_engine = create_async_engine(
                    self.mgmt_database_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=3600,
                    connect_args={
                        "command_timeout": 30,  # 30 second timeout for commands
                        "server_settings": {
                            "application_name": "huggingface_bot_supabase"
                        }
                    }
                )
                
                # Test management database connection with timeout
                logger.info("🔗 Testing Supabase management database connection...")
                async with self.mgmt_engine.begin() as connection:
                    await connection.execute(text("SELECT 1"))
                
                self.connected = True
                logger.info("✅ Successfully connected to Supabase management database")
                
            except asyncio.TimeoutError:
                raise ConnectionError(
                    "Supabase connection timed out after 30 seconds. "
                    "Please check your network connection and Supabase service availability."
                )
            except Exception as db_error:
                # Enhanced error classification
                error_str = str(db_error).lower()
                
                if 'timeout' in error_str or 'timed out' in error_str:
                    raise ConnectionError(f"Supabase connection timeout: {db_error}")
                elif 'connection refused' in error_str:
                    raise ConnectionError(f"Supabase server refused connection: {db_error}")
                elif 'name resolution' in error_str or 'nodename nor servname' in error_str:
                    raise ConnectionError(f"DNS resolution failed for Supabase: {db_error}")
                elif 'network' in error_str or 'unreachable' in error_str:
                    raise ConnectionError(f"Network error connecting to Supabase: {db_error}")
                elif 'authentication' in error_str or 'password' in error_str:
                    raise ConnectionError(f"Supabase authentication failed: {db_error}")
                elif 'invalid' in error_str and 'database' in error_str:
                    raise ValueError(f"Invalid Supabase database configuration: {db_error}")
                else:
                    raise ConnectionError(f"Supabase connection failed: {db_error}")
            
        except (ValueError, ConnectionError):
            # Re-raise configuration and connection errors as-is
            self.connected = False
            self._cleanup_connection_state()
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Supabase: {e}")
            self.connected = False
            self._cleanup_connection_state()
            raise ConnectionError(f"Unexpected Supabase connection error: {e}")
    
    def _cleanup_connection_state(self) -> None:
        """Clean up connection state on failure"""
        try:
            self.mgmt_engine = None
            self.mgmt_connection = None
            self.user_engines.clear()
            self.user_connections.clear()
        except Exception as e:
            logger.debug(f"Error during connection state cleanup: {e}")
    
    async def disconnect(self) -> None:
        """Close all Supabase connections gracefully"""
        try:
            # Close user connections
            for user_id, engine in self.user_engines.items():
                try:
                    await engine.dispose()
                except Exception as e:
                    logger.warning(f"Error closing user {user_id} connection: {e}")
            
            # Close management connection
            if self.mgmt_engine:
                await self.mgmt_engine.dispose()
                logger.info("✅ Supabase connections closed gracefully")
                
        except Exception as e:
            logger.warning(f"Error closing Supabase connections: {e}")
        finally:
            self.mgmt_engine = None
            self.user_engines.clear()
            self.user_connections.clear()
            self.connected = False
    
    async def initialize(self) -> None:
        """Initialize Supabase (create management tables, setup schemas, etc.)"""
        if not self.connected:
            raise ValueError("Must be connected before initialization")
        
        # Create management tables
        await self._create_management_tables()
        
        # Initialize encryption with persistent seed
        await self._initialize_encryption_with_persistent_seed()
        
        self._encryption_initialized = True
        logger.info("✅ Supabase user provider initialized successfully")
    
    async def health_check(self) -> bool:
        """Perform health check on Supabase"""
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            if not self.connected or not self.mgmt_engine:
                return False
            
            # Test management database
            async with self.mgmt_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            return True
        except Exception as e:
            logger.warning(f"Supabase health check failed: {e}")
            return False
    
    # User Schema Management
    async def _get_user_schema(self, user_id: int) -> str:
        """
        Get or create user schema name for a given user ID
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            str: Schema name for the user
        """
        user_id = self._validate_user_id(user_id)
        return f"{self.schema_prefix}{user_id}"
    
    async def _ensure_user_schema(self, user_id: int) -> str:
        """
        Ensure user schema exists, creating it if necessary
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            str: Schema name for the user
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        schema_name = await self._get_user_schema(user_id)
        
        # Check if user already exists in management database
        async with self.mgmt_engine.begin() as conn:
            result = await conn.execute(
                text("SELECT schema_name FROM user_schemas WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            existing_schema = result.fetchone()
            
            if existing_schema:
                return existing_schema[0]
        
        # Create new user schema
        await self._create_user_schema(user_id, schema_name)
        return schema_name
    
    async def _create_user_schema(self, user_id: int, schema_name: str) -> None:
        """
        Create new user schema with complete isolation
        
        Args:
            user_id (int): Telegram user ID
            schema_name (str): Schema name to create
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        async with self.mgmt_engine.begin() as conn:
            # Create the schema
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            
            # Create user-specific tables within the schema
            await self._create_user_tables(conn, schema_name)
            
            # Set up Row Level Security policies
            await self._setup_user_rls_policies(conn, schema_name, user_id)
            
            # Record the schema in management database
            await conn.execute(
                text("""
                INSERT INTO user_schemas (user_id, schema_name, created_at, active)
                VALUES (:user_id, :schema_name, :created_at, true)
                ON CONFLICT (user_id) DO UPDATE SET
                    schema_name = EXCLUDED.schema_name,
                    updated_at = :created_at
                """),
                {
                    "user_id": user_id,
                    "schema_name": schema_name,
                    "created_at": datetime.utcnow()
                }
            )
        
        logger.info(f"🏗️ Created new user schema: {schema_name} for user {user_id}")
    
    async def _create_user_tables(self, conn, schema_name: str) -> None:
        """
        Create user-specific tables within their schema
        
        Args:
            conn: Database connection
            schema_name (str): User schema name
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        # User preferences table
        await conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.preferences (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            preferences JSONB NOT NULL DEFAULT '{{}}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        
        # API keys table (encrypted)
        await conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.api_keys (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL UNIQUE,
            encrypted_api_key TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        
        # Conversations table
        await conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id INTEGER NOT NULL,
            conversation_data JSONB NOT NULL,
            summary TEXT,
            message_count INTEGER DEFAULT 0,
            started_at TIMESTAMP NOT NULL,
            last_message_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        
        # Files table
        await conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.files (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id INTEGER NOT NULL,
            file_id TEXT NOT NULL UNIQUE,
            file_data BYTEA,
            metadata JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        
        # Usage analytics table
        await conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.usage_analytics (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            model_used TEXT NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))
        
        # Create indexes for performance
        await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_conversations_user_id ON {schema_name}.conversations (user_id)"))
        await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_conversations_started_at ON {schema_name}.conversations (started_at)"))
        await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_files_user_id ON {schema_name}.files (user_id)"))
        await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_usage_user_id ON {schema_name}.usage_analytics (user_id, created_at)"))
    
    async def _setup_user_rls_policies(self, conn, schema_name: str, user_id: int) -> None:
        """
        Set up Row Level Security policies for complete user data isolation
        
        Args:
            conn: Database connection
            schema_name (str): User schema name
            user_id (int): User ID for the policies
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        # Enable RLS on all user tables
        tables = ["preferences", "api_keys", "conversations", "files", "usage_analytics"]
        
        for table in tables:
            # Enable RLS
            await conn.execute(text(f"ALTER TABLE {schema_name}.{table} ENABLE ROW LEVEL SECURITY"))
            
            # Create policy to only allow access to own data
            await conn.execute(text(f"""
            CREATE POLICY IF NOT EXISTS user_isolation_policy_{user_id}_{table}
            ON {schema_name}.{table}
            FOR ALL
            TO public
            USING (user_id = {user_id})
            WITH CHECK (user_id = {user_id})
            """))
    
    async def _create_management_tables(self) -> None:
        """Create management tables for user schema tracking"""
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        async with self.mgmt_engine.begin() as conn:
            # User schemas tracking table
            await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_schemas (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL UNIQUE,
                schema_name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT true
            )
            """))
            
            # Management system metadata
            await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS system_metadata (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """))
            
            # Create indexes
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_schemas_user_id ON user_schemas (user_id)"))
            await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_schemas_active ON user_schemas (active)"))
    
    async def _get_user_engine(self, user_id: int) -> Any:
        """
        Get or create database engine for a specific user
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Database engine for the user
        """
        # Type assertion for SQLAlchemy imports
        assert create_async_engine is not None, "SQLAlchemy create_async_engine function not available"
        
        if user_id in self.user_engines:
            return self.user_engines[user_id]
        
        # Ensure user schema exists
        schema_name = await self._ensure_user_schema(user_id)
        
        # Create engine with schema search path
        # Ensure the base URL uses asyncpg driver
        base_url = self.base_user_db_url
        if base_url is None:
            raise ValueError("base_user_db_url is not configured")
        
        if not base_url.startswith('postgresql+asyncpg://'):
            # Convert to asyncpg if not already
            if base_url.startswith('postgresql://'):
                base_url = base_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
            elif base_url.startswith('postgres://'):
                base_url = base_url.replace('postgres://', 'postgresql+asyncpg://', 1)
        
        user_db_url = f"{base_url}?options=-csearch_path%3D{schema_name}"
        
        user_engine = create_async_engine(
            user_db_url,
            pool_size=2,
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=3600
        )
        
        self.user_engines[user_id] = user_engine
        return user_engine
    
    # API Key Management  
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save user's API key in their isolated schema
        
        Args:
            user_id (int): Telegram user ID
            api_key (str): API key to save (will be encrypted)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("api_key must be a non-empty string")
            
            # Get user database engine
            user_engine = await self._get_user_engine(user_id)
            
            # Encrypt the API key
            encrypted_api_key = self._encrypt_api_key(api_key.strip(), user_id)
            
            # Save to user's schema
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO api_keys (user_id, encrypted_api_key)
                    VALUES (:user_id, :encrypted_api_key)
                    ON CONFLICT (user_id) DO UPDATE SET
                        encrypted_api_key = EXCLUDED.encrypted_api_key,
                        updated_at = CURRENT_TIMESTAMP
                    """),
                    {
                        "user_id": user_id,
                        "encrypted_api_key": encrypted_api_key
                    }
                )
            
            self.secure_logger.info(f"🔒 Successfully saved encrypted API key for user {user_id} in isolated schema")
            return True
            
        except Exception as e:
            self.secure_logger.error(f"Error saving API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        Retrieve and decrypt user's API key from their isolated schema
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Optional[str]: Decrypted API key or None if not found
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            
            # Get user database engine
            user_engine = await self._get_user_engine(user_id)
            
            # Retrieve from user's schema
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT encrypted_api_key FROM api_keys WHERE user_id = :user_id"),
                    {"user_id": user_id}
                )
                row = result.fetchone()
                
                if not row:
                    return None
                
                encrypted_api_key = row[0]
                
                # Decrypt the API key
                decrypted_api_key = self._decrypt_api_key(encrypted_api_key, user_id)
                
                self.secure_logger.info(f"🔓 Successfully retrieved API key for user {user_id} from isolated schema")
                return decrypted_api_key
                
        except Exception as e:
            self.secure_logger.error(f"Error retrieving API key for user {user_id}: {e}")
            return None
    
    # User Data Management Implementation
    
    async def reset_user_database(self, user_id: int) -> bool:
        """
        Reset all data for a user by dropping and recreating their schema
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        try:
            user_id = self._validate_user_id(user_id)
            schema_name = await self._get_user_schema(user_id)
            
            async with self.mgmt_engine.begin() as conn:
                # Drop the entire user schema (CASCADE removes all tables)
                await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
                
                # Remove from management tracking
                await conn.execute(
                    text("DELETE FROM user_schemas WHERE user_id = :user_id"),
                    {"user_id": user_id}
                )
            
            # Remove from user engine cache
            if user_id in self.user_engines:
                await self.user_engines[user_id].dispose()
                del self.user_engines[user_id]
            
            self.secure_logger.info(f"🗑️ Successfully reset user database for user {user_id}")
            return True
            
        except Exception as e:
            self.secure_logger.error(f"Error resetting user database for user {user_id}: {e}")
            return False
    
    async def get_user_data(self, user_id: int, data_key: str) -> Any:
        """
        Get user-specific data with encryption and isolation
        
        Args:
            user_id (int): Telegram user ID
            data_key (str): Key to identify the data
            
        Returns:
            Any: User data or None if not found
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            if not data_key or not isinstance(data_key, str):
                raise ValueError("data_key must be a non-empty string")
                
            user_engine = await self._get_user_engine(user_id)
            
            # Ensure user_data table exists in user schema
            await self._ensure_user_data_table(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT data, encrypted FROM user_data WHERE user_id = :user_id AND data_key = :data_key"),
                    {"user_id": user_id, "data_key": data_key}
                )
                row = result.fetchone()
                
                if not row:
                    return None
                
                data = row[0]
                encrypted = row[1]
                
                # Decrypt data if it was encrypted
                if encrypted and data:
                    try:
                        import json
                        decrypted_data = self._decrypt_data(data, user_id, "user_data")
                        # Try to deserialize if it's JSON
                        try:
                            return json.loads(decrypted_data)
                        except (json.JSONDecodeError, TypeError):
                            return decrypted_data
                    except Exception as e:
                        logger.error(f"Failed to decrypt user data for user {user_id}, key {data_key}: {e}")
                        return None
                
                return data
                
        except Exception as e:
            logger.error(f"Failed to get user data for user {user_id}, key {data_key}: {e}")
            return None
    
    async def save_user_data(self, user_id: int, data_key: str, data_value: Any, encrypt: bool = True) -> bool:
        """
        Save user-specific data with encryption and isolation
        
        Args:
            user_id (int): Telegram user ID
            data_key (str): Key to identify the data
            data_value (Any): Data to save
            encrypt (bool): Whether to encrypt the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            if not data_key or not isinstance(data_key, str):
                raise ValueError("data_key must be a non-empty string")
                
            user_engine = await self._get_user_engine(user_id)
            
            # Ensure user_data table exists in user schema
            await self._ensure_user_data_table(user_id)
            
            # Prepare data for storage
            storage_data = data_value
            encrypted_flag = False
            
            if encrypt:
                try:
                    # Serialize data if needed
                    import json
                    if not isinstance(data_value, str):
                        serialized_data = json.dumps(data_value, default=str)
                    else:
                        serialized_data = data_value
                    
                    # Encrypt the data
                    storage_data = self._encrypt_data(serialized_data, user_id, "user_data")
                    encrypted_flag = True
                    
                except Exception as e:
                    logger.warning(f"Failed to encrypt user data for user {user_id}, key {data_key}: {e}")
                    # Fall back to unencrypted storage
                    storage_data = data_value
                    encrypted_flag = False
            
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO user_data (user_id, data_key, data, encrypted, updated_at)
                    VALUES (:user_id, :data_key, :data, :encrypted, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, data_key) DO UPDATE SET
                        data = EXCLUDED.data,
                        encrypted = EXCLUDED.encrypted,
                        updated_at = EXCLUDED.updated_at
                    """),
                    {
                        "user_id": user_id,
                        "data_key": data_key,
                        "data": storage_data,
                        "encrypted": encrypted_flag
                    }
                )
            
            logger.info(f"✅ Saved user data for user {user_id}, key {data_key} (encrypted: {encrypted_flag})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user data for user {user_id}, key {data_key}: {e}")
            return False
    
    async def delete_user_data(self, user_id: int, data_key: str) -> bool:
        """
        Delete specific user data with security validation
        
        Args:
            user_id (int): Telegram user ID
            data_key (str): Key to identify the data to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            if not data_key or not isinstance(data_key, str):
                raise ValueError("data_key must be a non-empty string")
                
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    DELETE FROM user_data 
                    WHERE user_id = :user_id AND data_key = :data_key
                    """),
                    {
                        "user_id": user_id,
                        "data_key": data_key
                    }
                )
                
                # Check if any rows were deleted
                deleted = result.rowcount > 0
                
                if deleted:
                    logger.info(f"✅ Deleted user data for user {user_id}, key {data_key}")
                else:
                    logger.debug(f"No data found to delete for user {user_id}, key {data_key}")
                
                return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete user data for user {user_id}, key {data_key}: {e}")
            return False
    
    async def _ensure_user_data_table(self, user_id: int) -> None:
        """
        Ensure user_data table exists in the user's schema
        
        Args:
            user_id (int): Telegram user ID
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_engine = await self._get_user_engine(user_id)
            schema_name = await self._get_user_schema(user_id)
            
            async with user_engine.begin() as conn:
                # Create user_data table if it doesn't exist
                await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {schema_name}.user_data (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    data_key TEXT NOT NULL,
                    data TEXT,
                    encrypted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, data_key)
                )
                """))
                
                # Create index for performance
                await conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_user_data_user_key ON {schema_name}.user_data (user_id, data_key)"))
                
        except Exception as e:
            logger.error(f"Failed to ensure user_data table for user {user_id}: {e}")
            raise
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Get user preferences from their isolated schema
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Dict[str, Any]: User preferences
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT preferences FROM preferences WHERE user_id = :user_id ORDER BY updated_at DESC LIMIT 1"),
                    {"user_id": user_id}
                )
                row = result.fetchone()
                
                if row:
                    return row[0] or {}  # row[0] is the JSONB preferences column
                
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving preferences for user {user_id}: {e}")
            return {}
    
    async def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences in their isolated schema
        
        Args:
            user_id (int): Telegram user ID
            preferences (Dict[str, Any]): User preferences to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO preferences (user_id, preferences, updated_at)
                    VALUES (:user_id, :preferences, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE SET
                        preferences = EXCLUDED.preferences,
                        updated_at = EXCLUDED.updated_at
                    """),
                    {
                        "user_id": user_id,
                        "preferences": preferences
                    }
                )
            
            logger.info(f"✅ Saved preferences for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving preferences for user {user_id}: {e}")
            return False
    
    async def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """Get a specific user preference value by key from their isolated schema"""
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT preferences FROM preferences WHERE user_id = :user_id ORDER BY updated_at DESC LIMIT 1"),
                    {"user_id": user_id}
                )
                row = result.fetchone()
                
                if row and row[0]:
                    preferences = row[0]
                    value = preferences.get(key)
                    # Convert to string if not None, to match the interface signature
                    return str(value) if value is not None else None
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving preference '{key}' for user {user_id}: {e}")
            return None
    
    async def save_user_preference(self, user_id: int, key: str, value: str) -> bool:
        """Save a specific user preference value by key in their isolated schema"""
        try:
            user_id = self._validate_user_id(user_id)
            
            # Get current preferences or use empty dict
            current_preferences = await self.get_user_preferences(user_id)
            
            # Update the specific key
            current_preferences[key] = value
            
            # Save the updated preferences
            return await self.save_user_preferences(user_id, current_preferences)
                
        except Exception as e:
            logger.error(f"Error saving preference '{key}' for user {user_id}: {e}")
            return False
    
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """
        Save conversation history in user's isolated schema
        
        Args:
            user_id (int): Telegram user ID
            conversation_data (Dict[str, Any]): Conversation data including messages, summary, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            self._validate_conversation_data(conversation_data)
            user_engine = await self._get_user_engine(user_id)
            
            conversation_id = conversation_data.get('conversation_id', str(uuid.uuid4()))
            
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO conversations (
                        id, user_id, conversation_data, summary, message_count,
                        started_at, last_message_at
                    )
                    VALUES (
                        :conversation_id, :user_id, :conversation_data, :summary, 
                        :message_count, :started_at, :last_message_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        conversation_data = EXCLUDED.conversation_data,
                        summary = EXCLUDED.summary,
                        message_count = EXCLUDED.message_count,
                        last_message_at = EXCLUDED.last_message_at,
                        updated_at = CURRENT_TIMESTAMP
                    """),
                    {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "conversation_data": conversation_data,
                        "summary": conversation_data.get('summary', ''),
                        "message_count": conversation_data.get('message_count', 0),
                        "started_at": conversation_data.get('started_at', datetime.utcnow()),
                        "last_message_at": conversation_data.get('last_message_at', datetime.utcnow())
                    }
                )
            
            logger.info(f"✅ Saved conversation for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation for user {user_id}: {e}")
            return False
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversation summaries for history browsing with pagination
        
        Args:
            user_id (int): Telegram user ID
            limit (int): Maximum number of conversations to return
            skip (int): Number of conversations to skip (for pagination)
        
        Returns:
            List[Dict[str, Any]]: List of conversation summaries
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    SELECT id, summary, message_count, started_at, last_message_at
                    FROM conversations
                    WHERE user_id = :user_id
                    ORDER BY last_message_at DESC
                    LIMIT :limit OFFSET :skip
                    """),
                    {
                        "user_id": user_id,
                        "limit": limit,
                        "skip": skip
                    }
                )
                
                conversations = []
                for row in result:
                    conversations.append({
                        'conversation_id': str(row[0]),
                        'summary': row[1],
                        'message_count': row[2],
                        'started_at': row[3].isoformat() if row[3] else None,
                        'last_message_at': row[4].isoformat() if row[4] else None
                    })
                
                return conversations
                
        except Exception as e:
            logger.error(f"Error retrieving conversations for user {user_id}: {e}")
            return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed conversation data by ID from user's isolated schema
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): Conversation identifier
            
        Returns:
            Optional[Dict[str, Any]]: Full conversation data or None if not found
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    SELECT conversation_data, summary, message_count, started_at, last_message_at
                    FROM conversations
                    WHERE user_id = :user_id AND id = :conversation_id
                    """),
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id
                    }
                )
                row = result.fetchone()
                
                if row:
                    conversation_data = row[0]  # JSONB data
                    conversation_data.update({
                        'conversation_id': conversation_id,
                        'summary': row[1],
                        'message_count': row[2],
                        'started_at': row[3].isoformat() if row[3] else None,
                        'last_message_at': row[4].isoformat() if row[4] else None
                    })
                    return conversation_data
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving conversation details for user {user_id}, conversation {conversation_id}: {e}")
            return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """
        Delete a specific conversation by ID from user's isolated schema
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): Conversation identifier
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    DELETE FROM conversations
                    WHERE user_id = :user_id AND id = :conversation_id
                    """),
                    {
                        "user_id": user_id,
                        "conversation_id": conversation_id
                    }
                )
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deleting conversation for user {user_id}, conversation {conversation_id}: {e}")
            return False
    
    async def clear_user_history(self, user_id: int) -> bool:
        """
        Clear all conversation history for a user in their isolated schema
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("DELETE FROM conversations WHERE user_id = :user_id"),
                    {"user_id": user_id}
                )
            
            logger.info(f"🗑️ Cleared conversation history for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing history for user {user_id}: {e}")
            return False
    
    async def get_conversation_count(self, user_id: int) -> int:
        """
        Get total number of conversations for a user
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            int: Number of conversations (0 if error)
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT COUNT(*) FROM conversations WHERE user_id = :user_id"),
                    {"user_id": user_id}
                )
                count = result.scalar()
                return count or 0
                
        except Exception as e:
            logger.error(f"Error getting conversation count for user {user_id}: {e}")
            return 0
    
    # File Storage Implementation
    
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """
        Save file data with metadata in user's isolated schema
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            file_data (bytes): File content
            metadata (Dict[str, Any]): File metadata (name, type, size, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO files (user_id, file_id, file_data, metadata)
                    VALUES (:user_id, :file_id, :file_data, :metadata)
                    ON CONFLICT (file_id) DO UPDATE SET
                        file_data = EXCLUDED.file_data,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    """),
                    {
                        "user_id": user_id,
                        "file_id": file_id,
                        "file_data": file_data,
                        "metadata": metadata
                    }
                )
            
            logger.info(f"📁 Saved file {file_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving file for user {user_id}: {e}")
            return False
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file data and metadata from user's isolated schema
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            
        Returns:
            Optional[Dict[str, Any]]: File data and metadata or None if not found
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    SELECT file_data, metadata, created_at, updated_at
                    FROM files
                    WHERE user_id = :user_id AND file_id = :file_id
                    """),
                    {
                        "user_id": user_id,
                        "file_id": file_id
                    }
                )
                row = result.fetchone()
                
                if row:
                    return {
                        'file_id': file_id,
                        'file_data': row[0],
                        'metadata': row[1],
                        'created_at': row[2].isoformat() if row[2] else None,
                        'updated_at': row[3].isoformat() if row[3] else None
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving file for user {user_id}: {e}")
            return None
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """
        Delete file data from user's isolated schema
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    DELETE FROM files
                    WHERE user_id = :user_id AND file_id = :file_id
                    """),
                    {
                        "user_id": user_id,
                        "file_id": file_id
                    }
                )
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deleting file for user {user_id}: {e}")
            return False
    
    # Admin Data Management (using management database)
    
    async def get_admin_data(self) -> Optional[Dict[str, Any]]:
        """
        Get admin system configuration data from management database
        
        Returns:
            Optional[Dict[str, Any]]: Admin data including user list, settings, etc.
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        try:
            async with self.mgmt_engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT value FROM system_metadata WHERE key = 'admin_config'")
                )
                row = result.fetchone()
                
                if row:
                    return row[0]  # JSONB value
                
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving admin data: {e}")
            return None
    
    async def save_admin_data(self, admin_data: Dict[str, Any]) -> bool:
        """
        Save admin system configuration data to management database
        
        Args:
            admin_data (Dict[str, Any]): Admin data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        try:
            async with self.mgmt_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO system_metadata (key, value, updated_at)
                    VALUES ('admin_config', :admin_data, CURRENT_TIMESTAMP)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                    """),
                    {"admin_data": admin_data}
                )
            
            logger.info("✅ Saved admin configuration data")
            return True
            
        except Exception as e:
            logger.error(f"Error saving admin data: {e}")
            return False
    
    async def log_admin_action(self, admin_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log admin action for audit trail in management database
        
        Args:
            admin_id (int): Admin user ID
            action (str): Action performed
            details (Dict[str, Any]): Additional action details
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        try:
            # Create admin logs table if it doesn't exist
            async with self.mgmt_engine.begin() as conn:
                await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS admin_logs (
                    id SERIAL PRIMARY KEY,
                    admin_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    details JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))
                
                await conn.execute(
                    text("""
                    INSERT INTO admin_logs (admin_id, action, details)
                    VALUES (:admin_id, :action, :details)
                    """),
                    {
                        "admin_id": admin_id,
                        "action": action,
                        "details": details or {}
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging admin action: {e}")
            return False
    
    async def get_admin_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get admin action logs for audit trail from management database
        
        Args:
            limit (int): Maximum number of logs to return
            skip (int): Number of logs to skip (for pagination)
            
        Returns:
            List[Dict[str, Any]]: List of admin action logs
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        try:
            async with self.mgmt_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    SELECT id, admin_id, action, details, created_at
                    FROM admin_logs
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :skip
                    """),
                    {
                        "limit": limit,
                        "skip": skip
                    }
                )
                
                logs = []
                for row in result:
                    logs.append({
                        'id': row[0],
                        'admin_id': row[1],
                        'action': row[2],
                        'details': row[3] or {},
                        'created_at': row[4].isoformat() if row[4] else None
                    })
                
                return logs
                
        except Exception as e:
            logger.error(f"Error retrieving admin logs: {e}")
            return []
    
    # Analytics and Usage Tracking
    
    async def log_usage(self, user_id: int, action: str, model_used: str, tokens_used: int = 0) -> bool:
        """
        Log usage metrics for analytics in user's isolated schema
        
        Args:
            user_id (int): Telegram user ID
            action (str): Action performed (text_generation, image_generation, etc.)
            model_used (str): AI model used
            tokens_used (int): Number of tokens consumed
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            async with user_engine.begin() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO usage_analytics (user_id, action, model_used, tokens_used)
                    VALUES (:user_id, :action, :model_used, :tokens_used)
                    """),
                    {
                        "user_id": user_id,
                        "action": action,
                        "model_used": model_used,
                        "tokens_used": tokens_used
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging usage for user {user_id}: {e}")
            return False
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get usage statistics for a user from their isolated schema
        
        Args:
            user_id (int): Telegram user ID
            days (int): Number of days to look back
            
        Returns:
            Dict[str, Any]: Usage statistics
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        
        try:
            user_id = self._validate_user_id(user_id)
            user_engine = await self._get_user_engine(user_id)
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with user_engine.begin() as conn:
                result = await conn.execute(
                    text("""
                    SELECT 
                        action,
                        model_used,
                        COUNT(*) as request_count,
                        SUM(tokens_used) as total_tokens,
                        AVG(tokens_used) as avg_tokens
                    FROM usage_analytics
                    WHERE user_id = :user_id AND created_at >= :cutoff_date
                    GROUP BY action, model_used
                    ORDER BY request_count DESC
                    """),
                    {
                        "user_id": user_id,
                        "cutoff_date": cutoff_date
                    }
                )
                
                stats = {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'by_action': {},
                    'by_model': {},
                    'period_days': days
                }
                
                for row in result:
                    action, model, req_count, total_tokens, avg_tokens = row
                    stats['total_requests'] += req_count
                    stats['total_tokens'] += total_tokens or 0
                    
                    if action not in stats['by_action']:
                        stats['by_action'][action] = {'requests': 0, 'tokens': 0}
                    stats['by_action'][action]['requests'] += req_count
                    stats['by_action'][action]['tokens'] += total_tokens or 0
                    
                    if model not in stats['by_model']:
                        stats['by_model'][model] = {'requests': 0, 'tokens': 0}
                    stats['by_model'][model]['requests'] += req_count
                    stats['by_model'][model]['tokens'] += total_tokens or 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Error retrieving usage stats for user {user_id}: {e}")
            return {'error': str(e)}
    
    # Encryption Implementation (Full AESGCM with user-specific keys)
    
    async def _initialize_encryption_with_persistent_seed(self) -> None:
        """
        Initialize encryption system with persistent global seed
        Uses environment variable or creates/stores a new seed in the management database
        """
        # Type assertion for SQLAlchemy imports
        assert text is not None, "SQLAlchemy text function not available"
        assert self.mgmt_engine is not None, "Management engine not initialized"
        
        try:
            # Try to get encryption seed from environment variable
            env_seed = os.getenv('ENCRYPTION_SEED')
            
            if env_seed:
                self._global_seed = env_seed
                logger.info("🔐 Using encryption seed from environment variables (secure)")
            else:
                # Try to get seed from management database
                async with self.mgmt_engine.begin() as conn:
                    result = await conn.execute(
                        text("SELECT value FROM system_metadata WHERE key = 'encryption_seed'")
                    )
                    row = result.fetchone()
                    
                    if row and row[0] and isinstance(row[0], dict) and 'seed' in row[0]:
                        self._global_seed = row[0]['seed']
                        logger.info("🔐 Retrieved encryption seed from management database")
                    else:
                        # Generate new seed and store in management database
                        self._global_seed = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii')
                        
                        await conn.execute(
                            text("""
                            INSERT INTO system_metadata (key, value, updated_at)
                            VALUES ('encryption_seed', :seed_data, CURRENT_TIMESTAMP)
                            ON CONFLICT (key) DO UPDATE SET
                                value = EXCLUDED.value,
                                updated_at = EXCLUDED.updated_at
                            """),
                            {
                                "seed_data": {
                                    'seed': self._global_seed,
                                    'created_at': datetime.utcnow().isoformat(),
                                    'description': 'Global encryption seed for user data protection'
                                }
                            }
                        )
                        
                        logger.info("🔐 Generated and stored new encryption seed in management database")
            
            # Initialize AESGCM
            self._derive_master_key()
            
            logger.info("🔒 Encryption system initialized with AESGCM-256")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Encryption initialization failed: {e}")
            raise RuntimeError(f"Encryption system initialization failed: {e}")
    
    def _derive_master_key(self) -> None:
        """
        Derive master encryption key from global seed using PBKDF2
        """
        if not self._global_seed:
            raise ValueError("Global seed not initialized")
        
        # Use PBKDF2 with static salt for deterministic key derivation
        salt = b"supabase_user_provider_2025"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key for AESGCM
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
        )
        
        master_key = kdf.derive(self._global_seed.encode('utf-8'))
        self._aesgcm = AESGCM(master_key)
        
        logger.debug("🔑 Master encryption key derived successfully")
    
    def _derive_user_key(self, user_id: int) -> bytes:
        """
        Derive user-specific encryption key from master key and user ID
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bytes: User-specific 256-bit encryption key
        """
        if not self._aesgcm:
            raise ValueError("Encryption system not initialized")
        
        # Create user-specific salt using user ID
        user_salt = f"user_{user_id}_salt_2025".encode('utf-8')
        
        # Use PBKDF2 with user-specific salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=user_salt,
            iterations=50000,  # Slightly fewer iterations for performance
        )
        
        return kdf.derive(f"{self._global_seed}_{user_id}".encode('utf-8'))
    
    def _encrypt_api_key(self, api_key: str, user_id: int) -> str:
        """
        Encrypt API key with user-specific AESGCM encryption
        
        Args:
            api_key (str): Plain text API key
            user_id (int): Telegram user ID for user-specific encryption
            
        Returns:
            str: Base64-encoded encrypted data (nonce + ciphertext)
            
        Raises:
            ValueError: If encryption fails
        """
        try:
            if not api_key or not isinstance(api_key, str):
                raise ValueError("Invalid API key for encryption")
            
            # Derive user-specific key
            user_key = self._derive_user_key(user_id)
            user_aesgcm = AESGCM(user_key)
            
            # Generate random nonce (12 bytes for AESGCM)
            nonce = secrets.token_bytes(12)
            
            # Encrypt with additional authenticated data (AAD) using user ID
            aad = f"user_{user_id}_api_key".encode('utf-8')
            ciphertext = user_aesgcm.encrypt(nonce, api_key.encode('utf-8'), aad)
            
            # Combine nonce + ciphertext and encode
            encrypted_data = nonce + ciphertext
            return base64.urlsafe_b64encode(encrypted_data).decode('ascii')
            
        except Exception as e:
            logger.error(f"❌ API key encryption failed for user {user_id}: {e}")
            raise ValueError(f"Encryption failed: {e}")
    
    def _decrypt_api_key(self, encrypted_api_key: str, user_id: int) -> str:
        """
        Decrypt API key with user-specific AESGCM decryption
        
        Args:
            encrypted_api_key (str): Base64-encoded encrypted data
            user_id (int): Telegram user ID for user-specific decryption
            
        Returns:
            str: Decrypted API key
            
        Raises:
            ValueError: If decryption fails
        """
        try:
            if not encrypted_api_key or not isinstance(encrypted_api_key, str):
                raise ValueError("Invalid encrypted API key for decryption")
            
            # Decode from base64
            encrypted_data = base64.urlsafe_b64decode(encrypted_api_key.encode('ascii'))
            
            if len(encrypted_data) < 12:
                raise ValueError("Invalid encrypted data length")
            
            # Extract nonce (first 12 bytes) and ciphertext (remainder)
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Derive user-specific key
            user_key = self._derive_user_key(user_id)
            user_aesgcm = AESGCM(user_key)
            
            # Decrypt with AAD verification
            aad = f"user_{user_id}_api_key".encode('utf-8')
            decrypted_data = user_aesgcm.decrypt(nonce, ciphertext, aad)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ API key decryption failed for user {user_id}: {e}")
            raise ValueError(f"Decryption failed: {e}")
    
    def _encrypt_sensitive_data(self, data: str, user_id: int, data_type: str = "generic") -> str:
        """
        Encrypt any sensitive data with user-specific AESGCM encryption
        
        Args:
            data (str): Plain text data to encrypt
            user_id (int): Telegram user ID for user-specific encryption
            data_type (str): Type of data for AAD context
            
        Returns:
            str: Base64-encoded encrypted data
        """
        try:
            if not data or not isinstance(data, str):
                raise ValueError("Invalid data for encryption")
            
            # Derive user-specific key
            user_key = self._derive_user_key(user_id)
            user_aesgcm = AESGCM(user_key)
            
            # Generate random nonce
            nonce = secrets.token_bytes(12)
            
            # Encrypt with data-type-specific AAD
            aad = f"user_{user_id}_{data_type}".encode('utf-8')
            ciphertext = user_aesgcm.encrypt(nonce, data.encode('utf-8'), aad)
            
            # Combine and encode
            encrypted_data = nonce + ciphertext
            return base64.urlsafe_b64encode(encrypted_data).decode('ascii')
            
        except Exception as e:
            logger.error(f"❌ Data encryption failed for user {user_id}: {e}")
            raise ValueError(f"Encryption failed: {e}")
    
    def _decrypt_sensitive_data(self, encrypted_data: str, user_id: int, data_type: str = "generic") -> str:
        """
        Decrypt sensitive data with user-specific AESGCM decryption
        
        Args:
            encrypted_data (str): Base64-encoded encrypted data
            user_id (int): Telegram user ID for user-specific decryption
            data_type (str): Type of data for AAD context
            
        Returns:
            str: Decrypted data
        """
        try:
            if not encrypted_data or not isinstance(encrypted_data, str):
                raise ValueError("Invalid encrypted data for decryption")
            
            # Decode from base64
            raw_data = base64.urlsafe_b64decode(encrypted_data.encode('ascii'))
            
            if len(raw_data) < 12:
                raise ValueError("Invalid encrypted data length")
            
            # Extract nonce and ciphertext
            nonce = raw_data[:12]
            ciphertext = raw_data[12:]
            
            # Derive user-specific key
            user_key = self._derive_user_key(user_id)
            user_aesgcm = AESGCM(user_key)
            
            # Decrypt with AAD verification
            aad = f"user_{user_id}_{data_type}".encode('utf-8')
            decrypted_data = user_aesgcm.decrypt(nonce, ciphertext, aad)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ Data decryption failed for user {user_id}: {e}")
            raise ValueError(f"Decryption failed: {e}")
    
    def _secure_delete(self, sensitive_data: Any) -> None:
        """
        Securely delete sensitive data from memory
        
        Args:
            sensitive_data: Data to securely delete (string, bytes, etc.)
        """
        try:
            if isinstance(sensitive_data, str):
                # Overwrite string data (Python optimization may prevent this)
                sensitive_data = '0' * len(sensitive_data)
            elif isinstance(sensitive_data, bytearray):
                # For bytearray, we can clear the contents
                for i in range(len(sensitive_data)):
                    sensitive_data[i] = 0
            # Note: bytes objects are immutable, so we can't clear them
        except Exception:
            # Secure deletion may fail due to Python memory management
            # This is a best-effort attempt
            pass