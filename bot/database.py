"""
MongoDB database integration for user API key management
Secure storage for Telegram user IDs and Hugging Face API keys with encryption
"""

import asyncio
import base64
import os
from urllib.parse import urlparse, parse_qs
from cryptography.fernet import Fernet
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from bot.config import Config
import logging

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Handles encryption/decryption of sensitive data"""
    
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or generate one with security validation"""
        key_b64 = os.getenv('ENCRYPTION_KEY')
        is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
        
        if key_b64:
            try:
                # Validate that the environment key is a proper Fernet key
                key_bytes = key_b64.encode('utf-8')
                # Test key validity by creating a Fernet instance
                test_cipher = Fernet(key_bytes)
                # Perform a small encryption/decryption test
                test_data = b"security_validation_test"
                encrypted = test_cipher.encrypt(test_data)
                decrypted = test_cipher.decrypt(encrypted)
                if decrypted != test_data:
                    raise ValueError("Key validation failed")
                logger.info("Encryption key validated successfully")
                return key_bytes
            except Exception as e:
                error_msg = f"Invalid encryption key in environment: {e}"
                logger.error(error_msg)
                if is_production:
                    raise ValueError(f"CRITICAL SECURITY ERROR: {error_msg}. Production deployment requires valid ENCRYPTION_KEY.")
                logger.warning("Falling back to temporary key generation in development")
        
        # Production environment must have encryption key
        if is_production:
            raise ValueError(
                "CRITICAL SECURITY ERROR: ENCRYPTION_KEY environment variable is required in production. "
                "Generate a key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )
        
        # Generate new key only in development
        key = Fernet.generate_key()
        logger.warning(
            "Generated temporary encryption key for development session. "
            "Data will not persist across restarts. "
            "For production, set ENCRYPTION_KEY environment variable with: "
            "python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt a string"""
        if not data:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str | None:
        """Decrypt a string"""
        if not encrypted_data:
            return encrypted_data
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return None

# Global encryption manager
encryption_manager = EncryptionManager()

class Database:
    """MongoDB database manager for user data"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self.connected = False
    
    def _check_tls_enabled(self, mongo_uri: str) -> bool:
        """
        Safely check if TLS is enabled in MongoDB URI without mutating the original URI
        
        Args:
            mongo_uri (str): The original MongoDB connection string
            
        Returns:
            bool: True if TLS is enabled, False otherwise
        """
        try:
            # mongodb+srv:// always uses TLS by design
            if mongo_uri.startswith('mongodb+srv://'):
                return True
            
            # Parse the URI to check query parameters safely
            parsed_uri = urlparse(mongo_uri)
            query_params = parse_qs(parsed_uri.query)
            
            # Check for TLS/SSL parameters (case-insensitive)
            for key, values in query_params.items():
                key_lower = key.lower()
                if key_lower in ['tls', 'ssl'] and values:
                    # Check if any value indicates TLS is enabled
                    return any(value.lower() in ['true', '1', 'yes'] for value in values)
            
            return False
            
        except Exception as e:
            logger.warning(f"Error parsing MongoDB URI for TLS detection: {e}")
            # Default to False for safety in case of parsing errors
            return False
    
    async def connect(self):
        """Establish connection to MongoDB with TLS security validation"""
        try:
            if not Config.MONGO_URI:
                raise ValueError("MONGO_URI is not configured")
            
            # Security validation: Ensure TLS is enabled for production
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            # Safe TLS detection without corrupting the original URI
            has_tls = self._check_tls_enabled(Config.MONGO_URI)
            
            if is_production and not has_tls:
                raise ValueError(
                    "CRITICAL SECURITY ERROR: Production MongoDB connections must use TLS. "
                    "Use mongodb+srv:// connection string or add tls=true parameter."
                )
            
            if not has_tls:
                logger.warning(
                    "Database connection is not using TLS encryption. "
                    "This is acceptable for development but NOT for production."
                )
                
            self.client = AsyncIOMotorClient(Config.MONGO_URI)
            self.db = self.client.ai_assistant_bot
            self.users_collection = self.db.users
            
            # Create unique index on user_id with background creation
            try:
                await self.users_collection.create_index("user_id", unique=True, background=True)
            except Exception as e:
                logger.warning(f"Index creation warning (may already exist): {e}")
            
            # Test connection with retry logic
            max_ping_retries = 3
            for attempt in range(max_ping_retries):
                try:
                    await self.client.admin.command('ping')
                    break
                except Exception as e:
                    if attempt == max_ping_retries - 1:
                        raise e
                    await asyncio.sleep(1)
                    logger.warning(f"Ping attempt {attempt + 1} failed, retrying...")
            
            self.connected = True
            logger.info("✅ Successfully connected to MongoDB with security validation and performance optimizations")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            # Clean up partial state
            self.client = None
            self.db = None
            self.users_collection = None
            raise
    
    async def disconnect(self):
        """Close MongoDB connection gracefully"""
        if self.client:
            try:
                # Close connection gracefully
                self.client.close()
                self.connected = False
                self.client = None
                self.db = None
                self.users_collection = None
                logger.info("✅ Gracefully disconnected from MongoDB")
            except Exception as e:
                logger.error(f"Error during MongoDB disconnection: {e}")
                # Force cleanup
                self.connected = False
                self.client = None
                self.db = None
                self.users_collection = None
    
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save or update user's Hugging Face API key with encryption
        
        Args:
            user_id (int): Telegram user ID
            api_key (str): Hugging Face API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure connection before operation
            if not self.connected:
                await self.connect()
            
            # Encrypt the API key before storing
            encrypted_key = encryption_manager.encrypt(api_key)
            
            if self.users_collection is not None:
                await self.users_collection.update_one(
                    {"user_id": user_id},
                    {"$set": {"hf_api_key": encrypted_key}},
                    upsert=True
                )
            else:
                raise RuntimeError("Database not connected")
            logger.info(f"Encrypted API key saved for user {user_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Failed to save API key for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> str | None:
        """
        Retrieve and decrypt user's Hugging Face API key
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            str: Decrypted API key if found, None otherwise
        """
        try:
            # Ensure connection
            if not self.connected:
                await self.connect()
                
            if self.users_collection is not None:
                user_doc = await self.users_collection.find_one({"user_id": user_id})
            else:
                raise RuntimeError("Database not connected")
            if user_doc and user_doc.get("hf_api_key"):
                encrypted_key = user_doc.get("hf_api_key")
                # Decrypt the API key
                decrypted_key = encryption_manager.decrypt(encrypted_key)
                return decrypted_key
            return None
            
        except PyMongoError as e:
            logger.error(f"Failed to retrieve API key for user {user_id}: {e}")
            return None
    
    async def reset_user_database(self, user_id: int) -> bool:
        """
        Delete all user data from database
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure connection before operation
            if not self.connected:
                await self.connect()
                
            if self.users_collection is not None:
                result = await self.users_collection.delete_one({"user_id": user_id})
            else:
                raise RuntimeError("Database not connected")
            if result.deleted_count > 0:
                logger.info(f"User data reset for user {user_id}")
                return True
            else:
                logger.warning(f"No data found to reset for user {user_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Failed to reset user data for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error resetting user data for user {user_id}: {e}")
            return False
    
    async def get_user_count(self) -> int:
        """Get total number of users in database"""
        try:
            # Ensure connection before operation
            if not self.connected:
                await self.connect()
                
            if self.users_collection is not None:
                count = await self.users_collection.count_documents({})
            else:
                raise RuntimeError("Database not connected")
            return count
        except PyMongoError as e:
            logger.error(f"Failed to get user count: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error getting user count: {e}")
            return 0
    
    async def get_all_users(self) -> list:
        """Get all user IDs (for admin purposes)"""
        try:
            # Ensure connection before operation
            if not self.connected:
                await self.connect()
                
            if self.users_collection is not None:
                cursor = self.users_collection.find({}, {"user_id": 1, "_id": 0})
            else:
                raise RuntimeError("Database not connected")
            users = await cursor.to_list(length=None)
            return [user["user_id"] for user in users]
        except PyMongoError as e:
            logger.error(f"Failed to get all users: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting all users: {e}")
            return []

# Global database instance
db = Database()