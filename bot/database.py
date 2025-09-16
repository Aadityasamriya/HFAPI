"""
MongoDB database integration for AI Assistant Telegram Bot
Basic database connectivity for future use
"""

import asyncio
import os
import base64
import secrets
from urllib.parse import urlparse, parse_qs
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError
from bot.config import Config
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class Database:
    """MongoDB database manager for user data"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self._encryption_key = None
        self._aesgcm = None
    
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
    
    def _get_encryption_key(self) -> bytes:
        """
        Get or generate a secure encryption key for API key protection
        
        Returns:
            bytes: 32-byte AES-256 encryption key
        """
        try:
            # Try to get key from environment variable first
            env_key = os.getenv('API_ENCRYPTION_KEY')
            if env_key:
                # If provided as base64, decode it
                try:
                    decoded_key = base64.b64decode(env_key)
                    if len(decoded_key) == 32:
                        logger.info("✅ Using API encryption key from environment variable")
                        return decoded_key
                except Exception:
                    pass
                
                # If provided as hex, decode it  
                try:
                    if len(env_key) == 64:
                        decoded_key = bytes.fromhex(env_key)
                        logger.info("✅ Using API encryption key from environment variable (hex)")
                        return decoded_key
                except Exception:
                    pass
            
            # Generate a secure key using PBKDF2 with a deterministic salt based on MongoDB URI
            # This ensures the same key is generated across restarts while being cryptographically secure
            if not Config.MONGO_URI:
                raise ValueError("Cannot generate encryption key without MONGO_URI")
                
            # Use PBKDF2 with MongoDB URI as password and a static salt for deterministic key generation
            password = Config.MONGO_URI.encode('utf-8')
            salt = b'telegram_bot_api_key_encryption_2024'  # Fixed salt for deterministic keys
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits for AES-256
                salt=salt,
                iterations=100000,  # Strong iteration count
            )
            
            derived_key = kdf.derive(password)
            logger.info("✅ Generated secure encryption key using PBKDF2-SHA256")
            logger.warning("🔐 PRODUCTION TIP: Set API_ENCRYPTION_KEY environment variable for enhanced security")
            return derived_key
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to generate encryption key: {e}")
            raise ValueError(f"Cannot initialize encryption: {e}")
    
    def _initialize_encryption(self):
        """Initialize AES-GCM encryption with secure key"""
        try:
            if not self._encryption_key:
                self._encryption_key = self._get_encryption_key()
            
            if not self._aesgcm:
                self._aesgcm = AESGCM(self._encryption_key)
                
            logger.info("🔒 AES-256-GCM encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize encryption: {e}")
            raise
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """
        Encrypt API key using AES-256-GCM
        
        Args:
            api_key (str): Plaintext API key
            
        Returns:
            str: Base64-encoded encrypted API key with nonce
            
        Raises:
            ValueError: If encryption fails
        """
        try:
            if not api_key or not isinstance(api_key, str):
                raise ValueError("API key must be a non-empty string")
                
            self._initialize_encryption()
            
            # Ensure cipher was properly initialized
            if self._aesgcm is None:
                raise ValueError("Failed to initialize encryption cipher")
            
            # Generate a random 12-byte nonce for GCM
            nonce = secrets.token_bytes(12)
            
            # Encrypt the API key
            encrypted_data = self._aesgcm.encrypt(nonce, api_key.encode('utf-8'), None)
            
            # Combine nonce + encrypted data for storage
            combined = nonce + encrypted_data
            
            # Encode to base64 for storage in MongoDB
            encoded = base64.b64encode(combined).decode('ascii')
            
            logger.info("🔒 API key encrypted successfully using AES-256-GCM")
            return encoded
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to encrypt API key: {e}")
            raise ValueError(f"Encryption failed: {e}")
    
    def _decrypt_api_key(self, encrypted_api_key: str) -> str:
        """
        Decrypt API key using AES-256-GCM with backward compatibility
        
        Args:
            encrypted_api_key (str): Base64-encoded encrypted API key with nonce
            
        Returns:
            str: Decrypted plaintext API key
            
        Raises:
            ValueError: If decryption fails
        """
        try:
            if not encrypted_api_key or not isinstance(encrypted_api_key, str):
                raise ValueError("Encrypted API key must be a non-empty string")
            
            # Additional validation for empty or whitespace-only strings
            if not encrypted_api_key.strip():
                raise ValueError("Encrypted API key must be a non-empty string")
            
            # Check if this looks like an encrypted key (base64 encoded, reasonable length)
            # If not, assume it's a legacy plaintext key for backward compatibility
            if not self._is_encrypted_key(encrypted_api_key):
                logger.warning("🔓 Found legacy plaintext API key - will re-encrypt on next save")
                return encrypted_api_key.strip()
            
            self._initialize_encryption()
            
            # Ensure cipher was properly initialized
            if self._aesgcm is None:
                raise ValueError("Failed to initialize encryption cipher")
            
            # Decode from base64
            combined = base64.b64decode(encrypted_api_key.encode('ascii'))
            
            # Extract nonce (first 12 bytes) and encrypted data
            if len(combined) < 13:  # At least 12 bytes nonce + 1 byte data
                raise ValueError("Invalid encrypted data format")
                
            nonce = combined[:12]
            encrypted_data = combined[12:]
            
            # Decrypt the API key
            decrypted_bytes = self._aesgcm.decrypt(nonce, encrypted_data, None)
            decrypted_key = decrypted_bytes.decode('utf-8')
            
            logger.info("🔓 API key decrypted successfully using AES-256-GCM")
            return decrypted_key
            
        except ValueError:
            # Re-raise ValueError for proper error handling
            raise
        except Exception as e:
            logger.error(f"CRITICAL: Failed to decrypt API key: {e}")
            # For backward compatibility, if decryption fails, assume it might be plaintext
            # Only for non-empty strings
            if encrypted_api_key and encrypted_api_key.strip():
                logger.warning("🔓 Decryption failed - treating as legacy plaintext key")
                return encrypted_api_key.strip()
            else:
                raise ValueError("Cannot decrypt empty API key")
    
    def _is_encrypted_key(self, key: str) -> bool:
        """
        Check if a key appears to be encrypted (base64 format with reasonable length)
        Uses more strict detection to avoid false positives with API keys that happen to be valid base64
        
        Args:
            key (str): Key to check
            
        Returns:
            bool: True if key appears encrypted, False if likely plaintext
        """
        try:
            # Plaintext API keys typically start with recognizable prefixes
            plaintext_prefixes = ['hf_', 'sk-', 'api_', 'token_', 'key_']
            if any(key.lower().startswith(prefix) for prefix in plaintext_prefixes):
                return False
                
            # Encrypted keys should be base64 encoded and have a minimum length
            # (12 byte nonce + at least some encrypted data, base64 encoded = min ~20 chars)
            if len(key) < 20:  # Too short to be encrypted
                return False
            
            # Encrypted keys are typically much longer than common API keys
            # Most API keys are 20-50 chars, encrypted keys are typically 60+ chars
            if len(key) < 40:  # Most encrypted keys should be longer
                return False
                
            # Try to decode as base64 - if it fails, likely plaintext
            decoded = base64.b64decode(key, validate=True)
            
            # Should have at least 13 bytes (12 byte nonce + 1+ bytes encrypted data)
            if len(decoded) < 13:
                return False
            
            # Additional check: encrypted keys should have high entropy
            # Simple heuristic: check if it contains typical API key patterns
            if '_' in key or key.isalnum():
                # If it contains underscores or is purely alphanumeric, likely plaintext
                # Our encrypted keys should be base64 with +/= characters
                if not any(char in key for char in '+/='):
                    return False
                
            return True
            
        except Exception:
            # If base64 decode fails or validation fails, it's likely plaintext
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
            
            # Ensure proper indexing after successful connection
            await self._ensure_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            # Clean up partial state
            self.client = None
            self.db = None
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
                logger.info("✅ Gracefully disconnected from MongoDB")
            except Exception as e:
                logger.error(f"Error during MongoDB disconnection: {e}")
                # Force cleanup
                self.connected = False
                self.client = None
                self.db = None
    
    async def _ensure_indexes(self):
        """Ensure proper indexing on user_id for performance"""
        try:
            if self.db is None:
                raise ValueError("Database not connected")
            
            # Create index on user_id for fast lookups and uniqueness
            await self.db.users.create_index("user_id", unique=True)
            logger.info("✅ Database indexes created successfully")
            
        except PyMongoError as e:
            logger.error(f"Failed to create database indexes: {e}")
            # Don't raise - indexes are performance optimization
        except Exception as e:
            logger.error(f"Unexpected error creating indexes: {e}")
    
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save or update user's Hugging Face API key in MongoDB with AES-256-GCM encryption
        
        Args:
            user_id (int): Telegram user ID
            api_key (str): Hugging Face API key (will be encrypted before storage)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("api_key must be a non-empty string")
            
            # Encrypt the API key before storing
            encrypted_api_key = self._encrypt_api_key(api_key.strip())
            
            # Use upsert to insert or update the user document with encrypted key
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "hf_api_key": encrypted_api_key}},
                upsert=True
            )
            
            if result.acknowledged:
                action = "updated" if result.matched_count > 0 else "created"
                logger.info(f"🔒 Successfully {action} encrypted API key for user {user_id}")
                logger.info("✅ API key securely stored with AES-256-GCM encryption")
                return True
            else:
                logger.error(f"Failed to save API key for user {user_id} - operation not acknowledged")
                return False
                
        except ValueError as e:
            # Encryption errors are logged in _encrypt_api_key
            logger.error(f"Validation/encryption error saving API key for user {user_id}: {e}")
            return False
        except PyMongoError as e:
            logger.error(f"MongoDB error saving API key for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> str | None:
        """
        Retrieve and decrypt user's Hugging Face API key from MongoDB with backward compatibility
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            str | None: Decrypted API key if found, None otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            # Find user document by user_id
            user_doc = await self.db.users.find_one({"user_id": user_id})
            
            if user_doc and "hf_api_key" in user_doc:
                encrypted_api_key = user_doc["hf_api_key"]
                if isinstance(encrypted_api_key, str) and encrypted_api_key.strip():
                    
                    # Decrypt the API key (handles both encrypted and legacy plaintext keys)
                    decrypted_api_key = self._decrypt_api_key(encrypted_api_key.strip())
                    
                    # If it was a legacy plaintext key, re-encrypt it for future storage
                    if not self._is_encrypted_key(encrypted_api_key.strip()):
                        logger.info(f"🔄 Converting legacy plaintext API key to encrypted format for user {user_id}")
                        # Re-save with encryption (background task, don't wait)
                        asyncio.create_task(self.save_user_api_key(user_id, decrypted_api_key))
                    
                    logger.info(f"🔓 Successfully retrieved and decrypted API key for user {user_id}")
                    return decrypted_api_key
            
            logger.info(f"No API key found for user {user_id}")
            return None
                
        except ValueError as e:
            # Decryption errors are logged in _decrypt_api_key
            logger.error(f"Validation/decryption error retrieving API key for user {user_id}: {e}")
            return None
        except PyMongoError as e:
            logger.error(f"MongoDB error retrieving API key for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving API key for user {user_id}: {e}")
            return None
    
    async def reset_user_database(self, user_id: int) -> bool:
        """
        Delete user's entire document from MongoDB
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if successful or user didn't exist, False on error
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            # Delete the user document
            result = await self.db.users.delete_one({"user_id": user_id})
            
            if result.acknowledged:
                if result.deleted_count > 0:
                    logger.info(f"✅ Successfully deleted user data for user {user_id}")
                else:
                    logger.info(f"No data found to delete for user {user_id}")
                return True
            else:
                logger.error(f"Failed to delete user data for user {user_id} - operation not acknowledged")
                return False
                
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting user data for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting user data for user {user_id}: {e}")
            return False

# Global database instance
db = Database()