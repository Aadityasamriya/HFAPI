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
from datetime import datetime
from bot.security_utils import SecureLogger
from bot.crypto_utils import (
    initialize_crypto, get_crypto, encrypt_api_key, decrypt_api_key, 
    is_encrypted_data, CryptoError, EncryptionError, DecryptionError,
    TamperDetectionError, KeyDerivationError
)

logger = logging.getLogger(__name__)
secure_logger = SecureLogger(logger)

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
            secure_logger.warning(f"Error parsing MongoDB URI for TLS detection: {e}")
            # Default to False for safety in case of parsing errors
            return False
    
    # SECURITY FIX: Removed _get_or_create_encryption_seed method
    # Seed management is now centralized in Config.validate_config() only
    # This prevents storage layer mutations and ensures single source of truth
    
    def _get_encryption_key(self) -> bytes:
        """
        Get or generate a secure encryption key for API key protection
        Now uses persistent seed from database instead of environment variables
        
        Returns:
            bytes: 32-byte AES-256 encryption key
        """
        try:
            # Try to get key from environment variable first (for advanced users)
            env_key = os.getenv('API_ENCRYPTION_KEY')
            if env_key:
                # If provided as base64, decode it
                try:
                    decoded_key = base64.b64decode(env_key)
                    if len(decoded_key) == 32:
                        secure_logger.info("✅ Using API encryption key from environment variable")
                        return decoded_key
                except Exception:
                    pass
                
                # If provided as hex, decode it  
                try:
                    if len(env_key) == 64:
                        decoded_key = bytes.fromhex(env_key)
                        secure_logger.info("✅ Using API encryption key from environment variable (hex)")
                        return decoded_key
                except Exception:
                    pass
            
            # SECURITY FIX: Only use seed from centralized Config - no database mutations
            encryption_seed = Config.ENCRYPTION_SEED
            
            if not encryption_seed:
                raise ValueError(
                    "CRITICAL: Config.validate_config() must be called before database initialization. "
                    "Seed management is centralized in Config only for security."
                )
                
            logger.info("✅ Using ENCRYPTION_SEED from centralized Config (security hardened)")
            
            # Use PBKDF2 with the encryption seed
            password = encryption_seed.encode('utf-8')
            salt = b'telegram_bot_api_key_encryption_2024_v2'  # Fixed salt for deterministic keys
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits for AES-256
                salt=salt,
                iterations=100000,  # Strong iteration count
            )
            
            derived_key = kdf.derive(password)
            secure_logger.info("✅ Generated secure encryption key using PBKDF2-SHA256")
            return derived_key
            
        except Exception as e:
            secure_logger.error(f"CRITICAL: Failed to generate encryption key: {e}")
            raise ValueError(f"Cannot initialize encryption: {e}")
    
    def _initialize_encryption(self):
        """Initialize secure crypto system with hardened encryption"""
        try:
            # Initialize the new secure crypto system
            encryption_seed = Config.ENCRYPTION_SEED
            if not encryption_seed:
                raise ValueError(
                    "CRITICAL: Config.ensure_encryption_seed() must be called before database initialization. "
                    "Seed management is centralized in Config only for security."
                )
            
            # Initialize global crypto instance
            initialize_crypto(encryption_seed)
            
            # Keep legacy encryption for backward compatibility
            if not self._encryption_key:
                self._encryption_key = self._get_encryption_key()
            
            if not self._aesgcm:
                self._aesgcm = AESGCM(self._encryption_key)
                
            logger.info("🔒 Secure crypto system initialized successfully with backward compatibility")
            
        except Exception as e:
            secure_logger.error(f"CRITICAL: Failed to initialize secure crypto: {e}")
            raise
    
    def _encrypt_api_key(self, api_key: str, user_id: int) -> str:
        """
        Encrypt API key using secure authenticated encryption with versioned envelope
        
        SECURITY HARDENING: Uses new crypto_utils.encrypt_api_key with:
        - Versioned envelope format: v1 || salt || nonce || ciphertext || auth_tag  
        - HKDF(SHA256, salt) for key derivation
        - AES-256-GCM authenticated encryption
        - Strict error propagation - NO silent fallbacks
        
        Args:
            api_key (str): Plaintext API key
            user_id (int): User ID for per-user key derivation
            
        Returns:
            str: Secure encrypted API key with versioned envelope
            
        Raises:
            EncryptionError: If encryption fails
            KeyDerivationError: If key derivation fails
        """
        try:
            if not api_key or not isinstance(api_key, str):
                raise EncryptionError("API key must be a non-empty string")
            
            if not isinstance(user_id, int) or user_id <= 0:
                raise EncryptionError("User ID must be a positive integer")
                
            # Use new secure crypto system
            encrypted_key = encrypt_api_key(api_key.strip(), user_id)
            
            from bot.security_utils import SecureLogger
            secure_logger = SecureLogger(logger)
            secure_logger.info("🔒 API key encrypted successfully using hardened crypto with versioned envelope")
            return encrypted_key
            
        except (EncryptionError, KeyDerivationError):
            # Re-raise crypto errors as-is (strict error propagation)
            raise
        except Exception as e:
            from bot.security_utils import SecureLogger
            secure_logger = SecureLogger(logger)
            secure_logger.error(f"CRITICAL: Unexpected encryption error for user {user_id}: {e}")
            raise EncryptionError(f"Encryption failed: {e}")
    
    def _decrypt_api_key(self, encrypted_api_key: str, user_id: int) -> str:
        """
        Decrypt API key with secure backward compatibility and strict error handling
        
        SECURITY HARDENING: 
        - Tries secure crypto first (versioned envelope with integrity verification)
        - Falls back to legacy formats only for backward compatibility
        - NO silent failures - all errors propagate properly
        - Tamper detection raises TamperDetectionError immediately
        
        Args:
            encrypted_api_key (str): Encrypted API key or plaintext
            user_id (int): User ID for per-user key derivation
            
        Returns:
            str: Decrypted plaintext API key
            
        Raises:
            DecryptionError: If decryption fails
            TamperDetectionError: If data tampering is detected
            KeyDerivationError: If key derivation fails
        """
        try:
            if not encrypted_api_key or not isinstance(encrypted_api_key, str):
                raise DecryptionError("Encrypted API key must be a non-empty string")
                
            if not isinstance(user_id, int) or user_id <= 0:
                raise DecryptionError("User ID must be a positive integer")
            
            if not encrypted_api_key.strip():
                raise DecryptionError("Encrypted API key must be a non-empty string")
            
            cleaned_key = encrypted_api_key.strip()
            
            from bot.security_utils import SecureLogger
            secure_logger = SecureLogger(logger)
            
            # PRIORITY 1: Try new secure crypto system first
            if is_encrypted_data(cleaned_key):
                try:
                    decrypted_key = decrypt_api_key(cleaned_key, user_id)
                    secure_logger.info("🔓 API key decrypted successfully using secure crypto with integrity verification")
                    return decrypted_key
                except TamperDetectionError:
                    # SECURITY CRITICAL: Data tampering detected - DO NOT continue to fallbacks
                    secure_logger.error("🚨 SECURITY ALERT: Data tampering detected - refusing to decrypt")
                    raise
                except (DecryptionError, KeyDerivationError) as e:
                    # New format decryption failed, try legacy fallbacks
                    logger.debug(f"Secure crypto decryption failed for user {user_id}: {e}")
            
            # Check if this looks like plaintext first (most common fallback)
            if not self._is_encrypted_key(cleaned_key):
                secure_logger.info("🔓 Found plaintext API key - will re-encrypt on next save")
                return cleaned_key
            
            # FALLBACK 1: Try legacy per-user decryption (backward compatibility)
            try:
                user_key = self._derive_user_encryption_key(user_id)
                user_cipher = AESGCM(user_key)
                
                combined = base64.b64decode(cleaned_key.encode('ascii'))
                
                if len(combined) >= 13:  # At least 12 bytes nonce + 1 byte data
                    nonce = combined[:12]
                    encrypted_data = combined[12:]
                    
                    decrypted_bytes = user_cipher.decrypt(nonce, encrypted_data, None)
                    decrypted_key = decrypted_bytes.decode('utf-8')
                    
                    secure_logger.info("🔄 API key decrypted using legacy per-user cipher - will upgrade on next save")
                    return decrypted_key
                    
            except Exception as per_user_error:
                logger.debug(f"Legacy per-user decryption failed for user {user_id}: {per_user_error}")
            
            # FALLBACK 2: Try legacy global decryption (oldest format)
            try:
                decrypted_key = self._decrypt_legacy_global_key(cleaned_key)
                secure_logger.info("🔄 API key decrypted using legacy global cipher - will upgrade on next save")
                return decrypted_key
                
            except Exception as legacy_error:
                logger.debug(f"Legacy global decryption failed for user {user_id}: {legacy_error}")
            
            # All decryption methods failed - STRICT ERROR PROPAGATION
            secure_logger.error(f"CRITICAL: All decryption methods failed for user {user_id}")
            raise DecryptionError("Cannot decrypt API key with any available method")
            
        except (DecryptionError, TamperDetectionError, KeyDerivationError):
            # Re-raise crypto errors as-is (strict error propagation)
            raise
        except Exception as e:
            from bot.security_utils import SecureLogger
            secure_logger = SecureLogger(logger)
            secure_logger.error(f"CRITICAL: Unexpected error decrypting API key for user {user_id}: {e}")
            raise DecryptionError(f"Decryption failed: {e}")
    
    def _decrypt_legacy_global_key(self, encrypted_api_key: str) -> str:
        """
        Decrypt API key using legacy global AES-256-GCM encryption
        This handles keys that were encrypted before the per-user encryption upgrade
        
        Args:
            encrypted_api_key (str): Base64-encoded encrypted API key with nonce (legacy format)
            
        Returns:
            str: Decrypted plaintext API key
            
        Raises:
            ValueError: If decryption fails or global cipher not available
        """
        try:
            if not encrypted_api_key or not isinstance(encrypted_api_key, str):
                raise ValueError("Encrypted API key must be a non-empty string")
            
            # Check if global cipher is available
            if not hasattr(self, '_aesgcm') or self._aesgcm is None:
                raise ValueError("Legacy global cipher not initialized")
            
            # Decode from base64
            combined = base64.b64decode(encrypted_api_key.encode('ascii'))
            
            # Extract nonce (first 12 bytes) and encrypted data
            if len(combined) < 13:  # At least 12 bytes nonce + 1 byte data
                raise ValueError("Invalid legacy encrypted data format")
                
            nonce = combined[:12]
            encrypted_data = combined[12:]
            
            # Decrypt using legacy global key
            decrypted_bytes = self._aesgcm.decrypt(nonce, encrypted_data, None)
            decrypted_key = decrypted_bytes.decode('utf-8')
            
            from bot.security_utils import SecureLogger
            secure_logger = SecureLogger(logger)
            secure_logger.info("🔓 API key decrypted successfully using legacy global AES-256-GCM")
            return decrypted_key
            
        except Exception as e:
            from bot.security_utils import SecureLogger
            secure_logger = SecureLogger(logger)
            secure_logger.error(f"Failed to decrypt with legacy global cipher: {e}")
            raise ValueError(f"Legacy global decryption failed: {e}")
    
    def _derive_user_encryption_key(self, user_id: int) -> bytes:
        """
        Derive a unique encryption key for each user to reduce security blast radius
        Uses global encryption seed + user_id + salt for unique per-user keys
        
        Args:
            user_id (int): User ID for key derivation
            
        Returns:
            bytes: 32-byte AES-256 encryption key unique to this user
            
        Raises:
            ValueError: If global encryption seed is not available (fail-closed security)
        """
        try:
            # CRITICAL SECURITY: Only use persistent seed from database
            if hasattr(self, '_global_seed') and self._global_seed:
                encryption_seed = self._global_seed
            else:
                # Check environment as fallback (for advanced users)
                encryption_seed = os.getenv('ENCRYPTION_SEED')
                if not encryption_seed:
                    # FAIL CLOSED: No predictable fallback values allowed
                    raise ValueError(
                        "CRITICAL SECURITY ERROR: No encryption seed available. "
                        "Database initialization required before per-user encryption."
                    )
                logger.warning("⚠️ Using ENCRYPTION_SEED from environment - database seed preferred")
            
            # Validate seed quality
            if len(encryption_seed) < 16:
                raise ValueError("Encryption seed too short - minimum 16 characters required")
            
            # Create per-user password combining global seed + user_id
            user_password = f"{encryption_seed}:user:{user_id}:v2"
            password = user_password.encode('utf-8')
            
            # Use per-user salt for additional security
            user_salt = f"telegram_bot_user_{user_id}_encryption_2025".encode('utf-8')[:32]  # Limit to 32 bytes
            
            # Use strong PBKDF2 with 150,000 iterations for per-user keys
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits for AES-256
                salt=user_salt,
                iterations=150000,  # Enhanced security for per-user keys
            )
            
            derived_key = kdf.derive(password)
            return derived_key
            
        except Exception as e:
            secure_logger.error(f"CRITICAL: Failed to derive user encryption key for user {user_id}: {e}")
            raise ValueError(f"Cannot derive user encryption key: {e}")
    
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
            if not Config.get_mongodb_uri():
                raise ValueError("MONGO_URI is not configured")
            
            # Security validation: Ensure TLS is enabled for production
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            # Safe TLS detection without corrupting the original URI
            mongo_uri = Config.get_mongodb_uri()
            if not mongo_uri:
                raise ValueError("MONGODB_URI is not configured")
            has_tls = self._check_tls_enabled(mongo_uri)
            
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
                
            self.client = AsyncIOMotorClient(mongo_uri)
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
            
            # Initialize encryption with persistent seed after successful connection
            await self._initialize_encryption_with_persistent_seed()
            
        except Exception as e:
            secure_logger.error(f"Failed to connect to MongoDB: {e}")
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
        """Ensure proper indexing for performance on users and conversations"""
        try:
            if self.db is None:
                raise ValueError("Database not connected")
            
            # Create index on user_id for users collection (fast lookups and uniqueness)
            await self.db.users.create_index("user_id", unique=True)
            
            # Create indexes for conversations collection
            # Primary index: user_id for fast user-based queries
            await self.db.conversations.create_index("user_id")
            
            # Secondary index: user_id + last_message_at for sorted history browsing
            await self.db.conversations.create_index([
                ("user_id", 1),
                ("last_message_at", -1)
            ])
            
            # Index for conversation timestamp queries
            await self.db.conversations.create_index("created_at")
            
            logger.info("✅ Database indexes created successfully (users + conversations)")
            
        except PyMongoError as e:
            logger.error(f"Failed to create database indexes: {e}")
            # Don't raise - indexes are performance optimization
        except Exception as e:
            logger.error(f"Unexpected error creating indexes: {e}")
    
    async def _initialize_encryption_with_persistent_seed(self):
        """
        Initialize encryption system with seed from centralized Config only (SECURITY HARDENED)
        This method should be called after successful database connection
        """
        try:
            # SECURITY FIX: Get seed only from centralized Config - no database mutations
            encryption_seed = Config.ENCRYPTION_SEED
            
            if not encryption_seed:
                raise ValueError(
                    "CRITICAL: Config.validate_config() must be called before database initialization. "
                    "Seed management is centralized in Config only for security."
                )
            
            # Store the seed for per-user key derivation (from Config only)
            self._global_seed = encryption_seed
            
            # Generate the legacy global encryption key using PBKDF2
            password = encryption_seed.encode('utf-8')
            salt = b'telegram_bot_api_key_encryption_2024_v2'
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            self._encryption_key = kdf.derive(password)
            self._aesgcm = AESGCM(self._encryption_key)
            
            # Create index for system_config collection
            if self.db is not None:
                await self.db.system_config.create_index("type", unique=True)
            
            logger.info("✅ Encryption system initialized with seed from centralized Config (security hardened)")
            secure_logger.info("🔒 Per-user and legacy encryption keys derived from Config.validate_config() only")
            secure_logger.info("🔒 API keys secured with centralized seed management")
            
        except Exception as e:
            secure_logger.error(f"CRITICAL: Failed to initialize encryption with centralized seed: {e}")
            raise
    
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
            
            # Encrypt the API key before storing using per-user encryption
            encrypted_api_key = self._encrypt_api_key(api_key.strip(), user_id)
            
            # Use upsert to insert or update the user document with encrypted key
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "hf_api_key": encrypted_api_key}},
                upsert=True
            )
            
            if result.acknowledged:
                action = "updated" if result.matched_count > 0 else "created"
                secure_logger.info(f"🔒 Successfully {action} encrypted API key for user {user_id}")
                secure_logger.info("✅ API key securely stored with per-user AES-256-GCM encryption")
                
                # Security audit log for API key operations
                secure_logger.info(f"🔒 AUDIT: API_KEY_SAVE - User {user_id} - API key {action} successfully with per-user encryption")
                return True
            else:
                secure_logger.error(f"Failed to save API key for user {user_id} - operation not acknowledged")
                return False
                
        except ValueError as e:
            # Encryption errors are logged in _encrypt_api_key
            secure_logger.error(f"Validation/encryption error saving API key for user {user_id}: {e}")
            
            # Security audit log for failed API key save
            secure_logger.info(f"🔒 AUDIT: API_KEY_SAVE_FAILED - User {user_id} - API key save failed due to validation/encryption error: {str(e)[:100]}")
            return False
        except PyMongoError as e:
            logger.error(f"MongoDB error saving API key for user {user_id}: {e}")
            return False
        except Exception as e:
            secure_logger.error(f"Unexpected error saving API key for user {user_id}: {e}")
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
                    
                    # Decrypt the API key using per-user decryption (handles both encrypted and legacy plaintext keys)
                    decrypted_api_key = self._decrypt_api_key(encrypted_api_key.strip(), user_id)
                    
                    # If it was a legacy plaintext key, re-encrypt it for future storage
                    if not self._is_encrypted_key(encrypted_api_key.strip()):
                        logger.info(f"🔄 Converting legacy plaintext API key to encrypted format for user {user_id}")
                        # Re-save with encryption (background task, don't wait)
                        asyncio.create_task(self.save_user_api_key(user_id, decrypted_api_key))
                    
                    secure_logger.info(f"🔓 Successfully retrieved and decrypted API key for user {user_id}")
                    
                    # Security audit log for successful API key retrieval
                    secure_logger.info(f"🔒 AUDIT: API_KEY_RETRIEVE - User {user_id} - API key retrieved and decrypted successfully")
                    return decrypted_api_key
            
            logger.info(f"No API key found for user {user_id}")
            return None
                
        except ValueError as e:
            # Decryption errors are logged in _decrypt_api_key
            secure_logger.error(f"Validation/decryption error retrieving API key for user {user_id}: {e}")
            
            # Security audit log for failed API key retrieval
            secure_logger.info(f"🔒 AUDIT: API_KEY_RETRIEVE_FAILED - User {user_id} - API key retrieval failed due to validation/decryption error: {str(e)[:100]}")
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
                    secure_logger.info(f"✅ Successfully deleted user data for user {user_id}")
                    
                    # Security audit log for user data deletion
                    secure_logger.info(f"🔒 AUDIT: USER_DATA_DELETE - User {user_id} - User data (including encrypted API key) deleted successfully")
                else:
                    secure_logger.info(f"No data found to delete for user {user_id}")
                    
                    # Security audit log for attempted deletion of non-existent user
                    secure_logger.info(f"🔒 AUDIT: USER_DATA_DELETE_NO_DATA - User {user_id} - Attempted to delete user data but no data found")
                return True
            else:
                secure_logger.error(f"Failed to delete user data for user {user_id} - operation not acknowledged")
                return False
                
        except PyMongoError as e:
            secure_logger.error(f"MongoDB error deleting user data for user {user_id}: {e}")
            
            # Security audit log for failed user data deletion
            secure_logger.info(f"🔒 AUDIT: USER_DATA_DELETE_FAILED - User {user_id} - User data deletion failed due to database error: {str(e)[:100]}")
            return False
        except Exception as e:
            secure_logger.error(f"Unexpected error deleting user data for user {user_id}: {e}")
            return False
    
    async def save_conversation(self, user_id: int, conversation_data: dict) -> bool:
        """
        Save a complete conversation with metadata for persistent chat history
        
        Args:
            user_id (int): Telegram user ID
            conversation_data (dict): Conversation data with metadata
                Expected format:
                {
                    'messages': [{'role': 'user/assistant', 'content': str, 'timestamp': datetime}],
                    'summary': str,
                    'started_at': datetime,
                    'last_message_at': datetime,
                    'message_count': int
                }
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            if not isinstance(conversation_data, dict):
                raise ValueError("conversation_data must be a dictionary")
            
            # Validate required fields
            required_fields = ['messages', 'summary', 'started_at', 'last_message_at', 'message_count']
            for field in required_fields:
                if field not in conversation_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add user_id and ensure proper structure
            conversation_doc = {
                'user_id': user_id,
                **conversation_data,
                'created_at': conversation_data['started_at']
            }
            
            # Insert conversation into conversations collection
            result = await self.db.conversations.insert_one(conversation_doc)
            
            if result.acknowledged:
                logger.info(f"💾 Successfully saved conversation for user {user_id} (ID: {result.inserted_id})")
                logger.info(f"📊 Conversation saved: {conversation_data['message_count']} messages, {len(conversation_data['summary'])} char summary")
                return True
            else:
                logger.error(f"Failed to save conversation for user {user_id} - operation not acknowledged")
                return False
                
        except ValueError as e:
            logger.error(f"Validation error saving conversation for user {user_id}: {e}")
            return False
        except PyMongoError as e:
            logger.error(f"MongoDB error saving conversation for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving conversation for user {user_id}: {e}")
            return False
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> list:
        """
        Get conversation summaries for history browsing with pagination
        
        Args:
            user_id (int): Telegram user ID
            limit (int): Maximum number of conversations to return
            skip (int): Number of conversations to skip (for pagination)
        
        Returns:
            list: List of conversation summaries sorted by last_message_at (newest first)
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            # Query conversations with projection to get summaries only
            cursor = self.db.conversations.find(
                {"user_id": user_id},
                {
                    "_id": 1,
                    "summary": 1, 
                    "started_at": 1,
                    "last_message_at": 1,
                    "message_count": 1,
                    "created_at": 1
                }
            ).sort("last_message_at", -1).skip(skip).limit(limit)
            
            conversations = await cursor.to_list(length=limit)
            
            logger.info(f"📖 Retrieved {len(conversations)} conversation summaries for user {user_id} (skip: {skip}, limit: {limit})")
            
            return conversations
            
        except ValueError as e:
            logger.error(f"Validation error retrieving conversations for user {user_id}: {e}")
            return []
        except PyMongoError as e:
            logger.error(f"MongoDB error retrieving conversations for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversations for user {user_id}: {e}")
            return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> dict | None:
        """
        Get full conversation details by conversation ID
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): MongoDB ObjectId as string
        
        Returns:
            dict | None: Full conversation document or None if not found
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            from bson import ObjectId
            
            # Validate and convert conversation_id
            try:
                obj_id = ObjectId(conversation_id)
            except Exception:
                logger.warning(f"Invalid conversation_id format: {conversation_id}")
                return None
            
            # Find conversation ensuring it belongs to the user (security)
            conversation = await self.db.conversations.find_one({
                "_id": obj_id,
                "user_id": user_id
            })
            
            if conversation:
                logger.info(f"📄 Retrieved full conversation details for user {user_id}, conversation {conversation_id}")
                logger.info(f"📊 Conversation has {conversation.get('message_count', 0)} messages")
            else:
                logger.info(f"No conversation found for user {user_id} with ID {conversation_id}")
            
            return conversation
            
        except ValueError as e:
            logger.error(f"Validation error retrieving conversation details for user {user_id}: {e}")
            return None
        except PyMongoError as e:
            logger.error(f"MongoDB error retrieving conversation details for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversation details for user {user_id}: {e}")
            return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """
        Delete a specific conversation by ID
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): MongoDB ObjectId as string
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            from bson import ObjectId
            
            # Validate and convert conversation_id
            try:
                obj_id = ObjectId(conversation_id)
            except Exception:
                logger.warning(f"Invalid conversation_id format: {conversation_id}")
                return False
            
            # Delete conversation ensuring it belongs to the user (security)
            result = await self.db.conversations.delete_one({
                "_id": obj_id,
                "user_id": user_id
            })
            
            if result.acknowledged:
                if result.deleted_count > 0:
                    logger.info(f"🗑️ Successfully deleted conversation {conversation_id} for user {user_id}")
                    return True
                else:
                    logger.warning(f"No conversation found to delete: user {user_id}, conversation {conversation_id}")
                    return False
            else:
                logger.error(f"Failed to delete conversation {conversation_id} for user {user_id} - operation not acknowledged")
                return False
                
        except ValueError as e:
            logger.error(f"Validation error deleting conversation for user {user_id}: {e}")
            return False
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting conversation for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting conversation for user {user_id}: {e}")
            return False
    
    async def clear_user_history(self, user_id: int) -> bool:
        """
        Clear all conversation history for a user
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            # Delete all conversations for the user
            result = await self.db.conversations.delete_many({"user_id": user_id})
            
            if result.acknowledged:
                logger.info(f"🗑️ Successfully cleared {result.deleted_count} conversations for user {user_id}")
                return True
            else:
                logger.error(f"Failed to clear conversation history for user {user_id} - operation not acknowledged")
                return False
                
        except PyMongoError as e:
            logger.error(f"MongoDB error clearing conversation history for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error clearing conversation history for user {user_id}: {e}")
            return False
    
    async def get_conversation_count(self, user_id: int) -> int:
        """
        Get total number of conversations for a user
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            int: Number of conversations (0 if error)
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            count = await self.db.conversations.count_documents({"user_id": user_id})
            logger.info(f"📊 User {user_id} has {count} saved conversations")
            return count
            
        except Exception as e:
            logger.error(f"Error counting conversations for user {user_id}: {e}")
            return 0

# Create alias for backward compatibility - DatabaseManager points to Database class  
DatabaseManager = Database

# Global database instance
db = Database()