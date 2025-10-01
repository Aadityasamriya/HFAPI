"""
MongoDB Storage Provider Implementation
Ports existing MongoDB functionality from bot/database.py with full backward compatibility
"""

import asyncio
import os
import base64
import secrets
import json
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict, Any

# MongoDB imports with proper typing
try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
    from pymongo.errors import PyMongoError
    from gridfs import NoFile
    MONGODB_AVAILABLE = True
except ImportError:
    # Define stub types for when MongoDB is not available
    AsyncIOMotorClient = None
    PyMongoError = Exception
    NoFile = Exception
    AsyncIOMotorGridFSBucket = None
    MONGODB_AVAILABLE = False

# Encryption imports
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import logging
from .base import StorageProvider
from bot.config import Config
from bot.security_utils import SecureLogger
from bot.crypto_utils import (
    initialize_crypto, get_crypto, encrypt_api_key, decrypt_api_key, 
    is_encrypted_data, CryptoError, EncryptionError, DecryptionError,
    TamperDetectionError, KeyDerivationError
)

logger = logging.getLogger(__name__)

class MongoDBProvider(StorageProvider):
    """
    MongoDB storage provider implementation
    
    Ports all existing functionality from bot/database.py with full backward compatibility
    """
    
    def __init__(self):
        super().__init__()
        if not MONGODB_AVAILABLE or AsyncIOMotorClient is None:
            raise ImportError("MongoDB dependencies not available. Install with: pip install motor pymongo")
        
        self.client = None
        self.db = None
        self.gridfs_bucket = None
        self._encryption_key = None
        self._aesgcm = None
        self._global_seed = None
        self.secure_logger = SecureLogger(logger)
    
    # Connection Management
    async def connect(self) -> None:
        """Establish connection to MongoDB with TLS security validation"""
        try:
            mongo_uri = Config.get_mongodb_uri()
            if not mongo_uri:
                raise ValueError("MONGO_URI or MONGODB_URI is not configured")
            
            # Security validation: Ensure TLS is enabled for production
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            # Safe TLS detection without corrupting the original URI
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
                
            if AsyncIOMotorClient is None:
                raise ImportError("AsyncIOMotorClient not available - MongoDB dependencies missing")
            self.client = AsyncIOMotorClient(mongo_uri)
            self.db = self.client.ai_assistant_bot
            
            # Initialize GridFS bucket for efficient file storage
            if AsyncIOMotorGridFSBucket is not None:
                self.gridfs_bucket = AsyncIOMotorGridFSBucket(self.db)
            
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
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection gracefully"""
        if self.client:
            try:
                self.client.close()
                logger.info("✅ MongoDB connection closed gracefully")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")
            finally:
                self.client = None
                self.db = None
                self.gridfs_bucket = None
                self.connected = False
    
    # User Data Isolation Support (Phase 3 requirement)
    async def get_user_data(self, user_id: int, data_key: str) -> Any:
        """Get user-specific data with encryption and isolation"""
        try:
            if not self.connected:
                await self.connect()
            
            if self.db is None:
                raise RuntimeError("Database not connected")
            collection = self.db.user_data_isolation
            
            # Query with user isolation
            query = {
                'user_id': user_id,
                'data_key': data_key
            }
            
            result = await collection.find_one(query)
            if not result:
                return None
            
            # Decrypt data if encrypted
            data = result.get('data')
            if result.get('encrypted', False) and data:
                try:
                    crypto = get_crypto()
                    decrypted_data = crypto.decrypt(data)
                    # Try to deserialize if it's JSON
                    try:
                        return json.loads(decrypted_data)
                    except (json.JSONDecodeError, TypeError):
                        return decrypted_data
                except Exception as e:
                    logger.error(f"Failed to decrypt user data for user {user_id}: {e}")
                    return None
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get user data for user {user_id}, key {data_key}: {e}")
            return None
    
    async def save_user_data(self, user_id: int, data_key: str, data: Any, encrypt: bool = True) -> bool:
        """Save user-specific data with encryption and isolation"""
        try:
            if not self.connected:
                await self.connect()
            
            if self.db is None:
                raise RuntimeError("Database not connected")
            collection = self.db.user_data_isolation
            
            # Prepare data for storage
            storage_data = data
            encrypted = False
            
            if encrypt:
                try:
                    # Serialize data if needed
                    if not isinstance(data, str):
                        serialized_data = json.dumps(data, default=str)
                    else:
                        serialized_data = data
                    
                    # Encrypt the data
                    crypto = get_crypto()
                    storage_data = crypto.encrypt(serialized_data)
                    encrypted = True
                    
                except Exception as e:
                    logger.warning(f"Failed to encrypt user data for user {user_id}: {e}")
                    # Fall back to unencrypted storage
                    storage_data = data
                    encrypted = False
            
            # Create document with user isolation
            document = {
                'user_id': user_id,
                'data_key': data_key,
                'data': storage_data,
                'encrypted': encrypted,
                'updated_at': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
            
            # Upsert with user isolation
            query = {
                'user_id': user_id,
                'data_key': data_key
            }
            
            # Update timestamp for existing documents
            if encrypted:
                document['updated_at'] = datetime.utcnow()
            
            result = await collection.replace_one(
                query, 
                document, 
                upsert=True
            )
            
            logger.debug(f"Saved user data for user {user_id}, key {data_key}, encrypted: {encrypted}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save user data for user {user_id}, key {data_key}: {e}")
            return False
    
    async def initialize(self) -> None:
        """Initialize MongoDB (create indexes, setup encryption, etc.)"""
        if not self.connected:
            raise ValueError("Must be connected before initialization")
        
        # Ensure proper indexing
        await self._ensure_indexes()
        
        # Create rate limiting indexes for performance
        await self._ensure_rate_limit_indexes()
        
        # Initialize encryption with persistent seed
        await self._initialize_encryption_with_persistent_seed()
        
        self._encryption_initialized = True
        logger.info("✅ MongoDB provider initialized successfully")
    
    async def health_check(self) -> bool:
        """Perform health check on MongoDB"""
        try:
            if not self.connected or not self.client:
                return False
            
            # Simple ping test
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            return False
    
    # API Key Management
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save or update user's Hugging Face API key in MongoDB with AES-256-GCM encryption
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
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
                self.secure_logger.info(f"🔒 Successfully {action} encrypted API key for user {user_id}")
                
                # Security audit log for API key operations
                self.secure_logger.audit(
                    event_type="API_KEY_SAVE",
                    user_id=user_id,
                    details=f"API key {action} successfully with per-user encryption",
                    success=True
                )
                return True
            else:
                self.secure_logger.error(f"Failed to save API key for user {user_id} - operation not acknowledged")
                return False
                
        except ValueError as e:
            self.secure_logger.error(f"Validation/encryption error saving API key for user {user_id}: {e}")
            self.secure_logger.audit(
                event_type="API_KEY_SAVE_FAILED",
                user_id=user_id,
                details=f"API key save failed due to validation/encryption error: {str(e)[:100]}",
                success=False
            )
            return False
        except Exception as e:
            self.secure_logger.error(f"Unexpected error saving API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        Retrieve and decrypt user's Hugging Face API key
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Find user document
            user_doc = await self.db.users.find_one({"user_id": user_id})
            
            if user_doc and "hf_api_key" in user_doc:
                encrypted_api_key = user_doc["hf_api_key"]
                
                if encrypted_api_key and isinstance(encrypted_api_key, str):
                    # Decrypt the API key using per-user decryption (handles both encrypted and legacy plaintext keys)
                    decrypted_api_key = self._decrypt_api_key(encrypted_api_key.strip(), user_id)
                    
                    # If it was a legacy plaintext key, re-encrypt it for future storage
                    if not self._is_encrypted_key(encrypted_api_key.strip()):
                        self.secure_logger.info(f"🔄 Converting legacy plaintext API key to encrypted format for user {user_id}")
                        # Re-save with encryption (background task, don't wait)
                        asyncio.create_task(self.save_user_api_key(user_id, decrypted_api_key))
                    
                    self.secure_logger.info(f"🔓 Successfully retrieved and decrypted API key for user {user_id}")
                    
                    # Security audit log for successful API key retrieval
                    self.secure_logger.audit(
                        event_type="API_KEY_RETRIEVE",
                        user_id=user_id,
                        details="API key retrieved and decrypted successfully",
                        success=True
                    )
                    return decrypted_api_key
            
            self.secure_logger.info(f"No API key found for user {user_id}")
            return None
                
        except ValueError as e:
            self.secure_logger.error(f"Validation/decryption error retrieving API key for user {user_id}: {e}")
            self.secure_logger.audit(
                event_type="API_KEY_RETRIEVE_FAILED",
                user_id=user_id,
                details=f"API key retrieval failed due to validation/decryption error: {str(e)[:100]}",
                success=False
            )
            return None
        except Exception as e:
            self.secure_logger.error(f"Unexpected error retrieving API key for user {user_id}: {e}")
            return None
    
    # User Data Management
    async def reset_user_database(self, user_id: int) -> bool:
        """
        Delete user's entire document from MongoDB
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Delete user document and all conversations
            user_result = await self.db.users.delete_one({"user_id": user_id})
            conv_result = await self.db.conversations.delete_many({"user_id": user_id})
            
            self.secure_logger.info(f"🗑️ Successfully deleted user data for user {user_id}")
            self.secure_logger.info(f"📊 Deleted: user document, {conv_result.deleted_count} conversations")
            
            # Security audit log for user data deletion
            self.secure_logger.audit(
                event_type="USER_DATA_DELETE",
                user_id=user_id,
                details=f"User data deleted successfully: user doc + {conv_result.deleted_count} conversations",
                success=True
            )
            return True
            
        except ValueError as e:
            self.secure_logger.error(f"Validation error deleting user data for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting user data for user {user_id}: {e}")
            self.secure_logger.audit(
                event_type="USER_DATA_DELETE_FAILED",
                user_id=user_id,
                details=f"User data deletion failed due to database error: {str(e)[:100]}",
                success=False
            )
            return False
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user preferences and settings"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            user_doc = await self.db.users.find_one({"user_id": user_id})
            
            if user_doc and "preferences" in user_doc:
                return user_doc["preferences"]
            
            # Return default preferences
            return {
                "model_preference": "default",
                "response_style": "balanced",
                "language": "en"
            }
            
        except Exception as e:
            logger.error(f"Error retrieving user preferences for user {user_id}: {e}")
            return {}
    
    async def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """Save user preferences and settings"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "preferences": preferences, "updated_at": datetime.utcnow()}},
                upsert=True
            )
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error saving user preferences for user {user_id}: {e}")
            return False
    
    async def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """Get a specific user preference value by key"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            user_doc = await self.db.users.find_one({"user_id": user_id})
            
            if user_doc and "preferences" in user_doc:
                preferences = user_doc["preferences"]
                value = preferences.get(key)
                # Convert to string if not None, to match the interface signature
                return str(value) if value is not None else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving user preference '{key}' for user {user_id}: {e}")
            return None
    
    async def save_user_preference(self, user_id: int, key: str, value: str) -> bool:
        """Save a specific user preference value by key"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Get current preferences or use defaults
            current_preferences = await self.get_user_preferences(user_id)
            
            # Update the specific key
            current_preferences[key] = value
            
            # Save the updated preferences
            return await self.save_user_preferences(user_id, current_preferences)
            
        except Exception as e:
            logger.error(f"Error saving user preference '{key}' for user {user_id}: {e}")
            return False
    
    # Conversation Storage
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """
        Save conversation history with metadata
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            self._validate_conversation_data(conversation_data)
            
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
        except Exception as e:
            logger.error(f"Unexpected error saving conversation for user {user_id}: {e}")
            return False
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get conversation summaries for history browsing with pagination
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Find conversations for user, sorted by most recent
            cursor = self.db.conversations.find(
                {"user_id": user_id},
                {
                    "_id": 1, "summary": 1, "started_at": 1, 
                    "last_message_at": 1, "message_count": 1
                }
            ).sort("last_message_at", -1).skip(skip).limit(limit)
            
            conversations = []
            async for conv in cursor:
                conversations.append({
                    "id": str(conv["_id"]),
                    "summary": conv.get("summary", ""),
                    "started_at": conv.get("started_at"),
                    "last_message_at": conv.get("last_message_at"),
                    "message_count": conv.get("message_count", 0)
                })
            
            logger.info(f"📚 Retrieved {len(conversations)} conversation summaries for user {user_id}")
            return conversations
            
        except ValueError as e:
            logger.error(f"Validation error retrieving conversations for user {user_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversations for user {user_id}: {e}")
            return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed conversation data by ID
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
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
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversation details for user {user_id}: {e}")
            return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """
        Delete a specific conversation by ID
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
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
            
            if result.acknowledged and result.deleted_count > 0:
                logger.info(f"🗑️ Successfully deleted conversation {conversation_id} for user {user_id}")
                return True
            else:
                logger.warning(f"No conversation deleted - not found or doesn't belong to user {user_id}")
                return False
                
        except ValueError as e:
            logger.error(f"Validation error deleting conversation for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting conversation for user {user_id}: {e}")
            return False
    
    async def clear_user_history(self, user_id: int) -> bool:
        """
        Clear all conversation history for a user
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Delete all conversations for the user
            result = await self.db.conversations.delete_many({"user_id": user_id})
            
            if result.acknowledged:
                logger.info(f"🗑️ Successfully cleared {result.deleted_count} conversations for user {user_id}")
                return True
            else:
                logger.error(f"Failed to clear conversation history for user {user_id} - operation not acknowledged")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error clearing conversation history for user {user_id}: {e}")
            return False
    
    async def get_conversation_count(self, user_id: int) -> int:
        """
        Get total number of conversations for a user
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            count = await self.db.conversations.count_documents({"user_id": user_id})
            logger.info(f"📊 User {user_id} has {count} saved conversations")
            return count
            
        except Exception as e:
            logger.error(f"Error counting conversations for user {user_id}: {e}")
            return 0
    
    # File Storage (basic implementation for future use)
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """Save file data using GridFS for efficient storage with backward compatibility"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            if not file_data:
                raise ValueError("File data cannot be empty")
            
            if not isinstance(file_data, bytes):
                raise ValueError("File data must be bytes")
            
            # Check if we should use GridFS (recommended for files > 16MB or by default)
            file_size = len(file_data)
            use_gridfs = file_size > 16 * 1024 * 1024 or True  # Always use GridFS for consistency
            
            if use_gridfs and self.gridfs_bucket is not None:
                return await self._save_file_gridfs(user_id, file_id, file_data, metadata)
            else:
                # Fallback to base64 storage if GridFS unavailable
                logger.warning(f"GridFS unavailable, falling back to base64 storage for file {file_id}")
                return await self._save_file_base64(user_id, file_id, file_data, metadata)
            
        except Exception as e:
            logger.error(f"Error saving file for user {user_id}: {e}")
            return False
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file data with backward compatibility for base64 and GridFS storage"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # First try to get from GridFS
            gridfs_result = await self._get_file_gridfs(user_id, file_id)
            if gridfs_result is not None:
                logger.debug(f"Retrieved file {file_id} from GridFS for user {user_id}")
                return gridfs_result
            
            # Fallback to legacy base64 storage for backward compatibility
            base64_result = await self._get_file_base64(user_id, file_id)
            if base64_result is not None:
                logger.info(f"Retrieved legacy file {file_id} from base64 storage for user {user_id}")
                # Optionally migrate to GridFS in background (non-blocking)
                if self.gridfs_bucket is not None:
                    asyncio.create_task(self._migrate_file_to_gridfs(user_id, file_id, base64_result))
                return base64_result
            
            logger.debug(f"File {file_id} not found for user {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving file for user {user_id}: {e}")
            return None
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """Delete file data from both GridFS and legacy storage"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            deleted_any = False
            
            # Try to delete from GridFS first
            gridfs_deleted = await self._delete_file_gridfs(user_id, file_id)
            if gridfs_deleted:
                logger.debug(f"Deleted file {file_id} from GridFS for user {user_id}")
                deleted_any = True
            
            # Also try to delete from legacy base64 storage for cleanup
            base64_deleted = await self._delete_file_base64(user_id, file_id)
            if base64_deleted:
                logger.debug(f"Deleted legacy file {file_id} from base64 storage for user {user_id}")
                deleted_any = True
            
            if deleted_any:
                logger.info(f"Successfully deleted file {file_id} for user {user_id}")
            else:
                logger.warning(f"File {file_id} not found for user {user_id}")
            
            return deleted_any
            
        except Exception as e:
            logger.error(f"Error deleting file for user {user_id}: {e}")
            return False
    
    # GridFS Helper Methods
    async def _save_file_gridfs(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """Save file using GridFS for efficient storage"""
        try:
            if self.gridfs_bucket is None:
                raise ValueError("GridFS bucket not initialized")
            
            # Create unique filename combining user_id and file_id for isolation
            gridfs_filename = f"user_{user_id}_{file_id}"
            
            # Prepare metadata for GridFS
            gridfs_metadata = {
                "user_id": user_id,
                "file_id": file_id,
                "original_metadata": metadata,
                "storage_type": "gridfs",
                "created_at": datetime.utcnow(),
                "file_size": len(file_data)
            }
            
            # Check if file already exists and delete it
            await self._delete_file_gridfs(user_id, file_id)
            
            # Upload file to GridFS
            import io
            file_stream = io.BytesIO(file_data)
            object_id = await self.gridfs_bucket.upload_from_stream(
                gridfs_filename,
                file_stream,
                metadata=gridfs_metadata
            )
            
            logger.info(f"📁 Successfully saved file {file_id} to GridFS for user {user_id} (ObjectId: {object_id}, Size: {len(file_data)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save file {file_id} to GridFS for user {user_id}: {e}")
            return False
    
    async def _get_file_gridfs(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file from GridFS"""
        try:
            if self.gridfs_bucket is None:
                return None
            
            # Create unique filename
            gridfs_filename = f"user_{user_id}_{file_id}"
            
            # Find the file in GridFS
            try:
                grid_out = await self.gridfs_bucket.open_download_stream_by_name(gridfs_filename)
                file_data = await grid_out.read()
                
                # Get file metadata
                file_metadata = grid_out.metadata or {}
                
                # Return in the expected format
                return {
                    "user_id": user_id,
                    "file_id": file_id,
                    "data": file_data,
                    "metadata": file_metadata.get("original_metadata", {}),
                    "created_at": file_metadata.get("created_at"),
                    "storage_type": "gridfs",
                    "_gridfs_id": grid_out._id
                }
                
            except NoFile:
                return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id} from GridFS for user {user_id}: {e}")
            return None
    
    async def _delete_file_gridfs(self, user_id: int, file_id: str) -> bool:
        """Delete file from GridFS"""
        try:
            if self.gridfs_bucket is None:
                return False
            
            # Create unique filename
            gridfs_filename = f"user_{user_id}_{file_id}"
            
            # Find and delete the file
            try:
                # Find the file first
                if self.db is None:
                    return False
                
                file_doc = await self.db.fs.files.find_one({
                    "filename": gridfs_filename,
                    "metadata.user_id": user_id,
                    "metadata.file_id": file_id
                })
                
                if file_doc:
                    await self.gridfs_bucket.delete(file_doc["_id"])
                    return True
                else:
                    return False
                    
            except NoFile:
                return False
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id} from GridFS for user {user_id}: {e}")
            return False
    
    async def _save_file_base64(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """Fallback method: Save file using legacy base64 storage"""
        try:
            if self.db is None:
                raise ValueError("Database not connected")
            
            # Legacy file storage in files collection
            file_doc = {
                "user_id": user_id,
                "file_id": file_id,
                "data": base64.b64encode(file_data).decode('ascii'),
                "metadata": metadata,
                "storage_type": "base64",
                "created_at": datetime.utcnow()
            }
            
            result = await self.db.files.update_one(
                {"user_id": user_id, "file_id": file_id},
                {"$set": file_doc},
                upsert=True
            )
            
            logger.info(f"📁 Saved file {file_id} using legacy base64 storage for user {user_id}")
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Failed to save file {file_id} using base64 storage for user {user_id}: {e}")
            return False
    
    async def _get_file_base64(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file from legacy base64 storage"""
        try:
            if self.db is None:
                return None
            
            file_doc = await self.db.files.find_one({
                "user_id": user_id,
                "file_id": file_id
            })
            
            if file_doc and "data" in file_doc:
                # Decode base64 data
                file_doc["data"] = base64.b64decode(file_doc["data"].encode('ascii'))
                file_doc["storage_type"] = "base64"
            
            return file_doc
            
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id} from base64 storage for user {user_id}: {e}")
            return None
    
    async def _delete_file_base64(self, user_id: int, file_id: str) -> bool:
        """Delete file from legacy base64 storage"""
        try:
            if self.db is None:
                return False
            
            result = await self.db.files.delete_one({
                "user_id": user_id,
                "file_id": file_id
            })
            
            return result.acknowledged and result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id} from base64 storage for user {user_id}: {e}")
            return False
    
    async def _migrate_file_to_gridfs(self, user_id: int, file_id: str, base64_file_data: Dict[str, Any]) -> bool:
        """Background migration of legacy base64 files to GridFS"""
        try:
            if self.gridfs_bucket is None:
                logger.debug(f"GridFS not available, skipping migration for file {file_id}")
                return False
            
            # Extract data and metadata from base64 file
            file_data = base64_file_data.get("data")
            metadata = base64_file_data.get("metadata", {})
            
            if not file_data:
                logger.warning(f"No data found for migration of file {file_id}")
                return False
            
            # Save to GridFS
            gridfs_saved = await self._save_file_gridfs(user_id, file_id, file_data, metadata)
            
            if gridfs_saved:
                # Delete from legacy storage after successful migration
                base64_deleted = await self._delete_file_base64(user_id, file_id)
                logger.info(f"🔄 Successfully migrated file {file_id} from base64 to GridFS for user {user_id}")
                return True
            else:
                logger.warning(f"Failed to migrate file {file_id} to GridFS for user {user_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error migrating file {file_id} to GridFS for user {user_id}: {e}")
            return False
    
    # Analytics and Metrics
    async def log_usage(self, user_id: int, action: str, model_used: str, tokens_used: int = 0) -> bool:
        """Log usage metrics for analytics"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            usage_doc = {
                "user_id": user_id,
                "action": action,
                "model_used": model_used,
                "tokens_used": tokens_used,
                "timestamp": datetime.utcnow()
            }
            
            result = await self.db.usage_logs.insert_one(usage_doc)
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error logging usage for user {user_id}: {e}")
            return False
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Aggregate usage statistics
            pipeline = [
                {"$match": {"user_id": user_id, "timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": "$action",
                    "count": {"$sum": 1},
                    "total_tokens": {"$sum": "$tokens_used"}
                }}
            ]
            
            cursor = self.db.usage_logs.aggregate(pipeline)
            stats = {}
            total_requests = 0
            total_tokens = 0
            
            async for result in cursor:
                action = result["_id"]
                count = result["count"]
                tokens = result["total_tokens"]
                
                stats[action] = {"count": count, "tokens": tokens}
                total_requests += count
                total_tokens += tokens
            
            stats["summary"] = {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "days": days
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving usage stats for user {user_id}: {e}")
            return {}
    
    # Private Helper Methods (ported from original database.py)
    def _check_tls_enabled(self, mongo_uri: str) -> bool:
        """Safely check if TLS is enabled in MongoDB URI"""
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
            return False
    
    async def _ensure_indexes(self):
        """Ensure proper indexing for performance on users and conversations"""
        try:
            if self.db is None:
                raise ValueError("Database not connected")
            
            # Create index on user_id for users collection (fast lookups and uniqueness)
            await self.db.users.create_index("user_id", unique=True)
            
            # Create indexes for conversations collection
            await self.db.conversations.create_index("user_id")
            await self.db.conversations.create_index([
                ("user_id", 1),
                ("last_message_at", -1)
            ])
            await self.db.conversations.create_index("created_at")
            
            # Create indexes for files and usage logs
            await self.db.files.create_index([("user_id", 1), ("file_id", 1)], unique=True)
            await self.db.usage_logs.create_index([("user_id", 1), ("timestamp", -1)])
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")
    
    def _generate_encryption_seed_guidance(self) -> str:
        """Provide guidance for setting up encryption seed environment variable"""
        sample_seed = base64.b64encode(secrets.token_bytes(32)).decode('ascii')
        
        guidance = f"""
🔐 ENCRYPTION SETUP REQUIRED 🔐

For security, this bot requires an ENCRYPTION_SEED environment variable.
NEVER store encryption keys in the database alongside encrypted data!

Add this to your environment (.env file or deployment config):
ENCRYPTION_SEED='{sample_seed}'

🚨 IMPORTANT SECURITY NOTES:
• Use a unique, random 32+ character string
• NEVER commit this seed to version control
• Store it securely in your deployment environment
• If you change this seed, all existing encrypted data becomes unreadable

💡 For production: Use your hosting platform's secret management system
"""
        return guidance
    
    async def _initialize_encryption_with_persistent_seed(self):
        """Initialize secure crypto system with hardened encryption"""
        try:
            # SECURITY HARDENING: Initialize new secure crypto system first
            encryption_seed = Config.ENCRYPTION_SEED
            
            # Validate that Config.validate_config() was called first
            if not encryption_seed:
                raise ValueError(
                    "CRITICAL: Config.validate_config() must be called before storage initialization. "
                    "Seed management is centralized in Config only for security."
                )
            
            if len(encryption_seed) < 32:
                logger.warning("🚨 SECURITY WARNING: ENCRYPTION_SEED should be at least 32 characters for security")
                logger.warning("💡 Use a strong, random string with at least 32 characters")
            
            # Initialize global crypto instance
            initialize_crypto(encryption_seed)
            
            # Store the seed for per-user key derivation (from environment, not database)
            self._global_seed = encryption_seed
            
            # Keep legacy encryption for backward compatibility
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
            
            logger.info("✅ Secure crypto system initialized successfully with backward compatibility")
            self.secure_logger.info("🔒 MongoDB provider encryption ready - using hardened crypto with versioned envelopes")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize secure crypto system: {e}")
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
            
            self.secure_logger.info("🔒 API key encrypted successfully using hardened crypto with versioned envelope")
            return encrypted_key
            
        except (EncryptionError, KeyDerivationError):
            # Re-raise crypto errors as-is (strict error propagation)
            raise
        except Exception as e:
            self.secure_logger.error(f"CRITICAL: Unexpected encryption error for user {user_id}: {e}")
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
            
            # Check if this looks like plaintext first (most common and fast check)
            if not self._is_encrypted_key(cleaned_key):
                self.secure_logger.info("🔓 Found plaintext API key - will re-encrypt on next save")
                return cleaned_key
            
            # PRIORITY 1: Try new secure crypto system first if it matches the exact format
            if is_encrypted_data(cleaned_key):
                try:
                    decrypted_key = decrypt_api_key(cleaned_key, user_id)
                    self.secure_logger.info("🔓 API key decrypted successfully using secure crypto with integrity verification")
                    return decrypted_key
                except TamperDetectionError:
                    # SECURITY CRITICAL: Data tampering detected - DO NOT continue to fallbacks
                    self.secure_logger.error("🚨 SECURITY ALERT: Data tampering detected - refusing to decrypt")
                    raise
                except (DecryptionError, KeyDerivationError) as e:
                    # New format decryption failed, try legacy fallbacks
                    self.secure_logger.warning(f"Secure crypto decryption failed for user {user_id}, trying legacy methods: {e}")
            else:
                # Data appears encrypted but not in new format - go straight to legacy methods
                self.secure_logger.debug(f"Data appears to be legacy encrypted format for user {user_id}")
            
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
                    
                    self.secure_logger.info("🔄 API key decrypted using legacy per-user cipher - will upgrade on next save")
                    return decrypted_key
                    
            except Exception as per_user_error:
                logger.debug(f"Legacy per-user decryption failed for user {user_id}: {per_user_error}")
            
            # FALLBACK 2: Try legacy global decryption (oldest format)
            try:
                decrypted_key = self._decrypt_legacy_global_key(cleaned_key)
                self.secure_logger.info("🔄 API key decrypted using legacy global cipher - will upgrade on next save")
                return decrypted_key
                
            except Exception as legacy_error:
                logger.debug(f"Legacy global decryption failed for user {user_id}: {legacy_error}")
            
            # All decryption methods failed - STRICT ERROR PROPAGATION
            self.secure_logger.error(f"CRITICAL: All decryption methods failed for user {user_id}")
            raise DecryptionError("Cannot decrypt API key with any available method")
            
        except (DecryptionError, TamperDetectionError, KeyDerivationError):
            # Re-raise crypto errors as-is (strict error propagation)
            raise
        except Exception as e:
            self.secure_logger.error(f"CRITICAL: Unexpected error decrypting API key for user {user_id}: {e}")
            raise DecryptionError(f"Decryption failed: {e}")
    
    def _decrypt_legacy_global_key(self, encrypted_api_key: str) -> str:
        """Decrypt API key using legacy global encryption method"""
        try:
            if not self._aesgcm:
                raise ValueError("Legacy encryption not initialized")
            
            # Decode from base64
            combined = base64.b64decode(encrypted_api_key.encode('ascii'))
            
            # Extract nonce (first 12 bytes) and encrypted data
            if len(combined) >= 13:
                nonce = combined[:12]
                encrypted_data = combined[12:]
                
                # Decrypt with global key
                decrypted_bytes = self._aesgcm.decrypt(nonce, encrypted_data, None)
                return decrypted_bytes.decode('utf-8')
            else:
                raise ValueError("Invalid encrypted data format")
                
        except Exception as e:
            raise ValueError(f"Legacy decryption failed: {e}")
    
    def _derive_user_encryption_key(self, user_id: int) -> bytes:
        """Derive per-user encryption key from global seed + user_id"""
        try:
            if not self._global_seed:
                raise ValueError("Global encryption seed not available")
            
            # Combine global seed with user_id for per-user key derivation
            user_salt = f"{self._global_seed}:user:{user_id}".encode('utf-8')
            fixed_salt = b'telegram_bot_per_user_encryption_2024_v3'
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits for AES-256
                salt=fixed_salt,
                iterations=100000,
            )
            
            return kdf.derive(user_salt)
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to derive user encryption key for user {user_id}: {e}")
            raise ValueError(f"Cannot derive user encryption key: {e}")
    
    def _is_encrypted_key(self, key: str) -> bool:
        """Check if a key appears to be encrypted (base64) vs plaintext"""
        try:
            # Basic heuristics for encrypted vs plaintext keys
            if not key or len(key) < 10:
                return False
            
            # Hugging Face API keys typically start with 'hf_' and contain underscores
            if key.startswith('hf_') and '_' in key:
                return False  # Likely plaintext
            
            # Try to decode as base64 - encrypted keys should be valid base64
            try:
                decoded = base64.b64decode(key, validate=True)
                # Encrypted data should be at least 13 bytes (12-byte nonce + data)
                return len(decoded) >= 13
            except Exception:
                return False  # Not valid base64, likely plaintext
                
        except Exception:
            return False
    
    # Admin Data Management Implementation
    async def get_admin_data(self) -> Optional[Dict[str, Any]]:
        """Get admin system configuration data from MongoDB"""
        try:
            if not self.connected:
                raise ValueError("Must be connected to get admin data")
            
            # Look for admin data in admin_config collection
            if self.db is None:
                raise ValueError("Database connection not available")
            admin_doc = await self.db.admin_config.find_one({'_id': 'admin_system'})
            
            if admin_doc:
                # Remove MongoDB's _id field and return the data
                admin_data = dict(admin_doc)
                admin_data.pop('_id', None)
                logger.info("📊 Retrieved admin data from MongoDB")
                return admin_data
            else:
                logger.info("📊 No admin data found in MongoDB")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get admin data from MongoDB: {e}")
            return None
    
    async def save_admin_data(self, admin_data: Dict[str, Any]) -> bool:
        """Save admin system configuration data to MongoDB"""
        try:
            if not self.connected:
                raise ValueError("Must be connected to save admin data")
            
            # Add timestamp and prepare document
            admin_doc = admin_data.copy()
            admin_doc['_id'] = 'admin_system'
            admin_doc['last_updated'] = datetime.utcnow()
            
            # Upsert the admin configuration
            if self.db is None:
                raise ValueError("Database connection not available")
            result = await self.db.admin_config.replace_one(
                {'_id': 'admin_system'}, 
                admin_doc, 
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                logger.info("💾 Admin data saved to MongoDB successfully")
                return True
            else:
                logger.warning("⚠️ Admin data save to MongoDB had no effect")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save admin data to MongoDB: {e}")
            return False
    
    async def log_admin_action(self, admin_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Log admin action to MongoDB for audit trail"""
        try:
            if not self.connected:
                raise ValueError("Must be connected to log admin action")
            
            # Create admin action log entry
            log_entry = {
                'admin_id': admin_id,
                'action': action,
                'details': details or {},
                'timestamp': datetime.utcnow(),
                'ip_address': None,  # Can be added later if needed
                'user_agent': None   # Can be added later if needed
            }
            
            # Insert into admin_logs collection
            if self.db is None:
                raise ValueError("Database connection not available")
            result = await self.db.admin_logs.insert_one(log_entry)
            
            if result.inserted_id:
                logger.info(f"🔐 Admin action logged: {action} by admin {admin_id}")
                return True
            else:
                logger.warning(f"⚠️ Failed to log admin action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to log admin action to MongoDB: {e}")
            return False
    
    async def get_admin_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get admin action logs from MongoDB for audit trail"""
        try:
            if not self.connected:
                raise ValueError("Must be connected to get admin logs")
            
            # Query admin logs with pagination, sorted by timestamp (newest first)
            if self.db is None:
                raise ValueError("Database connection not available")
            cursor = self.db.admin_logs.find().sort('timestamp', -1).skip(skip).limit(limit)
            
            logs = []
            async for log_doc in cursor:
                # Remove MongoDB's _id field
                log_entry = dict(log_doc)
                log_entry.pop('_id', None)
                logs.append(log_entry)
            
            logger.info(f"📋 Retrieved {len(logs)} admin log entries from MongoDB")
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get admin logs from MongoDB: {e}")
            return []
    
    # Rate Limiting (Persistent Implementation)
    async def check_rate_limit(self, user_id: int, max_requests: int = 20, time_window: int = 60) -> tuple[bool, Optional[int]]:
        """
        Check if user is within rate limits using persistent database storage
        
        Args:
            user_id (int): Telegram user ID
            max_requests (int): Maximum requests per time window
            time_window (int): Time window in seconds
            
        Returns:
            tuple[bool, Optional[int]]: (is_allowed, seconds_until_reset)
        """
        try:
            if not self.connected or self.db is None:
                # Fallback to allow if database is unavailable
                logger.warning("Database not available for rate limiting - allowing request")
                return True, None
            
            user_id = self._validate_user_id(user_id)
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(seconds=time_window)
            
            # Remove old requests outside the time window
            await self.db.rate_limits.delete_many({
                "user_id": user_id,
                "timestamp": {"$lt": cutoff_time}
            })
            
            # Count current requests within the time window
            request_count = await self.db.rate_limits.count_documents({
                "user_id": user_id,
                "timestamp": {"$gte": cutoff_time}
            })
            
            # Check if user has exceeded rate limit
            if request_count >= max_requests:
                # Find the oldest request to calculate reset time
                oldest_request = await self.db.rate_limits.find_one(
                    {"user_id": user_id},
                    sort=[("timestamp", 1)]
                )
                
                if oldest_request:
                    oldest_time = oldest_request["timestamp"]
                    seconds_until_reset = int(time_window - (current_time - oldest_time).total_seconds())
                    return False, max(1, seconds_until_reset)
                else:
                    return False, time_window
            
            # Add current request to the database
            await self.db.rate_limits.insert_one({
                "user_id": user_id,
                "timestamp": current_time,
                "action": "message",
                "ip_address": None  # Can be added later if needed
            })
            
            return True, None
            
        except Exception as e:
            logger.error(f"Rate limiting check failed for user {user_id}: {e}")
            # Fallback to allow if rate limiting fails
            return True, None
    
    async def get_remaining_requests(self, user_id: int, max_requests: int = 20, time_window: int = 60) -> int:
        """
        Get number of remaining requests for user using persistent storage
        
        Args:
            user_id (int): Telegram user ID
            max_requests (int): Maximum requests per time window
            time_window (int): Time window in seconds
            
        Returns:
            int: Number of remaining requests
        """
        try:
            if not self.connected or self.db is None:
                return max_requests  # Return max if database unavailable
            
            user_id = self._validate_user_id(user_id)
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(seconds=time_window)
            
            # Count current requests within the time window
            request_count = await self.db.rate_limits.count_documents({
                "user_id": user_id,
                "timestamp": {"$gte": cutoff_time}
            })
            
            return max(0, max_requests - request_count)
            
        except Exception as e:
            logger.error(f"Failed to get remaining requests for user {user_id}: {e}")
            return max_requests  # Return max if check fails
    
    async def reset_user_rate_limit(self, user_id: int) -> bool:
        """
        Reset rate limit for specific user (admin function)
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Delete all rate limit entries for this user
            result = await self.db.rate_limits.delete_many({"user_id": user_id})
            
            logger.info(f"Rate limit reset for user {user_id}: {result.deleted_count} entries removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit for user {user_id}: {e}")
            return False
    
    async def cleanup_old_rate_limits(self, days_old: int = 7) -> int:
        """
        Clean up old rate limit entries (maintenance function)
        
        Args:
            days_old (int): Remove entries older than this many days
            
        Returns:
            int: Number of entries cleaned up
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            cutoff_time = datetime.utcnow() - timedelta(days=days_old)
            
            result = await self.db.rate_limits.delete_many({
                "timestamp": {"$lt": cutoff_time}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} old rate limit entries")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old rate limits: {e}")
            return 0
    
    async def _ensure_rate_limit_indexes(self):
        """Create rate limiting indexes for optimal performance with robust TTL migration"""
        try:
            if not self.connected or self.db is None:
                logger.warning("Cannot create rate limit indexes - database not connected")
                return
            
            # Rate limiting indexes for efficient queries
            try:
                await self.db.rate_limits.create_index("user_id")  # For user-specific queries
            except Exception:
                pass  # Index might already exist
            
            # Compound index for user + timestamp queries (most common)
            try:
                await self.db.rate_limits.create_index([
                    ("user_id", 1),
                    ("timestamp", -1)
                ])
            except Exception:
                pass  # Index might already exist
            
            # Handle TTL index with robust migration logic
            await self._ensure_ttl_index_migration()
            
            logger.info("✅ Rate limiting indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create rate limit indexes: {e}")
    
    async def _ensure_ttl_index_migration(self):
        """
        Robust TTL index migration that handles IndexOptionsConflict
        Checks existing index and recreates if TTL value differs
        """
        try:
            target_ttl = 86400  # 24 hours in seconds
            
            # Get existing indexes for the collection
            if self.db is None:
                logger.error("Database connection not available for TTL index migration")
                return
            existing_indexes = await self.db.rate_limits.index_information()
            
            # Check if timestamp_1 index exists and get its TTL value
            timestamp_index_name = None
            current_ttl = None
            
            for index_name, index_info in existing_indexes.items():
                if index_name.startswith("timestamp_"):
                    timestamp_index_name = index_name
                    current_ttl = index_info.get("expireAfterSeconds")
                    break
            
            # If no timestamp TTL index exists, create it
            if not timestamp_index_name:
                logger.info("🔄 Creating new TTL index for rate limiting cleanup")
                if self.db is not None:
                    await self.db.rate_limits.create_index(
                        "timestamp", 
                        expireAfterSeconds=target_ttl
                    )
                logger.info(f"✅ TTL index created with {target_ttl} seconds expiration")
                return
            
            # If TTL value is different, recreate the index
            if current_ttl != target_ttl:
                logger.info(f"🔄 Migrating TTL index: current={current_ttl}s, target={target_ttl}s")
                
                # Drop the existing TTL index
                if self.db is not None:
                    await self.db.rate_limits.drop_index(timestamp_index_name)
                    logger.info(f"🗑️ Dropped existing TTL index: {timestamp_index_name}")
                    
                    # Create new TTL index with correct value
                    await self.db.rate_limits.create_index(
                        "timestamp", 
                        expireAfterSeconds=target_ttl
                    )
                logger.info(f"✅ TTL index recreated with {target_ttl} seconds expiration")
            else:
                logger.info(f"✅ TTL index already exists with correct value ({target_ttl}s)")
                
        except Exception as e:
            logger.error(f"Failed to handle TTL index migration: {e}")
            # Fallback: try to create basic timestamp index without TTL
            try:
                if self.db is not None:
                    await self.db.rate_limits.create_index("timestamp")
                    logger.warning("⚠️ Created basic timestamp index without TTL due to migration error")
            except Exception:
                logger.error("💥 Failed to create any timestamp index - rate limiting may be degraded")