"""
MongoDB Storage Provider Implementation
Ports existing MongoDB functionality from bot/database.py with full backward compatibility
"""

import asyncio
import os
import base64
import secrets
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict, Any

# MongoDB imports
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo.errors import PyMongoError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# Encryption imports
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import logging
from .base import StorageProvider
from bot.config import Config
from bot.core.model_caller import SecureLogger

logger = logging.getLogger(__name__)

class MongoDBProvider(StorageProvider):
    """
    MongoDB storage provider implementation
    
    Ports all existing functionality from bot/database.py with full backward compatibility
    """
    
    def __init__(self):
        super().__init__()
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB dependencies not available. Install with: pip install motor pymongo")
        
        self.client = None
        self.db = None
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
                self.connected = False
    
    async def initialize(self) -> None:
        """Initialize MongoDB (create indexes, setup encryption, etc.)"""
        if not self.connected:
            raise ValueError("Must be connected before initialization")
        
        # Ensure proper indexing
        await self._ensure_indexes()
        
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
                logger.error(f"Failed to save API key for user {user_id} - operation not acknowledged")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error saving API key for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving API key for user {user_id}: {e}")
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
                        logger.info(f"🔄 Converting legacy plaintext API key to encrypted format for user {user_id}")
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
            
            logger.info(f"No API key found for user {user_id}")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error retrieving API key for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving API key for user {user_id}: {e}")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting user data for user {user_id}: {e}")
            self.secure_logger.audit(
                event_type="USER_DATA_DELETE_FAILED",
                user_id=user_id,
                details=f"User data deletion failed due to database error: {str(e)[:100]}",
                success=False
            )
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting user data for user {user_id}: {e}")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error saving conversation for user {user_id}: {e}")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error retrieving conversations for user {user_id}: {e}")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error retrieving conversation details for user {user_id}: {e}")
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
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting conversation for user {user_id}: {e}")
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
                
        except PyMongoError as e:
            logger.error(f"MongoDB error clearing conversation history for user {user_id}: {e}")
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
        """Save file data with metadata - basic MongoDB GridFS implementation"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Simple file storage in files collection (could be upgraded to GridFS)
            file_doc = {
                "user_id": user_id,
                "file_id": file_id,
                "data": base64.b64encode(file_data).decode('ascii'),
                "metadata": metadata,
                "created_at": datetime.utcnow()
            }
            
            result = await self.db.files.update_one(
                {"user_id": user_id, "file_id": file_id},
                {"$set": file_doc},
                upsert=True
            )
            
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Error saving file for user {user_id}: {e}")
            return False
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file data and metadata"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            file_doc = await self.db.files.find_one({
                "user_id": user_id,
                "file_id": file_id
            })
            
            if file_doc:
                # Decode base64 data
                file_doc["data"] = base64.b64decode(file_doc["data"].encode('ascii'))
            
            return file_doc
            
        except Exception as e:
            logger.error(f"Error retrieving file for user {user_id}: {e}")
            return None
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """Delete file data"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            result = await self.db.files.delete_one({
                "user_id": user_id,
                "file_id": file_id
            })
            
            return result.acknowledged and result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting file for user {user_id}: {e}")
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
            
        except PyMongoError as e:
            logger.error(f"Failed to create database indexes: {e}")
        except Exception as e:
            logger.error(f"Unexpected error creating indexes: {e}")
    
    async def _get_or_create_encryption_seed(self) -> str:
        """Get encryption seed from database or create and store a new one"""
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database must be connected to manage encryption seed")
            
            # Try to get existing seed from system collection
            system_doc = await self.db.system_config.find_one({"type": "encryption_config"})
            
            if system_doc and "encryption_seed" in system_doc:
                seed = system_doc["encryption_seed"]
                if isinstance(seed, str) and len(seed) >= 32:
                    logger.info("✅ Retrieved persistent encryption seed from database")
                    return seed
                else:
                    logger.warning("⚠️ Stored encryption seed is invalid, generating new one")
            
            # Generate new strong encryption seed
            logger.info("🔐 Generating new encryption seed for persistent storage...")
            new_seed = base64.b64encode(secrets.token_bytes(32)).decode('ascii')
            
            # Store the seed in system collection with metadata
            system_config = {
                "type": "encryption_config",
                "encryption_seed": new_seed,
                "created_at": datetime.utcnow(),
                "version": "1.0",
                "description": "Auto-generated encryption seed for API key protection"
            }
            
            await self.db.system_config.update_one(
                {"type": "encryption_config"},
                {"$set": system_config},
                upsert=True
            )
            
            logger.info("✅ Generated and stored new encryption seed in database")
            return new_seed
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to get/create encryption seed: {e}")
            raise ValueError(f"Cannot manage encryption seed: {e}")
    
    async def _initialize_encryption_with_persistent_seed(self):
        """Initialize encryption system with persistent seed from database"""
        try:
            # Get or create the encryption seed from database
            encryption_seed = await self._get_or_create_encryption_seed()
            
            # Store the persistent seed for per-user key derivation
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
            
            logger.info("✅ Encryption system initialized with persistent seed from database")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize encryption with persistent seed: {e}")
            raise
    
    def _encrypt_api_key(self, api_key: str, user_id: int) -> str:
        """Encrypt API key using per-user AES-256-GCM encryption"""
        try:
            if not api_key or not isinstance(api_key, str):
                raise ValueError("API key must be a non-empty string")
            
            if not isinstance(user_id, int):
                raise ValueError("User ID must be an integer")
                
            # Generate per-user encryption key
            user_key = self._derive_user_encryption_key(user_id)
            user_cipher = AESGCM(user_key)
            
            # Generate a random 12-byte nonce for GCM
            nonce = secrets.token_bytes(12)
            
            # Encrypt the API key
            encrypted_data = user_cipher.encrypt(nonce, api_key.encode('utf-8'), None)
            
            # Combine nonce + encrypted data for storage
            combined = nonce + encrypted_data
            
            # Encode to base64 for storage in MongoDB
            encoded = base64.b64encode(combined).decode('ascii')
            
            self.secure_logger.info("🔒 API key encrypted successfully using per-user AES-256-GCM")
            return encoded
            
        except Exception as e:
            self.secure_logger.error(f"CRITICAL: Failed to encrypt API key for user {user_id}: {e}")
            raise ValueError(f"Encryption failed: {e}")
    
    def _decrypt_api_key(self, encrypted_api_key: str, user_id: int) -> str:
        """Decrypt API key using 3-tier backward compatibility approach"""
        try:
            if not encrypted_api_key or not isinstance(encrypted_api_key, str):
                raise ValueError("Encrypted API key must be a non-empty string")
                
            if not isinstance(user_id, int):
                raise ValueError("User ID must be an integer")
            
            if not encrypted_api_key.strip():
                raise ValueError("Encrypted API key must be a non-empty string")
            
            cleaned_key = encrypted_api_key.strip()
            
            # Check if this looks like plaintext first (most efficient)
            if not self._is_encrypted_key(cleaned_key):
                self.secure_logger.info("🔓 Found plaintext API key - will re-encrypt on next save")
                return cleaned_key
            
            # Try per-user decryption (current/preferred method)
            try:
                user_key = self._derive_user_encryption_key(user_id)
                user_cipher = AESGCM(user_key)
                
                # Decode from base64
                combined = base64.b64decode(cleaned_key.encode('ascii'))
                
                # Extract nonce (first 12 bytes) and encrypted data
                if len(combined) >= 13:  # At least 12 bytes nonce + 1 byte data
                    nonce = combined[:12]
                    encrypted_data = combined[12:]
                    
                    # Decrypt with per-user key
                    decrypted_bytes = user_cipher.decrypt(nonce, encrypted_data, None)
                    decrypted_key = decrypted_bytes.decode('utf-8')
                    
                    self.secure_logger.info("🔓 API key decrypted successfully using per-user AES-256-GCM")
                    return decrypted_key
                    
            except Exception as per_user_error:
                logger.debug(f"Per-user decryption failed for user {user_id}: {per_user_error}")
            
            # Try legacy global decryption (backward compatibility)
            try:
                decrypted_key = self._decrypt_legacy_global_key(cleaned_key)
                self.secure_logger.info("🔄 API key decrypted using legacy global cipher - will re-encrypt with per-user key on next save")
                return decrypted_key
                
            except Exception as legacy_error:
                logger.debug(f"Legacy global decryption failed for user {user_id}: {legacy_error}")
            
            # All decryption methods failed
            self.secure_logger.error(f"CRITICAL: All decryption methods failed for user {user_id}")
            raise ValueError("Cannot decrypt API key with any available method")
            
        except ValueError:
            raise
        except Exception as e:
            self.secure_logger.error(f"CRITICAL: Unexpected error decrypting API key for user {user_id}: {e}")
            raise ValueError(f"Decryption failed: {e}")
    
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