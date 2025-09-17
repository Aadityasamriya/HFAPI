"""
Supabase Storage Provider Implementation
PostgreSQL-based storage backend with real-time capabilities
"""

import asyncio
import os
import base64
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Supabase imports
try:
    from supabase import create_client, Client
    import httpx
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Encryption imports
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import logging
from .base import StorageProvider
from bot.core.model_caller import SecureLogger

logger = logging.getLogger(__name__)

class SupabaseProvider(StorageProvider):
    """
    Supabase storage provider implementation
    
    Uses PostgreSQL with real-time capabilities and built-in auth
    Supports per-user database isolation and advanced features
    """
    
    def __init__(self):
        super().__init__()
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase dependencies not available. Install with: pip install supabase")
        
        self.client: Optional[Client] = None
        self._encryption_key = None
        self._aesgcm = None
        self._global_seed = None
        self.secure_logger = SecureLogger(logger)
        
        # Configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_anon_key = os.getenv('SUPABASE_ANON_KEY')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')  # For admin operations
        
        if not self.supabase_url or not self.supabase_anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be configured")
    
    # Connection Management
    async def connect(self) -> None:
        """Establish connection to Supabase"""
        try:
            # Use service key if available for admin operations, otherwise anon key
            api_key = self.supabase_service_key or self.supabase_anon_key
            
            self.client = create_client(self.supabase_url, api_key)
            
            # Test connection with simple query
            try:
                # Try to query a system table to verify connection
                result = self.client.from_('information_schema.tables').select('table_name').limit(1).execute()
                logger.info("✅ Successfully connected to Supabase")
            except Exception as e:
                # If we can't access system tables, try a simpler test
                logger.info("✅ Supabase client created successfully")
            
            self.connected = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.connected = False
            self.client = None
            raise
    
    async def disconnect(self) -> None:
        """Close Supabase connection gracefully"""
        if self.client:
            try:
                # Supabase client doesn't need explicit closure
                logger.info("✅ Supabase connection closed gracefully")
            except Exception as e:
                logger.warning(f"Error closing Supabase connection: {e}")
            finally:
                self.client = None
                self.connected = False
    
    async def initialize(self) -> None:
        """Initialize Supabase (create tables, setup encryption, etc.)"""
        if not self.connected:
            raise ValueError("Must be connected before initialization")
        
        # Create necessary tables
        await self._create_tables()
        
        # Initialize encryption
        await self._initialize_encryption_with_persistent_seed()
        
        self._encryption_initialized = True
        logger.info("✅ Supabase provider initialized successfully")
    
    async def health_check(self) -> bool:
        """Perform health check on Supabase"""
        try:
            if not self.connected or not self.client:
                return False
            
            # Simple query to test connection
            result = self.client.from_('system_config').select('type').limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Supabase health check failed: {e}")
            return False
    
    # API Key Management
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """Save or update user's Hugging Face API key with encryption"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("api_key must be a non-empty string")
            
            # Encrypt the API key using per-user encryption
            encrypted_api_key = self._encrypt_api_key(api_key.strip(), user_id)
            
            # Upsert user with encrypted API key
            user_data = {
                "user_id": user_id,
                "hf_api_key": encrypted_api_key,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Try to update first, then insert if doesn't exist
            result = self.client.from_('users').upsert(user_data, on_conflict='user_id').execute()
            
            if result.data:
                action = "updated" if len(result.data) > 0 else "created"
                self.secure_logger.info(f"🔒 Successfully {action} encrypted API key for user {user_id}")
                
                # Security audit log
                self.secure_logger.audit(
                    event_type="API_KEY_SAVE",
                    user_id=user_id,
                    details=f"API key {action} successfully with per-user encryption",
                    success=True
                )
                return True
            else:
                logger.error(f"Failed to save API key for user {user_id}")
                return False
                
        except ValueError as e:
            self.secure_logger.error(f"Validation/encryption error saving API key for user {user_id}: {e}")
            self.secure_logger.audit(
                event_type="API_KEY_SAVE_FAILED",
                user_id=user_id,
                details=f"API key save failed: {str(e)[:100]}",
                success=False
            )
            return False
        except Exception as e:
            logger.error(f"Supabase error saving API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """Retrieve and decrypt user's Hugging Face API key"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Find user by user_id
            result = self.client.from_('users').select('hf_api_key').eq('user_id', user_id).execute()
            
            if result.data and len(result.data) > 0:
                user_data = result.data[0]
                encrypted_api_key = user_data.get('hf_api_key')
                
                if encrypted_api_key and isinstance(encrypted_api_key, str):
                    # Decrypt the API key
                    decrypted_api_key = self._decrypt_api_key(encrypted_api_key.strip(), user_id)
                    
                    # If it was a legacy plaintext key, re-encrypt it
                    if not self._is_encrypted_key(encrypted_api_key.strip()):
                        logger.info(f"🔄 Converting legacy plaintext API key to encrypted format for user {user_id}")
                        # Re-save with encryption (background task)
                        asyncio.create_task(self.save_user_api_key(user_id, decrypted_api_key))
                    
                    self.secure_logger.info(f"🔓 Successfully retrieved and decrypted API key for user {user_id}")
                    
                    # Security audit log
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
                details=f"API key retrieval failed: {str(e)[:100]}",
                success=False
            )
            return None
        except Exception as e:
            logger.error(f"Supabase error retrieving API key for user {user_id}: {e}")
            return None
    
    # User Data Management
    async def reset_user_database(self, user_id: int) -> bool:
        """Delete all user data including API keys and conversations"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Delete user data and conversations in parallel
            user_result = self.client.from_('users').delete().eq('user_id', user_id).execute()
            conv_result = self.client.from_('conversations').delete().eq('user_id', user_id).execute()
            files_result = self.client.from_('files').delete().eq('user_id', user_id).execute()
            
            deleted_conversations = len(conv_result.data) if conv_result.data else 0
            deleted_files = len(files_result.data) if files_result.data else 0
            
            self.secure_logger.info(f"🗑️ Successfully deleted user data for user {user_id}")
            self.secure_logger.info(f"📊 Deleted: user document, {deleted_conversations} conversations, {deleted_files} files")
            
            # Security audit log
            self.secure_logger.audit(
                event_type="USER_DATA_DELETE",
                user_id=user_id,
                details=f"User data deleted: user + {deleted_conversations} conversations + {deleted_files} files",
                success=True
            )
            return True
            
        except ValueError as e:
            self.secure_logger.error(f"Validation error deleting user data for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Supabase error deleting user data for user {user_id}: {e}")
            self.secure_logger.audit(
                event_type="USER_DATA_DELETE_FAILED",
                user_id=user_id,
                details=f"User data deletion failed: {str(e)[:100]}",
                success=False
            )
            return False
    
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get user preferences and settings"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            result = self.client.from_('users').select('preferences').eq('user_id', user_id).execute()
            
            if result.data and len(result.data) > 0:
                preferences = result.data[0].get('preferences')
                if preferences:
                    return preferences if isinstance(preferences, dict) else json.loads(preferences)
            
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
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            user_data = {
                "user_id": user_id,
                "preferences": json.dumps(preferences),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.client.from_('users').upsert(user_data, on_conflict='user_id').execute()
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Error saving user preferences for user {user_id}: {e}")
            return False
    
    # Conversation Storage
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """Save conversation history with metadata"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            self._validate_conversation_data(conversation_data)
            
            # Prepare conversation document for PostgreSQL
            conversation_doc = {
                'user_id': user_id,
                'messages': json.dumps(conversation_data['messages']),
                'summary': conversation_data['summary'],
                'started_at': conversation_data['started_at'].isoformat() if isinstance(conversation_data['started_at'], datetime) else conversation_data['started_at'],
                'last_message_at': conversation_data['last_message_at'].isoformat() if isinstance(conversation_data['last_message_at'], datetime) else conversation_data['last_message_at'],
                'message_count': conversation_data['message_count'],
                'created_at': conversation_data['started_at'].isoformat() if isinstance(conversation_data['started_at'], datetime) else conversation_data['started_at']
            }
            
            # Insert conversation
            result = self.client.from_('conversations').insert(conversation_doc).execute()
            
            if result.data and len(result.data) > 0:
                conv_id = result.data[0].get('id')
                logger.info(f"💾 Successfully saved conversation for user {user_id} (ID: {conv_id})")
                logger.info(f"📊 Conversation saved: {conversation_data['message_count']} messages, {len(conversation_data['summary'])} char summary")
                return True
            else:
                logger.error(f"Failed to save conversation for user {user_id}")
                return False
                
        except ValueError as e:
            logger.error(f"Validation error saving conversation for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Supabase error saving conversation for user {user_id}: {e}")
            return False
    
    async def get_user_conversations(self, user_id: int, limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
        """Get conversation summaries for history browsing with pagination"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Query conversations with pagination
            result = self.client.from_('conversations').select(
                'id, summary, started_at, last_message_at, message_count'
            ).eq('user_id', user_id).order('last_message_at', desc=True).range(skip, skip + limit - 1).execute()
            
            conversations = []
            if result.data:
                for conv in result.data:
                    conversations.append({
                        "id": str(conv["id"]),
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
            logger.error(f"Supabase error retrieving conversations for user {user_id}: {e}")
            return []
    
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed conversation data by ID"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Find conversation by ID and user_id (security)
            result = self.client.from_('conversations').select('*').eq('id', conversation_id).eq('user_id', user_id).execute()
            
            if result.data and len(result.data) > 0:
                conversation = result.data[0]
                
                # Parse JSON messages back to list
                if conversation.get('messages'):
                    try:
                        conversation['messages'] = json.loads(conversation['messages'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse messages JSON for conversation {conversation_id}")
                
                logger.info(f"📄 Retrieved full conversation details for user {user_id}, conversation {conversation_id}")
                logger.info(f"📊 Conversation has {conversation.get('message_count', 0)} messages")
                return conversation
            else:
                logger.info(f"No conversation found for user {user_id} with ID {conversation_id}")
                return None
            
        except ValueError as e:
            logger.error(f"Validation error retrieving conversation details for user {user_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Supabase error retrieving conversation details for user {user_id}: {e}")
            return None
    
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """Delete a specific conversation by ID"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Delete conversation ensuring it belongs to the user (security)
            result = self.client.from_('conversations').delete().eq('id', conversation_id).eq('user_id', user_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"🗑️ Successfully deleted conversation {conversation_id} for user {user_id}")
                return True
            else:
                logger.warning(f"No conversation deleted - not found or doesn't belong to user {user_id}")
                return False
                
        except ValueError as e:
            logger.error(f"Validation error deleting conversation for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Supabase error deleting conversation for user {user_id}: {e}")
            return False
    
    async def clear_user_history(self, user_id: int) -> bool:
        """Clear all conversation history for a user"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Delete all conversations for the user
            result = self.client.from_('conversations').delete().eq('user_id', user_id).execute()
            
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"🗑️ Successfully cleared {deleted_count} conversations for user {user_id}")
            return True
                
        except Exception as e:
            logger.error(f"Supabase error clearing conversation history for user {user_id}: {e}")
            return False
    
    async def get_conversation_count(self, user_id: int) -> int:
        """Get total number of conversations for a user"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            result = self.client.from_('conversations').select('id', count='exact').eq('user_id', user_id).execute()
            
            count = result.count if result.count is not None else 0
            logger.info(f"📊 User {user_id} has {count} saved conversations")
            return count
            
        except Exception as e:
            logger.error(f"Error counting conversations for user {user_id}: {e}")
            return 0
    
    # File Storage
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """Save file data with metadata"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            # Store file in files table
            file_doc = {
                "user_id": user_id,
                "file_id": file_id,
                "data": base64.b64encode(file_data).decode('ascii'),
                "metadata": json.dumps(metadata),
                "created_at": datetime.utcnow().isoformat()
            }
            
            result = self.client.from_('files').upsert(file_doc, on_conflict='user_id,file_id').execute()
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Error saving file for user {user_id}: {e}")
            return False
    
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve file data and metadata"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            result = self.client.from_('files').select('*').eq('user_id', user_id).eq('file_id', file_id).execute()
            
            if result.data and len(result.data) > 0:
                file_doc = result.data[0]
                
                # Decode base64 data
                if file_doc.get('data'):
                    file_doc["data"] = base64.b64decode(file_doc["data"].encode('ascii'))
                
                # Parse metadata JSON
                if file_doc.get('metadata'):
                    try:
                        file_doc["metadata"] = json.loads(file_doc["metadata"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata JSON for file {file_id}")
                
                return file_doc
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving file for user {user_id}: {e}")
            return None
    
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """Delete file data"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            result = self.client.from_('files').delete().eq('user_id', user_id).eq('file_id', file_id).execute()
            return result.data is not None and len(result.data) > 0
            
        except Exception as e:
            logger.error(f"Error deleting file for user {user_id}: {e}")
            return False
    
    # Analytics and Metrics
    async def log_usage(self, user_id: int, action: str, model_used: str, tokens_used: int = 0) -> bool:
        """Log usage metrics for analytics"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            usage_doc = {
                "user_id": user_id,
                "action": action,
                "model_used": model_used,
                "tokens_used": tokens_used,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result = self.client.from_('usage_logs').insert(usage_doc).execute()
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Error logging usage for user {user_id}: {e}")
            return False
    
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database not connected")
            
            user_id = self._validate_user_id(user_id)
            
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Get usage logs for the time period
            result = self.client.from_('usage_logs').select('action, tokens_used').eq('user_id', user_id).gte('timestamp', cutoff_date).execute()
            
            stats = {}
            total_requests = 0
            total_tokens = 0
            
            if result.data:
                # Aggregate statistics
                action_stats = {}
                for log in result.data:
                    action = log.get('action', 'unknown')
                    tokens = log.get('tokens_used', 0)
                    
                    if action not in action_stats:
                        action_stats[action] = {"count": 0, "tokens": 0}
                    
                    action_stats[action]["count"] += 1
                    action_stats[action]["tokens"] += tokens
                    
                    total_requests += 1
                    total_tokens += tokens
                
                stats = action_stats
            
            stats["summary"] = {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "days": days
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving usage stats for user {user_id}: {e}")
            return {}
    
    # Private Helper Methods
    async def _create_tables(self):
        """Create necessary tables in Supabase if they don't exist"""
        try:
            # Note: In a real implementation, you would use Supabase migrations
            # or SQL DDL commands. For this example, we assume tables exist.
            # The table schemas would be created via Supabase dashboard or CLI:
            
            # Tables needed:
            # - users (id, user_id unique, hf_api_key, preferences, created_at, updated_at)
            # - conversations (id, user_id, messages, summary, started_at, last_message_at, message_count, created_at)
            # - files (id, user_id, file_id, data, metadata, created_at, unique(user_id, file_id))
            # - usage_logs (id, user_id, action, model_used, tokens_used, timestamp)
            # - system_config (id, type unique, encryption_seed, created_at, version, description)
            
            logger.info("✅ Table schema assumed to be created via Supabase migrations")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    async def _get_or_create_encryption_seed(self) -> str:
        """Get encryption seed from database or create and store a new one"""
        try:
            if not self.connected or not self.client:
                raise ValueError("Database must be connected to manage encryption seed")
            
            # Try to get existing seed from system_config
            result = self.client.from_('system_config').select('encryption_seed').eq('type', 'encryption_config').execute()
            
            if result.data and len(result.data) > 0:
                seed = result.data[0].get('encryption_seed')
                if isinstance(seed, str) and len(seed) >= 32:
                    logger.info("✅ Retrieved persistent encryption seed from database")
                    return seed
                else:
                    logger.warning("⚠️ Stored encryption seed is invalid, generating new one")
            
            # Generate new strong encryption seed
            logger.info("🔐 Generating new encryption seed for persistent storage...")
            new_seed = base64.b64encode(secrets.token_bytes(32)).decode('ascii')
            
            # Store the seed in system_config
            system_config = {
                "type": "encryption_config",
                "encryption_seed": new_seed,
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "description": "Auto-generated encryption seed for API key protection"
            }
            
            self.client.from_('system_config').upsert(system_config, on_conflict='type').execute()
            
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
            
            logger.info("✅ Encryption system initialized with persistent seed from database")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize encryption with persistent seed: {e}")
            raise
    
    # Encryption methods (same as MongoDB provider for compatibility)
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
            
            # Encode to base64 for storage
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