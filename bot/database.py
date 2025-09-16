"""
MongoDB database integration for AI Assistant Telegram Bot
Basic database connectivity for future use
"""

import asyncio
import os
from urllib.parse import urlparse, parse_qs
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError
from bot.config import Config
import logging

logger = logging.getLogger(__name__)

class Database:
    """MongoDB database manager for user data"""
    
    def __init__(self):
        self.client = None
        self.db = None
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
        Save or update user's Hugging Face API key in MongoDB
        
        Args:
            user_id (int): Telegram user ID
            api_key (str): Hugging Face API key
            
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
            
            # Use upsert to insert or update the user document
            result = await self.db.users.update_one(
                {"user_id": user_id},
                {"$set": {"user_id": user_id, "hf_api_key": api_key.strip()}},
                upsert=True
            )
            
            if result.acknowledged:
                action = "updated" if result.matched_count > 0 else "created"
                logger.info(f"✅ Successfully {action} API key for user {user_id}")
                return True
            else:
                logger.error(f"Failed to save API key for user {user_id} - operation not acknowledged")
                return False
                
        except PyMongoError as e:
            logger.error(f"MongoDB error saving API key for user {user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving API key for user {user_id}: {e}")
            return False
    
    async def get_user_api_key(self, user_id: int) -> str | None:
        """
        Retrieve user's Hugging Face API key from MongoDB
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            str | None: API key if found, None otherwise
        """
        try:
            if not self.connected or self.db is None:
                raise ValueError("Database not connected")
            
            if not isinstance(user_id, int):
                raise ValueError("user_id must be an integer")
            
            # Find user document by user_id
            user_doc = await self.db.users.find_one({"user_id": user_id})
            
            if user_doc and "hf_api_key" in user_doc:
                api_key = user_doc["hf_api_key"]
                if isinstance(api_key, str) and api_key.strip():
                    logger.info(f"✅ Retrieved API key for user {user_id}")
                    return api_key.strip()
            
            logger.info(f"No API key found for user {user_id}")
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