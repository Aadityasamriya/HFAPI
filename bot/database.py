"""
MongoDB database integration for user API key management
Secure storage for Telegram user IDs and Hugging Face API keys with encryption
"""

import asyncio
import base64
import os
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
        """Get encryption key from environment or generate one"""
        key_b64 = os.getenv('ENCRYPTION_KEY')
        if key_b64:
            try:
                return base64.urlsafe_b64decode(key_b64)
            except Exception:
                logger.warning("Invalid encryption key in environment, generating new one")
        
        # Generate new key for this session
        key = Fernet.generate_key()
        logger.info("Generated new encryption key for this session")
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt a string"""
        if not data:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
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
    
    async def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(Config.MONGO_URI)
            self.db = self.client.ai_assistant_bot
            self.users_collection = self.db.users
            
            # Create unique index on user_id
            await self.users_collection.create_index("user_id", unique=True)
            
            # Test connection
            await self.client.admin.command('ping')
            self.connected = True
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
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
            # Encrypt the API key before storing
            encrypted_key = encryption_manager.encrypt(api_key)
            
            await self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"hf_api_key": encrypted_key}},
                upsert=True
            )
            logger.info(f"Encrypted API key saved for user {user_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Failed to save API key for user {user_id}: {e}")
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
                
            user_doc = await self.users_collection.find_one({"user_id": user_id})
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
            result = await self.users_collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                logger.info(f"User data reset for user {user_id}")
                return True
            else:
                logger.warning(f"No data found to reset for user {user_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Failed to reset user data for user {user_id}: {e}")
            return False
    
    async def get_user_count(self) -> int:
        """Get total number of users in database"""
        try:
            count = await self.users_collection.count_documents({})
            return count
        except PyMongoError as e:
            logger.error(f"Failed to get user count: {e}")
            return 0
    
    async def get_all_users(self) -> list:
        """Get all user IDs (for admin purposes)"""
        try:
            cursor = self.users_collection.find({}, {"user_id": 1, "_id": 0})
            users = await cursor.to_list(length=None)
            return [user["user_id"] for user in users]
        except PyMongoError as e:
            logger.error(f"Failed to get all users: {e}")
            return []

# Global database instance
db = Database()