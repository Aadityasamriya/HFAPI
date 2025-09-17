"""
Abstract base class for storage providers
Defines the interface that all storage backends must implement
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import asyncio
import logging

logger = logging.getLogger(__name__)

class StorageProvider(ABC):
    """
    Abstract base class for storage providers
    
    This defines the interface that all storage backends must implement,
    ensuring consistent API across MongoDB, Supabase, and other providers.
    """
    
    def __init__(self):
        self.connected = False
        self._encryption_initialized = False
    
    # Connection Management
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the storage backend
        
        Raises:
            ConnectionError: If connection fails
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the storage backend gracefully
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize storage backend (create indexes, setup encryption, etc.)
        This should be called after successful connection
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform health check on storage backend
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass
    
    # API Key Management
    @abstractmethod
    async def save_user_api_key(self, user_id: int, api_key: str) -> bool:
        """
        Save or update user's Hugging Face API key with encryption
        
        Args:
            user_id (int): Telegram user ID
            api_key (str): Hugging Face API key (will be encrypted before storage)
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_user_api_key(self, user_id: int) -> Optional[str]:
        """
        Retrieve and decrypt user's Hugging Face API key
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Optional[str]: Decrypted API key or None if not found
        """
        pass
    
    # User Data Management
    @abstractmethod
    async def reset_user_database(self, user_id: int) -> bool:
        """
        Delete all user data including API keys and conversations
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Get user preferences and settings
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Dict[str, Any]: User preferences
        """
        pass
    
    @abstractmethod
    async def save_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences and settings
        
        Args:
            user_id (int): Telegram user ID
            preferences (Dict[str, Any]): User preferences to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    # Conversation Storage
    @abstractmethod
    async def save_conversation(self, user_id: int, conversation_data: Dict[str, Any]) -> bool:
        """
        Save conversation history with metadata
        
        Args:
            user_id (int): Telegram user ID
            conversation_data (Dict[str, Any]): Conversation data including messages, summary, etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def get_conversation_details(self, user_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed conversation data by ID
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): Conversation identifier
            
        Returns:
            Optional[Dict[str, Any]]: Full conversation data or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_conversation(self, user_id: int, conversation_id: str) -> bool:
        """
        Delete a specific conversation by ID
        
        Args:
            user_id (int): Telegram user ID (for security)
            conversation_id (str): Conversation identifier
        
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear_user_history(self, user_id: int) -> bool:
        """
        Clear all conversation history for a user
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_conversation_count(self, user_id: int) -> int:
        """
        Get total number of conversations for a user
        
        Args:
            user_id (int): Telegram user ID
        
        Returns:
            int: Number of conversations (0 if error)
        """
        pass
    
    # File Storage (for future use)
    @abstractmethod
    async def save_file(self, user_id: int, file_id: str, file_data: bytes, metadata: Dict[str, Any]) -> bool:
        """
        Save file data with metadata
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            file_data (bytes): File content
            metadata (Dict[str, Any]): File metadata (name, type, size, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_file(self, user_id: int, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve file data and metadata
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            
        Returns:
            Optional[Dict[str, Any]]: File data and metadata or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_file(self, user_id: int, file_id: str) -> bool:
        """
        Delete file data
        
        Args:
            user_id (int): Telegram user ID
            file_id (str): Unique file identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    # Analytics and Metrics (for future use)
    @abstractmethod
    async def log_usage(self, user_id: int, action: str, model_used: str, tokens_used: int = 0) -> bool:
        """
        Log usage metrics for analytics
        
        Args:
            user_id (int): Telegram user ID
            action (str): Action performed (text_generation, image_generation, etc.)
            model_used (str): AI model used
            tokens_used (int): Number of tokens consumed
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_usage_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """
        Get usage statistics for a user
        
        Args:
            user_id (int): Telegram user ID
            days (int): Number of days to look back
            
        Returns:
            Dict[str, Any]: Usage statistics
        """
        pass
    
    # Security and Encryption (implementation-specific)
    def _validate_user_id(self, user_id: Union[int, str]) -> int:
        """
        Validate and convert user_id to integer
        
        Args:
            user_id (Union[int, str]): User ID to validate
            
        Returns:
            int: Validated user ID
            
        Raises:
            ValueError: If user_id is invalid
        """
        try:
            uid = int(user_id)
            if uid <= 0:
                raise ValueError("User ID must be positive")
            return uid
        except (ValueError, TypeError):
            raise ValueError(f"Invalid user_id: {user_id}. Must be a positive integer.")
    
    def _validate_conversation_data(self, conversation_data: Dict[str, Any]) -> None:
        """
        Validate conversation data structure
        
        Args:
            conversation_data (Dict[str, Any]): Conversation data to validate
            
        Raises:
            ValueError: If conversation data is invalid
        """
        if not isinstance(conversation_data, dict):
            raise ValueError("conversation_data must be a dictionary")
        
        required_fields = ['messages', 'summary', 'started_at', 'last_message_at', 'message_count']
        for field in required_fields:
            if field not in conversation_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(conversation_data['messages'], list):
            raise ValueError("messages must be a list")
        
        if not isinstance(conversation_data['message_count'], int) or conversation_data['message_count'] < 0:
            raise ValueError("message_count must be a non-negative integer")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()