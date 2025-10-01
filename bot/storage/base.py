"""
Abstract base class for storage providers
Defines the interface that all storage backends must implement
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import asyncio
import logging
from bot.security_utils import InputValidator

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
    
    @abstractmethod
    async def get_user_preference(self, user_id: int, key: str) -> Optional[str]:
        """
        Get a specific user preference value by key
        
        Args:
            user_id (int): Telegram user ID
            key (str): Preference key to retrieve
            
        Returns:
            Optional[str]: Preference value or None if not found
        """
        pass
    
    @abstractmethod
    async def save_user_preference(self, user_id: int, key: str, value: str) -> bool:
        """
        Save a specific user preference value by key
        
        Args:
            user_id (int): Telegram user ID
            key (str): Preference key to save
            value (str): Preference value to save
            
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
    
    # Admin Data Management
    @abstractmethod
    async def get_admin_data(self) -> Optional[Dict[str, Any]]:
        """
        Get admin system configuration data
        
        Returns:
            Optional[Dict[str, Any]]: Admin data including user list, settings, etc.
        """
        pass
    
    @abstractmethod
    async def save_admin_data(self, admin_data: Dict[str, Any]) -> bool:
        """
        Save admin system configuration data
        
        Args:
            admin_data (Dict[str, Any]): Admin data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def log_admin_action(self, admin_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log admin action for audit trail
        
        Args:
            admin_id (int): Admin user ID
            action (str): Action performed
            details (Dict[str, Any]): Additional action details
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_admin_logs(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """
        Get admin action logs for audit trail
        
        Args:
            limit (int): Maximum number of logs to return
            skip (int): Number of logs to skip (for pagination)
            
        Returns:
            List[Dict[str, Any]]: List of admin action logs
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
    
    # CRITICAL FIX: Add comprehensive input validation for database operations
    def _validate_string_input(self, input_value: str, field_name: str, max_length: int = 255, allow_empty: bool = False) -> str:
        """
        Validate and sanitize string inputs to prevent SQL injection and data corruption
        
        Args:
            input_value (str): String to validate
            field_name (str): Name of the field being validated (for error messages)
            max_length (int): Maximum allowed length
            allow_empty (bool): Whether empty strings are allowed
            
        Returns:
            str: Sanitized string value
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(input_value, str):
            raise ValueError(f"{field_name} must be a string")
        
        if not allow_empty and not input_value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        
        if len(input_value) > max_length:
            raise ValueError(f"{field_name} exceeds maximum length of {max_length} characters")
        
        # Use InputValidator for comprehensive sanitization
        validator = InputValidator()
        is_safe, sanitized_value, threat_report = validator.validate_input(input_value)
        
        if not is_safe:
            raise ValueError(f"{field_name} contains potentially malicious content and cannot be processed")
        
        return sanitized_value
    
    def _validate_file_id(self, file_id: str) -> str:
        """Validate file ID with enhanced security checks"""
        return self._validate_string_input(file_id, "file_id", max_length=128, allow_empty=False)
    
    def _validate_conversation_id(self, conversation_id: str) -> str:
        """Validate conversation ID with enhanced security checks"""
        return self._validate_string_input(conversation_id, "conversation_id", max_length=128, allow_empty=False)
    
    def _validate_action_name(self, action: str) -> str:
        """Validate action name for usage logging"""
        return self._validate_string_input(action, "action", max_length=64, allow_empty=False)
    
    def _validate_model_name(self, model_name: str) -> str:
        """Validate model name for usage logging"""
        return self._validate_string_input(model_name, "model_name", max_length=128, allow_empty=False)
    
    def _validate_json_data(self, data: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """
        Validate JSON data structure for database storage
        
        Args:
            data (Dict[str, Any]): JSON data to validate
            field_name (str): Name of the field being validated
            
        Returns:
            Dict[str, Any]: Validated data
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValueError(f"{field_name} must be a dictionary")
        
        # Validate string values within the JSON structure
        def validate_nested_strings(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        validator = InputValidator()
                        is_safe, _, threat_report = validator.validate_input(value)
                        if not is_safe:
                            raise ValueError(f"Potentially malicious content found in {field_name} at path {path}.{key}")
                    elif isinstance(value, (dict, list)):
                        validate_nested_strings(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str):
                        validator = InputValidator()
                        is_safe, _, threat_report = validator.validate_input(item)
                        if not is_safe:
                            raise ValueError(f"Potentially malicious content found in {field_name} at index {i}")
                    elif isinstance(item, (dict, list)):
                        validate_nested_strings(item, f"{path}[{i}]")
        
        validate_nested_strings(data)
        return data
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()