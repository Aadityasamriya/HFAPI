"""
AI Provider Abstraction System for Phase 1 HF API Migration
Provides a unified interface for different AI providers with OpenAI-compatible format
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class APIMode(Enum):
    """API modes supported by the provider system"""
    INFERENCE_API = "inference_api"  # Direct HF Inference API (current)
    INFERENCE_PROVIDERS = "inference_providers"  # New HF Inference Providers API
    AUTO = "auto"  # Automatically choose based on configuration

@dataclass
class ProviderConfig:
    """Configuration for AI providers"""
    api_mode: APIMode
    api_key: Optional[str]
    base_url: Optional[str]
    provider_name: Optional[str]
    organization: Optional[str]
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class ChatMessage:
    """OpenAI-compatible chat message format"""
    role: str  # "system", "user", "assistant"
    content: str
    
@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request"""
    messages: List[ChatMessage]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    
@dataclass
class CompletionRequest:
    """Plain text completion request"""
    prompt: str
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

@dataclass
class ProviderResponse:
    """Standardized response from AI providers"""
    success: bool
    content: Any
    error_message: Optional[str] = None
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict] = None

class AIProvider(ABC):
    """
    Abstract base class for AI providers
    Defines the interface that all AI providers must implement
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the AI provider
        
        Args:
            config (ProviderConfig): Provider configuration
        """
        self.config = config
        self._session = None
        self._session_lock = asyncio.Lock()
        
    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ProviderResponse:
        """
        Generate chat completion using OpenAI-compatible format
        
        Args:
            request (ChatCompletionRequest): Chat completion request
            
        Returns:
            ProviderResponse: Standardized response
        """
        pass
    
    @abstractmethod
    async def text_completion(self, request: CompletionRequest) -> ProviderResponse:
        """
        Generate text completion from plain prompt
        
        Args:
            request (CompletionRequest): Text completion request
            
        Returns:
            ProviderResponse: Standardized response
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible
        
        Returns:
            bool: True if provider is healthy
        """
        pass
    
    @abstractmethod
    async def get_supported_models(self) -> List[str]:
        """
        Get list of models supported by this provider
        
        Returns:
            List[str]: List of supported model names
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self._cleanup()
    
    async def _cleanup(self):
        """Cleanup provider resources"""
        async with self._session_lock:
            if self._session and hasattr(self._session, 'close'):
                close_method = getattr(self._session, 'close', None)
                if close_method and callable(close_method):
                    await close_method()
                self._session = None
    
    def _format_chat_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Convert chat messages to a single prompt for models that don't support chat format
        
        Args:
            messages (List[ChatMessage]): Chat messages
            
        Returns:
            str: Formatted prompt
        """
        if not messages:
            return ""
        
        # Extract the last user message as the main prompt
        user_messages = [msg for msg in messages if msg.role == "user"]
        if user_messages:
            return user_messages[-1].content
        
        # Fallback: concatenate all messages
        return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in messages])
    
    def _format_prompt_to_chat_messages(self, prompt: str) -> List[ChatMessage]:
        """
        Convert plain prompt to chat messages format
        
        Args:
            prompt (str): Plain text prompt
            
        Returns:
            List[ChatMessage]: Chat messages
        """
        return [ChatMessage(role="user", content=prompt)]

class ProviderError(Exception):
    """Base exception for provider errors"""
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.model = model

class ModelNotAvailableError(ProviderError):
    """Raised when a model is not available"""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded"""
    pass

class AuthenticationError(ProviderError):
    """Raised when authentication fails"""
    pass

class TimeoutError(ProviderError):
    """Raised when request times out"""
    pass