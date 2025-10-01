"""
Provider Factory for AI Provider Abstraction
Creates appropriate AI providers based on configuration and feature flags
"""

import logging
from typing import Optional, Dict, Any
from bot.config import Config
from .ai_providers import AIProvider, ProviderConfig, APIMode
from .hf_inference_provider import HFInferenceProvider

logger = logging.getLogger(__name__)

class ProviderFactory:
    """
    Factory class for creating AI providers based on configuration
    Handles the logic of choosing the right provider implementation
    """
    
    @classmethod
    def create_provider(cls, provider_type: Optional[str] = None, **kwargs) -> AIProvider:
        """
        Create appropriate AI provider based on configuration
        
        Args:
            provider_type (Optional[str]): Specific provider type to create
            **kwargs: Additional configuration parameters
            
        Returns:
            AIProvider: Configured AI provider instance
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If provider cannot be created
        """
        # Determine API mode from config or parameter
        api_mode_str = kwargs.get('api_mode', Config.HF_API_MODE)
        
        try:
            if api_mode_str == 'inference_providers':
                api_mode = APIMode.INFERENCE_PROVIDERS
            elif api_mode_str == 'auto':
                api_mode = APIMode.AUTO
            else:
                api_mode = APIMode.INFERENCE_API  # Default/backward compatibility
        except (AttributeError, ValueError):
            api_mode = APIMode.INFERENCE_API
        
        # Get API key
        api_key = kwargs.get('api_key', Config.get_hf_token())
        if not api_key:
            logger.warning("⚠️ No HF API key provided - AI functionality may be limited")
        
        # Create provider configuration
        provider_config = ProviderConfig(
            api_mode=api_mode,
            api_key=api_key,
            base_url=kwargs.get('base_url', Config.HF_API_BASE_URL),
            provider_name=kwargs.get('provider_name', Config.HF_PROVIDER),
            organization=kwargs.get('organization', Config.HF_ORG),
            timeout=kwargs.get('timeout', Config.REQUEST_TIMEOUT),
            max_retries=kwargs.get('max_retries', Config.MAX_RETRIES),
            retry_delay=kwargs.get('retry_delay', Config.RETRY_DELAY)
        )
        
        # Create the appropriate provider
        if provider_type == 'hf_inference' or api_mode in [APIMode.INFERENCE_API, APIMode.INFERENCE_PROVIDERS, APIMode.AUTO]:
            logger.info(f"🏭 Creating HFInferenceProvider (mode: {api_mode.value})")
            return HFInferenceProvider(provider_config)
        else:
            # Default to HFInferenceProvider for now
            # Future providers (OpenAI, Anthropic, etc.) can be added here
            logger.info(f"🏭 Creating default HFInferenceProvider (mode: {api_mode.value})")
            return HFInferenceProvider(provider_config)
    
    @classmethod
    def get_provider_info(cls) -> Dict[str, Any]:
        """
        Get information about available providers and current configuration
        
        Returns:
            Dict[str, Any]: Provider information
        """
        return {
            'available_providers': ['hf_inference'],
            'current_api_mode': Config.HF_API_MODE,
            'has_api_key': bool(Config.get_hf_token()),
            'base_url': Config.HF_API_BASE_URL,
            'provider_name': Config.HF_PROVIDER,
            'organization': Config.HF_ORG,
            'timeout': Config.REQUEST_TIMEOUT,
            'max_retries': Config.MAX_RETRIES
        }
    
    @classmethod
    def validate_configuration(cls) -> tuple[bool, str]:
        """
        Validate provider configuration
        
        Returns:
            tuple[bool, str]: (is_valid, message)
        """
        # Check API mode
        valid_modes = ['inference_api', 'inference_providers', 'auto']
        if Config.HF_API_MODE not in valid_modes:
            return False, f"Invalid HF_API_MODE: {Config.HF_API_MODE}. Must be one of: {valid_modes}"
        
        # Check API key availability
        if not Config.get_hf_token():
            return False, "No HF API key found. Please set HF_TOKEN environment variable."
        
        # Validate API key format (basic check)
        api_key = Config.get_hf_token()
        if api_key is None or len(api_key) < 20 or not api_key.startswith(('hf_', 'api_')):
            return False, "HF API key appears to have invalid format."
        
        # Check timeout values
        if Config.REQUEST_TIMEOUT <= 0:
            return False, f"Invalid REQUEST_TIMEOUT: {Config.REQUEST_TIMEOUT}. Must be positive."
        
        if Config.MAX_RETRIES < 0:
            return False, f"Invalid MAX_RETRIES: {Config.MAX_RETRIES}. Must be non-negative."
        
        return True, "Provider configuration is valid."