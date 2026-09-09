"""Provider Factory for AI Provider Abstraction
Creates appropriate AI providers based on configuration and feature flags
"""

import logging
from typing import Optional, Dict, Any
from bot.config import Config
from .ai_providers import AIProvider, ProviderConfig, APIMode
from .hf_inference_provider import HFInferenceProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory class for creating AI providers based on configuration."""

    @classmethod
    def create_provider(cls, provider_type: Optional[str] = None, **kwargs) -> AIProvider:
        """Create the configured AI provider.

        Raises:
            ValueError: If ``api_mode`` is explicitly unsupported.
        """
        api_mode_str = kwargs.get('api_mode', Config.HF_API_MODE)

        mode_map = {
            'inference_api': APIMode.INFERENCE_API,
            'inference_providers': APIMode.INFERENCE_PROVIDERS,
            'auto': APIMode.AUTO,
        }
        try:
            api_mode = mode_map[api_mode_str]
        except (KeyError, TypeError):
            valid_modes = ', '.join(mode_map)
            raise ValueError(
                f"Invalid HF API mode: {api_mode_str!r}. "
                f"Must be one of: {valid_modes}"
            ) from None

        api_key = kwargs.get('api_key', Config.get_hf_token())
        if not api_key:
            logger.warning("⚠️ No HF API key provided - AI functionality may be limited")

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

        logger.info("🏭 Creating HFInferenceProvider (mode: %s)", api_mode.value)
        return HFInferenceProvider(provider_config)

    @classmethod
    def get_provider_info(cls) -> Dict[str, Any]:
        """Get information about available providers and current configuration."""
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
        """Validate provider configuration without contacting external services."""
        valid_modes = ['inference_api', 'inference_providers', 'auto']
        if Config.HF_API_MODE not in valid_modes:
            return False, f"Invalid HF_API_MODE: {Config.HF_API_MODE}. Must be one of: {valid_modes}"

        if not Config.get_hf_token():
            return False, "No HF API key found. Please set HF_TOKEN environment variable."

        api_key = Config.get_hf_token()
        if api_key is None or len(api_key) < 20 or not api_key.startswith(('hf_', 'api_')):
            return False, "HF API key appears to have invalid format."

        if Config.REQUEST_TIMEOUT <= 0:
            return False, f"Invalid REQUEST_TIMEOUT: {Config.REQUEST_TIMEOUT}. Must be positive."

        if Config.MAX_RETRIES < 0:
            return False, f"Invalid MAX_RETRIES: {Config.MAX_RETRIES}. Must be non-negative."

        return True, "Provider configuration is valid."
