"""Fast, deterministic regression tests for HFAPI's core contracts.

These tests intentionally avoid external APIs and databases. They are safe to run
in CI and catch structural regressions before deployment.
"""

from pathlib import Path

import pytest

from bot.core.bot_types import IntentType as CanonicalIntentType
from bot.core.types import IntentType as CompatibilityIntentType
from bot.core.ai_providers import APIMode, ProviderConfig
from bot.core.hf_inference_provider import HFInferenceProvider
from bot.core.provider_factory import ProviderFactory


ROOT = Path(__file__).resolve().parents[1]


def test_all_python_files_compile():
    """Every tracked Python source file must remain syntactically valid."""
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


def test_intent_type_has_single_canonical_identity():
    """Legacy imports must resolve to the same IntentType class."""
    assert CompatibilityIntentType is CanonicalIntentType
    assert CompatibilityIntentType.TEXT_GENERATION.value == "text_generation"
    assert CompatibilityIntentType.UNKNOWN.value == "unknown"


def test_provider_config_contract():
    config = ProviderConfig(
        api_mode=APIMode.INFERENCE_PROVIDERS,
        api_key=None,
        base_url=None,
        provider_name=None,
        organization=None,
        timeout=30,
        max_retries=3,
        retry_delay=1.0,
    )
    assert config.timeout > 0
    assert config.max_retries >= 0


def test_hf_generation_parameter_sanitization():
    provider = HFInferenceProvider(
        ProviderConfig(
            api_mode=APIMode.INFERENCE_PROVIDERS,
            api_key=None,
            base_url=None,
            provider_name=None,
            organization=None,
        )
    )
    params = provider._validate_and_sanitize_generation_params(
        {
            "max_new_tokens": -10,
            "temperature": 99,
            "top_p": -1,
            "top_k": 0,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        }
    )
    assert params["max_new_tokens"] == 512
    assert params["temperature"] == 2.0
    assert params["top_p"] == 0.0
    assert "top_k" not in params
    assert params["repetition_penalty"] == 1.1
    assert params["return_full_text"] is False


def test_provider_factory_rejects_unknown_api_mode():
    """Explicitly invalid modes must fail fast instead of silently changing modes."""
    with pytest.raises(ValueError, match="Invalid HF API mode"):
        ProviderFactory.create_provider(api_mode="unsupported-mode", api_key="hf_test_token")
