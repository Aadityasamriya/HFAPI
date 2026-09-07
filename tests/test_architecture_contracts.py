"""Architecture regression tests for the routing/provider layer."""


def test_core_imports_and_canonical_intents():
    from bot.core import IntentType as CoreIntentType
    from bot.core.bot_types import IntentType as BotIntentType
    from bot.core.types import IntentType as CompatibilityIntentType

    assert CoreIntentType is BotIntentType is CompatibilityIntentType


def test_router_can_be_constructed_without_external_services():
    from bot.core.router import IntelligentRouter

    router = IntelligentRouter()
    complexity = router.complexity_analyzer.analyze_complexity("Explain how a Python API works")
    assert 0.0 <= complexity.complexity_score <= 10.0
    assert complexity.context_length >= 0


def test_provider_factory_reports_hf_as_available_provider():
    from bot.core.provider_factory import ProviderFactory

    info = ProviderFactory.get_provider_info()
    assert "hf_inference" in info["available_providers"]
    assert info["current_api_mode"] in {"auto", "inference_api", "inference_providers"}
