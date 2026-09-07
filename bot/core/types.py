#!/usr/bin/env python3
"""
Compatibility types for the HFAPI core.

IntentType is intentionally re-exported from bot_types so the project has one
canonical intent enum.  The remaining dataclasses are kept here for backwards
compatibility with older modules that import them from ``core.types``.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .bot_types import IntentType


@dataclass
class PromptComplexity:
    """Represents the compact complexity analysis used by legacy callers."""

    score: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ModelPerformance:
    """Tracks model performance metrics for legacy callers."""

    model_name: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    quality_score: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class ContextState:
    """Maintains conversation and processing context for legacy callers."""

    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)


@dataclass
class ClassificationResult:
    """Backward-compatible classification result contract."""

    intent: IntentType
    confidence: float
    secondary_intent: Optional[IntentType] = None
    secondary_confidence: float = 0.0
    reasoning: str = ""
    detected_features: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    complexity: Optional[PromptComplexity] = None
    recommended_models: List[str] = field(default_factory=list)
    model_preferences: Dict[str, float] = field(default_factory=dict)
    context_state: Optional[ContextState] = None
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    uncertainty_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    classifier_version: str = "1.0.0"

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        return self.confidence >= threshold

    def has_secondary_intent(self, threshold: float = 0.3) -> bool:
        return (
            self.secondary_intent is not None
            and self.secondary_confidence >= threshold
        )

    def get_combined_confidence(self) -> float:
        if self.secondary_intent is not None:
            return (self.confidence * 0.7) + (self.secondary_confidence * 0.3)
        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "secondary_intent": (
                self.secondary_intent.value if self.secondary_intent else None
            ),
            "secondary_confidence": self.secondary_confidence,
            "reasoning": self.reasoning,
            "detected_features": self.detected_features,
            "processing_time_ms": self.processing_time_ms,
            "recommended_models": self.recommended_models,
            "quality_indicators": self.quality_indicators,
            "timestamp": self.timestamp.isoformat(),
            "classifier_version": self.classifier_version,
        }


# Backwards-compatible aliases.
Intent = IntentType
Classification = ClassificationResult

__all__ = [
    "IntentType",
    "Intent",
    "PromptComplexity",
    "ModelPerformance",
    "ContextState",
    "ClassificationResult",
    "Classification",
]
