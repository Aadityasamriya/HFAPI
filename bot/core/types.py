#!/usr/bin/env python3
"""
Core Types Module for AI Assistant Bot

This module defines shared types and contracts to prevent circular imports
between router.py and intent_classifier.py

Author: AI Assistant Bot Core Team
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


class IntentType(Enum):
    """Enhanced intent types for comprehensive model routing"""
    
    # Core Communication
    GENERAL_CHAT = "general_chat"
    QUESTION_ANSWERING = "question_answering"
    CONVERSATION = "conversation"
    
    # Content Generation
    TEXT_GENERATION = "text_generation"
    CREATIVE_WRITING = "creative_writing"
    CODE_GENERATION = "code_generation"
    DOCUMENTATION = "documentation"
    
    # Analysis & Processing
    TEXT_ANALYSIS = "text_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    
    # Visual & Media
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_GENERATION = "image_generation"
    
    # File & Data Processing (P1 Features)
    PDF_PROCESSING = "pdf_processing"
    ZIP_ANALYSIS = "zip_analysis"
    FILE_GENERATION = "file_generation"
    DOCUMENT_CONVERSION = "document_conversion"
    
    # Advanced Capabilities (Breakthrough Features)
    GUI_AUTOMATION = "gui_automation"
    TOOL_USE = "tool_use"
    API_INTEGRATION = "api_integration"
    SYSTEM_COMMANDS = "system_commands"
    
    # Administrative
    ADMIN_COMMAND = "admin_command"
    HELP_COMMAND = "help_command"
    
    # Error/Unknown
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class PromptComplexity:
    """Represents the complexity analysis of a prompt"""
    score: float = 0.0  # 0-10 complexity score
    factors: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ModelPerformance:
    """Tracks model performance metrics"""
    model_name: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    quality_score: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class ContextState:
    """Maintains conversation and processing context"""
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)


@dataclass
class ClassificationResult:
    """
    Comprehensive result from intent classification
    Encapsulates all aspects of intent analysis and routing decisions
    """
    
    # Primary classification
    intent: IntentType
    confidence: float
    
    # Secondary analysis
    secondary_intent: Optional[IntentType] = None
    secondary_confidence: float = 0.0
    
    # Reasoning and context
    reasoning: str = ""
    detected_features: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time_ms: float = 0.0
    complexity: Optional[PromptComplexity] = None
    
    # Model routing information
    recommended_models: List[str] = field(default_factory=list)
    model_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Context awareness
    context_state: Optional[ContextState] = None
    
    # Quality and confidence metrics
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    uncertainty_factors: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    classifier_version: str = "1.0.0"
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if classification meets high confidence threshold"""
        return self.confidence >= threshold
    
    def has_secondary_intent(self, threshold: float = 0.3) -> bool:
        """Check if secondary intent is significant"""
        return self.secondary_intent is not None and self.secondary_confidence >= threshold
    
    def get_combined_confidence(self) -> float:
        """Get weighted combined confidence score"""
        if self.secondary_intent:
            return (self.confidence * 0.7) + (self.secondary_confidence * 0.3)
        return self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'intent': self.intent.value,
            'confidence': self.confidence,
            'secondary_intent': self.secondary_intent.value if self.secondary_intent else None,
            'secondary_confidence': self.secondary_confidence,
            'reasoning': self.reasoning,
            'detected_features': self.detected_features,
            'processing_time_ms': self.processing_time_ms,
            'recommended_models': self.recommended_models,
            'quality_indicators': self.quality_indicators,
            'timestamp': self.timestamp.isoformat(),
            'classifier_version': self.classifier_version
        }


# Type aliases for convenience
Intent = IntentType
Classification = ClassificationResult