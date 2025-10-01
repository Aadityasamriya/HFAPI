"""
Shared Types and Contracts for AI Assistant Bot Core
Breaks circular imports between router.py and intent_classifier.py
"""

from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque

class IntentType(Enum):
    """2025 Enhanced intent types for model routing with specialized AI models"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"          # 2025: New intent
    CODE_REVIEW = "code_review"              # 2025: Enhanced intent - Code verification and review
    RESEARCH = "research"                    # 2025: Enhanced intent - Research and information gathering
    EXPLANATION = "explanation"              # 2025: Enhanced intent - Explaining complex concepts
    DOCUMENT_PROCESSING = "document_processing"  # 2025: New intent
    MULTI_MODAL = "multi_modal"              # 2025: New intent
    CONVERSATION = "conversation"            # 2025: New intent
    UNKNOWN = "unknown"                      # CRITICAL FIX: Add UNKNOWN intent for fallback handling
    PDF_PROCESSING = "pdf_processing"        # 2025: New P1 feature - PDF analysis
    ZIP_ANALYSIS = "zip_analysis"            # 2025: New P1 feature - ZIP file analysis  
    IMAGE_ANALYSIS = "image_analysis"        # 2025: New P1 feature - Image content analysis
    FILE_GENERATION = "file_generation"      # 2025: New P1 feature - File generation and delivery
    
    # 2025: NEW SPECIALIZED INTENTS for superior AI routing
    MATHEMATICAL_REASONING = "mathematical_reasoning"   # Math problems, calculations, proofs
    ADVANCED_REASONING = "advanced_reasoning"           # Complex logical reasoning, QwQ model
    ALGORITHM_DESIGN = "algorithm_design"               # Complex coding algorithms, CodeLlama
    SCIENTIFIC_ANALYSIS = "scientific_analysis"         # Scientific research, data analysis
    MEDICAL_ANALYSIS = "medical_analysis"               # Medical image/text analysis
    CREATIVE_DESIGN = "creative_design"                 # UI/UX, graphic design, artistic creation
    EDUCATIONAL_CONTENT = "educational_content"         # Teaching, explanations, tutorials
    BUSINESS_ANALYSIS = "business_analysis"             # Business intelligence, market analysis
    TECHNICAL_DOCUMENTATION = "technical_documentation" # API docs, technical writing
    MULTILINGUAL_PROCESSING = "multilingual_processing" # Advanced translation, cultural context
    
    # 2025: BREAKTHROUGH CAPABILITIES - Revolutionary new intent types
    GUI_AUTOMATION = "gui_automation"                   # BREAKTHROUGH: UI-TARS GUI automation tasks
    TOOL_USE = "tool_use"                               # BREAKTHROUGH: Function calling, API integration
    PREMIUM_VISION = "premium_vision"                   # BREAKTHROUGH: Advanced vision with MiniCPM-V
    SYSTEM_INTERACTION = "system_interaction"           # Advanced system-level interactions

@dataclass
class PromptComplexity:
    """Advanced prompt complexity analysis data structure"""
    complexity_score: float  # 0-10 scale
    technical_depth: int     # 1-5 technical complexity
    reasoning_required: bool # True if complex reasoning needed
    context_length: int      # Context requirement in tokens
    domain_specificity: float # 0-1 how domain-specific
    creativity_factor: float  # 0-1 creativity vs factual
    multi_step: bool         # True if multi-step task
    uncertainty: float       # 0-1 uncertainty level
    priority_level: str      # 'low', 'medium', 'high', 'critical'
    estimated_tokens: int    # Estimated response length needed
    
    # Enhanced fields for superior AI routing
    domain_expertise: str    # Detected domain (medical, legal, technical, etc.)
    reasoning_chain_length: int # Number of reasoning steps required
    requires_external_knowledge: bool # Needs web search or specialized data
    temporal_context: str    # Time-sensitive, historical, or current
    user_intent_confidence: float # 0-1 confidence in intent classification
    cognitive_load: float    # 0-1 mental processing complexity required

@dataclass
class ModelPerformance:
    """Model performance tracking data"""
    model_name: str
    intent_type: str
    success_rate: float      # 0-1 success rate
    avg_response_time: float # Average response time in seconds
    quality_score: float     # 0-10 quality assessment
    usage_count: int         # Number of times used
    last_used: datetime      # Last usage timestamp
    cost_efficiency: float   # Quality per cost metric

@dataclass
class ContextState:
    """Enhanced conversation context tracking for superior AI routing"""
    user_id: int
    conversation_history: deque  # Limited history queue
    domain_context: str          # Current domain (coding, creative, etc.)
    complexity_trend: List[float] # Recent complexity scores
    preferred_models: Dict[str, float] # User's successful models
    conversation_coherence: float # 0-1 coherence score
    last_intent: Optional[str]   # Previous intent type
    response_satisfaction: deque # Recent satisfaction scores
    
    # Enhanced context tracking for superior performance
    expertise_level: Dict[str, float]  # User expertise in different domains
    interaction_patterns: Dict[str, int]  # User behavior patterns
    follow_up_context: Optional[str]     # Context for follow-up questions
    conversation_flow: List[str]         # Intent sequence tracking
    domain_transition_history: List[Tuple[str, str]]  # Domain change tracking
    preferred_complexity: float         # User's preferred response complexity
    learning_profile: Dict[str, Any]    # User learning preferences and history

@dataclass
class DomainExpertise:
    """Domain expertise detection and scoring"""
    domain: str                    # Domain name (medical, legal, technical, etc.)
    confidence: float             # 0-1 confidence in domain detection
    expertise_required: float     # 0-1 level of expertise needed
    specialized_knowledge: List[str]  # Specific knowledge areas
    complexity_indicators: List[str]  # What made this complex
    recommended_models: List[str]     # Models best for this domain

@dataclass
class LearningMetrics:
    """Performance learning and adaptation metrics"""
    model_performance_history: Dict[str, List[float]]  # Model performance over time
    user_satisfaction_trends: Dict[str, float]         # User satisfaction by intent
    context_adaptation_success: float                  # How well we adapt to context
    domain_routing_accuracy: float                     # Accuracy of domain detection
    response_quality_improvement: float                # Quality improvement over time
    preferred_model_convergence: Dict[str, str]        # User's preferred models by task

@dataclass
class ClassificationResult:
    """Enhanced results of intent classification with confidence metrics"""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[Tuple[IntentType, float]]
    reasoning: str
    processing_time: float
    complexity_score: float
    special_features: List[str]
    
    # Enhanced classification data for superior routing
    domain_expertise: Optional[DomainExpertise]  # Detected domain expertise needs
    context_awareness_score: float               # How context-aware the classification is
    follow_up_likelihood: float                  # Probability of follow-up questions
    multi_turn_context: Optional[str]           # Context from previous interactions
    user_expertise_match: float                 # How well this matches user's expertise level