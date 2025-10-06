"""
Conversation Context Tracker for Superior AI Model Selection
Tracks conversation evolution and context for intelligent model routing decisions
"""

import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path

from .types import IntentType
from .bot_types import PromptComplexity
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: int
    timestamp: datetime
    intent_type: str
    complexity: PromptComplexity
    model_used: str
    success: bool
    response_time: float
    quality_score: float
    user_satisfaction: Optional[float]  # If available from feedback
    topic_shift: bool  # Whether this turn shifted topics
    followup_type: str  # 'clarification', 'continuation', 'new_topic', 'correction'

@dataclass
class ConversationContext:
    """Complete context for a conversation"""
    conversation_id: str
    start_time: datetime
    last_activity: datetime
    total_turns: int
    successful_turns: int
    models_used: List[str]
    intent_progression: List[str]
    complexity_progression: List[float]
    topic_clusters: List[str]
    conversation_quality: float
    user_engagement: float
    coherence_score: float
    
    # Advanced context features
    dominant_intent: str
    complexity_trend: str  # 'increasing', 'decreasing', 'stable'
    model_preferences: Dict[str, float]  # Which models work best for this conversation
    failure_patterns: List[Dict]  # Patterns of failures
    success_patterns: List[Dict]  # Patterns of successes

class ConversationContextTracker:
    """
    Advanced Conversation Context Tracking System
    Enables conversation-aware routing based on conversation evolution
    """
    
    def __init__(self, context_file: str = "conversation_contexts.json"):
        self.context_file = Path(context_file)
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_history: deque = deque(maxlen=1000)  # Keep last 1000 conversations
        
        # Analysis parameters
        self.topic_shift_threshold = 0.7  # Threshold for detecting topic shifts
        self.quality_weight = 0.4
        self.engagement_weight = 0.3
        self.coherence_weight = 0.3
        
        # Context patterns for analysis
        self.context_patterns = self._initialize_context_patterns()
        
        # Model selection strategies based on conversation stage
        self.stage_strategies = {
            'opening': {'prefer_engaging': True, 'quality_threshold': 7.0},
            'development': {'prefer_consistent': True, 'quality_threshold': 8.0},
            'clarification': {'prefer_precise': True, 'quality_threshold': 8.5},
            'conclusion': {'prefer_reliable': True, 'quality_threshold': 7.5}
        }
        
        # Load existing conversation data
        self._load_conversation_data()
        
        secure_logger.info("🗣️ Conversation Context Tracker initialized")
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for conversation analysis"""
        return {
            'topic_shift_indicators': [
                'by the way', 'actually', 'speaking of', 'on another note',
                'changing topics', 'different question', 'new subject',
                'let me ask about', 'i also wanted to', 'another thing'
            ],
            'clarification_requests': [
                'what do you mean', 'can you explain', 'i don\'t understand',
                'could you clarify', 'elaborate', 'more details',
                'what exactly', 'how so', 'in what way'
            ],
            'continuation_indicators': [
                'and also', 'furthermore', 'in addition', 'moreover',
                'continuing', 'following up', 'building on that',
                'related to that', 'similarly', 'along those lines'
            ],
            'correction_indicators': [
                'actually', 'no wait', 'i meant', 'correction',
                'let me rephrase', 'that\'s not right', 'i misspoke',
                'to clarify', 'what i really meant'
            ],
            'satisfaction_positive': [
                'perfect', 'exactly', 'great', 'excellent', 'thank you',
                'that helps', 'much better', 'good answer', 'spot on'
            ],
            'satisfaction_negative': [
                'not quite', 'that\'s wrong', 'no', 'incorrect',
                'try again', 'doesn\'t help', 'not what i meant'
            ]
        }
    
    def start_conversation(self, conversation_id: str, initial_intent: str, 
                          initial_complexity: PromptComplexity) -> ConversationContext:
        """
        Start tracking a new conversation
        
        Args:
            conversation_id: Unique identifier for the conversation
            initial_intent: Initial intent type
            initial_complexity: Initial prompt complexity
            
        Returns:
            Initialized conversation context
        """
        current_time = datetime.now()
        
        context = ConversationContext(
            conversation_id=conversation_id,
            start_time=current_time,
            last_activity=current_time,
            total_turns=0,
            successful_turns=0,
            models_used=[],
            intent_progression=[initial_intent],
            complexity_progression=[initial_complexity.complexity_score],
            topic_clusters=[self._extract_topic_cluster(initial_intent)],
            conversation_quality=5.0,  # Start with neutral quality
            user_engagement=5.0,
            coherence_score=10.0,  # Start with perfect coherence
            dominant_intent=initial_intent,
            complexity_trend='stable',
            model_preferences={},
            failure_patterns=[],
            success_patterns=[]
        )
        
        self.active_conversations[conversation_id] = context
        
        secure_logger.info(f"🆕 Started tracking conversation {conversation_id} "
                          f"with intent {initial_intent}")
        
        return context
    
    def record_turn(self, conversation_id: str, turn_data: ConversationTurn) -> None:
        """
        Record a conversation turn and update context
        
        Args:
            conversation_id: Conversation identifier
            turn_data: Turn information
        """
        if conversation_id not in self.active_conversations:
            # Auto-start conversation if not exists
            self.start_conversation(
                conversation_id, 
                turn_data.intent_type, 
                turn_data.complexity
            )
        
        context = self.active_conversations[conversation_id]
        
        # Update basic metrics
        context.total_turns += 1
        if turn_data.success:
            context.successful_turns += 1
        
        context.last_activity = turn_data.timestamp
        
        # Track model usage
        if turn_data.model_used not in context.models_used:
            context.models_used.append(turn_data.model_used)
        
        # Update progressions
        context.intent_progression.append(turn_data.intent_type)
        context.complexity_progression.append(turn_data.complexity.complexity_score)
        
        # Analyze topic shifts
        if turn_data.topic_shift:
            topic_cluster = self._extract_topic_cluster(turn_data.intent_type)
            if topic_cluster not in context.topic_clusters:
                context.topic_clusters.append(topic_cluster)
        
        # Update model preferences
        self._update_model_preferences(context, turn_data)
        
        # Track patterns
        self._track_patterns(context, turn_data)
        
        # Update conversation metrics
        self._update_conversation_metrics(context, turn_data)
        
        # Update trends
        self._update_trends(context)
        
        secure_logger.debug(f"📝 Recorded turn {turn_data.turn_id} for conversation {conversation_id}")
    
    def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get current context for a conversation"""
        return self.active_conversations.get(conversation_id)
    
    def predict_next_model(self, conversation_id: str, next_intent: str,
                          next_complexity: PromptComplexity, 
                          available_models: List[str]) -> List[Tuple[str, float]]:
        """
        Predict best models for next turn based on conversation context
        
        Args:
            conversation_id: Conversation identifier
            next_intent: Predicted intent for next turn
            next_complexity: Predicted complexity for next turn
            available_models: Available models to choose from
            
        Returns:
            List of (model_name, confidence_score) tuples
        """
        context = self.get_conversation_context(conversation_id)
        if not context:
            # No context, return basic ranking
            return [(model, 5.0) for model in available_models]
        
        # Analyze conversation stage
        conversation_stage = self._determine_conversation_stage(context)
        
        # Get stage-specific preferences
        stage_prefs = self.stage_strategies.get(conversation_stage, {})
        
        model_scores = []
        
        for model in available_models:
            score = self._calculate_conversation_aware_score(
                model, context, next_intent, next_complexity, 
                conversation_stage, stage_prefs
            )
            model_scores.append((model, score))
        
        # Sort by score descending
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        secure_logger.info(f"🎯 Conversation-aware model ranking for {conversation_id}: "
                          f"stage={conversation_stage}, "
                          f"top_3={[(m.split('/')[-1], f'{s:.2f}') for m, s in model_scores[:3]]}")
        
        return model_scores
    
    def _determine_conversation_stage(self, context: ConversationContext) -> str:
        """Determine what stage the conversation is in"""
        turn_count = context.total_turns
        
        if turn_count <= 2:
            return 'opening'
        elif turn_count <= 10:
            return 'development'
        elif context.intent_progression[-2:].count(IntentType.QUESTION_ANSWERING) >= 1:
            return 'clarification'
        else:
            return 'conclusion'
    
    def _calculate_conversation_aware_score(self, model: str, context: ConversationContext,
                                          next_intent: str, next_complexity: PromptComplexity,
                                          stage: str, stage_prefs: Dict) -> float:
        """Calculate conversation-aware score for a model"""
        base_score = 5.0  # Base score
        
        # Model preference from conversation history
        if model in context.model_preferences:
            preference_bonus = (context.model_preferences[model] - 0.5) * 4  # Scale to ±2
            base_score += preference_bonus
        
        # Stage-specific adjustments
        if stage_prefs.get('prefer_engaging') and self._is_engaging_model(model):
            base_score += 1.0
        
        if stage_prefs.get('prefer_consistent') and model in context.models_used[-3:]:
            base_score += 0.8  # Bonus for recent consistency
        
        if stage_prefs.get('prefer_precise') and self._is_precise_model(model):
            base_score += 1.2
        
        if stage_prefs.get('prefer_reliable') and self._is_reliable_model(model):
            base_score += 1.0
        
        # Intent continuity bonus
        if next_intent == context.dominant_intent and model in context.models_used:
            base_score += 0.5
        
        # Complexity trend matching
        if self._model_matches_complexity_trend(model, context, next_complexity):
            base_score += 0.7
        
        # Penalize recently failed models
        if self._model_recently_failed(model, context):
            base_score -= 1.5
        
        # Conversation quality consideration
        if context.conversation_quality > 7.0 and model in context.models_used[-2:]:
            base_score += 0.3  # Stick with what's working
        elif context.conversation_quality < 5.0:
            if model not in context.models_used[-3:]:
                base_score += 0.5  # Try something different
        
        return max(0.0, min(10.0, base_score))
    
    def _is_engaging_model(self, model: str) -> bool:
        """Check if model is known to be engaging for conversation openings"""
        engaging_models = [
            'meta-llama',  # Generally good at engaging conversation
            'microsoft/phi',  # Good at interactive responses
        ]
        return any(eng in model.lower() for eng in engaging_models)
    
    def _is_precise_model(self, model: str) -> bool:
        """Check if model is known for precision and accuracy"""
        precise_models = [
            'qwen',  # Generally precise
            'microsoft/phi',  # Good for precise responses
        ]
        return any(prec in model.lower() for prec in precise_models)
    
    def _is_reliable_model(self, model: str) -> bool:
        """Check if model is known for reliability"""
        reliable_models = [
            'microsoft/phi-3-mini',  # Consistently reliable
            'qwen2.5',  # Stable performance
        ]
        return any(rel in model.lower() for rel in reliable_models)
    
    def _model_matches_complexity_trend(self, model: str, context: ConversationContext,
                                      next_complexity: PromptComplexity) -> bool:
        """Check if model matches the complexity trend"""
        if context.complexity_trend == 'increasing':
            # Prefer more capable models for increasing complexity
            return any(indicator in model.lower() for indicator in ['llama', '70b', '405b'])
        elif context.complexity_trend == 'decreasing':
            # Prefer efficient models for decreasing complexity
            return any(indicator in model.lower() for indicator in ['mini', '0.5b', '1.5b'])
        
        return True  # Neutral for stable complexity
    
    def _model_recently_failed(self, model: str, context: ConversationContext) -> bool:
        """Check if model recently failed in this conversation"""
        for failure in context.failure_patterns[-3:]:  # Check last 3 failures
            if failure.get('model_used') == model:
                return True
        return False
    
    def _update_model_preferences(self, context: ConversationContext, turn_data: ConversationTurn) -> None:
        """Update model preferences based on turn outcome"""
        model = turn_data.model_used
        
        # Initialize if not exists
        if model not in context.model_preferences:
            context.model_preferences[model] = 0.5  # Neutral starting point
        
        # Update based on success and quality
        if turn_data.success:
            adjustment = 0.1 + (turn_data.quality_score - 5.0) * 0.02  # Scale quality to adjustment
            context.model_preferences[model] = min(1.0, context.model_preferences[model] + adjustment)
        else:
            adjustment = -0.15  # Penalty for failure
            context.model_preferences[model] = max(0.0, context.model_preferences[model] + adjustment)
        
        # Additional adjustment for user satisfaction if available
        if turn_data.user_satisfaction is not None:
            satisfaction_adjustment = (turn_data.user_satisfaction - 5.0) * 0.03
            context.model_preferences[model] = max(0.0, min(1.0, 
                context.model_preferences[model] + satisfaction_adjustment))
    
    def _track_patterns(self, context: ConversationContext, turn_data: ConversationTurn) -> None:
        """Track success and failure patterns"""
        pattern_data = {
            'turn_id': turn_data.turn_id,
            'timestamp': turn_data.timestamp.isoformat(),
            'intent_type': turn_data.intent_type,
            'complexity_score': turn_data.complexity.complexity_score,
            'model_used': turn_data.model_used,
            'response_time': turn_data.response_time,
            'quality_score': turn_data.quality_score,
            'followup_type': turn_data.followup_type
        }
        
        if turn_data.success:
            context.success_patterns.append(pattern_data)
            # Keep only recent successes
            if len(context.success_patterns) > 20:
                context.success_patterns = context.success_patterns[-20:]
        else:
            context.failure_patterns.append(pattern_data)
            # Keep only recent failures
            if len(context.failure_patterns) > 10:
                context.failure_patterns = context.failure_patterns[-10:]
    
    def _update_conversation_metrics(self, context: ConversationContext, turn_data: ConversationTurn) -> None:
        """Update overall conversation quality metrics"""
        # Update conversation quality (weighted average)
        weight = 0.2  # How much the latest turn affects overall quality
        
        if turn_data.success:
            turn_quality = turn_data.quality_score
        else:
            turn_quality = 2.0  # Low quality for failed turns
        
        context.conversation_quality = (
            (1 - weight) * context.conversation_quality + 
            weight * turn_quality
        )
        
        # Update user engagement based on followup type
        engagement_adjustments = {
            'continuation': 0.2,     # Good engagement
            'clarification': -0.1,   # Slight engagement drop
            'new_topic': 0.1,        # Moderate engagement
            'correction': -0.3       # Poor engagement
        }
        
        adjustment = engagement_adjustments.get(turn_data.followup_type, 0.0)
        context.user_engagement = max(0.0, min(10.0, context.user_engagement + adjustment))
        
        # Update coherence score based on topic shifts and failures
        if turn_data.topic_shift and not turn_data.success:
            context.coherence_score = max(0.0, context.coherence_score - 0.5)
        elif turn_data.success and not turn_data.topic_shift:
            context.coherence_score = min(10.0, context.coherence_score + 0.1)
    
    def _update_trends(self, context: ConversationContext) -> None:
        """Update complexity and other trends"""
        if len(context.complexity_progression) >= 3:
            recent_complexities = context.complexity_progression[-3:]
            
            # Simple trend detection
            if recent_complexities[-1] > recent_complexities[0] + 0.5:
                context.complexity_trend = 'increasing'
            elif recent_complexities[-1] < recent_complexities[0] - 0.5:
                context.complexity_trend = 'decreasing'
            else:
                context.complexity_trend = 'stable'
        
        # Update dominant intent
        if len(context.intent_progression) >= 3:
            recent_intents = context.intent_progression[-5:]  # Look at last 5
            intent_counts = defaultdict(int)
            for intent in recent_intents:
                intent_counts[intent] += 1
            
            if intent_counts:
                context.dominant_intent = max(intent_counts.keys(), 
                                            key=lambda x: intent_counts[x])
    
    def _extract_topic_cluster(self, intent_type: str) -> str:
        """Extract topic cluster from intent type"""
        # Map intent types to broader topic clusters
        topic_mapping = {
            IntentType.CODE_GENERATION: 'technical',
            IntentType.MATHEMATICAL_REASONING: 'analytical',
            IntentType.CREATIVE_WRITING: 'creative',
            IntentType.QUESTION_ANSWERING: 'informational',
            IntentType.CONVERSATION: 'social',
            IntentType.SENTIMENT_ANALYSIS: 'analytical',
            IntentType.IMAGE_GENERATION: 'creative',
            IntentType.TRANSLATION: 'linguistic'
        }
        
        return topic_mapping.get(intent_type, 'general')
    
    def analyze_conversation_pattern(self, conversation_id: str) -> Dict[str, Any]:
        """Analyze patterns in a conversation for insights"""
        context = self.get_conversation_context(conversation_id)
        if not context:
            return {'error': 'Conversation not found'}
        
        analysis = {
            'conversation_summary': {
                'duration_minutes': (context.last_activity - context.start_time).total_seconds() / 60,
                'total_turns': context.total_turns,
                'success_rate': context.successful_turns / max(1, context.total_turns),
                'quality_score': context.conversation_quality,
                'engagement_score': context.user_engagement,
                'coherence_score': context.coherence_score
            },
            'model_performance': dict(context.model_preferences),
            'intent_distribution': self._analyze_intent_distribution(context),
            'complexity_analysis': self._analyze_complexity_progression(context),
            'topic_evolution': context.topic_clusters,
            'success_patterns': self._extract_success_patterns(context),
            'failure_patterns': self._extract_failure_patterns(context),
            'recommendations': self._generate_recommendations(context)
        }
        
        return analysis
    
    def _analyze_intent_distribution(self, context: ConversationContext) -> Dict[str, float]:
        """Analyze distribution of intent types in conversation"""
        intent_counts = defaultdict(int)
        for intent in context.intent_progression:
            intent_counts[intent] += 1
        
        total = len(context.intent_progression)
        return {intent: count/total for intent, count in intent_counts.items()}
    
    def _analyze_complexity_progression(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze how complexity evolved during conversation"""
        if len(context.complexity_progression) < 2:
            return {'trend': 'insufficient_data'}
        
        complexities = context.complexity_progression
        
        return {
            'trend': context.complexity_trend,
            'average_complexity': np.mean(complexities),
            'complexity_range': max(complexities) - min(complexities),
            'complexity_variance': np.var(complexities),
            'peak_complexity': max(complexities),
            'initial_complexity': complexities[0],
            'final_complexity': complexities[-1]
        }
    
    def _extract_success_patterns(self, context: ConversationContext) -> List[Dict]:
        """Extract patterns from successful turns"""
        if not context.success_patterns:
            return []
        
        # Analyze common characteristics of successful turns
        patterns = []
        
        # Group by model
        model_successes = defaultdict(list)
        for success in context.success_patterns:
            model_successes[success['model_used']].append(success)
        
        for model, successes in model_successes.items():
            if len(successes) >= 2:  # Need at least 2 successes for a pattern
                avg_quality = np.mean([s['quality_score'] for s in successes])
                avg_response_time = np.mean([s['response_time'] for s in successes])
                
                patterns.append({
                    'type': 'model_success',
                    'model': model,
                    'success_count': len(successes),
                    'avg_quality': avg_quality,
                    'avg_response_time': avg_response_time
                })
        
        return patterns
    
    def _extract_failure_patterns(self, context: ConversationContext) -> List[Dict]:
        """Extract patterns from failed turns"""
        if not context.failure_patterns:
            return []
        
        patterns = []
        
        # Group failures by model
        model_failures = defaultdict(list)
        for failure in context.failure_patterns:
            model_failures[failure['model_used']].append(failure)
        
        for model, failures in model_failures.items():
            if len(failures) >= 2:  # Need at least 2 failures for a pattern
                patterns.append({
                    'type': 'model_failure',
                    'model': model,
                    'failure_count': len(failures),
                    'common_complexity': np.mean([f['complexity_score'] for f in failures])
                })
        
        return patterns
    
    def _generate_recommendations(self, context: ConversationContext) -> List[str]:
        """Generate recommendations for improving conversation experience"""
        recommendations = []
        
        # Quality-based recommendations
        if context.conversation_quality < 6.0:
            best_model = max(context.model_preferences.keys(), 
                           key=lambda x: context.model_preferences[x])
            recommendations.append(f"Consider using {best_model} more frequently for better quality")
        
        # Engagement-based recommendations
        if context.user_engagement < 6.0:
            recommendations.append("Conversation shows low engagement - consider more interactive responses")
        
        # Coherence-based recommendations
        if context.coherence_score < 7.0:
            recommendations.append("Improve coherence by maintaining topic consistency")
        
        # Success rate recommendations
        success_rate = context.successful_turns / max(1, context.total_turns)
        if success_rate < 0.8:
            recommendations.append("Success rate is low - review model selection strategy")
        
        return recommendations
    
    def end_conversation(self, conversation_id: str) -> None:
        """End tracking for a conversation and archive it"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            
            # Add to history
            self.conversation_history.append(asdict(context))
            
            # Remove from active
            del self.active_conversations[conversation_id]
            
            # Save to disk
            self._save_conversation_data()
            
            secure_logger.info(f"🏁 Ended tracking for conversation {conversation_id} "
                              f"({context.total_turns} turns, "
                              f"quality={context.conversation_quality:.1f})")
    
    def _load_conversation_data(self) -> None:
        """Load conversation data from persistent storage"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                
                # Load conversation history
                history_data = data.get('conversation_history', [])
                for conv_data in history_data:
                    # Convert timestamp strings back to datetime objects
                    conv_data['start_time'] = datetime.fromisoformat(conv_data['start_time'])
                    conv_data['last_activity'] = datetime.fromisoformat(conv_data['last_activity'])
                    self.conversation_history.append(conv_data)
                
                secure_logger.info(f"📚 Loaded {len(self.conversation_history)} conversation histories")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to load conversation data: {safe_error}")
    
    def _save_conversation_data(self) -> None:
        """Save conversation data to persistent storage"""
        try:
            data = {
                'conversation_history': [],
                'last_updated': datetime.now().isoformat()
            }
            
            # Convert conversation history to JSON-serializable format
            for conv_data in self.conversation_history:
                serializable_data = dict(conv_data)
                if isinstance(serializable_data.get('start_time'), datetime):
                    serializable_data['start_time'] = serializable_data['start_time'].isoformat()
                if isinstance(serializable_data.get('last_activity'), datetime):
                    serializable_data['last_activity'] = serializable_data['last_activity'].isoformat()
                data['conversation_history'].append(serializable_data)
            
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            secure_logger.debug(f"💾 Saved conversation data")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to save conversation data: {safe_error}")
    
    def get_global_insights(self) -> Dict[str, Any]:
        """Get insights across all conversations"""
        if not self.conversation_history:
            return {'message': 'No conversation data available'}
        
        insights = {
            'total_conversations': len(self.conversation_history),
            'average_conversation_length': np.mean([conv['total_turns'] for conv in self.conversation_history]),
            'average_quality': np.mean([conv['conversation_quality'] for conv in self.conversation_history]),
            'most_used_models': self._get_most_used_models(),
            'best_performing_models': self._get_best_performing_models(),
            'common_intent_patterns': self._get_common_intent_patterns(),
            'quality_trends': self._analyze_quality_trends()
        }
        
        return insights
    
    def _get_most_used_models(self) -> Dict[str, int]:
        """Get most frequently used models across all conversations"""
        model_counts = defaultdict(int)
        for conv in self.conversation_history:
            for model in conv.get('models_used', []):
                model_counts[model] += 1
        
        # Return top 10
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_models[:10])
    
    def _get_best_performing_models(self) -> Dict[str, float]:
        """Get best performing models by conversation quality"""
        model_qualities = defaultdict(list)
        
        for conv in self.conversation_history:
            quality = conv.get('conversation_quality', 5.0)
            for model in conv.get('models_used', []):
                model_qualities[model].append(quality)
        
        # Calculate average quality for each model
        avg_qualities = {}
        for model, qualities in model_qualities.items():
            if len(qualities) >= 3:  # Need at least 3 conversations
                avg_qualities[model] = np.mean(qualities)
        
        # Return top 10
        sorted_models = sorted(avg_qualities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_models[:10])
    
    def _get_common_intent_patterns(self) -> Dict[str, int]:
        """Get common intent transition patterns"""
        patterns = defaultdict(int)
        
        for conv in self.conversation_history:
            intent_prog = conv.get('intent_progression', [])
            # Look at intent transitions
            for i in range(len(intent_prog) - 1):
                transition = f"{intent_prog[i]} -> {intent_prog[i+1]}"
                patterns[transition] += 1
        
        # Return top 10 patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_patterns[:10])
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if len(self.conversation_history) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Sort by start time and analyze recent vs older conversations
        sorted_conversations = sorted(
            self.conversation_history, 
            key=lambda x: x.get('start_time', '1970-01-01T00:00:00')
        )
        
        # Split into old and recent halves
        mid_point = len(sorted_conversations) // 2
        old_half = sorted_conversations[:mid_point]
        recent_half = sorted_conversations[mid_point:]
        
        old_avg_quality = np.mean([conv.get('conversation_quality', 5.0) for conv in old_half])
        recent_avg_quality = np.mean([conv.get('conversation_quality', 5.0) for conv in recent_half])
        
        return {
            'old_average_quality': old_avg_quality,
            'recent_average_quality': recent_avg_quality,
            'quality_improvement': recent_avg_quality - old_avg_quality,
            'trend': 'improving' if recent_avg_quality > old_avg_quality else 'declining'
        }

# Global instance
conversation_context_tracker = ConversationContextTracker()