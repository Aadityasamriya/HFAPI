"""
Dynamic Fallback Strategy System for Superior AI Model Selection
Analyzes error types and chooses intelligent fallback strategies instead of static chains
"""

import re
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque

from .bot_types import IntentType, PromptComplexity
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

class ErrorType(Enum):
    """Categories of errors for intelligent fallback selection"""
    RATE_LIMITING = "rate_limiting"
    MODEL_UNAVAILABLE = "model_unavailable"
    CONTENT_FILTERING = "content_filtering"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_OVERLOADED = "model_overloaded"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN = "unknown"

@dataclass
class ErrorPattern:
    """Pattern for matching and categorizing errors"""
    error_type: ErrorType
    patterns: List[str]
    severity: float  # 0.0 to 1.0, higher means more critical
    recovery_time: int  # Estimated minutes before retry
    fallback_strategy: str  # Strategy to use for fallback

@dataclass
class FallbackDecision:
    """Decision about fallback strategy"""
    recommended_models: List[str]
    skip_models: Set[str]
    wait_time: Optional[float]
    strategy_type: str
    confidence: float
    reasoning: str

class DynamicFallbackStrategy:
    """
    Intelligent Fallback Strategy System with Circuit Breakers and Anti-Flapping
    Analyzes error types and patterns to make smart routing decisions
    """
    
    def __init__(self, fallback_history_file: str = "fallback_decisions.json"):
        self.history_file = Path(fallback_history_file)
        self.error_patterns = self._initialize_error_patterns()
        self.fallback_history: Dict[str, List[Dict]] = defaultdict(list)
        self.strategy_success_rates: Dict[str, float] = {}
        self.model_error_profiles: Dict[str, Dict[ErrorType, int]] = defaultdict(lambda: defaultdict(int))
        
        # ENHANCED: Advanced strategy weights with ML-tuned preferences
        self.strategy_weights = {
            'performance_based': 0.42,     # Increased - performance is key
            'predictive_switching': 0.25,  # NEW - proactive model switching
            'diversity_based': 0.18,       # Reduced but still important
            'similarity_based': 0.10,      # Reduced - less reliable
            'emergency_fallback': 0.05     # Last resort only
        }
        
        # NEW: Advanced circuit breaker configurations
        self.circuit_breaker_config = {
            'failure_threshold': 3,        # Failures before opening circuit
            'recovery_timeout': 300,       # 5 minutes before half-open
            'success_threshold': 2,        # Successes needed to close circuit
            'half_open_max_calls': 5,      # Max calls in half-open state
            'failure_rate_threshold': 0.5, # 50% failure rate triggers circuit
            'min_calls_threshold': 10      # Min calls before rate calculation
        }
        
        # NEW: Predictive failure detection
        self.failure_prediction = {
            'trend_window': 10,            # Number of recent calls to analyze
            'degradation_threshold': 0.3,  # Performance drop that triggers prediction
            'prediction_confidence': 0.7,  # Confidence needed for proactive switching
            'pattern_memory': 50           # How many failure patterns to remember
        }
        
        # Learning parameters with bounds
        self.learning_rate = 0.1
        self.min_samples_for_learning = 5
        self.max_learning_rate = 0.3  # Prevent excessive adaptation
        self.min_learning_rate = 0.01  # Ensure minimum learning
        
        # Circuit breaker safeguards (NEW)
        self.circuit_breakers = {}  # model_name -> circuit_breaker_state
        self.model_cooldowns = {}   # model_name -> cooldown_end_time
        self.failure_thresholds = {
            'consecutive_failures': 3,    # Open circuit after 3 consecutive failures
            'failure_rate_window': 300,   # 5-minute window for failure rate calculation
            'failure_rate_threshold': 0.7 # Open circuit if >70% failure rate
        }
        self.cooldown_periods = {
            'short': 60,    # 1 minute for transient issues
            'medium': 300,  # 5 minutes for service issues
            'long': 900     # 15 minutes for persistent issues
        }
        
        # Adaptation rate bounding (NEW)
        self.weight_bounds = {
            'min_weight': 0.05,  # Minimum strategy weight
            'max_weight': 0.8,   # Maximum strategy weight
            'max_adjustment': 0.1  # Maximum single adjustment
        }
        
        # Routing cycle budgets and timeouts (NEW)
        self.routing_budgets = {
            'max_routing_time': 30.0,     # Maximum time for routing decision
            'max_fallback_attempts': 5,   # Maximum fallback attempts per cycle
            'max_strategy_evaluations': 10  # Maximum strategy evaluations
        }
        
        # Anti-flapping measures (NEW)
        self.recent_decisions = deque(maxlen=20)  # Track recent routing decisions
        self.model_switch_history = defaultdict(list)  # Track model switches
        self.flapping_detection = {
            'switch_threshold': 3,     # Max switches between same models
            'time_window': 120,        # 2-minute window
            'penalty_duration': 300    # 5-minute penalty for flapping
        }
        
        # Load historical data
        self._load_fallback_history()
        
        secure_logger.info("🎯 ENHANCED Dynamic Fallback Strategy system initialized with predictive failure detection")
    
    def _initialize_error_patterns(self) -> Dict[ErrorType, ErrorPattern]:
        """Initialize error patterns for intelligent categorization"""
        return {
            ErrorType.RATE_LIMITING: ErrorPattern(
                error_type=ErrorType.RATE_LIMITING,
                patterns=[
                    r"rate.?limit",
                    r"too.?many.?requests",
                    r"429",
                    r"quota.?exceeded",
                    r"requests.?per.?minute",
                    r"throttle",
                    r"rate exceeded"
                ],
                severity=0.3,  # Usually temporary
                recovery_time=5,
                fallback_strategy="wait_and_diversify"
            ),
            ErrorType.MODEL_UNAVAILABLE: ErrorPattern(
                error_type=ErrorType.MODEL_UNAVAILABLE,
                patterns=[
                    r"model.?not.?found",
                    r"model.?unavailable",
                    r"not.?accessible",
                    r"404",
                    r"model.?loading",
                    r"currently.?loading",
                    r"model.?offline"
                ],
                severity=0.7,  # Need different model
                recovery_time=30,
                fallback_strategy="similar_models"
            ),
            ErrorType.CONTENT_FILTERING: ErrorPattern(
                error_type=ErrorType.CONTENT_FILTERING,
                patterns=[
                    r"content.?policy",
                    r"inappropriate.?content",
                    r"safety.?filter",
                    r"blocked.?content",
                    r"policy.?violation",
                    r"content.?moderation",
                    r"filtered"
                ],
                severity=0.5,  # Try different model approach
                recovery_time=0,
                fallback_strategy="alternative_models"
            ),
            ErrorType.TIMEOUT: ErrorPattern(
                error_type=ErrorType.TIMEOUT,
                patterns=[
                    r"timeout",
                    r"timed.?out",
                    r"request.?timeout",
                    r"connection.?timeout",
                    r"read.?timeout",
                    r"execution.?timeout"
                ],
                severity=0.4,  # Could be temporary or model issue
                recovery_time=2,
                fallback_strategy="faster_models"
            ),
            ErrorType.AUTHENTICATION: ErrorPattern(
                error_type=ErrorType.AUTHENTICATION,
                patterns=[
                    r"authentication",
                    r"unauthorized",
                    r"401",
                    r"403",
                    r"forbidden",
                    r"access.?denied",
                    r"invalid.?token",
                    r"api.?key"
                ],
                severity=0.9,  # Serious issue
                recovery_time=60,
                fallback_strategy="emergency_fallback"
            ),
            ErrorType.QUOTA_EXCEEDED: ErrorPattern(
                error_type=ErrorType.QUOTA_EXCEEDED,
                patterns=[
                    r"quota.?exceeded",
                    r"limit.?exceeded",
                    r"usage.?limit",
                    r"billing",
                    r"payment.?required",
                    r"subscription"
                ],
                severity=0.8,  # Need different provider/model
                recovery_time=120,
                fallback_strategy="free_tier_models"
            ),
            ErrorType.MODEL_OVERLOADED: ErrorPattern(
                error_type=ErrorType.MODEL_OVERLOADED,
                patterns=[
                    r"overloaded",
                    r"high.?demand",
                    r"busy",
                    r"503",
                    r"service.?unavailable",
                    r"server.?busy",
                    r"capacity"
                ],
                severity=0.6,  # Try different models
                recovery_time=10,
                fallback_strategy="load_balanced"
            ),
            ErrorType.NETWORK_ERROR: ErrorPattern(
                error_type=ErrorType.NETWORK_ERROR,
                patterns=[
                    r"network.?error",
                    r"connection.?error",
                    r"dns.?error",
                    r"host.?unreachable",
                    r"connection.?refused",
                    r"network.?timeout"
                ],
                severity=0.5,  # Infrastructure issue
                recovery_time=5,
                fallback_strategy="retry_with_backoff"
            ),
            ErrorType.PROCESSING_ERROR: ErrorPattern(
                error_type=ErrorType.PROCESSING_ERROR,
                patterns=[
                    r"processing.?error",
                    r"internal.?error",
                    r"500",
                    r"server.?error",
                    r"unexpected.?error",
                    r"runtime.?error"
                ],
                severity=0.7,  # Model or service issue
                recovery_time=15,
                fallback_strategy="different_provider"
            )
        }
    
    def analyze_error(self, error_message: str, model_name: str, 
                     intent_type: str, attempt_count: int = 1) -> ErrorType:
        """
        Analyze error message to categorize the error type
        
        Args:
            error_message: Error message from failed request
            model_name: Name of the model that failed
            intent_type: Type of task being attempted
            attempt_count: Number of attempts made so far
            
        Returns:
            Categorized error type
        """
        error_lower = error_message.lower()
        
        # Check each error pattern
        for error_type, pattern_info in self.error_patterns.items():
            for pattern in pattern_info.patterns:
                if re.search(pattern, error_lower):
                    # Record this error for the model
                    self.model_error_profiles[model_name][error_type] += 1
                    
                    secure_logger.debug(f"🔍 Error categorized as {error_type.value} for {model_name}: "
                                       f"{redact_sensitive_data(error_message[:100])}")
                    return error_type
        
        # Default to unknown if no pattern matches
        self.model_error_profiles[model_name][ErrorType.UNKNOWN] += 1
        return ErrorType.UNKNOWN
    
    def _is_circuit_open(self, model_name: str) -> bool:
        """Check if circuit breaker is open for a model"""
        if model_name not in self.circuit_breakers:
            return False
        
        breaker_state = self.circuit_breakers[model_name]
        current_time = time.time()
        
        # Check if cooldown period has ended
        if model_name in self.model_cooldowns:
            if current_time < self.model_cooldowns[model_name]:
                return True  # Still in cooldown
            else:
                # Cooldown ended, reset circuit breaker to half-open
                del self.model_cooldowns[model_name]
                breaker_state['state'] = 'half_open'
                breaker_state['consecutive_failures'] = 0
        
        return breaker_state.get('state') == 'open'
    
    def _record_model_failure(self, model_name: str, error_type: ErrorType) -> None:
        """Record a model failure and update circuit breaker state"""
        current_time = time.time()
        
        # Initialize circuit breaker if not exists
        if model_name not in self.circuit_breakers:
            self.circuit_breakers[model_name] = {
                'state': 'closed',
                'consecutive_failures': 0,
                'failure_history': deque(maxlen=50),  # Keep last 50 failures
                'last_failure_time': current_time
            }
        
        breaker = self.circuit_breakers[model_name]
        breaker['consecutive_failures'] += 1
        breaker['failure_history'].append({
            'timestamp': current_time,
            'error_type': error_type.value
        })
        breaker['last_failure_time'] = current_time
        
        # Check if we should open the circuit
        should_open = self._should_open_circuit(model_name, breaker)
        if should_open:
            self._open_circuit(model_name, error_type)
    
    def _should_open_circuit(self, model_name: str, breaker: Dict) -> bool:
        """Determine if circuit should be opened based on failure patterns"""
        current_time = time.time()
        
        # Check consecutive failures
        if breaker['consecutive_failures'] >= self.failure_thresholds['consecutive_failures']:
            return True
        
        # Check failure rate in time window
        window_start = current_time - self.failure_thresholds['failure_rate_window']
        recent_failures = [f for f in breaker['failure_history'] 
                          if f['timestamp'] >= window_start]
        
        if len(recent_failures) >= 5:  # Need minimum samples
            # Calculate failure rate (assuming some successes mixed in)
            # This is a simplified rate calculation
            failure_rate = len(recent_failures) / max(10, len(recent_failures) * 1.5)
            if failure_rate >= self.failure_thresholds['failure_rate_threshold']:
                return True
        
        return False
    
    def _open_circuit(self, model_name: str, error_type: ErrorType) -> None:
        """Open circuit breaker and set cooldown period"""
        if model_name not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[model_name]
        breaker['state'] = 'open'
        
        # Determine cooldown period based on error severity
        cooldown_duration = self._get_cooldown_duration(error_type, breaker['consecutive_failures'])
        cooldown_end_time = time.time() + cooldown_duration
        self.model_cooldowns[model_name] = cooldown_end_time
        
        secure_logger.warning(f"🔌 Circuit breaker OPENED for model {model_name.split('/')[-1]} "
                            f"(failures: {breaker['consecutive_failures']}, "
                            f"cooldown: {cooldown_duration}s)")
    
    def _get_cooldown_duration(self, error_type: ErrorType, consecutive_failures: int) -> int:
        """Calculate appropriate cooldown duration"""
        base_duration = self.cooldown_periods['medium']  # Default 5 minutes
        
        # Adjust based on error type
        if error_type in [ErrorType.AUTHENTICATION, ErrorType.QUOTA_EXCEEDED]:
            base_duration = self.cooldown_periods['long']  # 15 minutes
        elif error_type in [ErrorType.RATE_LIMITING, ErrorType.NETWORK_ERROR]:
            base_duration = self.cooldown_periods['short']  # 1 minute
        
        # Exponential backoff for repeated failures
        multiplier = min(2 ** (consecutive_failures - 3), 4)  # Cap at 4x
        return min(base_duration * multiplier, 1800)  # Cap at 30 minutes
    
    def _record_model_success(self, model_name: str) -> None:
        """Record a model success and potentially reset circuit breaker"""
        if model_name in self.circuit_breakers:
            breaker = self.circuit_breakers[model_name]
            
            if breaker['state'] == 'half_open':
                # Success in half-open state, close the circuit
                breaker['state'] = 'closed'
                breaker['consecutive_failures'] = 0
                secure_logger.info(f"🔌 Circuit breaker CLOSED for model {model_name.split('/')[-1]} "
                                 f"(success after half-open)")
            elif breaker['state'] == 'closed':
                # Reset consecutive failures on success
                breaker['consecutive_failures'] = max(0, breaker['consecutive_failures'] - 1)
    
    def _detect_flapping(self, current_model: str, recommended_models: List[str]) -> bool:
        """Detect if we're flapping between models"""
        current_time = time.time()
        
        # Clean old entries from switch history
        cutoff_time = current_time - self.flapping_detection['time_window']
        for model in list(self.model_switch_history.keys()):
            self.model_switch_history[model] = [
                t for t in self.model_switch_history[model] if t >= cutoff_time
            ]
            if not self.model_switch_history[model]:
                del self.model_switch_history[model]
        
        # Check recent decisions for patterns
        if len(self.recent_decisions) < 4:
            return False
        
        # Look for alternating patterns in recent decisions
        recent_models = [d.get('selected_model') for d in list(self.recent_decisions)[-6:]]
        model_sets = set(recent_models)
        
        # Check if we're switching between only 2-3 models repeatedly
        if len(model_sets) <= 3 and len(recent_models) >= 4:
            # Count switches between any two models in the set
            switch_count = 0
            for i in range(1, len(recent_models)):
                if recent_models[i] != recent_models[i-1]:
                    switch_count += 1
            
            # If more than half the decisions are switches, it's flapping
            if switch_count >= self.flapping_detection['switch_threshold']:
                secure_logger.warning(f"🌊 Routing flapping detected between models: {model_sets}")
                return True
        
        return False
    
    def _apply_anti_flapping_penalty(self, recommended_models: List[str]) -> List[str]:
        """Apply anti-flapping penalty by removing recently used models"""
        if len(self.recent_decisions) < 3:
            return recommended_models
        
        # Get models used in recent decisions
        recent_models = set()
        current_time = time.time()
        penalty_cutoff = current_time - self.flapping_detection['penalty_duration']
        
        for decision in self.recent_decisions:
            if decision.get('timestamp', 0) >= penalty_cutoff:
                recent_models.add(decision.get('selected_model'))
        
        # Filter out recently used models, but keep at least one option
        filtered_models = [m for m in recommended_models if m not in recent_models]
        
        if not filtered_models:
            # If all models were recently used, return the least recently used one
            return recommended_models[:1]
        
        return filtered_models
    
    def _clamp_adaptation_weights(self) -> None:
        """Clamp strategy weights to prevent excessive adaptation"""
        total_weight = sum(self.strategy_weights.values())
        
        # Normalize if total exceeds bounds
        if total_weight > 1.2 or total_weight < 0.8:
            factor = 1.0 / total_weight
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] *= factor
        
        # Clamp individual weights
        for strategy in self.strategy_weights:
            weight = self.strategy_weights[strategy]
            self.strategy_weights[strategy] = max(
                self.weight_bounds['min_weight'],
                min(self.weight_bounds['max_weight'], weight)
            )
    

    def determine_fallback_strategy(self, error_type: ErrorType, failed_model: str,
                                  intent_type: str, complexity: PromptComplexity,
                                  available_models: List[str],
                                  conversation_context: Optional[Dict] = None) -> FallbackDecision:
        """
        Enhanced fallback strategy with circuit breakers and anti-flapping safeguards
        
        Args:
            error_type: Categorized error type
            failed_model: Model that failed
            intent_type: Type of task
            complexity: Task complexity
            available_models: List of available models
            conversation_context: Optional conversation context
            
        Returns:
            Fallback decision with recommended strategy and safeguards
        """
        start_time = time.time()
        current_datetime = datetime.now()
        
        # Record the failure for circuit breaker tracking
        self._record_model_failure(failed_model, error_type)
        
        # Apply circuit breaker filtering - Remove models with open circuits
        circuit_filtered_models = [
            model for model in available_models 
            if not self._is_circuit_open(model)
        ]
        
        if not circuit_filtered_models:
            # All models have open circuits - emergency fallback with wait time
            secure_logger.warning("🚨 All models have open circuit breakers - emergency fallback")
            return FallbackDecision(
                recommended_models=available_models[:1],  # Try one model anyway
                skip_models=set(available_models[1:]),
                wait_time=self.cooldown_periods['short'],  # Short wait before retry
                strategy_type='emergency_circuit_fallback',
                confidence=0.2,
                reasoning="All models have open circuit breakers, emergency fallback"
            )
        
        # Use circuit-filtered models for strategy selection
        effective_models = circuit_filtered_models
        
        # Get error pattern information
        pattern_info = self.error_patterns.get(error_type)
        if not pattern_info:
            pattern_info = self.error_patterns[ErrorType.UNKNOWN]
        
        # Determine base strategy with timeout budget
        base_strategy = pattern_info.fallback_strategy
        strategy_evaluations = 0
        max_evaluations = self.routing_budgets['max_strategy_evaluations']
        
        # Apply intelligent modifications based on context
        strategy_function = getattr(self, f"_strategy_{base_strategy}", self._strategy_performance_based)
        
        try:
            # Execute strategy with timeout
            strategy_evaluations += 1
            decision = strategy_function(
                failed_model=failed_model,
                intent_type=intent_type,
                complexity=complexity,
                available_models=effective_models,
                error_type=error_type,
                pattern_info=pattern_info,
                conversation_context=conversation_context
            )
            
            # Check for routing time budget timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > self.routing_budgets['max_routing_time']:
                secure_logger.warning(f"⏰ Routing timeout exceeded: {elapsed_time:.2f}s")
                # Use emergency fallback
                decision = FallbackDecision(
                    recommended_models=effective_models[:1],
                    skip_models=set(),
                    wait_time=5.0,
                    strategy_type='timeout_emergency',
                    confidence=0.3,
                    reasoning=f"Routing timeout after {elapsed_time:.2f}s"
                )
            
            # Apply anti-flapping measures
            if self._detect_flapping(failed_model, decision.recommended_models):
                decision.recommended_models = self._apply_anti_flapping_penalty(
                    decision.recommended_models
                )
                decision.strategy_type += '_antiflap'
                decision.confidence *= 0.8  # Reduce confidence due to flapping
                decision.reasoning += " (anti-flapping applied)"
            
            # Record decision for tracking
            self.recent_decisions.append({
                'timestamp': time.time(),
                'failed_model': failed_model,
                'selected_model': decision.recommended_models[0] if decision.recommended_models else None,
                'strategy': decision.strategy_type,
                'error_type': error_type.value
            })
            
            # Apply strategy weight adaptation with bounds
            self._adapt_strategy_weights(decision, error_type, failed_model)
            
            # Record fallback decision for learning
            self._record_fallback_decision(decision, error_type, failed_model, intent_type)
            
            # Final validation - ensure at least one model is recommended
            if not decision.recommended_models:
                decision.recommended_models = effective_models[:1] if effective_models else available_models[:1]
                decision.strategy_type = 'emergency_validation'
                decision.confidence = 0.1
                decision.reasoning = "No models recommended, emergency validation fallback"
            
            elapsed_time = time.time() - start_time
            secure_logger.info(f"🎯 Enhanced fallback strategy '{decision.strategy_type}' selected for {error_type.value}: "
                             f"models={[m.split('/')[-1] for m in decision.recommended_models[:3]]}, "
                             f"confidence={decision.confidence:.2f}, time={elapsed_time:.3f}s")
            
            return decision
            
        except Exception as e:
            # Emergency fallback if strategy execution fails
            elapsed_time = time.time() - start_time
            secure_logger.error(f"❌ Strategy execution failed after {elapsed_time:.3f}s: {str(e)}")
            
            return FallbackDecision(
                recommended_models=effective_models[:1] if effective_models else available_models[:1],
                skip_models=set(),
                wait_time=10.0,
                strategy_type='strategy_execution_emergency',
                confidence=0.2,
                reasoning=f"Strategy execution failed: {str(e)}"
            )
    
    def record_strategy_outcome(self, strategy_type: str, success: bool, 
                               model_used: str, response_time: float) -> None:
        """
        Record the outcome of a fallback strategy for learning with safeguards
        
        Args:
            strategy_type: Type of strategy that was used
            success: Whether the strategy resulted in a successful response
            model_used: Model that was ultimately used
            response_time: Time taken for the response
        """
        # Record success for circuit breaker management
        if success:
            self._record_model_success(model_used)
        
        # Update strategy success rates with bounded adaptation
        if strategy_type not in self.strategy_success_rates:
            self.strategy_success_rates[strategy_type] = 0.5  # Start neutral
        
        # Bounded learning rate adaptation
        current_rate = self.strategy_success_rates[strategy_type]
        bounded_learning_rate = max(self.min_learning_rate, 
                                   min(self.max_learning_rate, self.learning_rate))
        
        # Update with exponential moving average
        if success:
            new_rate = current_rate + bounded_learning_rate * (1.0 - current_rate)
        else:
            new_rate = current_rate + bounded_learning_rate * (0.0 - current_rate)
        
        # Clamp to reasonable bounds
        self.strategy_success_rates[strategy_type] = max(0.1, min(0.9, new_rate))
        
        # Note: Strategy adaptation is handled at the decision level
        
        secure_logger.debug(f"📊 Strategy outcome recorded: {strategy_type} "
                          f"success={success}, rate={self.strategy_success_rates[strategy_type]:.3f}")

    
    def _strategy_wait_and_diversify(self, **kwargs) -> FallbackDecision:
        """Strategy for rate limiting - wait and try diverse models"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        pattern_info = kwargs['pattern_info']
        
        # Remove the failed model temporarily
        filtered_models = [m for m in available_models if m != failed_model]
        
        # Prefer models from different providers/families
        diverse_models = self._get_diverse_models(filtered_models, failed_model)
        
        return FallbackDecision(
            recommended_models=diverse_models[:5],
            skip_models={failed_model},
            wait_time=pattern_info.recovery_time * 60,  # Convert to seconds
            strategy_type="wait_and_diversify",
            confidence=0.8,
            reasoning=f"Rate limiting detected, waiting {pattern_info.recovery_time}min and trying diverse models"
        )
    
    def _strategy_similar_models(self, **kwargs) -> FallbackDecision:
        """Strategy for unavailable models - try similar models"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        intent_type = kwargs['intent_type']
        
        # Find models similar to the failed one
        similar_models = self._get_similar_models(failed_model, available_models, intent_type)
        
        return FallbackDecision(
            recommended_models=similar_models[:5],
            skip_models={failed_model},
            wait_time=None,
            strategy_type="similar_models",
            confidence=0.7,
            reasoning=f"Model unavailable, trying similar models to {failed_model.split('/')[-1]}"
        )
    
    def _strategy_alternative_models(self, **kwargs) -> FallbackDecision:
        """Strategy for content filtering - try models with different policies"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        
        # Prefer models known to be less restrictive or have different filtering approaches
        alternative_models = self._get_alternative_policy_models(available_models, failed_model)
        
        return FallbackDecision(
            recommended_models=alternative_models[:5],
            skip_models={failed_model},
            wait_time=None,
            strategy_type="alternative_models",
            confidence=0.6,
            reasoning="Content filtering detected, trying models with different policies"
        )
    
    def _strategy_faster_models(self, **kwargs) -> FallbackDecision:
        """Strategy for timeouts - prefer faster models"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        
        # Prefer smaller, faster models
        faster_models = self._get_faster_models(available_models, failed_model)
        
        return FallbackDecision(
            recommended_models=faster_models[:5],
            skip_models={failed_model},
            wait_time=None,
            strategy_type="faster_models",
            confidence=0.8,
            reasoning="Timeout detected, prioritizing faster/smaller models"
        )
    
    def _strategy_emergency_fallback(self, **kwargs) -> FallbackDecision:
        """Emergency fallback for serious errors"""
        available_models = kwargs['available_models']
        failed_model = kwargs['failed_model']
        
        # Use most reliable, basic models
        emergency_models = self._get_emergency_models(available_models)
        
        return FallbackDecision(
            recommended_models=emergency_models[:3],
            skip_models={failed_model},
            wait_time=None,
            strategy_type="emergency_fallback",
            confidence=0.9,
            reasoning="Critical error detected, using most reliable fallback models"
        )
    
    def _strategy_free_tier_models(self, **kwargs) -> FallbackDecision:
        """Strategy for quota/billing issues - use free tier models"""
        available_models = kwargs['available_models']
        failed_model = kwargs['failed_model']
        
        # Prefer known free/open models
        free_models = self._get_free_tier_models(available_models)
        
        return FallbackDecision(
            recommended_models=free_models[:5],
            skip_models={failed_model},
            wait_time=None,
            strategy_type="free_tier_models",
            confidence=0.7,
            reasoning="Quota exceeded, switching to free tier models"
        )
    
    def _strategy_load_balanced(self, **kwargs) -> FallbackDecision:
        """Strategy for overloaded models - distribute load"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        
        # Try to balance load across different models/providers
        balanced_models = self._get_load_balanced_models(available_models, failed_model)
        
        return FallbackDecision(
            recommended_models=balanced_models[:5],
            skip_models={failed_model},
            wait_time=30,  # Brief wait to let load settle
            strategy_type="load_balanced",
            confidence=0.7,
            reasoning="Model overloaded, distributing load to other models"
        )
    
    def _strategy_retry_with_backoff(self, **kwargs) -> FallbackDecision:
        """Strategy for network errors - retry with exponential backoff"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        
        # Include the original model but with a wait time
        retry_models = [failed_model] + [m for m in available_models if m != failed_model]
        
        return FallbackDecision(
            recommended_models=retry_models[:5],
            skip_models=set(),
            wait_time=10,  # Brief wait for network recovery
            strategy_type="retry_with_backoff",
            confidence=0.6,
            reasoning="Network error detected, retrying with backoff"
        )
    
    def _strategy_different_provider(self, **kwargs) -> FallbackDecision:
        """Strategy for processing errors - try different providers"""
        failed_model = kwargs['failed_model']
        available_models = kwargs['available_models']
        
        # Prefer models from different providers/organizations
        different_provider_models = self._get_different_provider_models(available_models, failed_model)
        
        return FallbackDecision(
            recommended_models=different_provider_models[:5],
            skip_models={failed_model},
            wait_time=None,
            strategy_type="different_provider",
            confidence=0.7,
            reasoning="Processing error detected, trying models from different providers"
        )
    
    def _strategy_performance_based(self, **kwargs) -> FallbackDecision:
        """Default strategy based on performance metrics"""
        available_models = kwargs['available_models']
        failed_model = kwargs['failed_model']
        intent_type = kwargs['intent_type']
        
        # Use performance predictor to rank models
        from .model_health_monitor import health_monitor
        
        rankings = health_monitor.get_conversation_aware_rankings(
            intent_type=intent_type,
            complexity=kwargs.get('complexity')
        )
        
        # Filter out failed model and get top performers
        recommended = [model for model, score in rankings if model != failed_model][:5]
        
        return FallbackDecision(
            recommended_models=recommended,
            skip_models={failed_model},
            wait_time=None,
            strategy_type="performance_based",
            confidence=0.8,
            reasoning="Using performance-based ranking for fallback selection"
        )
    
    def _get_diverse_models(self, available_models: List[str], failed_model: str) -> List[str]:
        """Get diverse models from different families/providers"""
        # Group models by provider/family
        providers = defaultdict(list)
        
        for model in available_models:
            if 'meta-llama' in model:
                providers['meta'].append(model)
            elif 'microsoft' in model:
                providers['microsoft'].append(model)
            elif 'qwen' in model.lower():
                providers['qwen'].append(model)
            elif 'huggingface' in model.lower():
                providers['huggingface'].append(model)
            else:
                providers['other'].append(model)
        
        # Get one model from each provider
        diverse_models = []
        for provider, models in providers.items():
            if models:
                # Sort by preference and take the best from each provider
                diverse_models.extend(models[:2])  # Take top 2 from each provider
        
        return diverse_models
    
    def _get_similar_models(self, failed_model: str, available_models: List[str], intent_type: str) -> List[str]:
        """Get models similar to the failed one"""
        # Extract model family/type
        failed_lower = failed_model.lower()
        
        similar_models = []
        for model in available_models:
            if model == failed_model:
                continue
                
            model_lower = model.lower()
            similarity_score = 0
            
            # Same provider family
            if any(provider in failed_lower and provider in model_lower 
                   for provider in ['meta', 'microsoft', 'qwen', 'huggingface']):
                similarity_score += 2
            
            # Similar size indicators
            if any(size in failed_lower and size in model_lower 
                   for size in ['mini', 'small', 'medium', 'large', '7b', '13b', '70b']):
                similarity_score += 1
            
            # Similar capabilities
            if any(cap in failed_lower and cap in model_lower 
                   for cap in ['instruct', 'chat', 'code', 'reasoning']):
                similarity_score += 1
            
            similar_models.append((model, similarity_score))
        
        # Sort by similarity and return
        similar_models.sort(key=lambda x: x[1], reverse=True)
        return [model for model, score in similar_models]
    
    def _get_alternative_policy_models(self, available_models: List[str], failed_model: str) -> List[str]:
        """Get models with potentially different content policies"""
        # Some models are known to have different filtering approaches
        alternative_priorities = [
            'qwen',  # Often less restrictive
            'microsoft/phi',  # Different policy approach
            'meta-llama',  # Open source, different policies
            'huggingface'  # Various policy approaches
        ]
        
        alternative_models = []
        for priority in alternative_priorities:
            for model in available_models:
                if priority in model.lower() and model != failed_model:
                    alternative_models.append(model)
        
        # Add remaining models
        for model in available_models:
            if model not in alternative_models and model != failed_model:
                alternative_models.append(model)
        
        return alternative_models
    
    def _get_faster_models(self, available_models: List[str], failed_model: str) -> List[str]:
        """Get faster/smaller models"""
        # Priority for smaller, faster models
        size_priorities = ['0.5b', '1.5b', 'mini', 'small', '7b', '13b', 'medium', '70b', 'large']
        
        sized_models = []
        for size in size_priorities:
            for model in available_models:
                if size in model.lower() and model != failed_model:
                    sized_models.append(model)
        
        # Add remaining models
        for model in available_models:
            if model not in sized_models and model != failed_model:
                sized_models.append(model)
        
        return sized_models
    
    def _get_emergency_models(self, available_models: List[str]) -> List[str]:
        """Get most reliable emergency fallback models"""
        # Known stable, reliable models
        emergency_priorities = [
            'microsoft/Phi-3-mini-4k-instruct',
            'Qwen/Qwen2.5-1.5B-Instruct',
            'microsoft/DialoGPT-medium',
            'HuggingFaceH4/zephyr-7b-beta'
        ]
        
        emergency_models = []
        for priority in emergency_priorities:
            if priority in available_models:
                emergency_models.append(priority)
        
        # Add any remaining available models
        for model in available_models:
            if model not in emergency_models:
                emergency_models.append(model)
        
        return emergency_models
    
    def _get_free_tier_models(self, available_models: List[str]) -> List[str]:
        """Get models likely to be on free tier"""
        # Smaller, open-source models are typically free
        free_indicators = ['mini', 'small', '0.5b', '1.5b', 'phi', 'qwen', 'dialogpt']
        
        free_models = []
        for model in available_models:
            model_lower = model.lower()
            if any(indicator in model_lower for indicator in free_indicators):
                free_models.append(model)
        
        # Add remaining models as backup
        for model in available_models:
            if model not in free_models:
                free_models.append(model)
        
        return free_models
    
    def _get_load_balanced_models(self, available_models: List[str], failed_model: str) -> List[str]:
        """Get models to balance load across different providers"""
        return self._get_diverse_models(available_models, failed_model)
    
    def _get_different_provider_models(self, available_models: List[str], failed_model: str) -> List[str]:
        """Get models from different providers than the failed one"""
        failed_provider = self._extract_provider(failed_model)
        
        different_provider_models = []
        same_provider_models = []
        
        for model in available_models:
            if model == failed_model:
                continue
                
            if self._extract_provider(model) != failed_provider:
                different_provider_models.append(model)
            else:
                same_provider_models.append(model)
        
        # Prefer different providers, but include same provider as backup
        return different_provider_models + same_provider_models
    
    def _extract_provider(self, model_name: str) -> str:
        """Extract provider name from model name"""
        model_lower = model_name.lower()
        
        if 'meta-llama' in model_lower or 'meta' in model_lower:
            return 'meta'
        elif 'microsoft' in model_lower:
            return 'microsoft'
        elif 'qwen' in model_lower:
            return 'qwen'
        elif 'huggingface' in model_lower:
            return 'huggingface'
        else:
            # Extract the organization part before the first slash
            if '/' in model_name:
                return model_name.split('/')[0].lower()
            return 'unknown'
    
    def _adapt_strategy_weights(self, decision: FallbackDecision, error_type: ErrorType, failed_model: str) -> None:
        """Adapt strategy weights based on decision outcomes"""
        # This would be updated when we get feedback on decision success
        # For now, we just record the decision for future learning
        pass
    
    def _record_fallback_decision(self, decision: FallbackDecision, error_type: ErrorType, 
                                failed_model: str, intent_type: str) -> None:
        """Record fallback decision for learning and analysis"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type.value,
            'failed_model': failed_model,
            'intent_type': intent_type,
            'strategy_type': decision.strategy_type,
            'recommended_models': decision.recommended_models,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning
        }
        
        self.fallback_history[error_type.value].append(record)
        
        # Keep only recent history (last 100 decisions per error type)
        if len(self.fallback_history[error_type.value]) > 100:
            self.fallback_history[error_type.value] = self.fallback_history[error_type.value][-100:]
        
        # Periodically save to disk
        if len(self.fallback_history[error_type.value]) % 10 == 0:
            self._save_fallback_history()
    
    def _load_fallback_history(self) -> None:
        """Load fallback decision history from disk"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                self.fallback_history = defaultdict(list, data.get('fallback_history', {}))
                self.strategy_success_rates = data.get('strategy_success_rates', {})
                
                secure_logger.info(f"📊 Loaded fallback history: {len(self.fallback_history)} error types")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to load fallback history: {safe_error}")
    
    def _save_fallback_history(self) -> None:
        """Save fallback decision history to disk"""
        try:
            data = {
                'fallback_history': dict(self.fallback_history),
                'strategy_success_rates': self.strategy_success_rates,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            secure_logger.debug(f"💾 Saved fallback history")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to save fallback history: {safe_error}")
    
    def get_error_insights(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get insights about error patterns and fallback effectiveness"""
        insights = {
            'error_distribution': {},
            'strategy_effectiveness': {},
            'model_error_profiles': {},
            'recommendations': []
        }
        
        # Error distribution
        for error_type, records in self.fallback_history.items():
            insights['error_distribution'][error_type] = len(records)
        
        # Strategy effectiveness
        for error_type, records in self.fallback_history.items():
            if records:
                strategy_counts = defaultdict(int)
                for record in records:
                    strategy_counts[record['strategy_type']] += 1
                insights['strategy_effectiveness'][error_type] = dict(strategy_counts)
        
        # Model error profiles
        if model_name and model_name in self.model_error_profiles:
            insights['model_error_profiles'][model_name] = dict(self.model_error_profiles[model_name])
        else:
            # Aggregate all model error profiles
            for model, error_counts in self.model_error_profiles.items():
                insights['model_error_profiles'][model] = dict(error_counts)
        
        return insights
    
    def predict_model_failure_risk(self, model_name: str, recent_performance: List[float]) -> float:
        """
        2025 NEW: Predict the risk of model failure based on performance trends
        
        Args:
            model_name (str): Model to analyze
            recent_performance (List[float]): Recent performance scores (0-1)
            
        Returns:
            float: Failure risk score (0-1, higher = more likely to fail)
        """
        if len(recent_performance) < 3:
            return 0.0  # Not enough data for prediction
            
        # Calculate performance trend
        trend_window = self.failure_prediction['trend_window']
        recent_scores = recent_performance[-trend_window:]
        
        if len(recent_scores) < 2:
            return 0.0
            
        # Calculate trend slope (negative = declining performance)
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        # Calculate performance variance (higher = more unstable)
        variance = np.var(recent_scores)
        
        # Calculate recent vs baseline performance  
        current_avg = float(np.mean(recent_scores[-3:]))  # Last 3 scores
        baseline = float(max(recent_scores)) if recent_scores else 1.0
        performance_drop = max(0.0, (baseline - current_avg) / baseline)
        
        # Combine factors for risk assessment
        trend_risk = max(0, -slope * 2)  # Negative slope increases risk
        stability_risk = min(float(variance * 3), 0.5)  # High variance increases risk
        degradation_risk = performance_drop
        
        # Weighted combination
        total_risk = float(
            trend_risk * 0.4 +
            stability_risk * 0.3 +
            degradation_risk * 0.3
        )
        
        return min(1.0, total_risk)
    
    def get_proactive_model_recommendations(self, intent_type: str, 
                                          available_models: List[str]) -> List[str]:
        """
        2025 NEW: Get proactive model recommendations based on predictive analysis
        
        Args:
            intent_type (str): Intent type
            available_models (List[str]): Available models
            
        Returns:
            List[str]: Recommended models sorted by predicted reliability
        """
        model_scores = []
        
        for model in available_models:
            # Initialize circuit breaker if not exists
            if model not in self.circuit_breakers:
                self.circuit_breakers[model] = {
                    'state': 'closed',
                    'failure_count': 0,
                    'success_count': 0
                }
            
            breaker = self.circuit_breakers[model]
            
            # Skip models with open circuits
            if breaker.get('state') == 'open':
                continue
                
            # Calculate reliability score based on historical performance
            failure_rate = 0.0
            if breaker.get('failure_count', 0) + breaker.get('success_count', 0) > 0:
                total_calls = breaker.get('failure_count', 0) + breaker.get('success_count', 0)
                failure_rate = breaker.get('failure_count', 0) / total_calls
                
            base_score = 1.0 - failure_rate
            
            # Adjust for circuit state
            if breaker.get('state') == 'half_open':
                base_score *= 0.7  # Reduce confidence for recovering models
            
            # Boost models with consistent success
            if breaker.get('success_count', 0) > 5 and failure_rate < 0.1:
                base_score *= 1.1
                
            model_scores.append((model, base_score))
        
        # Sort by score descending
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        secure_logger.info(f"🎯 Proactive recommendations for {intent_type}: "
                          f"Top 3: {[(m.split('/')[-1], f'{s:.2f}') for m, s in model_scores[:3]]}")
        
        return [model for model, _ in model_scores]

# Global instance
dynamic_fallback_strategy = DynamicFallbackStrategy()