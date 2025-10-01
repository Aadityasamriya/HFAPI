"""
Model Health Monitoring System
Implements startup health checks and active model registry for superior AI routing
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict

from .ai_providers import ProviderConfig, AIProvider
from .provider_factory import ProviderFactory
from ..config import Config
from ..security_utils import redact_sensitive_data, get_secure_logger
from .performance_predictor import performance_predictor, PredictionContext
from .bot_types import PromptComplexity

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

@dataclass
class ModelHealthStatus:
    """Enhanced health status for a specific model with real-time tracking"""
    model_name: str
    is_available: bool
    last_checked: datetime
    response_time: float  # Average response time in seconds
    success_rate: float   # Success rate (0.0 to 1.0)
    error_count: int
    last_error: Optional[str]
    consecutive_failures: int
    quality_score: float  # 0.0 to 10.0
    
    # Real-time adaptation fields
    recent_performance: deque  # Recent performance samples
    real_time_score: float  # Real-time performance score
    adaptation_weight: float  # Weight for real-time vs historical data
    trend_direction: str  # 'improving', 'declining', 'stable'
    peak_performance_time: Optional[datetime]  # When model performed best
    last_adaptation: datetime  # Last time we adapted based on this model

class ModelHealthMonitor:
    """
    Comprehensive Model Health Monitoring System
    Provides startup health checks, active model registry, and dynamic availability tracking
    """
    
    def __init__(self, health_check_timeout: int = 10):
        self.health_check_timeout = health_check_timeout
        self.model_registry: Dict[str, ModelHealthStatus] = {}
        self.available_models: Set[str] = set()
        self.provider: Optional[AIProvider] = None
        self.last_full_check: Optional[datetime] = None
        self.check_interval = timedelta(minutes=30)  # Check models every 30 minutes
        
        # Grace period for startup - models available by default during this time
        self.startup_time = datetime.now()
        self.startup_grace_period = timedelta(minutes=5)  # 5 minute grace period
        
        # Real-time adaptation features
        self.real_time_feedback: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))  # Store recent feedback
        self.adaptation_thresholds = {
            'poor_performance': 0.3,  # Threshold below which we adapt immediately
            'excellent_performance': 0.9,  # Threshold above which we boost confidence
            'trend_sensitivity': 0.15  # How sensitive we are to performance trends
        }
        self.conversation_context: Dict[str, Any] = {}  # Track conversation-level context
        self.dynamic_weights: Dict[str, float] = {}  # Dynamic weights for different models
        
        # Load previous health status if available
        self._load_health_cache()
        
        # Add critical models to available set by default (optimistic availability)
        self._ensure_critical_models_available()
        
        secure_logger.info("🏥 Enhanced Model Health Monitor initialized with real-time adaptation and grace period")
    
    def _ensure_critical_models_available(self) -> None:
        """Ensure critical models are always in the available set as a fallback"""
        critical_models = self._get_critical_models()
        for model in critical_models:
            if model:
                self.available_models.add(model)
        secure_logger.info(f"🔒 Ensured {len(critical_models)} critical models are available by default")
    
    def _is_in_grace_period(self) -> bool:
        """Check if we're still in the startup grace period"""
        return datetime.now() - self.startup_time < self.startup_grace_period
    
    def _load_health_cache(self) -> None:
        """Load previous health status from cache file"""
        try:
            cache_file = Path("model_health_cache.json")
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for model_name, data in cache_data.items():
                    # Convert string datetime back to datetime object
                    data['last_checked'] = datetime.fromisoformat(data['last_checked'])
                    self.model_registry[model_name] = ModelHealthStatus(**data)
                    
                    # Be more lenient: Add to available models if not recently failed badly
                    if data['consecutive_failures'] < 3:
                        self.available_models.add(model_name)
                
                secure_logger.info(f"📊 Loaded health data for {len(self.model_registry)} models from cache")
                
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to load health cache: {safe_error}")
    
    def _save_health_cache(self) -> None:
        """Save current health status to cache file"""
        try:
            cache_file = Path("model_health_cache.json")
            cache_data = {}
            
            for model_name, status in self.model_registry.items():
                # Convert datetime to string for JSON serialization
                data = asdict(status)
                data['last_checked'] = status.last_checked.isoformat()
                cache_data[model_name] = data
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            secure_logger.debug(f"💾 Saved health data for {len(cache_data)} models to cache")
            
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to save health cache: {safe_error}")
    
    async def initialize_provider(self) -> bool:
        """Initialize the AI provider for health checks"""
        try:
            if not self.provider:
                self.provider = ProviderFactory.create_provider(
                    provider_type='hf_inference',
                    api_key=Config.get_hf_token(),
                    api_mode=Config.HF_API_MODE,
                    timeout=self.health_check_timeout,
                    max_retries=1,  # Fast health checks
                    retry_delay=0.5
                )
            return True
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.error(f"❌ Failed to initialize provider for health checks: {safe_error}")
            return False
    
    async def check_model_health(self, model_name: str) -> ModelHealthStatus:
        """
        Check health of a specific model with quick timeout
        More lenient approach: models are available by default unless proven otherwise
        
        Args:
            model_name (str): Model to check
            
        Returns:
            ModelHealthStatus: Health status of the model
        """
        start_time = time.time()
        
        # Get existing status or create new one with OPTIMISTIC availability
        existing_status = self.model_registry.get(model_name)
        if existing_status:
            status = existing_status
        else:
            status = ModelHealthStatus(
                model_name=model_name,
                is_available=True,  # OPTIMISTIC: Available by default
                last_checked=datetime.now(),
                response_time=0.0,
                success_rate=0.0,
                error_count=0,
                last_error=None,
                consecutive_failures=0,
                quality_score=5.0,  # Start with reasonable quality
                recent_performance=deque(maxlen=20),
                real_time_score=5.0,
                adaptation_weight=0.3,
                trend_direction='stable',
                peak_performance_time=None,
                last_adaptation=datetime.now()
            )
        
        try:
            if not await self.initialize_provider():
                status.consecutive_failures += 1
                status.last_error = "Provider initialization failed"
                # LENIENT: Only mark unavailable after 3 consecutive failures
                if status.consecutive_failures >= 3:
                    status.is_available = False
                    self.available_models.discard(model_name)
                    secure_logger.warning(f"⚠️ Model {model_name} marked unavailable after {status.consecutive_failures} failures")
                return status
            
            # Quick health check with minimal prompt
            test_prompt = "Hello"
            
            # CRITICAL FIX: Use a robust timeout wrapper with proper provider method
            try:
                # Import the proper request type for health checks
                from .ai_providers import CompletionRequest
                
                # Create a proper completion request for health checks
                health_request = CompletionRequest(
                    prompt=test_prompt,
                    model=model_name,
                    max_tokens=5,
                    temperature=0.1
                )
                
                if self.provider:
                    response = await asyncio.wait_for(
                        self.provider.text_completion(health_request),
                        timeout=self.health_check_timeout
                    )
                else:
                    raise ValueError("Provider not initialized")
            except (AttributeError, ValueError):
                # Fallback to legacy method if provider doesn't have text_completion
                if self.provider:
                    from .ai_providers import ChatCompletionRequest, ChatMessage
                    chat_request = ChatCompletionRequest(
                        model=model_name,
                        messages=[ChatMessage(role='user', content=test_prompt)],
                        max_tokens=5,
                        temperature=0.1
                    )
                    response = await asyncio.wait_for(
                        self.provider.chat_completion(chat_request),
                        timeout=self.health_check_timeout
                    )
                else:
                    raise ValueError("Provider not initialized")
            
            response_time = time.time() - start_time
            
            if response and response.success and response.content:
                # Model is healthy
                status.is_available = True
                status.response_time = (status.response_time + response_time) / 2  # Running average
                status.consecutive_failures = 0
                status.last_error = None
                
                # Calculate success rate (simple estimation based on recent performance)
                if status.error_count == 0:
                    status.success_rate = 1.0
                else:
                    total_attempts = status.error_count + 10  # Assume some successful attempts
                    status.success_rate = max(0.1, 10 / total_attempts)
                
                # Calculate quality score based on response time and success rate
                speed_score = max(0, 5 - response_time)  # Faster = better
                quality_score = min(10.0, (status.success_rate * 5) + speed_score)
                status.quality_score = quality_score
                
                self.available_models.add(model_name)
                secure_logger.info(f"✅ Model {model_name} is healthy (⚡{response_time:.2f}s, 🏆{quality_score:.1f}/10)")
                
            else:
                # Model responded but with errors - treat as temporary issue
                status.consecutive_failures += 1
                status.error_count += 1
                status.last_error = f"Invalid response: {response.error_message if response else 'No response'}"
                # LENIENT: Only mark unavailable after 3 consecutive failures
                if status.consecutive_failures >= 3:
                    status.is_available = False
                    self.available_models.discard(model_name)
                    secure_logger.warning(f"⚠️ Model {model_name} marked unavailable after {status.consecutive_failures} invalid responses")
                else:
                    secure_logger.warning(f"⚠️ Model {model_name} responded with errors (failure {status.consecutive_failures}/3)")
        
        except asyncio.TimeoutError:
            # LENIENT: Timeout is a temporary issue, don't immediately mark unavailable
            status.consecutive_failures += 1
            status.error_count += 1
            status.last_error = f"Health check timeout after {self.health_check_timeout}s"
            # Only mark unavailable after 3 consecutive timeouts
            if status.consecutive_failures >= 3:
                status.is_available = False
                self.available_models.discard(model_name)
                secure_logger.warning(f"⏰ Model {model_name} marked unavailable after {status.consecutive_failures} timeouts")
            else:
                secure_logger.warning(f"⏰ Model {model_name} health check timed out (failure {status.consecutive_failures}/3)")
            
        except Exception as e:
            # LENIENT: General errors are temporary issues
            safe_error = redact_sensitive_data(str(e))
            status.consecutive_failures += 1
            status.error_count += 1
            status.last_error = safe_error
            # Only mark unavailable after 3 consecutive failures
            if status.consecutive_failures >= 3:
                status.is_available = False
                self.available_models.discard(model_name)
                secure_logger.warning(f"❌ Model {model_name} marked unavailable after {status.consecutive_failures} errors: {safe_error}")
            else:
                secure_logger.warning(f"❌ Model {model_name} health check failed (failure {status.consecutive_failures}/3): {safe_error}")
        
        finally:
            status.last_checked = datetime.now()
            self.model_registry[model_name] = status
        
        return status
    
    async def startup_health_checks(self, priority_models: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Perform startup health checks on critical models
        OPTIMISTIC APPROACH: Models are available by default, checks verify and optimize
        
        Args:
            priority_models (List[str]): Models to check first (high priority)
            
        Returns:
            Dict[str, bool]: Model availability status
        """
        secure_logger.info("🚀 Starting comprehensive model health checks...")
        
        # Get models to check
        models_to_check = priority_models or self._get_critical_models()
        
        # OPTIMISTIC: Mark all models as available BEFORE checking
        # This ensures AI functionality works even if health checks fail
        for model in models_to_check:
            if model:
                self.available_models.add(model)
        secure_logger.info(f"✅ Pre-marked {len(models_to_check)} critical models as available (optimistic)")
        
        # Perform health checks concurrently (but with limits to avoid overwhelming API)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent checks
        
        async def check_with_semaphore(model_name: str) -> Tuple[str, bool]:
            async with semaphore:
                status = await self.check_model_health(model_name)
                return model_name, status.is_available
        
        # Run health checks
        start_time = time.time()
        tasks = [check_with_semaphore(model) for model in models_to_check]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        availability_map = {}
        successful_checks = 0
        
        for result in results:
            if isinstance(result, Exception):
                safe_error = redact_sensitive_data(str(result))
                secure_logger.error(f"❌ Health check task failed: {safe_error}")
                continue
            
            try:
                # Ensure result is a tuple with exactly 2 elements
                if isinstance(result, (tuple, list)) and len(result) == 2:
                    model_name, is_available = result
                else:
                    secure_logger.error(f"❌ Invalid health check result format - expected tuple, got: {type(result)} - {result}")
                    continue
            except (TypeError, ValueError) as e:
                secure_logger.error(f"❌ Invalid health check result format: {result}")
                continue
            availability_map[model_name] = is_available
            if is_available:
                successful_checks += 1
        
        total_time = time.time() - start_time
        
        secure_logger.info(f"🏥 Health checks completed in {total_time:.2f}s")
        secure_logger.info(f"✅ Available models: {successful_checks}/{len(models_to_check)}")
        secure_logger.info(f"🎯 Active models: {list(self.available_models)}")
        
        self.last_full_check = datetime.now()
        self._save_health_cache()
        
        return availability_map
    
    def _get_critical_models(self) -> List[str]:
        """Get list of critical models to check during startup"""
        return [
            # Verified working models from config
            Config.DEFAULT_TEXT_MODEL,
            Config.BALANCED_TEXT_MODEL,
            Config.FALLBACK_TEXT_MODEL,
            Config.DEFAULT_CODE_MODEL,
            Config.FALLBACK_CODE_MODEL,
            Config.MATH_TEXT_MODEL,
            Config.REASONING_TEXT_MODEL,
            
            # Additional verified models
            "microsoft/Phi-3-mini-4k-instruct",
            "Qwen/Qwen2.5-1.5B-Instruct", 
            "Qwen/Qwen2.5-0.5B-Instruct",
            "facebook/bart-large-cnn",
            "facebook/bart-base"
        ]
    
    def get_available_models(self, intent_type: Optional[str] = None) -> List[str]:
        """
        Get list of currently available models, optionally filtered by intent type
        ALWAYS returns models - uses fallback to critical models if needed
        
        Args:
            intent_type (str): Optional intent type to filter models
            
        Returns:
            List[str]: Available models for the intent type (never empty)
        """
        # During grace period, always include critical models
        if self._is_in_grace_period():
            self._ensure_critical_models_available()
        
        # If no specific intent, return all available models (with fallback)
        if not intent_type:
            if self.available_models:
                return list(self.available_models)
            else:
                # FALLBACK: Return critical models if available_models is empty
                critical = self._get_critical_models()
                secure_logger.warning(f"⚠️ No available models, using critical models as fallback")
                return [m for m in critical if m]
        
        # Filter models based on intent type and availability
        available_for_intent = []
        
        # Get all models for the intent type from config
        try:
            from .router import IntelligentRouter
            router = IntelligentRouter()
            all_models_for_intent = router.model_selector._get_available_models_for_intent(intent_type)
            
            # Return only the ones that are verified available
            for model in all_models_for_intent:
                if model in self.available_models:
                    available_for_intent.append(model)
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Could not get models for intent {intent_type}: {safe_error}")
        
        # If no models are available for this intent, use fallback strategy
        if not available_for_intent:
            if self.available_models:
                # Use any available model as fallback
                try:
                    fallback_model = next(iter(self.available_models))
                    available_for_intent = [fallback_model]
                    secure_logger.warning(f"⚠️ No available models for {intent_type}, using fallback: {fallback_model}")
                except StopIteration:
                    pass
            
            # Ultimate fallback: Use critical models
            if not available_for_intent:
                critical = self._get_critical_models()
                available_for_intent = [m for m in critical if m][:1]  # Return first critical model
                if available_for_intent:
                    secure_logger.warning(f"⚠️ Using critical model fallback for {intent_type}: {available_for_intent[0]}")
        
        return available_for_intent
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is currently available"""
        return model_name in self.available_models
    
    def get_model_quality_score(self, model_name: str) -> float:
        """Get quality score for a specific model"""
        status = self.model_registry.get(model_name)
        return status.quality_score if status else 0.0
    
    def should_recheck_models(self) -> bool:
        """Check if it's time to recheck model health"""
        if not self.last_full_check:
            return True
        return datetime.now() - self.last_full_check > self.check_interval
    
    def should_avoid_model(self, model_name: str) -> bool:
        """
        Check if a model should be avoided due to health issues
        LENIENT: During grace period, never avoid critical models
        
        Args:
            model_name (str): Model to check
            
        Returns:
            bool: True if model should be avoided
        """
        # During grace period, never avoid critical models
        if self._is_in_grace_period():
            if model_name in self._get_critical_models():
                return False
        
        status = self.model_registry.get(model_name)
        if not status:
            # LENIENT: Don't avoid unknown critical models
            if model_name in self._get_critical_models():
                return False
            return True
        
        # Avoid only if has many consecutive failures (increased threshold)
        return not status.is_available or status.consecutive_failures >= 5
    
    def get_best_model(self, intent_type: str) -> str:
        """
        Get the best available model for a given intent type
        
        Args:
            intent_type (str): Type of task ("general", "code_generation", "math", etc.)
            
        Returns:
            str: Best available model name
        """
        from ..config import Config
        
        # Define preferred models for each intent type since PREFERRED_MODELS doesn't exist
        preferred_models = {
            "general": [Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL, Config.EFFICIENT_TEXT_MODEL],
            "text_generation": [Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL, Config.EFFICIENT_TEXT_MODEL],
            "code_generation": [Config.DEFAULT_CODE_MODEL, Config.ADVANCED_CODE_MODEL, Config.EFFICIENT_CODE_MODEL],
            "mathematical_reasoning": [Config.MATH_TEXT_MODEL, Config.REASONING_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL],
            "image_generation": [Config.DEFAULT_IMAGE_MODEL, Config.FLAGSHIP_IMAGE_MODEL, Config.FALLBACK_IMAGE_MODEL],
            "question_answering": [Config.DEFAULT_QA_MODEL, Config.ADVANCED_QA_MODEL, Config.FALLBACK_QA_MODEL],
            "summarization": [Config.DEFAULT_SUMMARIZATION_MODEL, Config.ADVANCED_SUMMARIZATION_MODEL, Config.FAST_SUMMARIZATION_MODEL],
            "translation": [Config.DEFAULT_TRANSLATION_MODEL, Config.ADVANCED_TRANSLATION_MODEL, Config.FAST_TRANSLATION_MODEL],
        }
        
        # Get models for this intent type
        candidates = preferred_models.get(intent_type, preferred_models["general"])
        
        # Filter out None values and find the best available model
        available_candidates = []
        for model in candidates:
            if model and not self.should_avoid_model(model):
                available_candidates.append(model)
        
        if available_candidates:
            # Return the first available model from the preferred list
            return available_candidates[0]
        
        # Fallback to any available model if no preferred models are available
        if self.available_models:
            for model in self.available_models:
                if not self.should_avoid_model(model):
                    return model
        
        # Final fallback to default models
        fallback_models = [
            Config.DEFAULT_TEXT_MODEL,
            Config.EFFICIENT_TEXT_MODEL,
            Config.FALLBACK_TEXT_MODEL,
            Config.LIGHTWEIGHT_TEXT_MODEL
        ]
        
        for model in fallback_models:
            if model:
                return model
        
        # Last resort - return first available model even if it should be avoided
        if self.available_models:
            return next(iter(self.available_models))
        
        # Ultimate fallback
        return Config.DEFAULT_TEXT_MODEL or "gpt2"
    
    async def update_metrics(self, model_name: str, success: bool, response_time: float|None=None, error: str|None=None) -> None:
        """
        Update performance metrics for a model
        
        Args:
            model_name (str): Model name
            success (bool): Whether the request was successful
            response_time (float|None): Response time in seconds
            error (str|None): Error message if failed
        """
        current_time = datetime.now()
        
        # Get existing status or create new one with OPTIMISTIC defaults
        if model_name in self.model_registry:
            status = self.model_registry[model_name]
        else:
            status = ModelHealthStatus(
                model_name=model_name,
                is_available=True,  # OPTIMISTIC: Available by default
                last_checked=current_time,
                response_time=0.0,
                success_rate=0.0,
                error_count=0,
                last_error=None,
                consecutive_failures=0,
                quality_score=5.0,
                recent_performance=deque(maxlen=50),
                real_time_score=5.0,
                adaptation_weight=0.3,
                trend_direction='stable',
                peak_performance_time=None,
                last_adaptation=current_time
            )
            self.model_registry[model_name] = status
            # Add to available models on first use
            self.available_models.add(model_name)
        
        # Update basic metrics
        status.last_checked = current_time
        
        if success:
            status.consecutive_failures = 0
            status.is_available = True
            
            # Update response time (rolling average)
            if response_time is not None:
                if status.response_time == 0.0:
                    status.response_time = response_time
                else:
                    # Simple moving average with 80% weight to previous
                    status.response_time = 0.8 * status.response_time + 0.2 * response_time
            
            # Update success rate (moving average)
            if status.success_rate == 0.0:
                status.success_rate = 1.0
            else:
                status.success_rate = min(1.0, 0.9 * status.success_rate + 0.1)
            
            # Update quality score based on success and response time
            speed_factor = max(0.1, min(1.0, 10.0 / max(0.1, status.response_time)))
            status.quality_score = min(10.0, status.success_rate * 10.0 * speed_factor)
            
            # Add to available models
            self.available_models.add(model_name)
            
        else:
            status.consecutive_failures += 1
            status.error_count += 1
            status.last_error = error
            
            # Update success rate (moving average with failure)
            status.success_rate = max(0.0, 0.9 * status.success_rate)
            
            # Degrade quality score
            status.quality_score = max(0.0, status.quality_score * 0.8)
            
            # Remove from available models if too many failures
            if status.consecutive_failures >= 3:
                status.is_available = False
                self.available_models.discard(model_name)
        
        # Save updated metrics to cache
        self._save_health_cache()
    
    def record_real_time_feedback(self, model_name: str, success: bool, 
                                 response_time: float, quality_score: float,
                                 intent_type: str, complexity: PromptComplexity,
                                 conversation_id: Optional[str] = None,
                                 error_type: Optional[str] = None) -> None:
        """
        Record real-time performance feedback for immediate adaptation
        
        Args:
            model_name: Name of the model
            success: Whether the request was successful
            response_time: Response time in seconds
            quality_score: Quality score (0-10)
            intent_type: Type of task
            complexity: Task complexity information
            conversation_id: Optional conversation identifier
            error_type: Type of error if failed
        """
        current_time = datetime.now()
        
        # Ensure model is in registry with OPTIMISTIC defaults
        if model_name not in self.model_registry:
            # Create basic entry for immediate use, will be updated by health check later
            self.model_registry[model_name] = ModelHealthStatus(
                model_name=model_name,
                is_available=True,  # OPTIMISTIC: Available by default
                last_checked=datetime.now(),
                response_time=0.0,
                success_rate=0.0,
                error_count=0,
                last_error=None,
                consecutive_failures=0,
                quality_score=5.0,
                recent_performance=deque(maxlen=20),
                real_time_score=5.0,
                adaptation_weight=0.3,
                trend_direction='stable',
                peak_performance_time=None,
                last_adaptation=datetime.now()
            )
            # Add to available models immediately
            self.available_models.add(model_name)
        
        status = self.model_registry[model_name]
        
        # Record feedback in recent performance queue
        feedback = {
            'timestamp': current_time,
            'success': success,
            'response_time': response_time,
            'quality_score': quality_score,
            'intent_type': intent_type,
            'complexity_score': complexity.complexity_score,
            'error_type': error_type
        }
        
        status.recent_performance.append(feedback)
        self.real_time_feedback[model_name].append(feedback)
        
        # Update real-time score immediately
        self._update_real_time_score(model_name)
        
        # Check if immediate adaptation is needed
        if self._should_adapt_immediately(model_name):
            self._adapt_model_weights(model_name)
        
        # Record performance in predictor for learning
        performance_predictor.record_performance(
            model_name, intent_type, success, response_time, 
            quality_score, complexity, error_type
        )
        
        # Update conversation context if provided
        if conversation_id:
            self._update_conversation_context(conversation_id, model_name, success, intent_type)
        
        secure_logger.debug(f"🔄 Real-time feedback recorded for {model_name}: "
                           f"success={success}, rt_score={status.real_time_score:.2f}")
    
    def _update_real_time_score(self, model_name: str) -> None:
        """Update real-time performance score based on recent feedback"""
        status = self.model_registry[model_name]
        recent_feedback = list(status.recent_performance)
        
        if not recent_feedback:
            return
        
        # Calculate real-time metrics from recent feedback
        recent_successes = [f for f in recent_feedback if f['success']]
        recent_success_rate = len(recent_successes) / len(recent_feedback)
        
        # Average response time for successful requests
        if recent_successes:
            avg_response_time = np.mean([f['response_time'] for f in recent_successes])
            avg_quality = np.mean([f['quality_score'] for f in recent_successes])
        else:
            avg_response_time = 10.0  # Penalty for no successes
            avg_quality = 0.0
        
        # Calculate time-weighted score (more recent = higher weight)
        current_time = datetime.now()
        weighted_scores = []
        
        for feedback in recent_feedback:
            age_hours = (current_time - feedback['timestamp']).total_seconds() / 3600
            weight = max(0.1, 1.0 - age_hours / 24)  # Linear decay over 24 hours
            
            # Calculate individual score
            individual_score = 0.0
            if feedback['success']:
                speed_score = max(0, 10 - feedback['response_time'] * 2)
                individual_score = (feedback['quality_score'] * 0.6 + speed_score * 0.4)
            
            weighted_scores.append(individual_score * weight)
        
        # Calculate weighted average
        if weighted_scores:
            status.real_time_score = float(np.average(weighted_scores))
        else:
            status.real_time_score = 5.0  # Default score
        
        # Detect performance trend
        status.trend_direction = self._detect_performance_trend(recent_feedback)
        
        # Track peak performance
        if status.real_time_score > 8.0:
            if not status.peak_performance_time or status.real_time_score > 8.5:
                status.peak_performance_time = current_time
    
    def _detect_performance_trend(self, recent_feedback: List[Dict]) -> str:
        """Detect if model performance is improving, declining, or stable"""
        if len(recent_feedback) < 5:
            return 'stable'
        
        # Split into early and recent halves
        mid_point = len(recent_feedback) // 2
        early_half = recent_feedback[:mid_point]
        recent_half = recent_feedback[mid_point:]
        
        # Calculate average scores for each half
        early_avg = np.mean([1.0 if f['success'] else 0.0 for f in early_half])
        recent_avg = np.mean([1.0 if f['success'] else 0.0 for f in recent_half])
        
        difference = recent_avg - early_avg
        threshold = self.adaptation_thresholds['trend_sensitivity']
        
        if difference > threshold:
            return 'improving'
        elif difference < -threshold:
            return 'declining'
        else:
            return 'stable'
    
    def _should_adapt_immediately(self, model_name: str) -> bool:
        """Determine if immediate adaptation is needed based on performance"""
        status = self.model_registry[model_name]
        
        # Adapt if performance is very poor
        if status.real_time_score < self.adaptation_thresholds['poor_performance']:
            return True
        
        # Adapt if declining trend with recent failures
        if (status.trend_direction == 'declining' and 
            status.consecutive_failures >= 2):
            return True
        
        # Adapt if we haven't adapted recently and score is low
        time_since_adaptation = datetime.now() - status.last_adaptation
        if (time_since_adaptation > timedelta(minutes=10) and 
            status.real_time_score < 0.6):
            return True
        
        return False
    
    def _adapt_model_weights(self, model_name: str) -> None:
        """Adapt model weights based on real-time performance"""
        status = self.model_registry[model_name]
        current_time = datetime.now()
        
        # Increase adaptation weight for poor performing models
        if status.real_time_score < 0.4:
            status.adaptation_weight = min(0.8, status.adaptation_weight + 0.2)
        elif status.real_time_score > 0.8:
            status.adaptation_weight = max(0.1, status.adaptation_weight - 0.1)
        
        # Update dynamic weights for routing decisions
        performance_factor = status.real_time_score / 10.0
        trend_factor = {'improving': 1.2, 'stable': 1.0, 'declining': 0.8}[status.trend_direction]
        
        self.dynamic_weights[model_name] = performance_factor * trend_factor
        
        status.last_adaptation = current_time
        
        secure_logger.info(f"🎛️ Adapted weights for {model_name}: "
                          f"rt_score={status.real_time_score:.2f}, "
                          f"weight={self.dynamic_weights[model_name]:.2f}, "
                          f"trend={status.trend_direction}")
    
    def _update_conversation_context(self, conversation_id: str, model_name: str, 
                                   success: bool, intent_type: str) -> None:
        """Update conversation-level context for better routing decisions"""
        if conversation_id not in self.conversation_context:
            self.conversation_context[conversation_id] = {
                'models_used': [],
                'success_pattern': [],
                'intent_progression': [],
                'start_time': datetime.now(),
                'turn_count': 0
            }
        
        context = self.conversation_context[conversation_id]
        context['models_used'].append(model_name)
        context['success_pattern'].append(success)
        context['intent_progression'].append(intent_type)
        context['turn_count'] += 1
        
        # Keep only recent conversation history (last 20 turns)
        if len(context['models_used']) > 20:
            context['models_used'] = context['models_used'][-20:]
            context['success_pattern'] = context['success_pattern'][-20:]
            context['intent_progression'] = context['intent_progression'][-20:]
    
    def get_conversation_aware_rankings(self, conversation_id: Optional[str] = None,
                                      intent_type: Optional[str] = None,
                                      complexity: Optional[PromptComplexity] = None) -> List[Tuple[str, float]]:
        """
        Get model rankings that consider conversation context and real-time performance
        
        Args:
            conversation_id: Optional conversation identifier
            intent_type: Type of task being performed
            complexity: Task complexity information
            
        Returns:
            List of (model_name, score) tuples sorted by score descending
        """
        rankings = []
        
        for model_name, status in self.model_registry.items():
            if not status.is_available:
                continue
            
            # Base score from traditional metrics
            base_score = status.quality_score * status.success_rate
            
            # Apply real-time adaptation
            real_time_weight = status.adaptation_weight
            adapted_score = (base_score * (1 - real_time_weight) + 
                           status.real_time_score * real_time_weight)
            
            # Apply dynamic weights
            if model_name in self.dynamic_weights:
                adapted_score *= self.dynamic_weights[model_name]
            
            # Apply conversation-specific adjustments
            context = None
            if conversation_id and conversation_id in self.conversation_context:
                context = self.conversation_context[conversation_id]
                adapted_score = self._apply_conversation_adjustments(
                    adapted_score, model_name, context, intent_type
                )
            
            # Apply performance prediction if available
            if intent_type and complexity:
                prediction_context = PredictionContext(
                    intent_type=intent_type,
                    complexity=complexity,
                    conversation_length=context.get('turn_count', 0) if context else 0,
                    user_preferences={},
                    time_constraints=None,
                    quality_requirements=7.0,
                    previous_models_used=context.get('models_used', [])[-3:] if context else []
                )
                
                predictions = performance_predictor.predict_performance(prediction_context)
                prediction_scores = {model: score for model, score in predictions}
                
                if model_name in prediction_scores:
                    prediction_weight = 0.3  # 30% weight for predictions
                    adapted_score = (adapted_score * (1 - prediction_weight) + 
                                   prediction_scores[model_name] * prediction_weight)
            
            rankings.append((model_name, adapted_score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _apply_conversation_adjustments(self, base_score: float, model_name: str,
                                      conversation_context: Dict, intent_type: Optional[str]) -> float:
        """Apply conversation-specific adjustments to model scores"""
        adjusted_score = base_score
        
        # Penalize recently failed models in this conversation
        recent_models = conversation_context['models_used'][-5:]  # Last 5 models
        recent_successes = conversation_context['success_pattern'][-5:]  # Last 5 results
        
        if model_name in recent_models:
            # Find the most recent usage
            for i in range(len(recent_models) - 1, -1, -1):
                if recent_models[i] == model_name:
                    if not recent_successes[i]:  # Recent failure
                        adjusted_score *= 0.7  # 30% penalty
                    break
        
        # Boost models that haven't been tried recently in long conversations
        if conversation_context['turn_count'] > 5 and model_name not in recent_models:
            adjusted_score *= 1.1  # 10% boost for variety
        
        # Consider intent progression - if we're switching task types, prefer models good at new type
        intent_progression = conversation_context['intent_progression']
        if len(intent_progression) > 1 and intent_type:
            if intent_progression[-1] != intent_type:  # Task type is changing
                # This would require model-specific intent performance data
                # For now, apply a small boost to encourage specialization
                adjusted_score *= 1.05
        
        return adjusted_score
    
    def model_rankings(self) -> list[tuple[str, float]]:
        """
        Get models ranked by their quality score and success rate
        
        Returns:
            list[tuple[str, float]]: List of (model_name, score) sorted by score descending
        """
        rankings = []
        
        for model_name, status in self.model_registry.items():
            # Calculate combined score (quality * success_rate * availability)
            availability_factor = 1.0 if status.is_available else 0.1
            combined_score = status.quality_score * status.success_rate * availability_factor
            rankings.append((model_name, combined_score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    async def background_health_monitoring(self) -> None:
        """Background task to periodically check model health"""
        while True:
            try:
                if self.should_recheck_models():
                    secure_logger.info("🔄 Starting background model health check...")
                    await self.startup_health_checks()
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                safe_error = redact_sensitive_data(str(e))
                secure_logger.error(f"❌ Background health monitoring error: {safe_error}")
                await asyncio.sleep(60)  # Shorter sleep on error

# Global health monitor instance
health_monitor = ModelHealthMonitor()