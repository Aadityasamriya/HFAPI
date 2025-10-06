"""
Hugging Face API integration with intelligent model calling
Phase 1 HF API Migration: Added provider abstraction with backward compatibility
Supports text generation, image creation, code generation, and more
"""

import aiohttp
import asyncio
import io
import base64
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union, cast
from ..config import Config
from ..security_utils import redact_sensitive_data, get_secure_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
# Use centralized secure logger from security_utils
secure_logger = get_secure_logger(logger)

# Phase 1 HF API Migration: Provider system imports
from .provider_factory import ProviderFactory
from .ai_providers import (
    AIProvider, ChatMessage, ChatCompletionRequest, 
    CompletionRequest, ProviderResponse, ProviderError
)
from .math_calculator import MathReasoningEnhancer
from .types import IntentType
PROVIDER_SYSTEM_AVAILABLE = True


class ModelCaller:
    """Handles all Hugging Face API interactions with intelligent routing
    Phase 1 HF API Migration: Added provider abstraction with backward compatibility"""
    
    def __init__(self, provider: str = "auto", bill_to: Optional[str] = None):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
        self.provider = provider  # 2025 feature: provider selection
        self.bill_to = bill_to    # 2025 feature: organization billing
        self.request_count = 0
        self.last_request_time = 0
        
        # Initialize HF_TOKEN validation state
        self._hf_token_validated = False
        self._hf_token_error = None
        
        # CRITICAL FIX: Track current fallback chain for tier degradation
        self._current_fallback_chain = []
        self._current_fallback_index = 0
        
        # CRITICAL FIX: Use single lock to prevent deadlock conditions
        self._lock = asyncio.Lock()  # Single lock for all thread-safe operations
        
        # Phase 1 HF API Migration: Initialize provider system
        self._ai_provider: Optional[AIProvider] = None
        self._provider_initialized = False
        self._use_provider_system = PROVIDER_SYSTEM_AVAILABLE and Config.HF_API_MODE in ['inference_providers', 'auto']
        
        if self._use_provider_system:
            secure_logger.info("🚀 Initializing ModelCaller with provider system support")
        
        # Initialize math reasoning enhancer for mathematical accuracy
        self.math_enhancer = MathReasoningEnhancer()
    
    async def _ensure_provider_initialized(self, api_key: Optional[str] = None) -> bool:
        """
        Ensure the AI provider is initialized for the new provider system
        
        Args:
            api_key (Optional[str]): HF API key
            
        Returns:
            bool: True if provider is initialized successfully
        """
        if not self._use_provider_system or not PROVIDER_SYSTEM_AVAILABLE:
            return False
            
        if self._provider_initialized and self._ai_provider:
            return True
            
        try:
            # Use the API key parameter or get from config
            effective_api_key = api_key or Config.get_hf_token()
            
            self._ai_provider = ProviderFactory.create_provider(
                provider_type='hf_inference',
                api_key=effective_api_key,
                api_mode=Config.HF_API_MODE,
                base_url=Config.HF_API_BASE_URL,
                provider_name=Config.HF_PROVIDER,
                organization=Config.HF_ORG,
                timeout=Config.REQUEST_TIMEOUT,
                max_retries=Config.MAX_RETRIES,
                retry_delay=Config.RETRY_DELAY
            )
            
            self._provider_initialized = True
            secure_logger.info(f"✅ AI Provider initialized: {Config.HF_API_MODE}")
            return True
            
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.error(f"❌ Failed to initialize AI provider: {safe_error}")
            self._provider_initialized = False
            return False
    
    async def _call_provider(self, model_name: str, prompt: str, parameters: Dict, is_chat: bool = False, intent_type: str = "text") -> Tuple[bool, Any]:
        """
        Call the AI provider with fallback to legacy API
        
        Args:
            model_name (str): Model to use
            prompt (str): Text prompt or formatted prompt
            parameters (Dict): Generation parameters
            is_chat (bool): Whether this is a chat completion request
            intent_type (str): Type of task for proper HF task routing
            
        Returns:
            Tuple[bool, Any]: (success, response_data)
        """
        if not self._use_provider_system or not await self._ensure_provider_initialized():
            # Fallback to legacy aiohttp API
            return await self._make_api_call_legacy(model_name, prompt, parameters, intent_type)
        
        # CRITICAL FIX: Use helper methods for proper task routing
        task_type = self._get_hf_task_type(model_name, intent_type)
        normalized_params = self._normalize_hf_parameters(parameters, model_name, task_type)
        
        secure_logger.info(f"🔧 Using task type '{task_type}' for model {model_name} with intent '{intent_type}'")
        
        try:
            if is_chat:
                # Convert prompt to chat messages if needed
                if isinstance(prompt, str):
                    messages = [ChatMessage(role="user", content=prompt)]
                else:
                    messages = prompt  # Assume it's already formatted
                
                request = ChatCompletionRequest(
                    messages=messages,
                    model=model_name,
                    max_tokens=normalized_params.get('max_new_tokens'),
                    temperature=normalized_params.get('temperature'),
                    top_p=normalized_params.get('top_p')
                )
                
                assert self._ai_provider is not None  # Guaranteed by _ensure_provider_initialized()
                response = await self._ai_provider.chat_completion(request)
            else:
                request = CompletionRequest(
                    prompt=prompt,
                    model=model_name,
                    max_tokens=normalized_params.get('max_new_tokens'),
                    temperature=normalized_params.get('temperature'),
                    top_p=normalized_params.get('top_p')
                )
                
                assert self._ai_provider is not None  # Guaranteed by _ensure_provider_initialized()
                response = await self._ai_provider.text_completion(request)
            
            if response.success:
                # Convert provider response to legacy format for compatibility
                legacy_response = [{
                    'generated_text': response.content
                }]
                secure_logger.info(f"✅ Provider API call successful: {model_name}")
                return True, legacy_response
            else:
                secure_logger.warning(f"⚠️ Provider API call failed: {response.error_message}")
                return False, response.error_message
                
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.error(f"❌ Provider API call error: {safe_error}")
            # Fallback to legacy API with proper parameters
            return await self._make_api_call_legacy(model_name, prompt, normalized_params, intent_type)
    
    async def _make_api_call_legacy(self, model_name: str, prompt: str, parameters: Dict, intent_type: str = "text") -> Tuple[bool, Any]:
        """
        Legacy API call method for backward compatibility
        
        Args:
            model_name (str): Model to use  
            prompt (str): Text prompt
            parameters (Dict): Generation parameters
            intent_type (str): Type of task for proper HF task routing
            
        Returns:
            Tuple[bool, Any]: (success, response_data)
        """
        # CRITICAL FIX: Use helper methods for proper task routing
        task_type = self._get_hf_task_type(model_name, intent_type)
        normalized_params = self._normalize_hf_parameters(parameters, model_name, task_type)
        
        secure_logger.info(f"🔧 Legacy API: Using task type '{task_type}' for model {model_name} with intent '{intent_type}'")
        
        # This will call the existing _make_api_call method
        payload = {
            "inputs": prompt,
            "parameters": normalized_params
        }
        api_key = Config.get_hf_token()
        return await self._make_api_call(model_name, payload, api_key)
    
    async def __aenter__(self):
        """Async context manager entry with thread safety"""
        async with self._lock:
            if self.session is None:
                self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with thread safety"""
        async with self._lock:
            if self.session:
                await self.session.close()
                self.session = None
    
    def validate_api_setup(self) -> Tuple[bool, str]:
        """
        Validate that AI functionality is properly configured
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if AI functionality is available in config
        is_available, status_msg = Config.is_ai_functionality_available()
        if not is_available:
            return False, status_msg
            
        # Additional validation for API key format
        hf_token = Config.get_hf_token()
        if hf_token:
            is_valid, error_msg = Config.validate_hf_token(hf_token)
            if not is_valid:
                return False, f"HF_TOKEN validation failed: {error_msg}"
        
        return True, "AI functionality ready"
    
    async def _setup_fallback_chain(self, model_name: str, intent_type: str = "text") -> None:
        """
        CRITICAL FIX: Setup tier-aware fallback chain for systematic model degradation with thread safety
        
        Args:
            model_name (str): Primary model to try
            intent_type (str): Type of task (text, code, reasoning, etc.)
        """
        from ..config import Config
        
        async with self._lock:
            # Get tier-aware fallback chain from Config
            fallback_chain = Config.get_model_fallback_chain(intent_type)
            
            if fallback_chain:
                # Ensure primary model is first, then add fallback chain
                if model_name not in fallback_chain:
                    self._current_fallback_chain = [model_name] + fallback_chain
                else:
                    # Start from the requested model in the chain
                    start_index = fallback_chain.index(model_name)
                    self._current_fallback_chain = fallback_chain[start_index:]
            else:
                # Fallback to guaranteed free-tier models if no chain found
                guaranteed_free_models = [
                    Config.DEFAULT_TEXT_MODEL,      # Qwen3-1.7B-Instruct
                    Config.EFFICIENT_TEXT_MODEL,    # Qwen3-0.6B-Instruct  
                    Config.LIGHTWEIGHT_TEXT_MODEL,  # microsoft/Phi-3-mini-4k-instruct
                    Config.LEGACY_EFFICIENT_MODEL,  # HuggingFaceH4/zephyr-7b-beta
                    Config.TERTIARY_FALLBACK_MODEL  # microsoft/Phi-3-mini-4k-instruct
                ]
                self._current_fallback_chain = [model_name] + guaranteed_free_models
            
            self._current_fallback_index = 0
            secure_logger.info(f"🔗 Fallback chain setup for {intent_type}: {[m.split('/')[-1] for m in self._current_fallback_chain[:3]]}...")
    
    async def _get_next_fallback_model(self) -> Optional[str]:
        """
        CRITICAL FIX: Get next model in tier-aware fallback chain for degradation with thread safety
        
        Returns:
            Optional[str]: Next fallback model or None if exhausted
        """
        async with self._lock:
            self._current_fallback_index += 1
            
            if self._current_fallback_index < len(self._current_fallback_chain):
                next_model = self._current_fallback_chain[self._current_fallback_index]
                secure_logger.info(f"🔄 Tier degradation: Trying fallback model {self._current_fallback_index + 1}/{len(self._current_fallback_chain)}: {next_model.split('/')[-1]}")
                return next_model
            
            secure_logger.warning(f"⚠️ All fallback models exhausted ({len(self._current_fallback_chain)} tried)")
            return None
    
    async def _make_api_call_with_fallbacks(self, model_name: str, payload: Dict, api_key: Optional[str], 
                                          intent_type: str = "text", endpoint_type: str = "inference",
                                          conversation_id: Optional[str] = None,
                                          prompt: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Enhanced API call with intelligent fallback strategies using sophisticated error analysis
        
        Args:
            model_name (str): Primary model to try
            payload (Dict): Request payload  
            api_key (Optional[str]): HF API key
            intent_type (str): Type of task for fallback chain selection
            endpoint_type (str): Endpoint type
            conversation_id (Optional[str]): Conversation identifier for context-aware fallback
            prompt (Optional[str]): Original prompt for complexity analysis
            
        Returns:
            Tuple[bool, Any]: (success, response_data)
        """
        from .dynamic_fallback_strategy import dynamic_fallback_strategy, ErrorType
        from .conversation_context_tracker import conversation_context_tracker
        from .model_selection_explainer import model_selection_explainer
        from .router import complexity_analyzer
        from .types import IntentType
        import time
        
        # Setup tier-aware fallback chain as backup
        await self._setup_fallback_chain(model_name, intent_type)
        
        current_model = model_name
        result = "No models were attempted"
        
        # Enhanced fallback management
        max_fallback_attempts = 8  # Reasonable limit with intelligent selection
        fallback_attempt = 0
        used_models = set()  # Track models already tried
        fallback_explanations = []  # Track fallback decisions for transparency
        
        # Record attempt start time for performance tracking
        start_time = time.time()
        
        while current_model is not None and fallback_attempt < max_fallback_attempts:
            # Avoid retrying the same model immediately
            if current_model in used_models:
                secure_logger.warning(f"⚠️ Skipping already tried model: {current_model.split('/')[-1]}")
                break
            
            used_models.add(current_model)
            
            # Increment request count with thread safety
            async with self._lock:
                self.request_count += 1
            
            model_start_time = time.time()
            
            # Try current model
            success, result = await self._make_api_call(current_model, payload, api_key, 0, endpoint_type)
            
            response_time = time.time() - model_start_time
            
            if success:
                # Record successful performance
                if hasattr(self, '_record_model_performance'):
                    await self._record_successful_performance(
                        current_model, intent_type, response_time, result, conversation_id
                    )
                
                # Log successful completion with context
                secure_logger.info(f"✅ Model success: {current_model.split('/')[-1]} "
                                 f"(attempt {fallback_attempt + 1}, response_time: {response_time:.2f}s)")
                
                return True, result
            
            # Enhanced error analysis using DynamicFallbackStrategy
            error_type = dynamic_fallback_strategy.analyze_error(
                error_message=result,
                model_name=current_model,
                intent_type=intent_type,
                attempt_count=fallback_attempt + 1
            )
            
            # Record failed performance for learning
            await self._record_failed_performance(
                current_model, intent_type, response_time, error_type, conversation_id
            )
            
            # Get available models for intelligent fallback
            available_models = health_monitor.get_available_models(intent_type)
            if not available_models:
                available_models = self._current_fallback_chain
            
            # Remove already tried models from available options
            available_models = [m for m in available_models if m not in used_models]
            
            if not available_models:
                secure_logger.warning(f"⚠️ No untried models available for fallback")
                break
            
            # Get conversation context for intelligent fallback
            conversation_context = None
            if conversation_id:
                conversation_context = conversation_context_tracker.get_conversation_context(conversation_id)
            
            # Analyze prompt complexity if available
            complexity = None
            if prompt:
                complexity = complexity_analyzer.analyze_complexity(prompt)
            
            # Determine intelligent fallback strategy
            fallback_decision = dynamic_fallback_strategy.determine_fallback_strategy(
                error_type=error_type,
                failed_model=current_model,
                intent_type=intent_type,
                complexity=complexity,
                available_models=available_models,
                conversation_context=conversation_context.__dict__ if conversation_context else None
            )
            
            # Record fallback decision for transparency
            fallback_explanations.append({
                'attempt': fallback_attempt + 1,
                'failed_model': current_model.split('/')[-1],
                'error_type': error_type.value,
                'strategy': fallback_decision.strategy_type,
                'reasoning': fallback_decision.reasoning,
                'confidence': fallback_decision.confidence
            })
            
            # Wait if strategy recommends it
            if fallback_decision.wait_time and fallback_decision.wait_time > 0:
                wait_seconds = min(fallback_decision.wait_time, 30)  # Cap wait time
                secure_logger.info(f"⏳ Strategic wait: {wait_seconds}s before fallback ({fallback_decision.strategy_type})")
                await asyncio.sleep(wait_seconds)
            
            # Select next model from intelligent recommendations
            if fallback_decision.recommended_models:
                # Filter recommended models to only those not yet tried
                untried_recommended = [m for m in fallback_decision.recommended_models if m not in used_models]
                
                if untried_recommended:
                    current_model = untried_recommended[0]
                    fallback_attempt += 1
                    
                    secure_logger.info(f"🎯 Intelligent fallback to {current_model.split('/')[-1]} "
                                     f"(strategy: {fallback_decision.strategy_type}, "
                                     f"confidence: {fallback_decision.confidence:.2f}, "
                                     f"attempt: {fallback_attempt}/{max_fallback_attempts})")
                    continue
                else:
                    secure_logger.warning(f"⚠️ All recommended models already tried")
            
            # If intelligent fallback failed, try traditional fallback as backup
            next_model = await self._get_next_fallback_model()
            if next_model and next_model not in used_models:
                current_model = next_model
                fallback_attempt += 1
                
                secure_logger.info(f"🔄 Traditional fallback to {current_model.split('/')[-1]} "
                                 f"(attempt {fallback_attempt}/{max_fallback_attempts})")
                continue
            else:
                break  # No more viable fallbacks
        
        # Log comprehensive failure analysis
        total_time = time.time() - start_time
        
        # Create detailed failure report
        failure_report = {
            'primary_model': model_name.split('/')[-1],
            'models_tried': [m.split('/')[-1] for m in used_models],
            'total_attempts': len(used_models),
            'total_time': f"{total_time:.2f}s",
            'fallback_strategies': fallback_explanations,
            'final_error_type': error_type.value if 'error_type' in locals() else 'unknown',
            'last_error': redact_sensitive_data(str(result))
        }
        
        # CRITICAL FIX: Handle max attempts reached
        if fallback_attempt >= max_fallback_attempts:
            secure_logger.error(f"🚨 Maximum intelligent fallback attempts reached ({max_fallback_attempts})")
            secure_logger.debug(f"📊 Failure analysis: {json.dumps(failure_report, indent=2)}")
            return False, f"Maximum fallback attempts exceeded ({max_fallback_attempts}). Failure analysis: {failure_report}"
        
        # All models failed - provide detailed analysis
        secure_logger.error(f"❌ All models failed after {len(used_models)} attempts in {total_time:.2f}s")
        secure_logger.debug(f"📊 Complete failure analysis: {json.dumps(failure_report, indent=2)}")
        
        return False, f"All intelligent fallback strategies exhausted. Analysis: {failure_report}"
    
    async def _record_successful_performance(self, model_name: str, intent_type: str, 
                                        response_time: float, result: Any, 
                                        conversation_id: Optional[str] = None) -> None:
        """Record successful model performance for learning"""
        try:
            # Calculate quality score based on response characteristics
            quality_score = self._calculate_response_quality(result, intent_type)
            
            # Record in health monitor for real-time adaptation
            health_monitor.record_real_time_feedback(
                model_name=model_name,
                success=True,
                response_time=response_time,
                quality_score=quality_score,
                intent_type=intent_type,
                complexity=getattr(self, '_last_complexity', None),
                conversation_id=conversation_id,
                error_type=None
            )
            
            secure_logger.debug(f"📊 Recorded success: {model_name.split('/')[-1]} "
                              f"(quality: {quality_score:.1f}, time: {response_time:.2f}s)")
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to record performance: {redact_sensitive_data(str(e))}")
    
    async def _record_failed_performance(self, model_name: str, intent_type: str,
                                       response_time: float, error_type: Any,
                                       conversation_id: Optional[str] = None) -> None:
        """Record failed model performance for learning"""
        try:
            # Import ErrorType if needed
            from .dynamic_fallback_strategy import ErrorType
            
            # Convert error_type to string if it's an ErrorType enum
            error_type_str = error_type.value if hasattr(error_type, 'value') else str(error_type)
            
            # Record in health monitor for real-time adaptation
            health_monitor.record_real_time_feedback(
                model_name=model_name,
                success=False,
                response_time=response_time,
                quality_score=0.0,  # Failed attempts get 0 quality
                intent_type=intent_type,
                complexity=getattr(self, '_last_complexity', None),
                conversation_id=conversation_id,
                error_type=error_type_str
            )
            
            secure_logger.debug(f"📊 Recorded failure: {model_name.split('/')[-1]} "
                              f"(error: {error_type_str}, time: {response_time:.2f}s)")
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to record failure: {redact_sensitive_data(str(e))}")
    
    def _calculate_response_quality(self, result: Any, intent_type: str) -> float:
        """Calculate quality score for a successful response"""
        if not result:
            return 3.0  # Low quality for empty responses
        
        try:
            # Basic quality assessment based on response characteristics
            base_score = 7.0  # Default good quality for successful responses
            
            # Check response length and content
            if isinstance(result, str):
                response_length = len(result)
                
                # Adjust based on response length appropriateness
                if intent_type in ['code_generation', 'creative_writing']:
                    # These tasks benefit from longer responses
                    if response_length > 200:
                        base_score += 1.0
                    elif response_length < 50:
                        base_score -= 1.0
                elif intent_type in ['question_answering', 'conversation']:
                    # These tasks need appropriate length responses
                    if 50 <= response_length <= 500:
                        base_score += 0.5
                    elif response_length < 20 or response_length > 1000:
                        base_score -= 0.5
                
                # Check for potential quality indicators
                if 'error' in result.lower() or 'failed' in result.lower():
                    base_score -= 2.0
                
                # Check for completeness indicators
                if result.strip().endswith(('...', '..', 'Continued')):
                    base_score -= 0.5
            
            elif isinstance(result, dict):
                # For dictionary responses, check for expected structure
                if 'error' in result:
                    base_score = 2.0
                elif 'content' in result or 'response' in result:
                    base_score += 0.5
            
            return max(0.0, min(10.0, base_score))
            
        except Exception:
            return 6.0  # Default moderate quality if assessment fails
    
    def _should_try_fallback(self, error_message: str) -> bool:
        """
        CRITICAL FIX: Determine if error indicates we should try tier degradation
        
        Args:
            error_message (str): Error message from API call
            
        Returns:
            bool: True if should try fallback model
        """
        # Errors that indicate model access issues (tier-related)
        tier_related_errors = [
            "not found or not accessible",
            "not supported on HF Inference API", 
            "Model is currently loading",
            "Request timed out",
            "Rate limit exceeded",
            "Server error",
            "HTTP 403",
            "HTTP 404", 
            "HTTP 503",
            "HTTP 429",
            "gated",
            "private",
            "access denied",
            "TIER_DEGRADATION_REQUIRED"  # CRITICAL FIX: Trigger fallback for 403 errors
        ]
        
        return any(error_indicator in error_message for error_indicator in tier_related_errors)
    
    async def _make_api_call(self, model_name: str, payload: Dict, api_key: Optional[str], retries: int = 0, endpoint_type: str = "inference") -> Tuple[bool, Any]:
        """
        Make API call to Hugging Face with retry logic and 2025 API support
        
        Args:
            model_name (str): Name of the Hugging Face model
            payload (dict): Request payload
            api_key (Optional[str]): User's Hugging Face API key (required)
            retries (int): Current retry count
            endpoint_type (str): Type of endpoint ("inference", "chat", "text2img")
            
        Returns:
            Tuple[bool, Any]: (success, response_data)
        """
        
        if not api_key:
            return False, "API key is required for Hugging Face API calls"
        # 2025 API Update: Enhanced endpoint routing with provider support
        # All Hugging Face models use the same endpoint pattern - chat formatting happens in payload
        url = f"https://api-inference.huggingface.co/models/{model_name}"
            
        # 2025 API: Enhanced headers with provider and cache control
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "x-use-cache": "false",  # 2025: Disable caching for non-deterministic models
            "x-wait-for-model": "true",  # 2025: Wait for cold models to load
        }
        
        # Add Accept header for binary image generation endpoints
        if endpoint_type == "text2img":
            headers["Accept"] = "image/png, image/jpeg, application/json"
        
        # 2025: Add provider header if specified
        if self.provider and self.provider != "auto":
            headers["x-provider"] = self.provider
        
        # 2025: Add organization billing if specified  
        if self.bill_to:
            headers["x-bill-to"] = self.bill_to
        
        # 2025: Add payload provider selection
        if "parameters" not in payload:
            payload["parameters"] = {}
        if self.provider != "auto":
            payload["parameters"]["provider"] = self.provider
        
        try:
            # FIXED: Thread-safe session initialization with proper lifecycle management
            async with self._lock:
                if not self.session or self.session.closed:
                    # Create new session only if none exists or current one is closed
                    if hasattr(self, 'session') and self.session and not self.session.closed:
                        await self.session.close()
                    self.session = aiohttp.ClientSession(
                        timeout=self.timeout,
                        connector=aiohttp.TCPConnector(
                            limit=100,  # Connection pool limit
                            limit_per_host=30,  # Limit per host for better resource management
                            ttl_dns_cache=300,  # DNS cache TTL
                            use_dns_cache=True,
                            keepalive_timeout=30
                        )
                    )
                
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # Handle binary image responses
                    if ('image' in content_type or 
                        endpoint_type == "text2img" or 
                        content_type.startswith('application/octet-stream')):
                        try:
                            image_data = await response.read()
                            if len(image_data) > 0 and image_data[:4] in [b'\x89PNG', b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xdb']:
                                # Valid image data (PNG or JPEG headers)
                                secure_logger.debug(f"Successfully received {len(image_data)} bytes of image data")
                                return True, image_data
                            else:
                                # Try to parse as JSON error response
                                try:
                                    text_response = image_data.decode('utf-8')
                                    json_response = json.loads(text_response)
                                    return False, json_response.get('error', 'Invalid image data received')
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    return False, "Invalid binary response received"
                        except Exception as e:
                            secure_logger.error(f"Error reading binary response: {e}")
                            # SECURITY FIX: Apply redaction to exception details in error returns
                            safe_exception_msg = redact_sensitive_data(str(e))
                            return False, f"Error processing image response: {safe_exception_msg}"
                    else:
                        # Handle JSON responses
                        try:
                            result = await response.json()
                            return True, result
                        except json.JSONDecodeError as e:
                            # Try to read as text for better error messages
                            text_response = await response.text()
                            # SECURITY FIX: Never log raw response data that could contain API keys
                            safe_response_snippet = redact_sensitive_data(text_response[:200]) if text_response else '[NO_RESPONSE]'
                            secure_logger.error(f"JSON decode error for model {model_name}: {e}, response_preview: {safe_response_snippet}")
                            # CRITICAL SECURITY FIX: Apply redaction to ALL error return messages
                            safe_response_excerpt = redact_sensitive_data(text_response[:100]) if text_response else '[NO_RESPONSE]'
                            return False, f"Invalid JSON response: {safe_response_excerpt}"
                
                elif response.status == 503:  # Model loading
                    if retries < Config.MAX_RETRIES:
                        # FIXED: Add jitter to prevent thundering herd
                        import random
                        jitter = random.uniform(0.1, 0.5)
                        wait_time = Config.RETRY_DELAY * (retries + 1) + jitter
                        await asyncio.sleep(wait_time)
                        return await self._make_api_call(model_name, payload, api_key, retries + 1)
                    else:
                        return False, "Model is currently loading. Please try again in a few moments."
                
                elif response.status == 429:  # Rate limit
                    if retries < Config.MAX_RETRIES:
                        # FIXED: Exponential backoff with jitter and cap
                        import random
                        jitter = random.uniform(0.1, 1.0)
                        wait_time = min(Config.RETRY_DELAY * (2 ** retries) + jitter, 30)  # Cap at 30s
                        secure_logger.warning(f"Rate limit hit for model {model_name}, retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        return await self._make_api_call(model_name, payload, api_key, retries + 1)
                    else:
                        return False, "Rate limit exceeded. Please try again later."
                
                elif response.status >= 500:  # Server errors
                    if retries < Config.MAX_RETRIES:
                        # FIXED: Linear backoff with jitter
                        import random
                        jitter = random.uniform(0.1, 0.5)
                        wait_time = Config.RETRY_DELAY * (retries + 1) + jitter
                        secure_logger.warning(f"Server error {response.status} for model {model_name}, retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        return await self._make_api_call(model_name, payload, api_key, retries + 1)
                    else:
                        return False, f"Server error (HTTP {response.status}). Please try again later."
                
                elif response.status == 401:  # Unauthorized
                    error_text = await response.text()
                    safe_error_text = redact_sensitive_data(error_text)
                    secure_logger.error(f"Unauthorized access for model {model_name}: {safe_error_text}")
                    return False, "Invalid API key. Please check your Hugging Face API key."
                
                elif response.status == 403:  # Forbidden - CRITICAL FIX: Trigger actual fallback instead of just message
                    error_text = await response.text()
                    safe_error_text = redact_sensitive_data(error_text)
                    secure_logger.error(f"Access forbidden for model {model_name}: {safe_error_text}")
                    
                    # CRITICAL FIX: 403 should trigger fallback, not just return error message
                    fallback_msg = f"Model '{model_name}' is gated or requires special access (HTTP 403). "
                    if Config.is_free_tier():
                        fallback_msg += "Free-tier users should use tier-appropriate models. "
                    fallback_msg += "Triggering tier degradation to accessible alternatives."
                    
                    # Return special error that triggers fallback in _make_api_call_with_fallbacks
                    return False, f"TIER_DEGRADATION_REQUIRED: {fallback_msg}"
                
                elif response.status == 404:  # Model not found
                    error_text = await response.text()
                    safe_error_text = redact_sensitive_data(error_text)
                    secure_logger.error(f"Model not found: {model_name} - {safe_error_text}")
                    
                    # Enhanced 404 handling: Try fallback for chat/conversation models
                    if retries == 0 and endpoint_type == "chat":
                        secure_logger.info(f"404 for chat model {model_name}, retrying with standard inference")
                        return await self._make_api_call(model_name, payload, api_key, retries + 1, "inference")
                    
                    # CRITICAL FIX: Use tier-aware fallback message instead of hardcoded suggestions
                    fallback_msg = f"Model '{model_name}' not found or not accessible on HF Inference API. "
                    fallback_msg += f"Tier degradation recommended (current tier: {Config.HF_TIER})."
                    return False, fallback_msg
                
                elif response.status == 400:  # Bad request - often API format issues
                    error_text = await response.text()
                    safe_error_text = redact_sensitive_data(error_text)
                    secure_logger.error(f"Bad request for model {model_name}: {safe_error_text}")
                    
                    # Check if it's a "model not supported" error specific to inference API
                    if "not supported" in error_text.lower() or "inference" in error_text.lower():
                        fallback_msg = f"Model '{model_name}' not supported on HF Inference API. Try using 'Inference Providers' or switch to a compatible model."
                        return False, fallback_msg
                    
                    return False, f"Invalid request format for model '{model_name}'. This model may require a different API endpoint or format."
                
                else:
                    error_text = await response.text()
                    # SECURITY FIX: Always redact error text that could contain sensitive data
                    safe_error_text = redact_sensitive_data(error_text) if error_text else '[NO_ERROR_TEXT]'
                    secure_logger.error(f"API call failed with status {response.status}: {safe_error_text}")
                    # CRITICAL SECURITY FIX: Apply redaction to ALL error return messages
                    return False, f"API error (HTTP {response.status}): {safe_error_text}"
                    
        except asyncio.TimeoutError:
            secure_logger.error(f"Timeout calling model {model_name} after {Config.REQUEST_TIMEOUT}s")
            if retries < Config.MAX_RETRIES:
                secure_logger.info(f"Retrying {model_name} due to timeout (attempt {retries + 1})")
                # FIXED: Add jitter to timeout retries
                import random
                jitter = random.uniform(0.1, 0.5)
                wait_time = Config.RETRY_DELAY + jitter
                await asyncio.sleep(wait_time)
                return await self._make_api_call(model_name, payload, api_key, retries + 1)
            
            # CRITICAL FIX: Timeout suggests model may be overloaded - tier degradation recommended
            fallback_msg = f"Request timed out after multiple attempts. Model '{model_name}' may be overloaded. "
            fallback_msg += "Tier degradation to lighter/faster models recommended."
            return False, fallback_msg
        
        except aiohttp.ClientConnectorError as e:
            # SECURITY FIX: Redact exception details that could contain sensitive information
            safe_exception_msg = redact_sensitive_data(str(e))
            secure_logger.error(f"Connection error calling model {model_name}: {safe_exception_msg}")
            if retries < Config.MAX_RETRIES:
                secure_logger.info(f"Retrying {model_name} due to connection error (attempt {retries + 1})")
                # FIXED: Add jitter to connection error retries
                import random
                jitter = random.uniform(0.5, 1.0)
                wait_time = Config.RETRY_DELAY * 2 + jitter  # Longer wait for connection issues
                await asyncio.sleep(wait_time)
                return await self._make_api_call(model_name, payload, api_key, retries + 1)
            return False, "Network connection error. Please check your internet connection and try again."
        
        except aiohttp.ContentTypeError as e:
            # SECURITY FIX: Redact exception details that could contain sensitive information
            safe_exception_msg = redact_sensitive_data(str(e))
            secure_logger.error(f"Content type error calling model {model_name}: {safe_exception_msg}")
            return False, "Invalid response format from API. Please try again with a different model."
        
        except json.JSONDecodeError as e:
            # SECURITY FIX: Redact exception details that could contain sensitive information
            safe_exception_msg = redact_sensitive_data(str(e))
            secure_logger.error(f"JSON decode error calling model {model_name}: {safe_exception_msg}")
            return False, "Invalid JSON response from API. Please try again."
        
        except Exception as e:
            # Use secure logging to prevent token leakage in tracebacks
            # SECURITY FIX: Redact exception details that could contain sensitive information
            safe_exception_msg = redact_sensitive_data(str(e))
            secure_logger.error(f"Unexpected error calling model {model_name}: {safe_exception_msg}")
            return False, f"Unexpected error occurred. Please try again later."
    
    def _normalize_hf_parameters(self, raw_params: Dict, model_name: str, task_type: str) -> Dict:
        """
        CRITICAL FIX: Normalize parameters for specific HuggingFace models and tasks
        
        Args:
            raw_params (Dict): Raw generation parameters
            model_name (str): Model name for task-specific optimization
            task_type (str): HF task type
            
        Returns:
            Dict: Normalized parameters for HF Inference API
        """
        normalized = {}
        
        # Essential parameters that are widely supported
        if "max_new_tokens" in raw_params:
            normalized["max_new_tokens"] = min(raw_params["max_new_tokens"], 2048)  # Cap for stability
        
        if "temperature" in raw_params:
            normalized["temperature"] = max(0.01, min(raw_params["temperature"], 1.5))  # Safe range
            
        if "top_p" in raw_params:
            normalized["top_p"] = max(0.1, min(raw_params["top_p"], 1.0))
            
        if "top_k" in raw_params:
            normalized["top_k"] = max(1, min(raw_params["top_k"], 100))  # Reasonable range
            
        if "repetition_penalty" in raw_params:
            normalized["repetition_penalty"] = max(0.8, min(raw_params["repetition_penalty"], 1.3))
        
        # Task-specific parameter optimization
        if task_type == "text-generation":
            normalized["return_full_text"] = raw_params.get("return_full_text", False)
            normalized["do_sample"] = raw_params.get("do_sample", True)
            
        elif task_type == "text2text-generation":
            # T5-style models (BART, T5, etc.)
            normalized["do_sample"] = raw_params.get("do_sample", True)
            normalized["early_stopping"] = True
            
        elif task_type == "conversational":
            # DialoGPT-style models
            normalized["return_full_text"] = False
            normalized["do_sample"] = True
            normalized["pad_token_id"] = 50256  # Common for GPT models
            
        # Model-specific optimizations
        model_lower = model_name.lower()
        if "bart" in model_lower:
            # BART models work better with specific settings
            normalized["do_sample"] = True
            normalized["early_stopping"] = True
            normalized["num_beams"] = 1  # Faster sampling
        elif "phi-3" in model_lower:
            # Phi-3 models optimization
            normalized["do_sample"] = True
            normalized["temperature"] = max(normalized.get("temperature", 0.7), 0.3)
        elif "qwen" in model_lower:
            # Qwen models optimization
            normalized["do_sample"] = True
            normalized["top_p"] = min(normalized.get("top_p", 0.9), 0.95)
        
        return normalized

    def _get_hf_task_type(self, model_name: str, intent_type: str) -> str:
        """
        CRITICAL FIX: Determine correct HuggingFace task type based on model and intent
        
        Args:
            model_name (str): Model name
            intent_type (str): Intent type
            
        Returns:
            str: Correct HF task type
        """
        model_lower = model_name.lower()
        
        # Model-specific task type mapping
        if "bart" in model_lower:
            if "cnn" in model_lower or intent_type in ["summarization", "text"]:
                return "summarization"
            return "text2text-generation"
        elif "t5" in model_lower:
            return "text2text-generation"
        elif "dialogpt" in model_lower:
            return "conversational"
        elif "phi-3" in model_lower or "qwen" in model_lower:
            return "text-generation"
        elif "clip" in model_lower:
            return "zero-shot-image-classification"
        elif "blip" in model_lower:
            return "image-to-text"
        elif "detr" in model_lower:
            return "object-detection"
        elif "vilt" in model_lower:
            return "visual-question-answering"
        elif "roberta" in model_lower or "bert" in model_lower:
            return "text-classification"
        elif "whisper" in model_lower:
            return "automatic-speech-recognition"
        
        # Default fallback based on intent
        if intent_type in ["code", "coding"]:
            return "text-generation"
        elif intent_type in ["vision", "image"]:
            return "image-to-text"
        elif intent_type == "sentiment":
            return "text-classification"
        else:
            return "text-generation"

    def _format_chat_history(self, chat_history: List[Dict], new_prompt: str) -> str:
        """
        Format chat history into a coherent prompt for instruction models
        
        Args:
            chat_history (list): List of message dictionaries
            new_prompt (str): New user prompt
            
        Returns:
            str: Formatted prompt string
        """
        if not chat_history:
            return f"### Human: {new_prompt}\n### Assistant:"
        
        formatted_messages = []
        for msg in chat_history[-Config.MAX_CHAT_HISTORY:]:  # Limit history
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                formatted_messages.append(f"### Human: {content}")
            else:
                formatted_messages.append(f"### Assistant: {content}")
        
        # Add new prompt
        formatted_messages.append(f"### Human: {new_prompt}")
        formatted_messages.append("### Assistant:")
        
        return "\n".join(formatted_messages)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
    async def generate_text(self, prompt: str, api_key: Optional[str] = None, chat_history: Optional[List[Dict]] = None, model_override: Optional[str] = None, special_params: Optional[Dict] = None, intent_type: str = "text") -> Tuple[bool, str]:
        """
        CRITICAL FIX: Generate text using tier-aware fallback system for guaranteed functionality
        Phase 1 HF API Migration: Added provider system support with backward compatibility
        
        Args:
            prompt (str): User prompt
            api_key (Optional[str]): Hugging Face API key (optional)
            chat_history (list): Previous conversation history
            model_override (str): Optional model override
            special_params (dict): Special parameters for model
            intent_type (str): Type of task for fallback chain selection
            
        Returns:
            Tuple[bool, str]: (success, generated_text)
        """
        # GUARD: Validate prompt is not empty or whitespace-only
        if not prompt or not prompt.strip():
            # If we have chat history, synthesize a response from context
            if chat_history and len(chat_history) > 0:
                secure_logger.warning("🔸 Empty prompt received, synthesizing response from chat context")
                last_message = chat_history[-1].get('content', '') if chat_history else ''
                if last_message.strip():
                    return True, "I understand you'd like to continue our conversation. Could you please provide more details about what you'd like to discuss?"
                else:
                    return True, "I'm here to help! Please let me know what you'd like to talk about or ask me anything."
            else:
                secure_logger.warning("🔸 Empty prompt received with no context")
                return False, "Please provide a question or prompt for me to respond to."
        
        if not api_key:
            return False, "API key is required for text generation"
        
        chat_history = chat_history or []
        special_params = special_params or {}
        model_name = model_override or Config.DEFAULT_TEXT_MODEL
        
        # Format the prompt with chat history
        formatted_prompt = self._format_chat_history(chat_history, prompt)
        
        # CRITICAL FIX: Normalize parameters for HuggingFace Inference API compatibility
        parameters = self._normalize_hf_parameters({
            "max_new_tokens": special_params.get('max_new_tokens', 1000),
            "temperature": special_params.get('temperature', 0.7),
            "do_sample": True,
            "top_p": special_params.get('top_p', 0.9),
            "top_k": special_params.get('top_k', 50),
            "repetition_penalty": special_params.get('repetition_penalty', 1.05),
            "return_full_text": False
        }, model_name, "text-generation")
        
        # Phase 1 HF API Migration: Try provider system first, then fallback to legacy
        if self._use_provider_system and Config.HF_API_MODE in ['inference_providers', 'auto']:
            try:
                # Determine if this should be treated as chat (with history)
                is_chat_context = bool(chat_history)
                success, result = await self._call_provider(model_name, formatted_prompt, parameters, is_chat=is_chat_context, intent_type=intent_type)
                if success:
                    generated_text = result[0].get('generated_text', '').strip()
                    
                    # Advanced text cleanup for latest models
                    cleanup_patterns = ['### Assistant:', '### Human:', 'Assistant:', 'Human:']
                    for pattern in cleanup_patterns:
                        if generated_text.startswith(pattern):
                            generated_text = generated_text.replace(pattern, '').strip()
                    
                    # Limit response length
                    if len(generated_text) > Config.MAX_RESPONSE_LENGTH:
                        generated_text = generated_text[:Config.MAX_RESPONSE_LENGTH] + "..."
                    
                    secure_logger.info(f"✅ Provider text generation successful: {model_name.split('/')[-1]} ({len(generated_text)} chars)")
                    return True, generated_text
            except Exception as e:
                safe_error = redact_sensitive_data(str(e))
                secure_logger.warning(f"⚠️ Provider system failed, falling back to legacy API: {safe_error}")
        
        # Legacy API fallback (preserves all existing functionality)
        payload = {
            "inputs": formatted_prompt,
            "parameters": parameters,
            "options": {
                "wait_for_model": True,  # 2025: Always wait for model loading
                "use_gpu": True,  # 2025: Prefer GPU inference
            }
        }
        
        # CRITICAL FIX: Bypass complex tier system for guaranteed basic models
        if model_name in ["gpt2", "distilgpt2", "microsoft/DialoGPT-medium"]:
            secure_logger.info(f"🤖 Using direct API call for guaranteed basic model: {model_name}")
            success, result = await self._make_api_call(model_name, payload, api_key)
        else:
            secure_logger.info(f"🤖 Starting tier-aware text generation (tier: {Config.HF_TIER}, intent: {intent_type})")
            success, result = await self._make_api_call_with_fallbacks(model_name, payload, api_key, intent_type)
        
        if success and isinstance(result, list) and len(result) > 0:
            # CRITICAL FIX: Handle different response structures from HF API
            if isinstance(result[0], dict):
                generated_text = result[0].get('generated_text', '').strip()
            elif isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], dict):
                generated_text = result[0][0].get('generated_text', '').strip()
            else:
                # Fallback: convert to string if unexpected structure
                generated_text = str(result[0]).strip()
                secure_logger.warning(f"⚠️ Unexpected response structure: {type(result[0])}, using string fallback")
            
            # Advanced text cleanup for latest models
            cleanup_patterns = ['### Assistant:', '### Human:', 'Assistant:', 'Human:']
            for pattern in cleanup_patterns:
                if generated_text.startswith(pattern):
                    generated_text = generated_text.replace(pattern, '').strip()
            
            # Limit response length
            if len(generated_text) > Config.MAX_RESPONSE_LENGTH:
                generated_text = generated_text[:Config.MAX_RESPONSE_LENGTH] + "..."
            
            # CRITICAL FIX: Enhance mathematical responses with calculator integration
            if intent_type == "mathematical_reasoning" or intent_type == IntentType.MATHEMATICAL_REASONING:
                try:
                    enhanced_text = await self.math_enhancer.enhance_mathematical_response(prompt, generated_text)
                    generated_text = enhanced_text
                    secure_logger.info("✅ Enhanced mathematical response with calculator integration")
                except Exception as e:
                    secure_logger.warning(f"⚠️ Math enhancement failed: {str(e)}, using original response")
            
            secure_logger.info(f"✅ Tier-aware text generation successful")
            return True, generated_text
        
        # CRITICAL FIX: If tier-aware system fails, this indicates fundamental API issues
        error_message = result if isinstance(result, str) else "All tier-appropriate models failed."
        secure_logger.error(f"❌ Tier-aware generation failed: {error_message}")
        return False, f"Text generation failed after tier degradation: {error_message}"
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=15))
    async def generate_code(self, prompt: str, api_key: Optional[str] = None, language: str = "python", special_params: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Generate code using latest StarCoder2-15B and enhanced models
        Phase 1 HF API Migration: Added provider system support with backward compatibility
        
        Args:
            prompt (str): Code generation prompt
            api_key (Optional[str]): Hugging Face API key (optional)
            language (str): Programming language
            special_params (dict): Special parameters for model
            
        Returns:
            Tuple[bool, str]: (success, generated_code)
        """
        if not api_key:
            return False, "API key is required for code generation"
        
        special_params = special_params or {}
        model_name = Config.DEFAULT_CODE_MODEL  # StarCoder2-15B
        
        # Enhanced prompt formatting for StarCoder2
        if language.lower() in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c++']:
            enhanced_prompt = f"""<fim_prefix># Task: {prompt}
# Language: {language}
# Instructions: Generate clean, well-documented {language} code

<fim_suffix>

<fim_middle>"""
        else:
            enhanced_prompt = f"# Generate {language} code for: {prompt}\n\n"
        
        # Optimized parameters for StarCoder2
        parameters = {
            "max_new_tokens": special_params.get('max_new_tokens', 1200),
            "temperature": special_params.get('temperature', 0.2),
            "do_sample": True,
            "top_p": special_params.get('top_p', 0.95),
            "repetition_penalty": special_params.get('repetition_penalty', 1.05),
            "return_full_text": False,
            "stop": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]  # Stop tokens for StarCoder
        }
        
        # Phase 1 HF API Migration: Try provider system first, then fallback to legacy
        if self._use_provider_system and Config.HF_API_MODE in ['inference_providers', 'auto']:
            try:
                success, result = await self._call_provider(model_name, enhanced_prompt, parameters, is_chat=False, intent_type="code")
                if success:
                    generated_code = result[0].get('generated_text', '').strip()
                    
                    # Clean up StarCoder2 specific tokens
                    cleanup_tokens = ['<fim_prefix>', '<fim_suffix>', '<fim_middle>']
                    for token in cleanup_tokens:
                        generated_code = generated_code.replace(token, '')
                    
                    generated_code = generated_code.strip()
                    
                    # Format code response with proper markdown
                    if not generated_code.startswith('```'):
                        generated_code = f"```{language}\n{generated_code}\n```"
                    
                    secure_logger.info(f"✅ Provider code generation successful: {model_name.split('/')[-1]} ({len(generated_code)} chars)")
                    return True, generated_code
            except Exception as e:
                safe_error = redact_sensitive_data(str(e))
                secure_logger.warning(f"⚠️ Provider system failed for code generation, falling back to legacy API: {safe_error}")
        
        # Legacy API fallback (preserves all existing functionality)
        
        # Optimized parameters for StarCoder2
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "max_new_tokens": special_params.get('max_new_tokens', 1200),
                "temperature": special_params.get('temperature', 0.2),
                "do_sample": True,
                "top_p": special_params.get('top_p', 0.95),
                "repetition_penalty": special_params.get('repetition_penalty', 1.05),
                "return_full_text": False,
                "stop": ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]  # Stop tokens for StarCoder
            }
        }
        
        # CRITICAL FIX: Bypass complex tier system for guaranteed basic models  
        if model_name in ["gpt2", "distilgpt2", "microsoft/DialoGPT-medium"]:
            secure_logger.info(f"🤖 Using direct API call for guaranteed basic coding model: {model_name}")
            success, result = await self._make_api_call(model_name, payload, api_key)
        else:
            success, result = await self._make_api_call_with_fallbacks(model_name, payload, api_key, "code_generation")
        
        if success and isinstance(result, list) and len(result) > 0:
            # CRITICAL FIX: Handle different response structures from HF API
            if isinstance(result[0], dict):
                generated_code = result[0].get('generated_text', '').strip()
            elif isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], dict):
                generated_code = result[0][0].get('generated_text', '').strip() 
            else:
                # Fallback: convert to string if unexpected structure
                generated_code = str(result[0]).strip()
                secure_logger.warning(f"⚠️ Unexpected code response structure: {type(result[0])}, using string fallback")
            
            # Clean up StarCoder2 specific tokens
            cleanup_tokens = ['<fim_prefix>', '<fim_suffix>', '<fim_middle>']
            for token in cleanup_tokens:
                generated_code = generated_code.replace(token, '')
            
            generated_code = generated_code.strip()
            
            # Format code response with proper markdown
            if not generated_code.startswith('```'):
                generated_code = f"```{language}\n{generated_code}\n```"
            
            return True, generated_code
        
        # Try fallback models with different strategies
        fallback_models = [
            (Config.FALLBACK_CODE_MODEL, f"# Generate {language} code for: {prompt}\n\n"),
            (Config.ADVANCED_CODE_MODEL, f"Write {language} code to {prompt}"),
            (Config.ADVANCED_TEXT_MODEL, f"Please write {language} code that does the following: {prompt}")
        ]
        
        for fallback_model, fallback_prompt in fallback_models:
            if model_name != fallback_model:
                secure_logger.info(f"Trying fallback code model {fallback_model}")
                fallback_payload = {
                    "inputs": fallback_prompt,
                    "parameters": {
                        "max_new_tokens": 800,
                        "temperature": 0.3,
                        "do_sample": True,
                        "top_p": 0.95,
                        "return_full_text": False
                    }
                }
                
                fallback_success, fallback_result = await self._make_api_call(fallback_model, fallback_payload, api_key)
                if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
                    # CRITICAL FIX: Handle different response structures from HF API
                    if isinstance(fallback_result[0], dict):
                        generated_code = fallback_result[0].get('generated_text', '').strip()
                    elif isinstance(fallback_result[0], list) and len(fallback_result[0]) > 0 and isinstance(fallback_result[0][0], dict):
                        generated_code = fallback_result[0][0].get('generated_text', '').strip()
                    else:
                        generated_code = str(fallback_result[0]).strip()
                    if not generated_code.startswith('```'):
                        generated_code = f"```{language}\n{generated_code}\n```"
                    secure_logger.info(f"Successfully used fallback code model {fallback_model}")
                    return True, generated_code
        
        error_message = result if isinstance(result, str) else "Failed to generate code with available models."
        return False, f"Code generation failed: {error_message}"
    
    async def generate_image(self, prompt: str, api_key: Optional[str] = None, special_params: Optional[Dict] = None, **kwargs) -> Tuple[bool, Dict]:
        """
        Enhanced image generation with intelligent free-tier handling
        Provides detailed artistic descriptions when image generation models aren't available
        
        Args:
            prompt (str): Image generation prompt
            api_key (str): Hugging Face API key
            special_params (dict): Special parameters for model
            **kwargs: Additional parameters for backward compatibility (style, model, etc.)
            
        Returns:
            Tuple[bool, Dict]: (success, {type: 'description'|'image', content: description_text|image_bytes, ...})
        """
        special_params = special_params or {}
        
        # Handle backward compatibility for old-style calls
        if kwargs:
            secure_logger.debug(f"Received additional kwargs for generate_image: {list(kwargs.keys())}")
            # Merge kwargs into special_params for compatibility
            for key, value in kwargs.items():
                if key == 'style':
                    special_params['style'] = value
                elif key == 'model':
                    special_params['preferred_model'] = value
                elif key in ['width', 'height', 'guidance_scale', 'num_inference_steps']:
                    special_params[key] = value
                else:
                    secure_logger.debug(f"Ignoring unknown parameter: {key}={value}")
        
        # Check if we should attempt actual image generation first (in case models become available)
        attempt_image_generation = special_params.get('force_image_generation', False)
        
        if attempt_image_generation:
            # Try actual image generation as fallback (in case premium models become available)
            success, result = await self._attempt_image_generation(prompt, api_key, special_params)
            if success:
                return True, {
                    'type': 'image',
                    'content': result,
                    'format': 'bytes'
                }
        
        # Primary path: Generate enhanced text description (free tier solution)
        secure_logger.info(f"Generating enhanced artistic description for image prompt")
        return await self._generate_enhanced_image_description(prompt, api_key, special_params)
    
    async def _attempt_image_generation(self, prompt: str, api_key: Optional[str], special_params: Dict) -> Tuple[bool, bytes]:
        """
        Attempt actual image generation (fallback for when premium models might be available)
        
        Returns:
            Tuple[bool, bytes]: (success, image_bytes)
        """
        # Smart serverless-aware model selection (prefer stable models)
        is_complex_request = len(prompt) > 100 or any(word in prompt.lower() for word in ['detailed', 'complex', 'professional', 'artistic', 'photorealistic'])
        
        # Try a few known working image generation models
        test_models = [
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4", 
            "stabilityai/stable-diffusion-2-1"
        ]
        
        for model_name in test_models:
            try:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "guidance_scale": special_params.get('guidance_scale', 7.5),
                        "num_inference_steps": special_params.get('num_inference_steps', 20),
                        "width": special_params.get('width', 512),
                        "height": special_params.get('height', 512)
                    }
                }
                
                success, result = await self._make_api_call(model_name, payload, api_key, endpoint_type="text2img")
                
                if success and isinstance(result, bytes) and len(result) > 0:
                    # Validate image data
                    if result[:4] in [b'\x89PNG', b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1', b'\xff\xd8\xff\xdb']:
                        secure_logger.info(f"Successfully generated image using {model_name} ({len(result)} bytes)")
                        return True, result
                    else:
                        secure_logger.debug(f"Invalid image data received from {model_name} (first 4 bytes: {result[:4]})")
                        continue
                elif success and isinstance(result, dict) and 'error' in result:
                    secure_logger.debug(f"Image generation error from {model_name}: {result.get('error', 'Unknown error')}")
                    continue
                    
            except Exception as e:
                secure_logger.debug(f"Image model {model_name} not available: {str(e)}")
                continue
        
        return False, b""
    
    async def _generate_enhanced_image_description(self, prompt: str, api_key: Optional[str], special_params: Dict) -> Tuple[bool, Dict]:
        """
        Generate extremely detailed artistic descriptions with technical guidance
        
        Returns:
            Tuple[bool, Dict]: (success, enhanced_description_data)
        """
        # Determine complexity and style for enhanced description
        is_complex_request = len(prompt) > 100 or any(word in prompt.lower() for word in 
            ['detailed', 'complex', 'professional', 'artistic', 'photorealistic', 'cinematic', 'masterpiece'])
        
        # Select best text model for detailed descriptions
        if is_complex_request:
            model_name = Config.FLAGSHIP_IMAGE_MODEL  # Use advanced model for complex descriptions
            max_tokens = 1500
        else:
            model_name = Config.DEFAULT_IMAGE_MODEL   # Use standard model for simple descriptions
            max_tokens = 1000
        
        # Create comprehensive description prompt
        description_prompt = self._build_detailed_description_prompt(prompt, special_params)
        
        payload = {
            "inputs": description_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,  # Slightly creative for artistic descriptions
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False,
                "repetition_penalty": 1.1
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            description_text = result[0].get('generated_text', '').strip()
            
            # Enhance and structure the description
            enhanced_result = self._structure_image_description(description_text, prompt, special_params)
            
            secure_logger.info(f"Successfully generated enhanced image description ({len(description_text)} chars)")
            return True, enhanced_result
        
        # Fallback with simpler model
        if model_name != Config.FALLBACK_IMAGE_MODEL:
            fallback_success, fallback_result = await self._make_api_call(
                Config.FALLBACK_IMAGE_MODEL, payload, api_key
            )
            
            if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
                description_text = fallback_result[0].get('generated_text', '').strip()
                enhanced_result = self._structure_image_description(description_text, prompt, special_params)
                return True, enhanced_result
        
        # Ultimate fallback: create structured description manually
        fallback_description = self._create_fallback_description(prompt, special_params)
        return True, fallback_description
    
    async def analyze_sentiment(self, text: str, api_key: Optional[str] = None, use_emotion_detection: bool = False) -> Tuple[bool, Dict]:
        """
        Analyze sentiment with advanced emotion detection
        
        Args:
            text (str): Text to analyze
            api_key (Optional[str]): Hugging Face API key (optional)
            use_emotion_detection (bool): Use advanced emotion model
            
        Returns:
            Tuple[bool, Dict]: (success, sentiment_data)
        """
        if not api_key:
            return False, {"error": "API key is required for sentiment analysis"}
        
        # Choose model based on requirements
        model_name = Config.EMOTION_MODEL if use_emotion_detection else Config.DEFAULT_SENTIMENT_MODEL
        
        payload = {"inputs": text}
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list):
            # Enhanced result formatting for emotion detection
            if use_emotion_detection and result:
                # Process emotion results (go_emotions has 28 emotion categories)
                emotion_data = result[0] if result else {}
                return True, {
                    'emotion_type': 'advanced',
                    'emotions': result,  # All emotion scores
                    'primary_emotion': emotion_data,
                    'model_used': 'go_emotions'
                }
            else:
                # Standard sentiment analysis
                return True, {
                    'emotion_type': 'sentiment',
                    'result': result[0] if result else {},
                    'model_used': 'roberta_sentiment'
                }
        
        # Try fallback sentiment model
        if not success and model_name != Config.FALLBACK_SENTIMENT_MODEL:
            secure_logger.info(f"Trying fallback sentiment model {Config.FALLBACK_SENTIMENT_MODEL}")
            fallback_success, fallback_result = await self._make_api_call(Config.FALLBACK_SENTIMENT_MODEL, payload, api_key)
            
            if fallback_success and isinstance(fallback_result, list):
                # CRITICAL FIX: Handle different response structures from HF API
                sentiment_data = {}
                if fallback_result and len(fallback_result) > 0:
                    if isinstance(fallback_result[0], dict):
                        sentiment_data = fallback_result[0]
                    elif isinstance(fallback_result[0], list) and len(fallback_result[0]) > 0:
                        sentiment_data = fallback_result[0][0] if isinstance(fallback_result[0][0], dict) else {}
                        
                return True, {
                    'emotion_type': 'sentiment',
                    'result': sentiment_data,
                    'model_used': 'distilbert_sentiment'
                }
        
        error_message = result if isinstance(result, str) else "Failed to analyze sentiment"
        return False, {'error': error_message}
    
    def _build_detailed_description_prompt(self, original_prompt: str, special_params: Dict) -> str:
        """
        Build a comprehensive prompt for generating detailed artistic descriptions
        
        Args:
            original_prompt (str): User's original image request
            special_params (Dict): Additional parameters for customization
            
        Returns:
            str: Enhanced prompt for description generation
        """
        style_preferences = special_params.get('style', 'photorealistic')
        detail_level = special_params.get('detail_level', 'high')
        
        enhanced_prompt = f"""Create an extremely detailed, professional artistic description for the following image concept: "{original_prompt}"

Provide a comprehensive description that includes:

🎨 **VISUAL COMPOSITION:**
- Detailed subject description and positioning
- Background elements and environment
- Foreground, middle ground, and background layers
- Overall composition and framing

💡 **LIGHTING & ATMOSPHERE:**
- Light sources and direction (natural/artificial)
- Shadows, highlights, and contrast
- Mood and atmospheric effects
- Color temperature (warm/cool tones)

🌈 **COLOR PALETTE & STYLE:**
- Primary and secondary colors
- Color harmony and saturation levels
- Artistic style ({style_preferences} preferred)
- Texture and material descriptions

📸 **TECHNICAL DETAILS:**
- Camera angle and perspective
- Depth of field and focus points
- Artistic techniques and rendering style
- Quality and resolution suggestions

✨ **ARTISTIC ELEMENTS:**
- Emotional tone and feeling
- Symbolic elements or meaning
- Art movement or style references
- Professional photography/art terms

Format the response as a structured, detailed description that could guide an artist or photographer in creating this exact image. Use professional artistic terminology and be specific about every visual element.

Description:"""
        
        return enhanced_prompt
    
    def _structure_image_description(self, raw_description: str, original_prompt: str, special_params: Dict) -> Dict:
        """
        Structure the AI-generated description into organized sections
        
        Args:
            raw_description (str): Raw AI-generated description
            original_prompt (str): Original user prompt
            special_params (Dict): Additional parameters
            
        Returns:
            Dict: Structured description data
        """
        # Clean and enhance the description
        cleaned_description = raw_description.strip()
        
        # Extract or enhance key elements
        alternative_services = self._get_alternative_services(original_prompt)
        optimized_prompts = self._generate_optimized_prompts(original_prompt, special_params)
        creation_steps = self._generate_creation_steps(original_prompt, cleaned_description)
        
        return {
            'type': 'enhanced_description',
            'original_prompt': original_prompt,
            'detailed_description': cleaned_description,
            'alternative_services': alternative_services,
            'optimized_prompts': optimized_prompts,
            'creation_steps': creation_steps,
            'technical_specs': self._extract_technical_specs(cleaned_description),
            'artistic_style': special_params.get('style', 'photorealistic'),
            'confidence_score': 0.95,  # High confidence for enhanced descriptions
            'estimated_creation_time': self._estimate_creation_time(original_prompt)
        }
    
    def _create_fallback_description(self, prompt: str, special_params: Dict) -> Dict:
        """
        Create a structured fallback description when AI models fail
        
        Args:
            prompt (str): Original user prompt
            special_params (Dict): Additional parameters
            
        Returns:
            Dict: Fallback description data
        """
        # Analyze the prompt to extract key elements
        subjects = self._extract_subjects(prompt)
        style = special_params.get('style', 'photorealistic')
        
        fallback_description = f"""🎨 **Detailed Artistic Vision for: "{prompt}"**

**VISUAL COMPOSITION:**
• Main Subject(s): {', '.join(subjects) if subjects else 'Central focus as described'}
• Setting: {self._infer_setting(prompt)}
• Composition: Balanced framing with clear focal points
• Layout: Rule of thirds for optimal visual appeal

**LIGHTING & ATMOSPHERE:**
• Lighting: {self._infer_lighting(prompt)}
• Mood: {self._infer_mood(prompt)}
• Atmosphere: Professional, high-quality rendering
• Contrast: Well-balanced highlights and shadows

**COLOR & STYLE:**
• Style: {style.title()} rendering
• Color Palette: {self._suggest_colors(prompt)}
• Saturation: Rich, vibrant colors without oversaturation
• Texture: Detailed surface materials and finishes

**TECHNICAL SPECIFICATIONS:**
• Resolution: High definition (1024x1024 or higher)
• Quality: Professional-grade artistic rendering
• Perspective: {self._suggest_perspective(prompt)}
• Depth: Multi-layered composition with clear depth of field

**ARTISTIC ELEMENTS:**
• Emotional Impact: {self._assess_emotional_impact(prompt)}
• Professional Techniques: Advanced rendering and post-processing
• Style References: Contemporary digital art standards
• Attention to Detail: Meticulous craftsmanship in every element"""
        
        alternative_services = self._get_alternative_services(prompt)
        optimized_prompts = self._generate_optimized_prompts(prompt, special_params)
        creation_steps = self._generate_creation_steps(prompt, fallback_description)
        
        return {
            'type': 'enhanced_description',
            'original_prompt': prompt,
            'detailed_description': fallback_description,
            'alternative_services': alternative_services,
            'optimized_prompts': optimized_prompts,
            'creation_steps': creation_steps,
            'technical_specs': self._extract_technical_specs(fallback_description),
            'artistic_style': style,
            'confidence_score': 0.85,  # Good confidence for fallback
            'estimated_creation_time': self._estimate_creation_time(prompt)
        }
    
    def _get_alternative_services(self, prompt: str) -> List[Dict]:
        """
        Get list of alternative free image generation services
        
        Returns:
            List[Dict]: Alternative services with optimized prompts
        """
        return [
            {
                'name': 'Bing Image Creator (DALL-E 3)',
                'url': 'https://www.bing.com/images/create',
                'description': 'Microsoft\'s free DALL-E 3 implementation - 15 fast generations per day',
                'optimized_prompt': f'Create a high-quality image of: {prompt}. Style: photorealistic, highly detailed, professional quality.',
                'pros': ['High quality', 'DALL-E 3 technology', 'Free daily credits'],
                'tips': 'Use specific descriptive language and mention desired style explicitly'
            },
            {
                'name': 'Leonardo.ai',
                'url': 'https://leonardo.ai',
                'description': 'Professional AI art generator - Free tier with daily credits',
                'optimized_prompt': f'{prompt}, masterpiece, best quality, highly detailed, professional digital art',
                'pros': ['Multiple art styles', 'High-res generation', 'Professional features'],
                'tips': 'Select appropriate model (Absolute Reality for photorealism, Leonardo Creative for artistic)'
            },
            {
                'name': 'Playground AI',
                'url': 'https://playground.com',
                'description': 'User-friendly AI image generator - Free daily generations',
                'optimized_prompt': f'Professional quality image of: {prompt}. High detail, excellent composition, perfect lighting.',
                'pros': ['Easy to use', 'Multiple style options', 'Good free tier'],
                'tips': 'Experiment with different models like Stable Diffusion XL or PlaygroundV2'
            },
            {
                'name': 'Stable Diffusion Online',
                'url': 'https://stablediffusionweb.com',
                'description': 'Free Stable Diffusion access - No registration required',
                'optimized_prompt': f'{prompt}, high quality, detailed, 8k resolution, professional photography',
                'pros': ['No signup required', 'Open source', 'Multiple models'],
                'tips': 'Use negative prompts to avoid unwanted elements (e.g., "blurry, low quality, distorted")'
            },
            {
                'name': 'Ideogram',
                'url': 'https://ideogram.ai',
                'description': 'Advanced AI image generator with excellent text rendering',
                'optimized_prompt': f'Create: {prompt}. Style: realistic, high detail, professional quality, perfect composition.',
                'pros': ['Excellent text in images', 'High quality results', 'Free tier available'],
                'tips': 'Great for images with text elements like logos, signs, or typography'
            }
        ]
    
    def _generate_optimized_prompts(self, original_prompt: str, special_params: Dict) -> Dict[str, str]:
        """
        Generate optimized prompts for different platforms
        
        Returns:
            Dict[str, str]: Platform-specific optimized prompts
        """
        base_prompt = original_prompt.strip()
        style = special_params.get('style', 'photorealistic')
        
        return {
            'dalle3_optimized': f'Create a {style} image of: {base_prompt}. Professional quality, highly detailed, perfect composition, excellent lighting.',
            'stable_diffusion': f'{base_prompt}, {style}, masterpiece, best quality, highly detailed, 8k resolution, professional photography, perfect lighting, sharp focus',
            'midjourney_style': f'{base_prompt} --style {style} --quality 2 --ar 1:1 --stylize 500',
            'leonardo_ai': f'{base_prompt}, {style} style, professional digital art, high resolution, masterpiece quality, perfect composition',
            'playground_ai': f'High-quality {style} image of: {base_prompt}. Excellent detail, professional composition, perfect lighting and colors.'
        }
    
    def _generate_creation_steps(self, prompt: str, description: str) -> List[str]:
        """
        Generate step-by-step manual creation instructions
        
        Returns:
            List[str]: Step-by-step creation guide
        """
        return [
            '🎯 **Plan Your Composition**: Study the detailed description above and sketch the basic layout',
            '🖼️ **Choose Your Medium**: Digital art software (Photoshop, Procreate) or traditional media',
            '🎨 **Establish Color Palette**: Select primary and secondary colors based on the mood description',
            '💡 **Set Up Lighting**: Plan light sources and shadow placement as described',
            '📐 **Create Base Shapes**: Block in major forms and establish perspective',
            '🔍 **Add Details Gradually**: Work from general shapes to specific details',
            '✨ **Apply Finishing Touches**: Add highlights, refine edges, and adjust overall contrast',
            '📊 **Review and Refine**: Compare with description and make final adjustments'
        ]
    
    def _extract_technical_specs(self, description: str) -> Dict[str, str]:
        """
        Extract or suggest technical specifications from description
        
        Returns:
            Dict[str, str]: Technical specifications
        """
        return {
            'recommended_resolution': '1024x1024 or higher',
            'aspect_ratio': '1:1 (square) or 16:9 (landscape)',
            'color_space': 'sRGB for digital, Adobe RGB for print',
            'file_format': 'PNG for digital art, TIFF for professional printing',
            'dpi': '300 DPI for print, 72 DPI for web/digital display',
            'bit_depth': '8-bit for standard use, 16-bit for professional editing'
        }
    
    def _estimate_creation_time(self, prompt: str) -> str:
        """
        Estimate creation time based on complexity
        
        Returns:
            str: Estimated time ranges
        """
        complexity_indicators = ['detailed', 'complex', 'intricate', 'professional', 'multiple', 'elaborate']
        is_complex = any(word in prompt.lower() for word in complexity_indicators)
        
        if is_complex:
            return 'Professional: 2-6 hours, Beginner: 8-15 hours'
        else:
            return 'Professional: 30 minutes - 2 hours, Beginner: 2-5 hours'
    
    # Helper methods for fallback description generation
    def _extract_subjects(self, prompt: str) -> List[str]:
        """Extract main subjects from prompt"""
        # Simple keyword extraction - can be enhanced with NLP
        common_subjects = ['person', 'people', 'man', 'woman', 'child', 'animal', 'cat', 'dog', 
                          'landscape', 'building', 'car', 'tree', 'flower', 'sunset', 'mountain']
        found_subjects = [subject for subject in common_subjects if subject in prompt.lower()]
        return found_subjects if found_subjects else ['main subject']
    
    def _infer_setting(self, prompt: str) -> str:
        """Infer setting from prompt"""
        if any(word in prompt.lower() for word in ['indoor', 'inside', 'room', 'house']):
            return 'Indoor environment with controlled lighting'
        elif any(word in prompt.lower() for word in ['outdoor', 'outside', 'nature', 'landscape']):
            return 'Outdoor natural environment'
        elif any(word in prompt.lower() for word in ['studio', 'professional']):
            return 'Professional studio setting'
        else:
            return 'Contextually appropriate environment'
    
    def _infer_lighting(self, prompt: str) -> str:
        """Infer lighting style from prompt"""
        if any(word in prompt.lower() for word in ['sunset', 'golden', 'warm']):
            return 'Warm golden hour lighting'
        elif any(word in prompt.lower() for word in ['dramatic', 'moody', 'dark']):
            return 'Dramatic low-key lighting with strong contrast'
        elif any(word in prompt.lower() for word in ['bright', 'cheerful', 'sunny']):
            return 'Bright, even lighting with soft shadows'
        else:
            return 'Professional three-point lighting setup'
    
    def _infer_mood(self, prompt: str) -> str:
        """Infer mood from prompt"""
        if any(word in prompt.lower() for word in ['happy', 'joyful', 'cheerful', 'bright']):
            return 'Uplifting and positive'
        elif any(word in prompt.lower() for word in ['dramatic', 'intense', 'powerful']):
            return 'Dramatic and impactful'
        elif any(word in prompt.lower() for word in ['peaceful', 'calm', 'serene']):
            return 'Peaceful and tranquil'
        else:
            return 'Professional and polished'
    
    def _suggest_colors(self, prompt: str) -> str:
        """Suggest color palette based on prompt"""
        if any(word in prompt.lower() for word in ['sunset', 'warm', 'golden']):
            return 'Warm oranges, golds, and soft reds'
        elif any(word in prompt.lower() for word in ['ocean', 'water', 'blue', 'cool']):
            return 'Cool blues, teals, and aqua tones'
        elif any(word in prompt.lower() for word in ['nature', 'forest', 'green']):
            return 'Natural greens with earth tones'
        else:
            return 'Balanced palette appropriate to subject matter'
    
    def _suggest_perspective(self, prompt: str) -> str:
        """Suggest camera perspective"""
        if any(word in prompt.lower() for word in ['portrait', 'person', 'face']):
            return 'Eye-level portrait perspective'
        elif any(word in prompt.lower() for word in ['landscape', 'wide', 'panoramic']):
            return 'Wide-angle landscape perspective'
        elif any(word in prompt.lower() for word in ['close', 'detail', 'macro']):
            return 'Close-up detailed perspective'
        else:
            return 'Standard viewing angle with good depth'
    
    def _assess_emotional_impact(self, prompt: str) -> str:
        """Assess intended emotional impact"""
        if any(word in prompt.lower() for word in ['powerful', 'strong', 'bold']):
            return 'Strong emotional resonance'
        elif any(word in prompt.lower() for word in ['gentle', 'soft', 'subtle']):
            return 'Subtle emotional connection'
        else:
            return 'Engaging and memorable visual experience'
    
    async def analyze_pdf(self, pdf_text: str, pdf_metadata: dict, api_key: str, analysis_type: str = "comprehensive") -> Tuple[bool, Dict]:
        """
        Analyze PDF content using AI models
        
        Args:
            pdf_text (str): Extracted text from PDF
            pdf_metadata (dict): PDF metadata (pages, title, author, etc.)
            api_key (str): Hugging Face API key
            analysis_type (str): Type of analysis ("comprehensive", "summary", "key_points", "tables")
            
        Returns:
            Tuple[bool, Dict]: (success, analysis_data)
        """
        if not pdf_text.strip():
            return False, {'error': 'No text content found in PDF'}
        
        # Choose appropriate model based on content length and complexity
        content_length = len(pdf_text)
        if content_length > 50000:  # Long documents
            model_name = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-72B for complex analysis
            max_tokens = min(2000, Config.QWEN_MAX_TOKENS)
        elif content_length > 10000:  # Medium documents
            model_name = Config.ADVANCED_TEXT_MODEL  # Qwen2.5-7B
            max_tokens = 1500
        else:  # Short documents
            model_name = Config.FALLBACK_TEXT_MODEL  # Llama-3.2-3B
            max_tokens = 1000
        
        # Create analysis prompt based on type
        if analysis_type == "summary":
            analysis_prompt = f"""📄 **PDF Summary Task**
        
Please provide a comprehensive summary of this PDF document:

**Metadata:**
- Pages: {pdf_metadata.get('pages', 'Unknown')}
- Title: {pdf_metadata.get('title', 'Not provided')}
- Author: {pdf_metadata.get('author', 'Not provided')}

**Content:**
{pdf_text[:int(os.getenv('MAX_PDF_CONTENT_LENGTH', '8000'))]}{('...' if len(pdf_text) > int(os.getenv('MAX_PDF_CONTENT_LENGTH', '8000')) else '')}

Please provide:
1. Executive Summary (2-3 sentences)
2. Key Points (bullet format)
3. Main Topics Covered
4. Important Details or Findings
5. Conclusion/Recommendations (if applicable)

Keep the summary concise but comprehensive."""

        elif analysis_type == "key_points":
            analysis_prompt = f"""📝 **PDF Key Points Extraction**
        
Extract the most important points from this PDF document:

**Document Info:** {pdf_metadata.get('pages', 'Unknown')} pages
**Content:**
{pdf_text[:int(os.getenv('MAX_PDF_EXTRACT_LENGTH', '10000'))]}{('...' if len(pdf_text) > int(os.getenv('MAX_PDF_EXTRACT_LENGTH', '10000')) else '')}

Please extract:
- Main arguments or thesis
- Supporting evidence
- Key statistics or data
- Important conclusions
- Action items or recommendations

Format as clear, numbered points."""

        elif analysis_type == "tables":
            analysis_prompt = f"""📊 **PDF Table and Data Analysis**
        
Analyze any tables, charts, or structured data in this PDF:

**Content:**
{pdf_text[:int(os.getenv('MAX_PDF_CONTENT_LENGTH', '8000'))]}{('...' if len(pdf_text) > int(os.getenv('MAX_PDF_CONTENT_LENGTH', '8000')) else '')}

Please identify and explain:
- Tables and their contents
- Charts or graphs mentioned
- Statistical data
- Structured information
- Data insights and patterns

If no tables are found, analyze the numerical/structured content."""

        else:  # comprehensive
            analysis_prompt = f"""🔍 **Comprehensive PDF Analysis**
        
Provide a thorough analysis of this PDF document:

**Metadata:**
- Pages: {pdf_metadata.get('pages', 'Unknown')}
- Title: {pdf_metadata.get('title', 'Not provided')}
- Author: {pdf_metadata.get('author', 'Not provided')}

**Content:**
{pdf_text[:int(os.getenv('MAX_PDF_CONTENT_LENGTH', '8000'))]}{('...' if len(pdf_text) > int(os.getenv('MAX_PDF_CONTENT_LENGTH', '8000')) else '')}

Please provide:
1. **Document Overview** - What type of document is this?
2. **Main Content Analysis** - Key themes, arguments, findings
3. **Structure Analysis** - How is the document organized?
4. **Important Details** - Critical information, data, statistics
5. **Quality Assessment** - Clarity, completeness, reliability
6. **Practical Value** - How can this information be used?

Be thorough but concise."""

        # Prepare payload for text generation
        payload = {
            "inputs": analysis_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for factual analysis
                "do_sample": True,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "return_full_text": False
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            analysis_text = result[0].get('generated_text', '').strip()
            
            return True, {
                'analysis_type': analysis_type,
                'analysis': analysis_text,
                'document_info': pdf_metadata,
                'content_length': content_length,
                'model_used': model_name.split('/')[-1] if '/' in model_name else model_name
            }
        
        # Try fallback with simpler prompt
        simple_prompt = f"Analyze this PDF content:\n\n{pdf_text[:5000]}\n\nProvide key insights and summary."
        simple_payload = {
            "inputs": simple_prompt,
            "parameters": {
                "max_new_tokens": 800,
                "temperature": 0.4,
                "return_full_text": False
            }
        }
        
        fallback_success, fallback_result = await self._make_api_call(Config.FALLBACK_TEXT_MODEL, simple_payload, api_key)
        if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
            analysis_text = fallback_result[0].get('generated_text', '').strip()
            return True, {
                'analysis_type': f"{analysis_type}_simplified",
                'analysis': analysis_text,
                'document_info': pdf_metadata,
                'model_used': 'llama_fallback'
            }
        
        error_message = result if isinstance(result, str) else "Failed to analyze PDF content"
        return False, {'error': error_message}
    
    async def analyze_zip_contents(self, file_contents: List[Dict], api_key: str, analysis_depth: str = "overview") -> Tuple[bool, Dict]:
        """
        Analyze contents of a ZIP archive using AI models
        
        Args:
            file_contents (List[Dict]): List of file info with 'name', 'size', 'type', 'content' (if text)
            api_key (str): Hugging Face API key  
            analysis_depth (str): Level of analysis ("overview", "detailed", "code_focus")
            
        Returns:
            Tuple[bool, Dict]: (success, analysis_data)
        """
        if not file_contents:
            return False, {'error': 'No files found in ZIP archive'}
        
        # Prepare content summary
        total_files = len(file_contents)
        total_size = sum(f.get('size', 0) for f in file_contents)
        file_types = {}
        text_files = []
        
        for file_info in file_contents:
            file_type = file_info.get('type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Include text content for analysis (limit to prevent overload)
            if file_info.get('content') and len(text_files) < 20:
                text_files.append(file_info)
        
        # Create analysis prompt based on depth
        if analysis_depth == "overview":
            analysis_prompt = f"""📦 **ZIP Archive Overview Analysis**

Analyze this ZIP archive structure and provide insights:

**Archive Statistics:**
- Total Files: {total_files}
- Total Size: {total_size:,} bytes ({total_size/1024:.1f} KB)
- File Types: {', '.join([f'{count} {ftype}' for ftype, count in file_types.items()])}

**File Listing:**
{chr(10).join([f"- {f['name']} ({f.get('size', 0)} bytes)" for f in file_contents[:50]])}
{'... (and more)' if total_files > 50 else ''}

**Sample Content (first few text files):**
{chr(10).join([f"=== {f['name']} ===" + chr(10) + f.get('content', '')[:500] + ('...' if len(f.get('content', '')) > 500 else '') + chr(10) for f in text_files[:5]])}

Please analyze:
1. **Archive Purpose** - What kind of project/content is this?
2. **Structure Analysis** - How is the content organized?
3. **Content Quality** - Assessment of the files and structure
4. **Notable Files** - Important or interesting files found
5. **Recommendations** - Suggestions for the user

Be concise but informative."""

        elif analysis_depth == "detailed":
            analysis_prompt = f"""🔍 **Detailed ZIP Archive Analysis**

Perform comprehensive analysis of this ZIP archive:

**Archive Details:**
- Files: {total_files} | Size: {total_size:,} bytes
- Types: {file_types}

**Complete File Structure:**
{chr(10).join([f"{i+1}. {f['name']} ({f.get('size', 0)} bytes, {f.get('type', 'unknown')})" for i, f in enumerate(file_contents)])}

**Content Analysis (text files):**
{chr(10).join([f"=== FILE: {f['name']} ===" + chr(10) + f.get('content', '')[:1000] + ('...' if len(f.get('content', '')) > 1000 else '') + chr(10) + "---" + chr(10) for f in text_files[:10]])}

Provide detailed analysis:
1. **Project/Content Identification** - What is this archive?
2. **Architecture Analysis** - Code/content structure and organization
3. **Quality Assessment** - Code quality, documentation, completeness
4. **Key Components** - Important files and their purposes
5. **Dependencies** - External requirements or libraries
6. **Usage Instructions** - How to use/deploy this content
7. **Security Review** - Any potential security concerns
8. **Improvement Suggestions** - Recommendations for enhancement"""

        elif analysis_depth == "code_focus":
            code_files = [f for f in text_files if any(ext in f.get('name', '').lower() for ext in ['.py', '.js', '.java', '.cpp', '.c', '.php', '.rb', '.go', '.rs', '.ts', '.jsx', '.vue'])]
            
            analysis_prompt = f"""💻 **Code-Focused ZIP Analysis**

Analyze the code structure and quality in this archive:

**Code Statistics:**
- Total Files: {total_files} | Code Files: {len(code_files)}
- Languages Detected: {', '.join(set([f.get('name', '').split('.')[-1] for f in code_files if '.' in f.get('name', '')]))}

**Code Files:**
{chr(10).join([f"📄 {f['name']} ({f.get('size', 0)} bytes)" for f in code_files])}

**Code Analysis (samples):**
{chr(10).join([f"=== {f['name']} ===" + chr(10) + f.get('content', '')[:800] + ('...' if len(f.get('content', '')) > 800 else '') + chr(10) + "---" + chr(10) for f in code_files[:8]])}

Please provide:
1. **Technology Stack** - Programming languages, frameworks used
2. **Code Architecture** - Structure, patterns, design approach  
3. **Code Quality** - Style, organization, best practices
4. **Functionality** - What does this code do?
5. **Completeness** - Is this a complete application/library?
6. **Dependencies** - Required libraries or external dependencies
7. **Deployment** - How to run or deploy this code
8. **Security Analysis** - Potential security issues or concerns
9. **Code Review** - Strengths, weaknesses, suggestions

Focus on technical aspects and code quality."""

        else:
            analysis_prompt = f"Analyze this ZIP archive with {total_files} files containing: {', '.join(file_types.keys())}"
        
        # Choose model based on analysis complexity
        if analysis_depth == "detailed" or len(text_files) > 10:
            model_name = Config.DEFAULT_TEXT_MODEL  # Use most capable model
            max_tokens = 2500
        elif analysis_depth == "code_focus":
            model_name = Config.DEFAULT_CODE_MODEL  # Use code-specialized model for code analysis
            max_tokens = 2000
        else:
            model_name = Config.ADVANCED_TEXT_MODEL
            max_tokens = 1500
        
        # For code-focused analysis, use code generation endpoint
        if analysis_depth == "code_focus":
            payload = {
                "inputs": analysis_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.2,  # Lower temperature for code analysis
                    "do_sample": True,
                    "top_p": 0.9,
                    "return_full_text": False
                }
            }
        else:
            payload = {
                "inputs": analysis_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.4,  # Moderate temperature for analysis
                    "do_sample": True,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            analysis_text = result[0].get('generated_text', '').strip()
            
            return True, {
                'analysis_depth': analysis_depth,
                'analysis': analysis_text,
                'archive_stats': {
                    'total_files': total_files,
                    'total_size': total_size,
                    'file_types': file_types,
                    'text_files_analyzed': len(text_files)
                },
                'model_used': model_name.split('/')[-1] if '/' in model_name else model_name
            }
        
        error_message = result if isinstance(result, str) else "Failed to analyze ZIP contents"
        return False, {'error': error_message}
    
    async def analyze_image_content(self, image_data: bytes, analysis_type: str, api_key: str) -> Tuple[bool, Dict]:
        """
        Analyze image content using vision models and AI
        
        Args:
            image_data (bytes): Image file data
            analysis_type (str): Type of analysis ("description", "ocr", "objects", "comprehensive")
            api_key (str): Hugging Face API key
            
        Returns:
            Tuple[bool, Dict]: (success, analysis_data)
        """
        try:
            # First, we need to encode the image for API calls
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Choose model based on analysis type
            if analysis_type == "ocr" or analysis_type == "text":
                # Use a model that can extract text from images
                model_name = "microsoft/trocr-base-handwritten"  # For OCR tasks
                payload = {
                    "inputs": image_b64
                }
            elif analysis_type == "objects" or analysis_type == "detection":
                # Use object detection model
                model_name = "facebook/detr-resnet-50"  # For object detection
                payload = {
                    "inputs": image_b64
                }
            else:
                # Use general vision model for description and comprehensive analysis
                model_name = "nlpconnect/vit-gpt2-image-captioning"  # For image captioning
                payload = {
                    "inputs": image_b64
                }
            
            success, result = await self._make_api_call(model_name, payload, api_key)
            
            if success:
                # Process different types of results
                if analysis_type == "ocr" and result:
                    # OCR results
                    if isinstance(result, list) and len(result) > 0:
                        extracted_text = result[0].get('generated_text', '') if 'generated_text' in result[0] else str(result[0])
                        
                        # If we have text, analyze it with a text model
                        if extracted_text.strip():
                            text_analysis_prompt = f"""📖 **Text Extracted from Image Analysis**

Extracted Text:
{extracted_text}

Please provide:
1. **Content Summary** - What type of text is this?
2. **Key Information** - Important details, numbers, or data
3. **Structure Analysis** - How is the text organized?
4. **Practical Value** - How can this information be used?
5. **Quality Assessment** - Is the text complete and clear?

Be concise but thorough."""

                            text_payload = {
                                "inputs": text_analysis_prompt,
                                "parameters": {
                                    "max_new_tokens": 800,
                                    "temperature": 0.3,
                                    "return_full_text": False
                                }
                            }
                            
                            text_success, text_result = await self._make_api_call(Config.ADVANCED_TEXT_MODEL, text_payload, api_key)
                            
                            analysis = text_result[0].get('generated_text', '') if text_success and isinstance(text_result, list) else "Text analysis unavailable"
                            
                            return True, {
                                'analysis_type': 'ocr',
                                'extracted_text': extracted_text,
                                'text_analysis': analysis,
                                'model_used': 'trocr_plus_text_analysis'
                            }
                        else:
                            return True, {
                                'analysis_type': 'ocr',
                                'extracted_text': extracted_text or "No text detected in image",
                                'model_used': 'trocr'
                            }
                            
                elif analysis_type in ["objects", "detection"] and result:
                    # Object detection results
                    return True, {
                        'analysis_type': 'object_detection',
                        'objects_detected': result,
                        'model_used': 'detr'
                    }
                    
                else:
                    # General image description/captioning
                    if isinstance(result, list) and len(result) > 0:
                        caption = result[0].get('generated_text', '') if 'generated_text' in result[0] else str(result[0])
                        
                        # Enhance the caption with AI analysis
                        if analysis_type == "comprehensive":
                            enhancement_prompt = f"""🖼️ **Comprehensive Image Analysis Enhancement**

Basic Image Caption: "{caption}"

Based on this caption, please provide enhanced analysis:

1. **Visual Content** - Detailed description of what's visible
2. **Scene Analysis** - Setting, context, environment
3. **Object Details** - Specific objects, people, or elements present
4. **Artistic/Technical Elements** - Colors, composition, style, quality
5. **Context Interpretation** - What might be happening or the purpose
6. **Practical Information** - Any useful details for the viewer
7. **Overall Assessment** - Quality, clarity, and notable features

Expand on the basic caption with rich, detailed insights."""

                            enhancement_payload = {
                                "inputs": enhancement_prompt,
                                "parameters": {
                                    "max_new_tokens": 1200,
                                    "temperature": 0.5,
                                    "return_full_text": False
                                }
                            }
                            
                            enhance_success, enhance_result = await self._make_api_call(Config.ADVANCED_TEXT_MODEL, enhancement_payload, api_key)
                            
                            enhanced_analysis = enhance_result[0].get('generated_text', '') if enhance_success and isinstance(enhance_result, list) else caption
                            
                            return True, {
                                'analysis_type': 'comprehensive',
                                'basic_caption': caption,
                                'detailed_analysis': enhanced_analysis,
                                'model_used': 'vit_gpt2_plus_text_enhancement'
                            }
                        else:
                            return True, {
                                'analysis_type': 'description',
                                'description': caption,
                                'model_used': 'vit_gpt2'
                            }
            
            # Fallback to text-only analysis if vision models fail
            fallback_prompt = f"""🖼️ **Image Analysis Request**

The user has uploaded an image for {analysis_type} analysis, but I cannot directly process images at the moment.

Please provide helpful guidance:

1. **Analysis Type Requested:** {analysis_type}
2. **What I would typically analyze:** 
   - For description: Visual content, objects, scene, colors, composition
   - For OCR: Text extraction and content analysis
   - For objects: Object detection and identification
   - For comprehensive: Complete visual analysis and insights

3. **Alternative Suggestions:**
   - How the user could get this type of analysis
   - What to look for when analyzing this type of content
   - Tools or methods that might help

4. **General Guidance:**
   - Tips for this type of image analysis
   - What information is typically valuable

Be helpful and provide value even without direct image access."""

            fallback_payload = {
                "inputs": fallback_prompt,
                "parameters": {
                    "max_new_tokens": 600,
                    "temperature": 0.6,
                    "return_full_text": False
                }
            }
            
            fallback_success, fallback_result = await self._make_api_call(Config.FALLBACK_TEXT_MODEL, fallback_payload, api_key)
            
            if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
                guidance = fallback_result[0].get('generated_text', '').strip()
                return True, {
                    'analysis_type': f'{analysis_type}_guidance',
                    'guidance': guidance,
                    'note': 'Direct image processing unavailable, providing guidance instead',
                    'model_used': 'text_guidance'
                }
            
            return False, {'error': f'Image {analysis_type} analysis failed - vision models unavailable'}
            
        except Exception as e:
            safe_exception_msg = redact_sensitive_data(str(e))
            secure_logger.error(f"Error in image analysis: {safe_exception_msg}")
            return False, {'error': f'Image analysis error: {safe_exception_msg}'}

    async def summarize_text(self, text: str, api_key: Optional[str] = None, max_length: int = 200) -> Tuple[bool, str]:
        """
        Summarize long text using AI models
        
        Args:
            text (str): Text to summarize
            api_key (Optional[str]): Hugging Face API key (optional)
            max_length (int): Maximum length of summary
            
        Returns:
            Tuple[bool, str]: (success, summary_text)
        """
        if not api_key:
            return False, "API key is required for text summarization"
        
        if not text or not text.strip():
            return False, "Text content is required for summarization"
        
        # Use Facebook BART model for summarization
        model_name = getattr(Config, 'SUMMARIZATION_MODEL', "facebook/bart-large-cnn")
        
        # Truncate text if too long (BART has input limits)
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": max(30, max_length // 4),
                "do_sample": False
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            summary = result[0].get('summary_text', '').strip()
            if summary:
                return True, summary
        
        # Try fallback with text generation model
        fallback_prompt = f"Summarize the following text in {max_length} characters or less:\n\n{text}\n\nSummary:"
        fallback_payload = {
            "inputs": fallback_prompt,
            "parameters": {
                "max_new_tokens": max_length // 4,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        fallback_success, fallback_result = await self._make_api_call(Config.DEFAULT_TEXT_MODEL, fallback_payload, api_key)
        if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
            summary = fallback_result[0].get('generated_text', '').strip()
            return True, summary
        
        error_message = result if isinstance(result, str) else "Failed to summarize text"
        return False, error_message

    async def answer_question(self, question: str, context: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Answer questions based on provided context using AI models
        
        Args:
            question (str): Question to answer
            context (str): Context/background information
            api_key (Optional[str]): Hugging Face API key (optional)
            
        Returns:
            Tuple[bool, str]: (success, answer_text)
        """
        if not api_key:
            return False, "API key is required for question answering"
        
        if not question or not question.strip():
            return False, "Question is required"
        
        if not context or not context.strip():
            return False, "Context is required for question answering"
        
        # Use BERT model for question answering
        model_name = getattr(Config, 'QA_MODEL', "deepset/roberta-base-squad2")
        
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, dict):
            answer = result.get('answer', '').strip()
            if answer:
                confidence = result.get('score', 0.0)
                if confidence > 0.1:  # Minimum confidence threshold
                    return True, answer
        
        # Try fallback with text generation model
        fallback_prompt = f"""Context: {context}

Question: {question}

Answer based on the context provided:"""
        
        fallback_payload = {
            "inputs": fallback_prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.2,
                "return_full_text": False
            }
        }
        
        fallback_success, fallback_result = await self._make_api_call(Config.DEFAULT_TEXT_MODEL, fallback_payload, api_key)
        if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
            answer = fallback_result[0].get('generated_text', '').strip()
            return True, answer
        
        error_message = result if isinstance(result, str) else "Failed to answer question"
        return False, error_message

    async def translate_text(self, text: str, target_language: str, api_key: Optional[str] = None, source_language: str = "auto") -> Tuple[bool, str]:
        """
        Translate text to target language using AI models
        
        Args:
            text (str): Text to translate
            target_language (str): Target language (e.g., "Spanish", "French", "German")
            api_key (Optional[str]): Hugging Face API key (optional)
            source_language (str): Source language (default: "auto" for auto-detection)
            
        Returns:
            Tuple[bool, str]: (success, translated_text)
        """
        if not api_key:
            return False, "API key is required for text translation"
        
        if not text or not text.strip():
            return False, "Text content is required for translation"
        
        if not target_language or not target_language.strip():
            return False, "Target language is required"
        
        # Map language names to codes
        language_map = {
            "spanish": "es", "french": "fr", "german": "de", "italian": "it",
            "portuguese": "pt", "russian": "ru", "chinese": "zh", "japanese": "ja",
            "korean": "ko", "arabic": "ar", "hindi": "hi", "dutch": "nl"
        }
        
        target_lang_code = language_map.get(target_language.lower(), target_language.lower())
        
        # Use Helsinki-NLP translation models
        model_name = f"Helsinki-NLP/opus-mt-en-{target_lang_code}"
        
        # For common language pairs, use specific models
        if target_language.lower() in ["spanish", "es"]:
            model_name = "Helsinki-NLP/opus-mt-en-es"
        elif target_language.lower() in ["french", "fr"]:
            model_name = "Helsinki-NLP/opus-mt-en-fr"
        elif target_language.lower() in ["german", "de"]:
            model_name = "Helsinki-NLP/opus-mt-en-de"
        else:
            # Use mBART for less common languages
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
        
        payload = {
            "inputs": text
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            translation = result[0].get('translation_text', '').strip()
            if translation:
                return True, translation
        
        # Try fallback with text generation model
        fallback_prompt = f"Translate the following text to {target_language}:\n\n{text}\n\nTranslation:"
        fallback_payload = {
            "inputs": fallback_prompt,
            "parameters": {
                "max_new_tokens": len(text) + 100,
                "temperature": 0.2,
                "return_full_text": False
            }
        }
        
        fallback_success, fallback_result = await self._make_api_call(Config.DEFAULT_TEXT_MODEL, fallback_payload, api_key)
        if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
            translation = fallback_result[0].get('generated_text', '').strip()
            return True, translation
        
        error_message = result if isinstance(result, str) else f"Failed to translate text to {target_language}"
        return False, error_message
    
    async def generate_with_fallback(self, 
                                   messages: Union[str, List[str], List[ChatMessage]], 
                                   intent_type: str = "text", 
                                   model: Optional[str] = None,
                                   max_tokens: Optional[int] = None,
                                   temperature: Optional[float] = None,
                                   top_p: Optional[float] = None,
                                   fallback_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate content with automatic fallback to alternative models
        
        Args:
            messages (str|List[str]|List[ChatMessage]): Input messages
            intent_type (str): Type of content to generate
            model (str|None): Preferred model to use
            max_tokens (int|None): Maximum tokens to generate
            temperature (float|None): Sampling temperature
            top_p (float|None): Top-p sampling parameter
            fallback_models (list[str]|None): Specific fallback models to try
            
        Returns:
            dict: {"success": True, "content": response_content, "model": model_name} on success
                 or {"success": False, "error": "All models failed"} on failure
        """
        import time
        from .model_health_monitor import health_monitor
        from .dynamic_fallback_strategy import dynamic_fallback_strategy, ErrorType
        from .conversation_context_tracker import conversation_context_tracker
        from .model_selection_explainer import model_selection_explainer
        
        # Normalize messages to List[ChatMessage] format
        if isinstance(messages, str):
            # Convert string to chat message format
            from .ai_providers import ChatMessage
            normalized_messages = [ChatMessage(role="user", content=messages)]
            prompt_text = messages  # For _call_provider which expects string
        elif isinstance(messages, list):
            # Assume it's already in the right format or convert from list of strings
            if messages and isinstance(messages[0], str):
                # Handle List[str] case - explicit cast for type checker
                string_messages = cast(List[str], messages)
                prompt_text = "\n".join(string_messages)
                from .ai_providers import ChatMessage
                normalized_messages = [ChatMessage(role="user", content=prompt_text)]
            else:
                # Handle List[ChatMessage] case - explicit cast for type checker
                from .ai_providers import ChatMessage
                chat_messages = cast(List[ChatMessage], messages)
                normalized_messages = chat_messages
                # Extract text for _call_provider from ChatMessage objects
                text_parts = []
                for msg in chat_messages:
                    # msg is guaranteed to be ChatMessage here
                    text_parts.append(msg.content)
                prompt_text = "\n".join(text_parts)
        else:
            return {"success": False, "error": "Invalid messages format"}
        
        # Build candidate model list
        candidate_models = []
        
        # Add specific model if provided
        if model and not health_monitor.should_avoid_model(model):
            candidate_models.append(model)
        
        # Add fallback models if provided
        if fallback_models:
            for fallback_model in fallback_models:
                if fallback_model and not health_monitor.should_avoid_model(fallback_model):
                    candidate_models.append(fallback_model)
        
        # If no specific models provided or all should be avoided, use health monitor
        if not candidate_models:
            best_model = health_monitor.get_best_model(intent_type)
            if best_model:
                candidate_models.append(best_model)
        
        # Add additional fallback models from health monitor rankings
        if len(candidate_models) < 3:  # Ensure we have enough fallbacks
            model_rankings = health_monitor.model_rankings()
            for ranked_model, score in model_rankings:
                if (ranked_model not in candidate_models and 
                    not health_monitor.should_avoid_model(ranked_model) and
                    len(candidate_models) < 5):  # Limit to 5 total models
                    candidate_models.append(ranked_model)
        
        # Final fallback to config defaults if still no models
        if not candidate_models:
            from ..config import Config
            fallback_defaults = [
                Config.DEFAULT_TEXT_MODEL,
                Config.EFFICIENT_TEXT_MODEL,
                Config.FALLBACK_TEXT_MODEL
            ]
            candidate_models = [m for m in fallback_defaults if m]
        
        if not candidate_models:
            return {"success": False, "error": "No models available"}
        
        # Build parameters for _call_provider
        parameters = {}
        if max_tokens is not None:
            parameters['max_new_tokens'] = max_tokens
        if temperature is not None:
            parameters['temperature'] = temperature
        if top_p is not None:
            parameters['top_p'] = top_p
        
        # Set defaults if not provided
        if 'max_new_tokens' not in parameters:
            parameters['max_new_tokens'] = 512
        if 'temperature' not in parameters:
            parameters['temperature'] = 0.7
        if 'top_p' not in parameters:
            parameters['top_p'] = 0.9
        
        # Try each candidate model
        last_error = None
        for model_name in candidate_models:
            start_time = time.time()
            
            try:
                # Check if model should be avoided (double-check in case status changed)
                if health_monitor.should_avoid_model(model_name):
                    secure_logger.warning(f"⚠️ Skipping model {model_name} (health monitor says avoid)")
                    continue
                
                # Call _call_provider with proper parameters
                success, response = await self._call_provider(
                    model_name=model_name,
                    prompt=prompt_text,
                    parameters=parameters,
                    is_chat=len(normalized_messages) > 1 or any(msg.role != "user" for msg in normalized_messages),
                    intent_type=intent_type
                )
                
                response_time = time.time() - start_time
                
                if success and response:
                    # Extract content from response
                    response_content = ""
                    if isinstance(response, list) and len(response) > 0:
                        if isinstance(response[0], dict) and "generated_text" in response[0]:
                            response_content = response[0]["generated_text"]
                        elif isinstance(response[0], dict) and "content" in response[0]:
                            response_content = response[0]["content"]
                        else:
                            response_content = str(response[0])
                    elif isinstance(response, dict):
                        response_content = response.get("content", response.get("generated_text", str(response)))
                    else:
                        response_content = str(response)
                    
                    # Update health monitor with success
                    await health_monitor.update_metrics(model_name, success=True, response_time=response_time)
                    
                    secure_logger.info(f"✅ generate_with_fallback succeeded with model {model_name}")
                    return {
                        "success": True,
                        "content": response_content,
                        "model": model_name
                    }
                else:
                    # Handle failure
                    error_msg = response if isinstance(response, str) else "Model call failed"
                    last_error = error_msg
                    
                    # Update health monitor with failure
                    await health_monitor.update_metrics(model_name, success=False, response_time=response_time, error=error_msg)
                    
                    secure_logger.warning(f"⚠️ Model {model_name} failed: {redact_sensitive_data(str(error_msg)[:100])}")
                    
            except Exception as e:
                response_time = time.time() - start_time
                error_msg = str(e)
                last_error = error_msg
                
                # Update health monitor with exception
                await health_monitor.update_metrics(model_name, success=False, response_time=response_time, error=error_msg)
                
                safe_error = redact_sensitive_data(str(e))
                secure_logger.error(f"❌ Exception with model {model_name}: {safe_error}")
        
        # All models failed
        secure_logger.error(f"🚨 All {len(candidate_models)} models failed for intent_type: {intent_type}")
        return {
            "success": False,
            "error": "All models failed"
        }

# Performance monitoring system
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

@dataclass 
class ModelPerformanceMetrics:
    """Track performance metrics for each model"""
    model_name: str
    success_rate: float
    avg_response_time: float
    avg_quality_score: float
    total_calls: int
    recent_failures: int
    last_success: datetime
    last_failure: Optional[datetime]
    quality_trend: List[float]  # Recent quality scores
    response_time_trend: List[float]  # Recent response times

class PerformanceMonitor:
    """Advanced performance monitoring system for superior AI routing"""
    
    def __init__(self):
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.failure_patterns = defaultdict(list)
        self.model_rankings = {}
        self.last_ranking_update = datetime.now()
        
    def update_metrics(self, model: str, success: bool, response_time: float, 
                      quality_score: float, intent_type: str):
        """Update performance metrics for a model"""
        if model not in self.model_metrics:
            self.model_metrics[model] = ModelPerformanceMetrics(
                model_name=model,
                success_rate=1.0 if success else 0.0,
                avg_response_time=response_time,
                avg_quality_score=quality_score,
                total_calls=1,
                recent_failures=0 if success else 1,
                last_success=datetime.now() if success else datetime.min,
                last_failure=datetime.now() if not success else None,
                quality_trend=[quality_score],
                response_time_trend=[response_time]
            )
            secure_logger.info(f"📊 Initialized metrics for {model}: quality={quality_score:.1f}, success={success}")
        else:
            metrics = self.model_metrics[model]
            
            # Update basic stats
            metrics.total_calls += 1
            old_avg_time = metrics.avg_response_time
            old_avg_quality = metrics.avg_quality_score
            
            metrics.avg_response_time = (old_avg_time * (metrics.total_calls - 1) + response_time) / metrics.total_calls
            metrics.avg_quality_score = (old_avg_quality * (metrics.total_calls - 1) + quality_score) / metrics.total_calls
            
            # Update success rate
            success_count = metrics.success_rate * (metrics.total_calls - 1) + (1 if success else 0)
            metrics.success_rate = success_count / metrics.total_calls
            
            # Update trends (keep last 10)
            metrics.quality_trend.append(quality_score)
            if len(metrics.quality_trend) > 10:
                metrics.quality_trend.pop(0)
            
            metrics.response_time_trend.append(response_time)
            if len(metrics.response_time_trend) > 10:
                metrics.response_time_trend.pop(0)
                
            # Track failures
            if success:
                metrics.last_success = datetime.now()
                metrics.recent_failures = 0
            else:
                metrics.last_failure = datetime.now()
                metrics.recent_failures += 1
                self.failure_patterns[model].append({
                    'timestamp': datetime.now(),
                    'intent_type': intent_type,
                    'response_time': response_time
                })
            
            secure_logger.info(f"📈 Updated {model}: calls={metrics.total_calls}, success={metrics.success_rate:.2f}, "
                       f"quality={metrics.avg_quality_score:.1f}, time={metrics.avg_response_time:.1f}s")
        
        # Update performance history
        self.performance_history[model].append({
            'timestamp': datetime.now(),
            'success': success,
            'response_time': response_time,
            'quality_score': quality_score,
            'intent_type': intent_type
        })
        
        # Update rankings periodically
        if (datetime.now() - self.last_ranking_update).total_seconds() > 300:  # Every 5 minutes
            self.update_model_rankings()
    
    def update_model_rankings(self):
        """Update model performance rankings for superior routing"""
        rankings = {}
        
        for model, metrics in self.model_metrics.items():
            # Calculate composite score (0-1 scale)
            success_component = metrics.success_rate * 0.4  # 40% success rate
            quality_component = min(metrics.avg_quality_score / 10.0, 1.0) * 0.3  # 30% quality
            speed_component = max(0, 1.0 - metrics.avg_response_time / 30.0) * 0.2  # 20% speed (30s baseline)
            reliability_component = (1.0 if metrics.recent_failures == 0 else 0.5) * 0.1  # 10% recent reliability
            
            score = success_component + quality_component + speed_component + reliability_component
            
            # Apply trend bonuses/penalties
            if len(metrics.quality_trend) >= 3:
                recent_quality = statistics.mean(metrics.quality_trend[-3:])
                older_quality = statistics.mean(metrics.quality_trend[:-3] if len(metrics.quality_trend) > 3 else metrics.quality_trend[-3:])
                trend = recent_quality - older_quality
                score += trend * 0.05  # Trend adjustment (±0.5 max)
            
            rankings[model] = max(0.0, min(1.0, score))  # Clamp to [0,1]
        
        # Sort by score (highest first)
        self.model_rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
        self.last_ranking_update = datetime.now()
        
        if rankings:
            top_3 = list(self.model_rankings.keys())[:3]
            secure_logger.info(f"🏆 TOP MODELS: {', '.join(f'{m}({self.model_rankings[m]:.2f})' for m in top_3)}")
    
    def get_best_model_for_intent(self, intent_type: str, available_models: List[str]) -> str:
        """Get the best performing model for a specific intent type - ALWAYS returns a valid model"""
        from ..config import Config
        
        # CRITICAL FIX: Always have a guaranteed fallback
        guaranteed_fallback = Config.DEFAULT_TEXT_MODEL
        
        if not self.model_metrics or not available_models:
            secure_logger.warning(f"No metrics/models available for {intent_type}, using guaranteed fallback: {guaranteed_fallback}")
            return guaranteed_fallback
        
        # Find models with good performance for this intent
        intent_performers = {}
        
        for model in available_models:
            if model not in self.performance_history:
                continue
                
            history = self.performance_history[model]
            intent_records = [r for r in history if r['intent_type'] == intent_type]
            
            if len(intent_records) >= 2:  # Need at least 2 samples
                recent_records = intent_records[-5:]  # Last 5 for quality
                avg_quality = statistics.mean([r['quality_score'] for r in recent_records])
                success_rate = sum(1 for r in intent_records[-10:] if r['success']) / min(len(intent_records), 10)
                
                # Combined score: quality (60%) + success rate (40%)
                intent_score = avg_quality * 0.06 + success_rate * 4.0
                intent_performers[model] = {
                    'quality': avg_quality,
                    'success_rate': success_rate,
                    'score': intent_score,
                    'samples': len(intent_records)
                }
        
        if intent_performers:
            best_model = max(intent_performers.keys(), key=lambda m: intent_performers[m]['score'])
            perf = intent_performers[best_model]
            secure_logger.info(f"🎯 INTENT SPECIALIST: {best_model} for {intent_type} "
                       f"(quality={perf['quality']:.1f}, success={perf['success_rate']:.2f}, samples={perf['samples']})")
            return best_model
        
        # Fallback to overall rankings
        for model in self.model_rankings:
            if model in available_models:
                return model
        
        # CRITICAL FIX: NEVER return None - always return guaranteed fallback
        if available_models:
            # Use first available model if no ranking matches
            fallback_model = available_models[0]
            secure_logger.warning(f"No ranked models available for {intent_type}, using first available: {fallback_model}")
            return fallback_model
        else:
            # Ultimate fallback if somehow no models are available
            secure_logger.error(f"CRITICAL: No models available for {intent_type}, using guaranteed fallback: {guaranteed_fallback}")
            return guaranteed_fallback
    
    def should_avoid_model(self, model: str) -> Tuple[bool, str]:
        """Check if a model should be avoided due to poor performance"""
        if model not in self.model_metrics:
            return False, "No metrics available"
        
        metrics = self.model_metrics[model]
        
        # Avoid if too many recent consecutive failures
        if metrics.recent_failures >= 3:
            return True, f"Too many recent failures ({metrics.recent_failures})"
        
        # Avoid if last failure was recent and success rate is critically low
        if (metrics.last_failure and 
            (datetime.now() - metrics.last_failure).total_seconds() < 300 and
            metrics.success_rate < 0.5):
            return True, f"Recent failure with low success rate ({metrics.success_rate:.2f})"
        
        # Avoid if average quality is very poor (many samples needed)
        if metrics.total_calls >= 10 and metrics.avg_quality_score < 3.0:
            return True, f"Consistently poor quality ({metrics.avg_quality_score:.1f}/10)"
        
        return False, "Model is performing adequately"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.model_metrics:
            return {'status': 'No metrics available'}
        
        summary = {
            'total_models': len(self.model_metrics),
            'top_performers': list(self.model_rankings.keys())[:5],
            'model_details': {},
            'system_health': 'healthy'
        }
        
        failing_models = 0
        for model, metrics in self.model_metrics.items():
            should_avoid, reason = self.should_avoid_model(model)
            if should_avoid:
                failing_models += 1
            
            summary['model_details'][model] = {
                'success_rate': metrics.success_rate,
                'avg_quality': metrics.avg_quality_score,
                'avg_response_time': metrics.avg_response_time,
                'total_calls': metrics.total_calls,
                'status': 'failing' if should_avoid else 'healthy',
                'ranking': list(self.model_rankings.keys()).index(model) + 1 if model in self.model_rankings else None
            }
        
        if failing_models > len(self.model_metrics) * 0.3:  # More than 30% failing
            summary['system_health'] = 'degraded'
        elif failing_models > len(self.model_metrics) * 0.5:  # More than 50% failing
            summary['system_health'] = 'critical'
        
        return summary

# Enhanced ModelCaller with integrated performance monitoring
class SuperiorModelCaller(ModelCaller):
    """Enhanced ModelCaller with superior AI capabilities and performance monitoring"""
    
    def __init__(self):
        super().__init__()
        self.performance_monitor = PerformanceMonitor()
        
    async def call_with_monitoring(self, method_name: str, model: str, intent_type: str, 
                                 *args, **kwargs) -> Tuple[bool, Any, Dict]:
        """
        Call model method with performance monitoring
        Returns: (success, result, performance_metrics)
        """
        start_time = time.time()
        
        try:
            # Call the original method
            method = getattr(self, method_name)
            success, result = await method(*args, **kwargs)
            
            response_time = time.time() - start_time
            
            # Assess quality if we have response processor available
            quality_score = 7.0  # Default quality score
            try:
                from bot.core.response_processor import response_processor
                if isinstance(result, str) and result:
                    # For text responses, assess quality
                    _, quality_metrics = response_processor.process_response(
                        result, 
                        kwargs.get('prompt', ''), 
                        intent_type, 
                        model
                    )
                    quality_score = quality_metrics.overall_score
            except ImportError:
                pass  # Response processor not available
            
            # Update performance metrics
            self.performance_monitor.update_metrics(
                model, success, response_time, quality_score, intent_type
            )
            
            perf_metrics = {
                'response_time': response_time,
                'quality_score': quality_score,
                'model_used': model,
                'success': success
            }
            
            return success, result, perf_metrics
            
        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.update_metrics(model, False, response_time, 0.0, intent_type)
            
            perf_metrics = {
                'response_time': response_time,
                'quality_score': 0.0,
                'model_used': model,
                'success': False,
                'error': str(e)
            }
            
            return False, str(e), perf_metrics
    
    def get_optimal_model_for_intent(self, intent_type: str, complexity_score: float = 5.0) -> str:
        """Get the optimal model for an intent with fallback logic"""
        # Get available models based on intent type
        available_models = self._get_available_models_for_intent(intent_type, complexity_score)
        
        # Try to get the best performer for this intent
        best_model = self.performance_monitor.get_best_model_for_intent(intent_type, available_models)
        
        if best_model:
            should_avoid, reason = self.performance_monitor.should_avoid_model(best_model)
            if not should_avoid:
                return best_model
            else:
                secure_logger.warning(f"⚠️ Avoiding {best_model}: {reason}")
        
        # Fallback to default model selection
        return self._get_fallback_model_for_intent(intent_type, complexity_score)
    
    def _get_available_models_for_intent(self, intent_type: str, complexity_score: float) -> List[str]:
        """Get available models for a specific intent type"""
        models = []
        
        if intent_type == 'code_generation':
            models = [Config.DEFAULT_CODE_MODEL, Config.ADVANCED_TEXT_MODEL, Config.DEFAULT_TEXT_MODEL]
        elif intent_type in ['mathematical_reasoning', 'advanced_reasoning']:
            models = [Config.ADVANCED_TEXT_MODEL, Config.DEFAULT_TEXT_MODEL]
        elif intent_type == 'creative_writing':
            models = [Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL]
        elif intent_type == 'image_generation':
            models = [Config.DEFAULT_IMAGE_MODEL]
        else:
            # General text generation
            if complexity_score > 7:
                models = [Config.ADVANCED_TEXT_MODEL, Config.DEFAULT_TEXT_MODEL]
            else:
                models = [Config.DEFAULT_TEXT_MODEL, Config.FAST_TEXT_MODEL]
        
        return [m for m in models if m]  # Remove None values
    
    def _get_fallback_model_for_intent(self, intent_type: str, complexity_score: float) -> str:
        """Get fallback model when optimal selection fails"""
        if intent_type == 'code_generation':
            return Config.DEFAULT_CODE_MODEL or Config.ADVANCED_TEXT_MODEL
        elif intent_type == 'image_generation':
            return Config.DEFAULT_IMAGE_MODEL
        elif complexity_score > 7:
            return Config.ADVANCED_TEXT_MODEL
        else:
            return Config.DEFAULT_TEXT_MODEL or Config.FALLBACK_TEXT_MODEL
    
    async def call_model(self, model_name: str, prompt: str, 
                        intent_type: str = "text", **kwargs) -> Tuple[bool, str]:
        """
        Simple convenience method for model calling - delegates to appropriate generation method
        
        Args:
            model_name (str): Model to use for generation
            prompt (str): Input prompt
            intent_type (str): Type of generation task
            **kwargs: Additional parameters for generation
            
        Returns:
            Tuple[bool, str]: (success, generated_content)
        """
        try:
            # Route to appropriate generation method based on intent type
            if intent_type in ["code_generation", "code", "programming"]:
                success, content = await self.generate_code(
                    prompt=prompt,
                    model_override=model_name,
                    special_params=kwargs.get('special_params'),
                    language=kwargs.get('language', 'python')
                )
            elif intent_type in ["image_generation", "image", "visual"]:
                success, result = await self.generate_image(
                    prompt=prompt,
                    model_override=model_name,
                    special_params=kwargs.get('special_params'),
                    **kwargs
                )
                content = str(result) if success else result
            else:
                # Default to text generation
                success, content = await self.generate_text(
                    prompt=prompt,
                    model_override=model_name,
                    special_params=kwargs.get('special_params'),
                    intent_type=intent_type,
                    chat_history=kwargs.get('chat_history')
                )
            
            return success, content
            
        except Exception as e:
            secure_logger.error(f"❌ Error in call_model: {redact_sensitive_data(str(e))}")
            return False, f"Error: {str(e)}"

# Global instances
performance_monitor = PerformanceMonitor()
model_caller = SuperiorModelCaller()