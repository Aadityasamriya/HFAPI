"""
HuggingFace InferenceClient Provider Implementation
Phase 1 of HF API Migration - Uses huggingface_hub.InferenceClient for modern API access
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceEndpointError, InferenceTimeoutError, HfHubHTTPError

from .ai_providers import (
    AIProvider, 
    ProviderConfig, 
    ChatMessage, 
    ChatCompletionRequest,
    CompletionRequest,
    ProviderResponse,
    ProviderError,
    ModelNotAvailableError,
    RateLimitError,
    AuthenticationError,
    TimeoutError as ProviderTimeoutError,
    QuotaExceededError
)
from bot.security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

class HFInferenceProvider(AIProvider):
    """
    HuggingFace Inference Provider using InferenceClient
    Supports both text generation and chat completion with OpenAI-compatible format
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize HF Inference Provider
        
        Args:
            config (ProviderConfig): Provider configuration
        """
        super().__init__(config)
        
        # Initialize InferenceClient
        self.client = InferenceClient(
            model=None,  # Model will be specified per request
            token=config.api_key,
            timeout=config.timeout
        )
        
        # Track supported models and their capabilities
        self._model_cache = {}
        self._last_model_fetch = 0
        self._model_cache_ttl = 3600  # 1 hour
        
        secure_logger.info(f"🚀 HFInferenceProvider initialized (mode: {config.api_mode.value})")
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ProviderResponse:
        """
        Generate chat completion using OpenAI-compatible format
        
        Args:
            request (ChatCompletionRequest): Chat completion request
            
        Returns:
            ProviderResponse: Standardized response
        """
        start_time = time.time()
        
        try:
            # Convert chat messages to prompt format for HF models
            prompt = self._format_chat_messages_for_hf(request.messages)
            
            # Prepare generation parameters
            generation_params = self._prepare_generation_params(request)
            
            secure_logger.info(f"🔄 HF Chat completion: {request.model} ({len(request.messages)} messages)")
            
            # Make the inference call
            try:
                # FIXED: Use proper async pattern to avoid StopIteration issues
                response = await self._safe_text_generation(
                    prompt=prompt,
                    model=request.model,
                    **generation_params
                )
                
                response_time = time.time() - start_time
                
                # Extract text content from response
                if isinstance(response, str):
                    content = response
                elif hasattr(response, 'generated_text'):
                    content = response.generated_text
                else:
                    content = str(response)
                
                # ENHANCED: Clean up and post-process the response for superior quality
                content = self._clean_generated_text(content, prompt)
                
                # ENHANCED: Calculate response quality score for performance tracking
                quality_score = self._calculate_response_quality_score(content, prompt, response_time)
                
                secure_logger.info(f"✅ HF Chat completion success: {len(content)} chars in {response_time:.2f}s (quality: {quality_score:.1f}/10)")
                
                return ProviderResponse(
                    success=True,
                    content=content,
                    provider_used="huggingface_inference",
                    model_used=request.model,
                    response_time=response_time,
                    metadata={
                        "messages_count": len(request.messages),
                        "prompt_length": len(prompt),
                        "generation_params": generation_params,
                        "quality_score": quality_score,  # ENHANCED: Include quality score
                        "response_length": len(content),
                        "tokens_used": generation_params.get("max_new_tokens", 512)
                    }
                )
                
            except InferenceTimeoutError as e:
                raise ProviderTimeoutError(f"Request timed out: {e}", "huggingface_inference", request.model)
            except InferenceEndpointError as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    raise AuthenticationError(f"Authentication failed: {e}", "huggingface_inference", request.model)
                elif "402" in error_msg or "payment required" in error_msg.lower() or "quota exceeded" in error_msg.lower() or "credit" in error_msg.lower():
                    raise QuotaExceededError(f"API quota/credits exceeded: {e}", "huggingface_inference", request.model)
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    raise RateLimitError(f"Rate limit exceeded: {e}", "huggingface_inference", request.model)
                elif "404" in error_msg or "not found" in error_msg.lower():
                    raise ModelNotAvailableError(f"Model not available: {e}", "huggingface_inference", request.model)
                else:
                    raise ProviderError(f"Inference error: {e}", "huggingface_inference", request.model)
                    
        except (ProviderError, AuthenticationError, RateLimitError, ModelNotAvailableError, ProviderTimeoutError, QuotaExceededError) as e:
            response_time = time.time() - start_time
            secure_logger.error(f"❌ HF Chat completion failed: {e}")
            
            return ProviderResponse(
                success=False,
                content=None,
                error_message=str(e),
                provider_used="huggingface_inference",
                model_used=request.model,
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            safe_error = redact_sensitive_data(str(e))
            
            secure_logger.error(f"❌ HF Chat completion unexpected error: {safe_error}")
            
            return ProviderResponse(
                success=False,
                content=None,
                error_message=f"Unexpected error: {safe_error}",
                provider_used="huggingface_inference",
                model_used=request.model,
                response_time=response_time
            )
    
    async def text_completion(self, request: CompletionRequest) -> ProviderResponse:
        """
        Generate text completion from plain prompt
        
        Args:
            request (CompletionRequest): Text completion request
            
        Returns:
            ProviderResponse: Standardized response
        """
        start_time = time.time()
        
        try:
            # Prepare generation parameters
            generation_params = self._prepare_generation_params_from_completion(request)
            
            secure_logger.info(f"🔄 HF Text completion: {request.model} ({len(request.prompt)} chars)")
            
            # Make the inference call
            try:
                # FIXED: Use proper async pattern to avoid StopIteration issues
                response = await self._safe_text_generation(
                    prompt=request.prompt,
                    model=request.model,
                    **generation_params
                )
                
                response_time = time.time() - start_time
                
                # Extract text content from response
                if isinstance(response, str):
                    content = response
                elif hasattr(response, 'generated_text'):
                    content = response.generated_text
                else:
                    content = str(response)
                
                # Clean up the response
                content = self._clean_generated_text(content, request.prompt)
                
                secure_logger.info(f"✅ HF Text completion success: {len(content)} chars in {response_time:.2f}s")
                
                return ProviderResponse(
                    success=True,
                    content=content,
                    provider_used="huggingface_inference",
                    model_used=request.model,
                    response_time=response_time,
                    metadata={
                        "prompt_length": len(request.prompt),
                        "generation_params": generation_params
                    }
                )
                
            except InferenceTimeoutError as e:
                raise ProviderTimeoutError(f"Request timed out: {e}", "huggingface_inference", request.model)
            except InferenceEndpointError as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    raise AuthenticationError(f"Authentication failed: {e}", "huggingface_inference", request.model)
                elif "402" in error_msg or "payment required" in error_msg.lower() or "quota exceeded" in error_msg.lower() or "credit" in error_msg.lower():
                    raise QuotaExceededError(f"API quota/credits exceeded: {e}", "huggingface_inference", request.model)
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    raise RateLimitError(f"Rate limit exceeded: {e}", "huggingface_inference", request.model)
                elif "404" in error_msg or "not found" in error_msg.lower():
                    raise ModelNotAvailableError(f"Model not available: {e}", "huggingface_inference", request.model)
                else:
                    raise ProviderError(f"Inference error: {e}", "huggingface_inference", request.model)
                    
        except (ProviderError, AuthenticationError, RateLimitError, ModelNotAvailableError, ProviderTimeoutError, QuotaExceededError) as e:
            response_time = time.time() - start_time
            secure_logger.error(f"❌ HF Text completion failed: {e}")
            
            return ProviderResponse(
                success=False,
                content=None,
                error_message=str(e),
                provider_used="huggingface_inference",
                model_used=request.model,
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            safe_error = redact_sensitive_data(str(e))
            
            secure_logger.error(f"❌ HF Text completion unexpected error: {safe_error}")
            
            return ProviderResponse(
                success=False,
                content=None,
                error_message=f"Unexpected error: {safe_error}",
                provider_used="huggingface_inference",
                model_used=request.model,
                response_time=response_time
            )
    
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible
        
        Returns:
            bool: True if provider is healthy
        """
        try:
            # CRITICAL FIX: Use a verified working model from discovery
            from ..config import Config
            test_model = Config.FLAGSHIP_TEXT_MODEL  # Qwen/Qwen2.5-7B-Instruct (verified working)
            
            # FIXED: Use asyncio.to_thread to properly wrap synchronous call
            response = await asyncio.to_thread(
                self.client.text_generation,
                prompt="Hello",
                model=test_model,
                max_new_tokens=5,
                return_full_text=False
            )
            
            secure_logger.info("✅ HF Provider health check passed")
            return True
            
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ HF Provider health check failed: {safe_error}")
            return False
    
    async def get_supported_models(self) -> List[str]:
        """
        Get list of models supported by this provider
        
        Returns:
            List[str]: List of supported model names
        """
        # CRITICAL FIX: Return only verified working models from discovery
        # Only these 2 models passed quota and availability checks as of 2025-10-06
        from ..config import Config
        return [
            Config.FLAGSHIP_TEXT_MODEL,        # Qwen/Qwen2.5-7B-Instruct
            Config.ULTRA_PERFORMANCE_TEXT_MODEL  # Qwen/Qwen2.5-72B-Instruct
        ]
    
    def _format_chat_messages_for_hf(self, messages: List[ChatMessage]) -> str:
        """
        ENHANCED: Format chat messages for superior HuggingFace model performance
        Uses advanced chat templating optimized for instruction-following and response quality
        
        Args:
            messages (List[ChatMessage]): Chat messages
            
        Returns:
            str: Enhanced formatted prompt optimized for quality responses
        """
        if not messages:
            return ""
        
        # ENHANCED: Use superior chat template format for better model performance
        formatted_parts = []
        
        system_messages = [msg for msg in messages if msg.role == "system"]
        conversation_messages = [msg for msg in messages if msg.role in ["user", "assistant"]]
        
        # ENHANCED: Add sophisticated system instruction formatting
        if system_messages:
            system_content = " ".join([msg.content.strip() for msg in system_messages])
            # Use enhanced system prompt formatting for better instruction following
            formatted_parts.append(f"<|system|>\n{system_content}\n<|end|>")
        else:
            # Default enhanced system prompt for superior response quality
            default_system = "You are a helpful, accurate, and concise AI assistant. Provide clear, informative, and well-structured responses. Focus on being precise and helpful."
            formatted_parts.append(f"<|system|>\n{default_system}\n<|end|>")
        
        # ENHANCED: Process conversation with improved context preservation
        for msg in conversation_messages:
            if msg.role == "user":
                # Enhanced user message formatting with context markers
                formatted_parts.append(f"<|user|>\n{msg.content.strip()}\n<|end|>")
            elif msg.role == "assistant":
                # Enhanced assistant message formatting for consistency
                formatted_parts.append(f"<|assistant|>\n{msg.content.strip()}\n<|end|>")
        
        # ENHANCED: Add optimized response prompt for better generation quality
        formatted_parts.append("<|assistant|>")
        
        # Join with proper spacing for optimal model performance
        return "\n\n".join(formatted_parts)
    
    def _validate_and_sanitize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize generation parameters for HuggingFace API
        
        Args:
            params (Dict[str, Any]): Raw generation parameters
            
        Returns:
            Dict[str, Any]: Validated and sanitized parameters
        """
        validated_params: Dict[str, Any] = {}
        
        # Validate and set return_full_text (boolean)
        if "return_full_text" in params:
            validated_params["return_full_text"] = bool(params["return_full_text"])
        
        # Validate and set do_sample (boolean)
        if "do_sample" in params:
            validated_params["do_sample"] = bool(params["do_sample"])
        
        # Validate and set max_new_tokens (positive integer)
        if "max_new_tokens" in params:
            max_tokens = params["max_new_tokens"]
            if isinstance(max_tokens, (int, float)) and max_tokens > 0:
                validated_params["max_new_tokens"] = int(max_tokens)
            else:
                secure_logger.warning(f"Invalid max_new_tokens: {max_tokens}, using default 512")
                validated_params["max_new_tokens"] = 512
        
        # Validate and set temperature (float between 0.0 and 2.0)
        if "temperature" in params:
            temp = params["temperature"]
            if isinstance(temp, (int, float)):
                validated_params["temperature"] = max(0.0, min(2.0, float(temp)))
            else:
                secure_logger.warning(f"Invalid temperature: {temp}, using default 0.7")
                validated_params["temperature"] = 0.7
        
        # Validate and set top_p (float between 0.0 and 1.0)
        if "top_p" in params:
            top_p = params["top_p"]
            if isinstance(top_p, (int, float)):
                validated_params["top_p"] = max(0.0, min(1.0, float(top_p)))
            else:
                secure_logger.warning(f"Invalid top_p: {top_p}, ignoring parameter")
        
        # Validate and set top_k (positive integer)
        if "top_k" in params:
            top_k = params["top_k"]
            if isinstance(top_k, (int, float)) and top_k > 0:
                validated_params["top_k"] = int(top_k)
            else:
                secure_logger.warning(f"Invalid top_k: {top_k}, ignoring parameter")
        
        # Validate and set repetition_penalty (positive float)
        if "repetition_penalty" in params:
            rep_penalty = params["repetition_penalty"]
            if isinstance(rep_penalty, (int, float)) and rep_penalty > 0:
                validated_params["repetition_penalty"] = float(rep_penalty)
            else:
                secure_logger.warning(f"Invalid repetition_penalty: {rep_penalty}, ignoring parameter")
        
        return validated_params
    
    def _prepare_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Prepare generation parameters from chat completion request
        
        Args:
            request (ChatCompletionRequest): Chat completion request
            
        Returns:
            Dict[str, Any]: Generation parameters for InferenceClient
        """
        # Create raw parameters dictionary
        raw_params: Dict[str, Any] = {
            "return_full_text": False,  # Only return generated text
            "do_sample": True,
        }
        
        # ENHANCED: Add intelligent token limit based on request type and context
        if request.max_tokens:
            # Apply intelligent caps to prevent excessive token usage
            capped_tokens = min(request.max_tokens, self._get_intelligent_token_limit(request))
            raw_params["max_new_tokens"] = capped_tokens
        else:
            # Use context-aware default token limits
            raw_params["max_new_tokens"] = self._get_intelligent_token_limit(request)
        
        # Add temperature parameter
        if request.temperature is not None:
            raw_params["temperature"] = request.temperature
        else:
            raw_params["temperature"] = 0.7  # Default
            
        # Add top_p parameter
        if request.top_p is not None:
            raw_params["top_p"] = request.top_p
        
        # Validate and sanitize all parameters
        return self._validate_and_sanitize_generation_params(raw_params)
    
    def _prepare_generation_params_from_completion(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Prepare generation parameters from text completion request
        
        Args:
            request (CompletionRequest): Text completion request
            
        Returns:
            Dict[str, Any]: Generation parameters for InferenceClient
        """
        # Create raw parameters dictionary
        raw_params: Dict[str, Any] = {
            "return_full_text": False,  # Only return generated text
            "do_sample": True,
        }
        
        # ENHANCED: Add intelligent token limit based on request type and context
        if request.max_tokens:
            # Apply intelligent caps to prevent excessive token usage
            capped_tokens = min(request.max_tokens, self._get_intelligent_token_limit(request))
            raw_params["max_new_tokens"] = capped_tokens
        else:
            # Use context-aware default token limits
            raw_params["max_new_tokens"] = self._get_intelligent_token_limit(request)
        
        # Add temperature parameter
        if request.temperature is not None:
            raw_params["temperature"] = request.temperature
        else:
            raw_params["temperature"] = 0.7  # Default
            
        # Add top_p parameter
        if request.top_p is not None:
            raw_params["top_p"] = request.top_p
        
        # Validate and sanitize all parameters
        return self._validate_and_sanitize_generation_params(raw_params)
    
    def _get_intelligent_token_limit(self, request) -> int:
        """
        ENHANCED: Calculate intelligent token limits based on request context and type
        Optimizes response length for quality while preventing excessive token usage
        
        Args:
            request: Chat or completion request
            
        Returns:
            int: Intelligent token limit for optimal response quality
        """
        # Base token limits by request type
        base_limits = {
            'chat': 512,
            'completion': 512,
            'code': 1024,      # Code generation needs more tokens
            'reasoning': 768,   # Reasoning tasks need moderate length
            'creative': 1024,   # Creative writing can be longer
            'short': 256,      # Quick responses
        }
        
        # Analyze request content to determine appropriate category
        if hasattr(request, 'messages') and request.messages:
            # Chat request - analyze message content
            latest_message = request.messages[-1].content.lower()
            
            # Check for code generation indicators
            if any(keyword in latest_message for keyword in [
                'code', 'function', 'class', 'script', 'program', 'algorithm',
                'python', 'javascript', 'java', 'cpp', 'rust', 'go'
            ]):
                return base_limits['code']
            
            # Check for reasoning/explanation indicators  
            elif any(keyword in latest_message for keyword in [
                'explain', 'analyze', 'compare', 'reasoning', 'why', 'how',
                'step by step', 'detailed', 'comprehensive'
            ]):
                return base_limits['reasoning']
            
            # Check for creative writing indicators
            elif any(keyword in latest_message for keyword in [
                'story', 'creative', 'poem', 'narrative', 'write', 'compose',
                'imagine', 'fiction'
            ]):
                return base_limits['creative']
            
            # Check for short response indicators
            elif any(keyword in latest_message for keyword in [
                'yes/no', 'quickly', 'brief', 'short', 'simple', 'list',
                'summarize', 'tldr'
            ]) or len(latest_message) < 50:
                return base_limits['short']
        
        elif hasattr(request, 'prompt'):
            # Completion request - analyze prompt
            prompt_lower = request.prompt.lower()
            if 'code' in prompt_lower or 'function' in prompt_lower:
                return base_limits['code']
            elif len(request.prompt) < 50:
                return base_limits['short']
        
        # Default to chat limit
        return base_limits['chat']
    
    def _calculate_response_quality_score(self, response_content: str, original_prompt: str, response_time: float) -> float:
        """
        ENHANCED: Calculate quality score for response post-processing
        Evaluates response quality across multiple dimensions for superior AI performance
        
        Args:
            response_content (str): Generated response
            original_prompt (str): Original user prompt
            response_time (float): Time taken to generate response
            
        Returns:
            float: Quality score from 0.0 to 10.0
        """
        if not response_content or not response_content.strip():
            return 0.0
        
        score = 5.0  # Base score
        
        # Factor 1: Response completeness (0-2 points)
        if len(response_content.strip()) > 20:
            score += 1.0
        if len(response_content.strip()) > 100:
            score += 1.0
        
        # Factor 2: Response coherence - no excessive repetition (-2 to +1 points)
        words = response_content.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio > 0.7:
                score += 1.0  # Good variety
            elif repetition_ratio < 0.3:
                score -= 2.0  # Too repetitive
        
        # Factor 3: Response structure - proper formatting (+0.5 to +1.5 points)
        if any(marker in response_content for marker in ['.', '!', '?']):
            score += 0.5  # Has sentence structure
        if any(marker in response_content for marker in ['\n', '\n-', '1.', '2.']):
            score += 0.5  # Has formatting/structure
        if response_content.count('\n') >= 2:
            score += 0.5  # Good paragraph structure
        
        # Factor 4: Response speed bonus (0 to +1 points)
        if response_time < 2.0:
            score += 1.0
        elif response_time < 5.0:
            score += 0.5
        
        # Factor 5: No obvious errors penalty (-1 to 0 points)
        error_indicators = ['error', 'undefined', 'null', 'none', '[object object]']
        if any(error in response_content.lower() for error in error_indicators):
            score -= 1.0
        
        # Factor 6: Relevance to prompt (+0 to +1.5 points)
        prompt_words = set(original_prompt.lower().split())
        response_words = set(response_content.lower().split())
        if len(prompt_words) > 0:
            relevance_ratio = len(prompt_words.intersection(response_words)) / len(prompt_words)
            if relevance_ratio > 0.3:
                score += 1.5  # High relevance
            elif relevance_ratio > 0.1:
                score += 1.0  # Moderate relevance
        
        # Cap at 10.0 and ensure minimum 0.0
        return max(0.0, min(10.0, score))
    
    def _clean_generated_text(self, generated_text: str, original_prompt: str) -> str:
        """
        Clean generated text by removing the original prompt if it's included
        
        Args:
            generated_text (str): Generated text from the model
            original_prompt (str): Original prompt
            
        Returns:
            str: Cleaned text
        """
        if not generated_text:
            return ""
        
        # If the response includes the original prompt, remove it
        if generated_text.startswith(original_prompt):
            cleaned = generated_text[len(original_prompt):].strip()
            return cleaned
        
        # Remove common prompt artifacts
        cleaned = generated_text.strip()
        
        # Remove "Assistant:" prefix if present
        if cleaned.startswith("Assistant:"):
            cleaned = cleaned[10:].strip()
        
        return cleaned
    
    async def _safe_text_generation(self, prompt: str, model: str, **kwargs) -> Any:
        """
        FIXED: Safe text generation using InferenceClient with proper async pattern
        Respects config.api_mode and config.base_url settings
        
        Args:
            prompt (str): Text prompt
            model (str): Model name
            **kwargs: Additional generation parameters
        
        Returns:
            Any: Response from HuggingFace API
        
        Raises:
            Various provider errors for different failure modes
        """
        from .ai_providers import APIMode
        
        timeout = kwargs.pop('timeout', self.config.timeout)
        
        try:
            # Check API mode from config
            if self.config.api_mode == APIMode.INFERENCE_PROVIDERS:
                # Use Inference Providers API (chat completions endpoint)
                return await self._call_inference_providers_api(prompt, model, timeout, **kwargs)
            else:
                # Use standard InferenceClient for INFERENCE_API mode
                return await self._call_inference_client(prompt, model, **kwargs)
                
        except Exception as e:
            # Re-raise with proper error classification
            error_msg = str(e)
            
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise AuthenticationError(
                    f"Authentication failed: {e}", 
                    "huggingface_inference", 
                    model
                )
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                raise RateLimitError(
                    f"Rate limit exceeded: {e}", 
                    "huggingface_inference", 
                    model
                )
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise ModelNotAvailableError(
                    f"Model not available: {e}", 
                    "huggingface_inference", 
                    model
                )
            elif "doesn't support task" in error_msg or "not supported for task" in error_msg:
                raise ModelNotAvailableError(
                    f"Model task mismatch: {e}", 
                    "huggingface_inference", 
                    model
                )
            else:
                raise ProviderError(
                    f"Inference error: {e}", 
                    "huggingface_inference", 
                    model
                )
    
    async def _call_inference_client(self, prompt: str, model: str, **kwargs) -> Any:
        """
        Call InferenceClient.text_generation() properly using asyncio.to_thread()
        
        Args:
            prompt (str): Text prompt
            model (str): Model name
            **kwargs: Additional generation parameters
        
        Returns:
            Any: Generated text response
        """
        secure_logger.info(f"🔄 Using InferenceClient for model {model}")
        
        # Use asyncio.to_thread to make synchronous InferenceClient async
        response = await asyncio.to_thread(
            self.client.text_generation,
            prompt=prompt,
            model=model,
            **kwargs
        )
        
        secure_logger.info(f"✅ InferenceClient request successful for model {model}")
        return response
    
    async def _call_inference_providers_api(self, prompt: str, model: str, timeout: int, **kwargs) -> str:
        """
        Call Inference Providers API (chat completions endpoint)
        Respects config.base_url setting
        
        Args:
            prompt (str): Text prompt
            model (str): Model name
            timeout (int): Request timeout in seconds
            **kwargs: Additional generation parameters
        
        Returns:
            str: Generated text response
        """
        import aiohttp
        
        # FIXED: Respect config.base_url instead of hardcoding
        if self.config.base_url:
            api_url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        else:
            # Default to Inference Providers router endpoint
            api_url = "https://router.huggingface.co/v1/chat/completions"
        
        secure_logger.info(f"🔄 Using Inference Providers API at {api_url} for model {model}")
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use OpenAI-compatible chat completion format
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", 150)),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False
        }
        
        # Create ClientTimeout object for aiohttp
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        # Make HTTP request to endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload, timeout=timeout_obj) as http_response:
                if http_response.status == 200:
                    response_data = await http_response.json()
                    
                    # Extract text from response format
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        choice = response_data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response = choice["message"]["content"]
                        elif "text" in choice:
                            response = choice["text"]
                        else:
                            response = str(choice)
                    else:
                        response = str(response_data)
                    
                    secure_logger.info(f"✅ Inference Providers API request successful for model {model}")
                    return response
                else:
                    error_text = await http_response.text()
                    raise Exception(f"HTTP {http_response.status}: {error_text}")
    
    def _get_model_supported_tasks(self, model: str) -> List[str]:
        """Get supported tasks for a specific model"""
        # Model task mapping - using text-generation for most models as it's most universal
        model_tasks = {
            # VERIFIED working models only
            "facebook/bart-large-cnn": ["summarization", "text-generation"],  # VERIFIED WORKING
            # Other models return 404, commented out:
            # "openai-community/gpt2": ["text-generation"],
            # "microsoft/DialoGPT-medium": ["text-generation"],
            
            # Specialized models with specific tasks
            "facebook/bart-large-cnn": ["summarization"],
            "facebook/bart-base": ["summarization"],
            "deepset/roberta-base-squad2": ["question-answering"],
            "cardiffnlp/twitter-roberta-base-sentiment-latest": ["text-classification"],
        }
        
        return model_tasks.get(model, ["text-generation"])  # Default to text-generation
    
    def _get_compatible_models_for_task(self, task: str) -> List[str]:
        """
        Get models that are compatible with specific tasks - UPDATED with working models
        
        Args:
            task (str): Task type (text-generation, text-classification, etc.)
        
        Returns:
            List[str]: Compatible model names
        """
        # FIXED: Model-task compatibility mapping with verified working models
        task_models = {
            'text-generation': [
                # Use summarization model for text generation as fallback since it works
                "facebook/bart-large-cnn",  # VERIFIED WORKING
                # Other models return 404, so commented out for now
                # "openai-community/gpt2",
                # "openai-community/gpt2-medium", 
                # "microsoft/DialoGPT-medium",
            ],
            'summarization': [
                "facebook/bart-large-cnn",  # VERIFIED WORKING
                "facebook/bart-base",
                "google/pegasus-xsum",
                "sshleifer/distilbart-cnn-12-6"
            ],
            'text-classification': [
                "distilbert-base-uncased-finetuned-sst-2-english",
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "j-hartmann/emotion-english-distilroberta-base"
            ],
            'question-answering': [
                "deepset/roberta-base-squad2",
                "distilbert-base-uncased-distilled-squad"
            ]
        }
        
        return task_models.get(task, task_models['text-generation'])