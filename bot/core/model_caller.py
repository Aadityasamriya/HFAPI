"""
Hugging Face API integration with intelligent model calling
Supports text generation, image creation, code generation, and more
"""

import aiohttp
import asyncio
import io
import base64
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from bot.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

def _redact_sensitive_data(text: str) -> str:
    """
    Enhanced redaction of sensitive data from log messages to prevent token/credential leakage
    
    Args:
        text (str): Text that might contain sensitive information
        
    Returns:
        str: Sanitized text with sensitive data redacted
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Redact Hugging Face tokens (hf_xxxx) - Updated pattern for newer formats
    text = re.sub(r'hf_[a-zA-Z0-9]{20,}', 'hf_[REDACTED]', text)
    text = re.sub(r'huggingface_hub[_-]token["\s]*[:=]["\s]*[a-zA-Z0-9_.-]{15,}', 'huggingface_hub_token: [REDACTED]', text, flags=re.IGNORECASE)
    
    # Redact OpenAI API keys (sk-xxxx)
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', 'sk-[REDACTED]', text)
    
    # Redact Anthropic API keys (sk-ant-xxxx)
    text = re.sub(r'sk-ant-[a-zA-Z0-9_-]{20,}', 'sk-ant-[REDACTED]', text)
    
    # Redact Google/Vertex AI keys
    text = re.sub(r'AIza[a-zA-Z0-9_-]{35}', 'AIza[REDACTED]', text)
    
    # Redact JWT tokens (eyJ...)
    text = re.sub(r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', 'eyJ[REDACTED_JWT]', text)
    
    # Redact Authorization headers (various formats)
    text = re.sub(r'Bearer\s+[a-zA-Z0-9_.-]{15,}', 'Bearer [REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'"Authorization":\s*"Bearer\s+[^"]+"', '"Authorization": "Bearer [REDACTED]"', text, flags=re.IGNORECASE)
    text = re.sub(r"'Authorization':\s*'Bearer\s+[^']+'", "'Authorization': 'Bearer [REDACTED]'", text, flags=re.IGNORECASE)
    
    # Redact API keys in various formats
    text = re.sub(r'api[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9_.-]{15,}', 'api_key: [REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'token["\s]*[:=]["\s]*[a-zA-Z0-9_.-]{15,}', 'token: [REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'secret["\s]*[:=]["\s]*[a-zA-Z0-9_.-]{15,}', 'secret: [REDACTED]', text, flags=re.IGNORECASE)
    
    # Redact MongoDB connection strings (prevent credential leakage)
    text = re.sub(r'mongodb://[^@]+@[^/]+', 'mongodb://[REDACTED_USER]:[REDACTED_PASSWORD]@[REDACTED_HOST]', text, flags=re.IGNORECASE)
    text = re.sub(r'mongodb\+srv://[^@]+@[^/]+', 'mongodb+srv://[REDACTED_USER]:[REDACTED_PASSWORD]@[REDACTED_HOST]', text, flags=re.IGNORECASE)
    
    # Redact URLs with tokens/credentials in query parameters
    text = re.sub(r'([?&])(token|key|secret|password|auth)=([^&\s]+)', r'\1\2=[REDACTED]', text, flags=re.IGNORECASE)
    
    # Redact Telegram bot tokens (format: NNNNNNNNN:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)
    text = re.sub(r'\b\d{8,10}:[a-zA-Z0-9_-]{35}\b', '[REDACTED_BOT_TOKEN]', text)
    
    # Redact encryption seeds/keys (base64 or hex patterns that look like keys)
    text = re.sub(r'\b[A-Za-z0-9+/]{32,}={0,2}\b', '[REDACTED_KEY_OR_SEED]', text)
    text = re.sub(r'\b[a-fA-F0-9]{32,}\b', '[REDACTED_HEX_KEY]', text)
    
    # Redact email addresses in some contexts (privacy protection)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    
    # Redact IP addresses (privacy protection)
    text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[REDACTED_IP]', text)
    
    return text

def _safe_log_error(level, message: str, *args, **kwargs):
    """
    Safely log an error message with sensitive data redacted
    
    Args:
        level: Logging level method (logger.error, logger.warning, etc.)
        message (str): Log message that might contain sensitive data
        *args: Additional args for logging
        **kwargs: Additional kwargs for logging (exc_info will be disabled for security)
    """
    # Disable exc_info to prevent token leakage in tracebacks
    kwargs.pop('exc_info', None)
    
    # Redact sensitive data from message and args
    safe_message = _redact_sensitive_data(message)
    safe_args = tuple(_redact_sensitive_data(str(arg)) for arg in args)
    
    level(safe_message, *safe_args, **kwargs)


class SecureLogger:
    """
    Centralized secure logging system that automatically redacts sensitive data
    from all log messages to prevent credential leakage
    """
    
    def __init__(self, logger_instance):
        self.logger = logger_instance
    
    def _safe_log(self, level_method, message: str, *args, **kwargs):
        """
        Safely log a message with automatic sensitive data redaction
        
        Args:
            level_method: Logger level method (e.g., self.logger.info)
            message (str): Log message that might contain sensitive data
            *args: Additional args for logging
            **kwargs: Additional kwargs for logging (exc_info will be disabled for security)
        """
        # Always disable exc_info to prevent credential leakage in tracebacks
        kwargs.pop('exc_info', None)
        
        # Redact sensitive data from message and all args
        safe_message = _redact_sensitive_data(str(message))
        safe_args = tuple(_redact_sensitive_data(str(arg)) for arg in args)
        
        # Add security marker to identify secure logging
        if not safe_message.startswith('🔒'):
            safe_message = f"🔒 {safe_message}"
        
        level_method(safe_message, *safe_args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Secure info logging with automatic redaction"""
        self._safe_log(self.logger.info, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Secure warning logging with automatic redaction"""
        self._safe_log(self.logger.warning, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Secure error logging with automatic redaction"""
        self._safe_log(self.logger.error, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Secure debug logging with automatic redaction"""
        self._safe_log(self.logger.debug, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Secure critical logging with automatic redaction"""
        self._safe_log(self.logger.critical, message, *args, **kwargs)
    
    def audit(self, event_type: str, user_id: int, details: str, success: bool = True):
        """Security audit logging for sensitive operations"""
        status = "SUCCESS" if success else "FAILED"
        audit_message = f"🔐 SECURITY_AUDIT | Event: {event_type} | User: {user_id} | Status: {status} | Details: {details}"
        self._safe_log(self.logger.info, audit_message)


# Create a secure logger instance for use throughout the module
secure_logger = SecureLogger(logger)

class ModelCaller:
    """Handles all Hugging Face API interactions with intelligent routing"""
    
    def __init__(self, provider: str = "auto", bill_to: Optional[str] = None):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
        self.provider = provider  # 2025 feature: provider selection
        self.bill_to = bill_to    # 2025 feature: organization billing
        self.request_count = 0
        self.last_request_time = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_api_call(self, model_name: str, payload: Dict, api_key: str, retries: int = 0, endpoint_type: str = "inference") -> Tuple[bool, Any]:
        """
        Make API call to Hugging Face with retry logic and 2025 API support
        
        Args:
            model_name (str): Name of the Hugging Face model
            payload (dict): Request payload
            api_key (str): User's Hugging Face API key
            retries (int): Current retry count
            endpoint_type (str): Type of endpoint ("inference", "chat", "text2img")
            
        Returns:
            Tuple[bool, Any]: (success, response_data)
        """
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
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=self.timeout)
                
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    if 'image' in content_type:
                        image_data = await response.read()
                        return True, image_data
                    else:
                        result = await response.json()
                        return True, result
                
                elif response.status == 503:  # Model loading
                    if retries < Config.MAX_RETRIES:
                        await asyncio.sleep(Config.RETRY_DELAY * (retries + 1))
                        return await self._make_api_call(model_name, payload, api_key, retries + 1)
                    else:
                        return False, "Model is currently loading. Please try again in a few moments."
                
                elif response.status == 429:  # Rate limit
                    if retries < Config.MAX_RETRIES:
                        # Exponential backoff for rate limits
                        wait_time = Config.RETRY_DELAY * (2 ** retries)
                        logger.warning(f"Rate limit hit for model {model_name}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        return await self._make_api_call(model_name, payload, api_key, retries + 1)
                    else:
                        return False, "Rate limit exceeded. Please try again later."
                
                elif response.status >= 500:  # Server errors
                    if retries < Config.MAX_RETRIES:
                        # Linear backoff for server errors
                        wait_time = Config.RETRY_DELAY * (retries + 1)
                        logger.warning(f"Server error {response.status} for model {model_name}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        return await self._make_api_call(model_name, payload, api_key, retries + 1)
                    else:
                        return False, f"Server error (HTTP {response.status}). Please try again later."
                
                elif response.status == 401:  # Unauthorized
                    error_text = await response.text()
                    _safe_log_error(logger.error, f"Unauthorized access for model {model_name}: {error_text}")
                    return False, "Invalid API key. Please check your Hugging Face API key."
                
                elif response.status == 404:  # Model not found
                    error_text = await response.text()
                    _safe_log_error(logger.error, f"Model not found: {model_name} - {error_text}")
                    
                    # Enhanced 404 handling: Try fallback for chat/conversation models
                    if retries == 0 and endpoint_type == "chat":
                        logger.info(f"404 for chat model {model_name}, retrying with standard inference")
                        return await self._make_api_call(model_name, payload, api_key, retries + 1, "inference")
                    
                    # 2025: Enhanced fallback system - suggest switching to validated small models
                    fallback_msg = f"Model '{model_name}' not found or not accessible on HF Inference API. "
                    fallback_msg += "Consider using validated models: 'Qwen/Qwen2.5-7B-Instruct', 'microsoft/Phi-3.5-mini-instruct', or 'google-bert/bert-base-uncased'."
                    return False, fallback_msg
                
                elif response.status == 400:  # Bad request - often API format issues
                    error_text = await response.text()
                    _safe_log_error(logger.error, f"Bad request for model {model_name}: {error_text}")
                    
                    # Check if it's a "model not supported" error specific to inference API
                    if "not supported" in error_text.lower() or "inference" in error_text.lower():
                        fallback_msg = f"Model '{model_name}' not supported on HF Inference API. Try using 'Inference Providers' or switch to a compatible model."
                        return False, fallback_msg
                    
                    return False, f"Invalid request format for model '{model_name}'. This model may require a different API endpoint or format."
                
                else:
                    error_text = await response.text()
                    _safe_log_error(logger.error, f"API call failed with status {response.status}: {error_text}")
                    return False, f"API error (HTTP {response.status}): {error_text}"
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling model {model_name} after {Config.REQUEST_TIMEOUT}s")
            if retries < Config.MAX_RETRIES:
                logger.info(f"Retrying {model_name} due to timeout (attempt {retries + 1})")
                await asyncio.sleep(Config.RETRY_DELAY)
                return await self._make_api_call(model_name, payload, api_key, retries + 1)
            return False, "Request timed out after multiple attempts. The model may be overloaded."
        
        except aiohttp.ClientConnectorError as e:
            _safe_log_error(logger.error, f"Connection error calling model {model_name}: {e}")
            if retries < Config.MAX_RETRIES:
                logger.info(f"Retrying {model_name} due to connection error (attempt {retries + 1})")
                await asyncio.sleep(Config.RETRY_DELAY * 2)  # Longer wait for connection issues
                return await self._make_api_call(model_name, payload, api_key, retries + 1)
            return False, "Network connection error. Please check your internet connection and try again."
        
        except aiohttp.ContentTypeError as e:
            _safe_log_error(logger.error, f"Content type error calling model {model_name}: {e}")
            return False, "Invalid response format from API. Please try again with a different model."
        
        except json.JSONDecodeError as e:
            _safe_log_error(logger.error, f"JSON decode error calling model {model_name}: {e}")
            return False, "Invalid JSON response from API. Please try again."
        
        except Exception as e:
            # Use secure logging to prevent token leakage in tracebacks
            secure_logger.error(f"Unexpected error calling model {model_name}: {e}")
            return False, f"Unexpected error occurred. Please try again later."
    
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
    async def generate_text(self, prompt: str, api_key: str, chat_history: Optional[List[Dict]] = None, model_override: Optional[str] = None, special_params: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Generate text using latest 2024-2025 language models with intelligent fallbacks
        
        Args:
            prompt (str): User prompt
            api_key (str): Hugging Face API key
            chat_history (list): Previous conversation history
            model_override (str): Optional model override
            special_params (dict): Special parameters for model
            
        Returns:
            Tuple[bool, str]: (success, generated_text)
        """
        chat_history = chat_history or []
        special_params = special_params or {}
        model_name = model_override or Config.DEFAULT_TEXT_MODEL
        
        # Format the prompt with chat history
        formatted_prompt = self._format_chat_history(chat_history, prompt)
        
        # 2025 Enhanced parameters for latest models with provider-specific optimizations
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": special_params.get('max_new_tokens', 1000),
                "temperature": special_params.get('temperature', 0.7),
                "do_sample": True,
                "top_p": special_params.get('top_p', 0.9),
                "top_k": special_params.get('top_k', 50),  # 2025: Top-k sampling
                "repetition_penalty": special_params.get('repetition_penalty', 1.05),
                "length_penalty": special_params.get('length_penalty', 1.0),  # 2025: Length penalty
                "num_return_sequences": 1,
                "return_full_text": False,
                "pad_token_id": 50256,
                "eos_token_id": 50256,
                "use_cache": False,  # 2025: Disable caching for fresh responses
                "typical_p": special_params.get('typical_p', 0.95)  # 2025: Typical sampling
            },
            "options": {
                "wait_for_model": True,  # 2025: Always wait for model loading
                "use_gpu": True,  # 2025: Prefer GPU inference
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '').strip()
            
            # Advanced text cleanup for latest models
            cleanup_patterns = ['### Assistant:', '### Human:', 'Assistant:', 'Human:']
            for pattern in cleanup_patterns:
                if generated_text.startswith(pattern):
                    generated_text = generated_text.replace(pattern, '').strip()
            
            # Limit response length
            if len(generated_text) > Config.MAX_RESPONSE_LENGTH:
                generated_text = generated_text[:Config.MAX_RESPONSE_LENGTH] + "..."
            
            return True, generated_text
        
        # Smart fallback system - try multiple models with simplified payloads
        fallback_models = [
            (Config.ADVANCED_TEXT_MODEL, "inference"),
            (Config.FALLBACK_TEXT_MODEL, "inference")
        ]
        
        for fallback_model, endpoint_type in fallback_models:
            if model_name != fallback_model:
                logger.info(f"Trying fallback model {fallback_model}")
                
                # Simplified payload for fallback models
                fallback_payload = {
                    "inputs": prompt,  # Use simple prompt for fallbacks
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "return_full_text": False
                    }
                }
                
                fallback_success, fallback_result = await self._make_api_call(
                    fallback_model, fallback_payload, api_key, endpoint_type=endpoint_type
                )
                
                if fallback_success and isinstance(fallback_result, list) and len(fallback_result) > 0:
                    generated_text = fallback_result[0].get('generated_text', '').strip()
                    logger.info(f"Successfully used fallback model {fallback_model}")
                    return True, generated_text
        
        error_message = result if isinstance(result, str) else "Failed to generate text with all available models."
        return False, f"Text generation failed: {error_message}"
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=15))
    async def generate_code(self, prompt: str, api_key: str, language: str = "python", special_params: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Generate code using latest StarCoder2-15B and enhanced models
        
        Args:
            prompt (str): Code generation prompt
            api_key (str): Hugging Face API key
            language (str): Programming language
            special_params (dict): Special parameters for model
            
        Returns:
            Tuple[bool, str]: (success, generated_code)
        """
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
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            generated_code = result[0].get('generated_text', '').strip()
            
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
                logger.info(f"Trying fallback code model {fallback_model}")
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
                    generated_code = fallback_result[0].get('generated_text', '').strip()
                    if not generated_code.startswith('```'):
                        generated_code = f"```{language}\n{generated_code}\n```"
                    logger.info(f"Successfully used fallback code model {fallback_model}")
                    return True, generated_code
        
        error_message = result if isinstance(result, str) else "Failed to generate code with available models."
        return False, f"Code generation failed: {error_message}"
    
    async def generate_image(self, prompt: str, api_key: str, special_params: Optional[Dict] = None) -> Tuple[bool, bytes]:
        """
        Generate image with smart fallbacks - prefer reliable models for serverless
        
        Args:
            prompt (str): Image generation prompt
            api_key (str): Hugging Face API key
            special_params (dict): Special parameters for model
            
        Returns:
            Tuple[bool, bytes]: (success, image_bytes)
        """
        special_params = special_params or {}
        
        # Smart serverless-aware model selection (prefer stable models)
        is_complex_request = len(prompt) > 100 or any(word in prompt.lower() for word in ['detailed', 'complex', 'professional', 'artistic', 'photorealistic'])
        
        if is_complex_request:
            model_name = Config.DEFAULT_IMAGE_MODEL  # FLUX.1-schnell for complex requests
            # FLUX.1-schnell specific payload (guidance-free diffusion)
            payload = {
                "inputs": f"{prompt}, high quality, detailed, sharp focus",
                "parameters": {
                    "num_inference_steps": special_params.get('num_inference_steps', 4),  # FLUX.1-schnell optimized
                    "width": special_params.get('width', 1024),
                    "height": special_params.get('height', 1024)
                    # Note: FLUX.1-schnell doesn't use guidance_scale (guidance-free)
                }
            }
        else:
            # Use more reliable fallback model for simple requests
            model_name = Config.FALLBACK_IMAGE_MODEL  # SD 3.5 - more serverless-compatible
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
        
        if success and isinstance(result, bytes):
            return True, result
        
        # Smart fallback system with serverless-optimized models
        fallback_models = [
            # Most reliable serverless model first
            (Config.FALLBACK_IMAGE_MODEL, {
                "inputs": prompt,
                "parameters": {
                    "guidance_scale": 7.5,
                    "num_inference_steps": 15,
                    "width": 512,
                    "height": 512
                }
            }),
            # Simplified FLUX.1 (if primary was SD)
            (Config.DEFAULT_IMAGE_MODEL if model_name != Config.DEFAULT_IMAGE_MODEL else Config.ADVANCED_IMAGE_MODEL, {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 4,  # Minimal steps for faster generation
                    "width": 512,
                    "height": 512
                }
            }),
            # Last resort - simplest possible payload
            ("runwayml/stable-diffusion-v1-5", {
                "inputs": prompt
            })
        ]
        
        for fallback_model, fallback_payload in fallback_models:
            if model_name != fallback_model:
                logger.info(f"Trying serverless-optimized fallback model {fallback_model}")
                fallback_success, fallback_result = await self._make_api_call(
                    fallback_model, fallback_payload, api_key, endpoint_type="text2img"
                )
                if fallback_success and isinstance(fallback_result, bytes):
                    logger.info(f"Successfully used fallback image model {fallback_model}")
                    return True, fallback_result
        
        error_message = result if isinstance(result, str) else "Failed to generate image with available models."
        return False, f"Image generation failed: {error_message}".encode('utf-8')
    
    async def analyze_sentiment(self, text: str, api_key: str, use_emotion_detection: bool = False) -> Tuple[bool, Dict]:
        """
        Analyze sentiment with advanced emotion detection
        
        Args:
            text (str): Text to analyze
            api_key (str): Hugging Face API key
            use_emotion_detection (bool): Use advanced emotion model
            
        Returns:
            Tuple[bool, Dict]: (success, sentiment_data)
        """
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
            logger.info(f"Trying fallback sentiment model {Config.FALLBACK_SENTIMENT_MODEL}")
            fallback_success, fallback_result = await self._make_api_call(Config.FALLBACK_SENTIMENT_MODEL, payload, api_key)
            
            if fallback_success and isinstance(fallback_result, list):
                return True, {
                    'emotion_type': 'sentiment',
                    'result': fallback_result[0] if fallback_result else {},
                    'model_used': 'distilbert_sentiment'
                }
        
        error_message = result if isinstance(result, str) else "Failed to analyze sentiment"
        return False, {'error': error_message}
    
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
{pdf_text[:8000]}{'...' if len(pdf_text) > 8000 else ''}

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
{pdf_text[:10000]}{'...' if len(pdf_text) > 10000 else ''}

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
{pdf_text[:8000]}{'...' if len(pdf_text) > 8000 else ''}

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
{pdf_text[:8000]}{'...' if len(pdf_text) > 8000 else ''}

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
            _safe_log_error(logger.error, f"Error in image analysis: {e}")
            return False, {'error': f'Image analysis error: {str(e)}'}

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
            logger.info(f"📊 Initialized metrics for {model}: quality={quality_score:.1f}, success={success}")
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
            
            logger.info(f"📈 Updated {model}: calls={metrics.total_calls}, success={metrics.success_rate:.2f}, "
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
            logger.info(f"🏆 TOP MODELS: {', '.join(f'{m}({self.model_rankings[m]:.2f})' for m in top_3)}")
    
    def get_best_model_for_intent(self, intent_type: str, available_models: List[str]) -> Optional[str]:
        """Get the best performing model for a specific intent type"""
        if not self.model_metrics or not available_models:
            return None
        
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
            logger.info(f"🎯 INTENT SPECIALIST: {best_model} for {intent_type} "
                       f"(quality={perf['quality']:.1f}, success={perf['success_rate']:.2f}, samples={perf['samples']})")
            return best_model
        
        # Fallback to overall rankings
        for model in self.model_rankings:
            if model in available_models:
                return model
        
        return None
    
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
                logger.warning(f"⚠️ Avoiding {best_model}: {reason}")
        
        # Fallback to default model selection
        return self._get_fallback_model_for_intent(intent_type, complexity_score)
    
    def _get_available_models_for_intent(self, intent_type: str, complexity_score: float) -> List[str]:
        """Get available models for a specific intent type"""
        models = []
        
        if intent_type == 'code_generation':
            models = [Config.CODE_MODEL, Config.ADVANCED_TEXT_MODEL, Config.DEFAULT_TEXT_MODEL]
        elif intent_type in ['mathematical_reasoning', 'advanced_reasoning']:
            models = [Config.ADVANCED_TEXT_MODEL, Config.DEFAULT_TEXT_MODEL]
        elif intent_type == 'creative_writing':
            models = [Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL]
        elif intent_type == 'image_generation':
            models = [Config.IMAGE_MODEL]
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
            return Config.CODE_MODEL or Config.ADVANCED_TEXT_MODEL
        elif intent_type == 'image_generation':
            return Config.IMAGE_MODEL
        elif complexity_score > 7:
            return Config.ADVANCED_TEXT_MODEL
        else:
            return Config.DEFAULT_TEXT_MODEL or Config.FALLBACK_TEXT_MODEL

# Global instances
performance_monitor = PerformanceMonitor()
model_caller = SuperiorModelCaller()