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
from typing import Dict, List, Optional, Tuple, Any
from bot.config import Config

logger = logging.getLogger(__name__)

class ModelCaller:
    """Handles all Hugging Face API interactions with intelligent routing"""
    
    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)
    
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
        # Use different endpoints based on type for 2025 API changes
        if endpoint_type == "chat":
            url = f"https://api-inference.huggingface.co/models/{model_name}/v1/chat/completions"
        elif endpoint_type == "text2img":
            url = f"https://api-inference.huggingface.co/models/{model_name}"
        else:
            url = f"https://api-inference.huggingface.co/models/{model_name}"
            
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
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
                    logger.error(f"Unauthorized access for model {model_name}: {error_text}")
                    return False, "Invalid API key. Please check your Hugging Face API key."
                
                elif response.status == 404:  # Model not found
                    error_text = await response.text()
                    logger.error(f"Model not found: {model_name} - {error_text}")
                    return False, f"Model '{model_name}' not found or not accessible. This may be due to 2025 API changes."
                
                elif response.status == 400:  # Bad request - often API format issues
                    error_text = await response.text()
                    logger.error(f"Bad request for model {model_name}: {error_text}")
                    return False, f"Invalid request format for model '{model_name}'. This model may require a different API endpoint."
                
                else:
                    error_text = await response.text()
                    logger.error(f"API call failed with status {response.status}: {error_text}")
                    return False, f"API error (HTTP {response.status}): {error_text}"
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling model {model_name} after {Config.REQUEST_TIMEOUT}s")
            if retries < Config.MAX_RETRIES:
                logger.info(f"Retrying {model_name} due to timeout (attempt {retries + 1})")
                await asyncio.sleep(Config.RETRY_DELAY)
                return await self._make_api_call(model_name, payload, api_key, retries + 1)
            return False, "Request timed out after multiple attempts. The model may be overloaded."
        
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error calling model {model_name}: {e}")
            if retries < Config.MAX_RETRIES:
                logger.info(f"Retrying {model_name} due to connection error (attempt {retries + 1})")
                await asyncio.sleep(Config.RETRY_DELAY * 2)  # Longer wait for connection issues
                return await self._make_api_call(model_name, payload, api_key, retries + 1)
            return False, "Network connection error. Please check your internet connection and try again."
        
        except aiohttp.ContentTypeError as e:
            logger.error(f"Content type error calling model {model_name}: {e}")
            return False, "Invalid response format from API. Please try again with a different model."
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error calling model {model_name}: {e}")
            return False, "Invalid JSON response from API. Please try again."
        
        except Exception as e:
            logger.error(f"Unexpected error calling model {model_name}: {e}", exc_info=True)
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
        
        # Advanced parameters for latest models
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": special_params.get('max_new_tokens', 1000),
                "temperature": special_params.get('temperature', 0.7),
                "do_sample": True,
                "top_p": special_params.get('top_p', 0.9),
                "repetition_penalty": special_params.get('repetition_penalty', 1.05),
                "return_full_text": False,
                "pad_token_id": 50256  # For better generation with latest models
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
            (Config.CODE_INSTRUCT_MODEL, f"Write {language} code to {prompt}"),
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
        Generate image using state-of-the-art FLUX.1 model
        
        Args:
            prompt (str): Image generation prompt
            api_key (str): Hugging Face API key
            special_params (dict): Special parameters for model
            
        Returns:
            Tuple[bool, bytes]: (success, image_bytes)
        """
        special_params = special_params or {}
        model_name = Config.DEFAULT_IMAGE_MODEL  # FLUX.1-schnell
        
        # Enhanced prompt for FLUX.1 - more detailed and artistic
        style_enhancers = "professional photography, high quality, detailed, sharp focus, vibrant colors"
        enhanced_prompt = f"{prompt}, {style_enhancers}"
        
        # FLUX.1-schnell optimized parameters (4 steps for speed)
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "guidance_scale": special_params.get('guidance_scale', 7.5),
                "num_inference_steps": special_params.get('num_inference_steps', 4),  # FLUX.1-schnell optimized
                "width": special_params.get('width', 1024),
                "height": special_params.get('height', 1024),
                "seed": -1  # Random seed for variety
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key, endpoint_type="text2img")
        
        if success and isinstance(result, bytes):
            return True, result
        
        # Try fallback image models with appropriate configurations
        fallback_models = [
            (Config.FALLBACK_IMAGE_MODEL, {
                "inputs": enhanced_prompt,
                "parameters": {
                    "guidance_scale": special_params.get('guidance_scale', 7.5),
                    "num_inference_steps": special_params.get('num_inference_steps', 20),
                    "width": special_params.get('width', 512),
                    "height": special_params.get('height', 512)
                }
            }),
            (Config.ALTERNATIVE_IMAGE_MODEL, {
                "inputs": prompt,  # Simpler prompt for fallback
                "parameters": {
                    "guidance_scale": 7.5,
                    "num_inference_steps": 15
                }
            })
        ]
        
        for fallback_model, fallback_payload in fallback_models:
            if model_name != fallback_model:
                logger.info(f"Trying fallback image model {fallback_model}")
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
        if not success and model_name != Config.SIMPLE_SENTIMENT_MODEL:
            logger.info(f"Trying fallback sentiment model {Config.SIMPLE_SENTIMENT_MODEL}")
            fallback_success, fallback_result = await self._make_api_call(Config.SIMPLE_SENTIMENT_MODEL, payload, api_key)
            
            if fallback_success and isinstance(fallback_result, list):
                return True, {
                    'emotion_type': 'sentiment',
                    'result': fallback_result[0] if fallback_result else {},
                    'model_used': 'distilbert_sentiment'
                }
        
        error_message = result if isinstance(result, str) else "Failed to analyze sentiment"
        return False, {'error': error_message}

# Global model caller instance
model_caller = ModelCaller()