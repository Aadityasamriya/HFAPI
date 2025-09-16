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
    
    async def _make_api_call(self, model_name: str, payload: Dict, api_key: str, retries: int = 0) -> Tuple[bool, Any]:
        """
        Make API call to Hugging Face with retry logic
        
        Args:
            model_name (str): Name of the Hugging Face model
            payload (dict): Request payload
            api_key (str): User's Hugging Face API key
            retries (int): Current retry count
            
        Returns:
            Tuple[bool, Any]: (success, response_data)
        """
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
                
                else:
                    error_text = await response.text()
                    logger.error(f"API call failed with status {response.status}: {error_text}")
                    return False, f"API error: {error_text}"
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling model {model_name}")
            return False, "Request timed out. Please try again."
        
        except Exception as e:
            logger.error(f"Error calling model {model_name}: {e}")
            return False, f"Network error: {str(e)}"
    
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
    
    async def generate_text(self, prompt: str, api_key: str, chat_history: List[Dict] = None, model_override: str = None) -> Tuple[bool, str]:
        """
        Generate text using advanced language models
        
        Args:
            prompt (str): User prompt
            api_key (str): Hugging Face API key
            chat_history (list): Previous conversation history
            model_override (str): Optional model override
            
        Returns:
            Tuple[bool, str]: (success, generated_text)
        """
        chat_history = chat_history or []
        model_name = model_override or Config.DEFAULT_TEXT_MODEL
        
        # Format the prompt with chat history
        formatted_prompt = self._format_chat_history(chat_history, prompt)
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "return_full_text": False
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '').strip()
            
            # Clean up the response
            if generated_text.startswith('### Assistant:'):
                generated_text = generated_text.replace('### Assistant:', '').strip()
            
            # Limit response length
            if len(generated_text) > Config.MAX_RESPONSE_LENGTH:
                generated_text = generated_text[:Config.MAX_RESPONSE_LENGTH] + "..."
            
            return True, generated_text
        
        # Try fallback model if primary fails
        if not success and model_name != Config.FALLBACK_TEXT_MODEL:
            logger.info(f"Trying fallback model {Config.FALLBACK_TEXT_MODEL}")
            return await self.generate_text(prompt, api_key, chat_history, Config.FALLBACK_TEXT_MODEL)
        
        return False, result if isinstance(result, str) else "Failed to generate text."
    
    async def generate_code(self, prompt: str, api_key: str, language: str = "python") -> Tuple[bool, str]:
        """
        Generate code using specialized code generation models
        
        Args:
            prompt (str): Code generation prompt
            api_key (str): Hugging Face API key
            language (str): Programming language
            
        Returns:
            Tuple[bool, str]: (success, generated_code)
        """
        model_name = Config.DEFAULT_CODE_MODEL
        
        # Enhance prompt for better code generation
        enhanced_prompt = f"# Generate {language} code for the following request:\n# {prompt}\n\n"
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "max_new_tokens": 800,
                "temperature": 0.3,
                "do_sample": True,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "return_full_text": False
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list) and len(result) > 0:
            generated_code = result[0].get('generated_text', '').strip()
            
            # Format code response
            if not generated_code.startswith('```'):
                generated_code = f"```{language}\n{generated_code}\n```"
            
            return True, generated_code
        
        return False, result if isinstance(result, str) else "Failed to generate code."
    
    async def generate_image(self, prompt: str, api_key: str) -> Tuple[bool, bytes]:
        """
        Generate image using text-to-image models
        
        Args:
            prompt (str): Image generation prompt
            api_key (str): Hugging Face API key
            
        Returns:
            Tuple[bool, bytes]: (success, image_bytes)
        """
        model_name = Config.DEFAULT_IMAGE_MODEL
        
        # Enhance prompt for better image generation
        enhanced_prompt = f"{prompt}, high quality, detailed, professional"
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
                "width": 1024,
                "height": 1024
            }
        }
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, bytes):
            return True, result
        
        return False, b""
    
    async def analyze_sentiment(self, text: str, api_key: str) -> Tuple[bool, Dict]:
        """
        Analyze sentiment of text
        
        Args:
            text (str): Text to analyze
            api_key (str): Hugging Face API key
            
        Returns:
            Tuple[bool, Dict]: (success, sentiment_data)
        """
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        payload = {"inputs": text}
        
        success, result = await self._make_api_call(model_name, payload, api_key)
        
        if success and isinstance(result, list):
            return True, result[0] if result else {}
        
        return False, {}

# Global model caller instance
model_caller = ModelCaller()