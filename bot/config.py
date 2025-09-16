"""
Configuration management for the AI Assistant Telegram Bot
Handles environment variables and secure configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for environment variables"""
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    OWNER_ID = os.getenv('OWNER_ID')
    
    # Database Configuration
    MONGO_URI = os.getenv('MONGO_URI')
    
    # Latest 2024-2025 Hugging Face Models - State-of-the-Art Performance
    DEFAULT_TEXT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Latest Llama 3.2 - Superior performance
    ADVANCED_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Multilingual powerhouse
    DEFAULT_CODE_MODEL = "bigcode/starcoder2-15b"  # Latest StarCoder2 - Best coding model
    FALLBACK_CODE_MODEL = "codellama/CodeLlama-13b-Instruct-hf"  # Enhanced CodeLlama
    DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"  # FLUX.1 - State-of-the-art 2024
    FALLBACK_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # Reliable fallback
    DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"  # Advanced emotion detection
    FALLBACK_TEXT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Still excellent fallback
    
    # Bot Configuration
    MAX_CHAT_HISTORY = 15
    MAX_RESPONSE_LENGTH = 4000
    REQUEST_TIMEOUT = 120  # Increased for latest models
    
    # Performance Settings - Optimized for 2024-2025 models
    MAX_RETRIES = 4  # More retries for better reliability
    RETRY_DELAY = 3  # Slightly longer delay for model loading
    
    # Model Quality Settings
    USE_ADVANCED_ROUTING = True  # Enable sophisticated model selection
    ENABLE_MODEL_FALLBACKS = True  # Smart fallback system
    PERFORMANCE_MONITORING = True  # Track model performance
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set"""
        required_vars = ['TELEGRAM_BOT_TOKEN', 'MONGO_URI']
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # OWNER_ID is optional - log if not provided
        if not cls.OWNER_ID:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("OWNER_ID not provided - admin features will be disabled")
        
        return True