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
    
    # Hugging Face Model Configuration
    DEFAULT_TEXT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    DEFAULT_CODE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
    DEFAULT_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    FALLBACK_TEXT_MODEL = "microsoft/DialoGPT-large"
    
    # Bot Configuration
    MAX_CHAT_HISTORY = 15
    MAX_RESPONSE_LENGTH = 4000
    REQUEST_TIMEOUT = 60
    
    # Performance Settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    @classmethod
    def validate_config(cls):
        """Validate that all required environment variables are set"""
        required_vars = ['TELEGRAM_BOT_TOKEN', 'OWNER_ID', 'MONGO_URI']
        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True