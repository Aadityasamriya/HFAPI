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
    
    # 2025 Hugging Face Models - Optimized for Free Tier Accessibility
    # Text Generation Models - Reliable, always-available models
    DEFAULT_TEXT_MODEL = "gpt2"  # Classic, always accessible
    ADVANCED_TEXT_MODEL = "distilgpt2"  # Fast and reliable
    FALLBACK_TEXT_MODEL = "microsoft/DialoGPT-medium"  # Conversational fallback
    
    # Code Generation Models - Free tier accessible code models  
    DEFAULT_CODE_MODEL = "microsoft/CodeBERT-base"  # Code understanding and completion
    FALLBACK_CODE_MODEL = "codeparrot/codeparrot-small"  # Lightweight code generation
    CODE_INSTRUCT_MODEL = "bigcode/starcoder2-3b"  # More advanced when available
    
    # Image Generation Models - Classic stable diffusion models
    DEFAULT_IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"  # Most reliable SD model
    FALLBACK_IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"  # SD 2.1 fallback
    ALTERNATIVE_IMAGE_MODEL = "CompVis/stable-diffusion-v1-4"  # Original SD fallback
    
    # Analysis Models - Always-available classification models
    DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # Always works
    EMOTION_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Robust sentiment
    SIMPLE_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"  # Simple fallback
    
    # Bot Configuration
    MAX_CHAT_HISTORY = 15
    MAX_RESPONSE_LENGTH = 4000
    REQUEST_TIMEOUT = 180  # Increased for complex 2024-2025 model inference
    API_RETRY_TIMEOUT = 30  # Individual API call timeout
    MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent API calls for stability
    
    # Performance Settings - Optimized for 2024-2025 models
    MAX_RETRIES = 5  # Enhanced retry logic for model loading  
    RETRY_DELAY = 2  # Optimized delay for faster recovery
    EXPONENTIAL_BACKOFF = True  # Smart backoff strategy
    
    # Quality and Safety Settings
    ENABLE_CONTENT_FILTERING = True  # Built-in content safety
    USE_ADVANCED_PROMPTING = True  # Enhanced prompt engineering
    ENABLE_SMART_CACHING = True  # Performance optimization
    
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
            raise ValueError(f"CRITICAL: Missing required environment variables: {', '.join(missing_vars)}")
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate Telegram bot token format
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_BOT_TOKEN.count(':') == 1:
            logger.warning("Telegram bot token format may be invalid")
        
        # Validate MongoDB URI format  
        if cls.MONGO_URI and not (cls.MONGO_URI.startswith('mongodb://') or cls.MONGO_URI.startswith('mongodb+srv://')):
            raise ValueError("MONGO_URI must be a valid MongoDB connection string")
        
        # Security validation for production
        import os
        is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
        if is_production:
            if cls.MONGO_URI and not ('tls=true' in cls.MONGO_URI or cls.MONGO_URI.startswith('mongodb+srv://')):
                logger.error("SECURITY WARNING: Production database should use TLS encryption")
            if not os.getenv('ENCRYPTION_KEY'):
                logger.error("SECURITY WARNING: ENCRYPTION_KEY not set in production")
        
        # Optional settings warnings
        if not cls.OWNER_ID:
            logger.info("OWNER_ID not provided - admin features will be disabled")
        
        logger.info("✅ Configuration validation completed successfully")
        logger.info(f"🤖 Using {len([m for m in dir(cls) if 'MODEL' in m and not m.startswith('_')])} AI models")
        
        return True