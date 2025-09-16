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
    
    # Database Configuration
    MONGO_URI = os.getenv('MONGO_URI')
    
    # 2024-2025 STATE-OF-THE-ART Hugging Face Models - SUPERIOR TO CHATGPT/GROK/GEMINI
    # Text Generation Models - Latest cutting-edge models outperforming GPT-4
    DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # 72B params, outperforms GPT-4, 131K context
    ADVANCED_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 7B params, excellent performance, fast
    FALLBACK_TEXT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Latest Llama 3.2, reliable fallback
    
    # Code Generation Models - Latest 2024-2025 StarCoder2 series (600+ languages)
    DEFAULT_CODE_MODEL = "bigcode/starcoder2-15b"  # 15B params, matches CodeLlama-34B performance
    ADVANCED_CODE_MODEL = "bigcode/starcoder2-7b"  # 7B params, excellent balance
    FALLBACK_CODE_MODEL = "bigcode/starcoder2-3b"  # 3B params, lightweight but powerful
    
    # Image Generation Models - Latest 2024-2025 FLUX.1 and SD3 series
    DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"  # 12B params, commercial license, superior text rendering
    ADVANCED_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"  # Best quality, research use, 12B params
    FALLBACK_IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-large"  # Latest SD 3.5, excellent quality
    
    # Analysis Models - Latest 2024-2025 sentiment and emotion detection models
    DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Trained on 124M tweets, state-of-the-art
    ADVANCED_SENTIMENT_MODEL = "tabularisai/multilingual-sentiment-analysis"  # 2025 model, 48% improvement over existing
    EMOTION_MODEL = "cardiffnlp/twitter-roberta-base-emotion"  # Multi-class emotion detection (joy, anger, fear, etc)
    FALLBACK_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Multilingual, 8+ languages
    
    # Bot Configuration
    MAX_CHAT_HISTORY = 15
    MAX_RESPONSE_LENGTH = 4000
    REQUEST_TIMEOUT = 300  # Increased for large 72B+ model inference
    API_RETRY_TIMEOUT = 60  # Individual API call timeout for complex models
    MAX_CONCURRENT_REQUESTS = 8  # Optimized for large model stability
    
    # Performance Settings - Optimized for 2024-2025 large models (72B+ parameters)
    MAX_RETRIES = 7  # Enhanced retry logic for large model loading  
    RETRY_DELAY = 3  # Optimized delay for 72B+ model recovery
    EXPONENTIAL_BACKOFF = True  # Smart backoff strategy for large models
    
    # Advanced Model Parameters for 2024-2025 State-of-the-Art Models
    QWEN_MAX_TOKENS = 131072  # Qwen2.5 supports up to 131K context length
    STARCODER_MAX_TOKENS = 16384  # StarCoder2 context window
    FLUX_MAX_RESOLUTION = 1024  # FLUX.1 native resolution
    
    # Model-specific optimal parameters
    QWEN_TEMPERATURE = 0.7  # Optimal for Qwen2.5 creativity vs accuracy
    STARCODER_TEMPERATURE = 0.2  # Lower temp for precise code generation
    FLUX_INFERENCE_STEPS = 4  # FLUX.1-schnell optimized for 1-4 steps
    
    # Advanced Features for Superior Performance
    USE_CONTEXT_OPTIMIZATION = True  # Leverage long context capabilities
    ENABLE_MULTIMODAL_ROUTING = True  # Smart routing for different model types
    USE_ADAPTIVE_PARAMETERS = True  # Dynamic parameter adjustment
    
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
                logger.warning("SECURITY WARNING: Production database should use TLS encryption")
        
        logger.info("✅ Configuration validation completed successfully")
        logger.info(f"🚀 Using {len([m for m in dir(cls) if 'MODEL' in m and not m.startswith('_')])} state-of-the-art 2024-2025 AI models")
        logger.info("💡 API keys are now session-based - no persistent storage required")
        logger.info("🏆 Bot powered by models SUPERIOR to ChatGPT, Grok, and Gemini")
        logger.info("⚡ Text: Qwen2.5-72B | Code: StarCoder2-15B | Images: FLUX.1 | Sentiment: CardiffNLP Latest")
        
        return True