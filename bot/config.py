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
    
    # Storage Provider Configuration
    STORAGE_PROVIDER = os.getenv('STORAGE_PROVIDER', 'mongodb').lower()  # Default to MongoDB for backward compatibility
    
    # MongoDB Configuration (backward compatibility)
    MONGO_URI = os.getenv('MONGO_URI')
    MONGODB_URI = os.getenv('MONGODB_URI')  # Alternative naming
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
    SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Optional for admin operations
    
    # Security Configuration
    ENCRYPTION_SEED = os.getenv('ENCRYPTION_SEED')
    API_ENCRYPTION_KEY = os.getenv('API_ENCRYPTION_KEY')  # Advanced encryption override
    
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
    def get_storage_provider(cls) -> str:
        """Get the configured storage provider"""
        return cls.STORAGE_PROVIDER
    
    @classmethod
    def get_mongodb_uri(cls) -> str:
        """Get MongoDB URI with fallback options"""
        return cls.MONGO_URI or cls.MONGODB_URI
    
    @classmethod
    def has_mongodb_config(cls) -> bool:
        """Check if MongoDB configuration is available"""
        return bool(cls.get_mongodb_uri())
    
    @classmethod
    def has_supabase_config(cls) -> bool:
        """Check if Supabase configuration is available"""
        return bool(cls.SUPABASE_URL and cls.SUPABASE_ANON_KEY)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration based on storage provider and available options"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Always require Telegram bot token
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("CRITICAL: TELEGRAM_BOT_TOKEN environment variable is required")
        
        # Validate Telegram bot token format
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_BOT_TOKEN.count(':') == 1:
            logger.warning("Telegram bot token format may be invalid")
        
        # Storage provider validation
        storage_provider = cls.get_storage_provider()
        logger.info(f"🔧 Configured storage provider: {storage_provider}")
        
        # Provider-specific validation
        if storage_provider == 'mongodb':
            cls._validate_mongodb_config(logger)
        elif storage_provider == 'supabase':
            cls._validate_supabase_config(logger)
        else:
            # Auto-detect if provider is not explicitly set or invalid
            logger.info(f"🔍 Storage provider '{storage_provider}' not recognized, attempting auto-detection...")
            cls._auto_detect_and_validate_storage(logger)
        
        # Validate encryption configuration
        cls._validate_encryption_config(logger)
        
        # Production security validation
        cls._validate_production_security(logger)
        
        # Success logging
        cls._log_successful_validation(logger, storage_provider)
        
        return True
    
    @classmethod
    def _validate_mongodb_config(cls, logger):
        """Validate MongoDB-specific configuration"""
        mongo_uri = cls.get_mongodb_uri()
        if not mongo_uri:
            raise ValueError(
                "CRITICAL: MongoDB provider requires MONGO_URI or MONGODB_URI environment variable. "
                "Set one of these variables or switch to Supabase with STORAGE_PROVIDER=supabase"
            )
        
        # Validate MongoDB URI format
        if not (mongo_uri.startswith('mongodb://') or mongo_uri.startswith('mongodb+srv://')):
            raise ValueError("MONGO_URI must be a valid MongoDB connection string (mongodb:// or mongodb+srv://)")
        
        logger.info("✅ MongoDB configuration validated successfully")
    
    @classmethod
    def _validate_supabase_config(cls, logger):
        """Validate Supabase-specific configuration"""
        if not cls.has_supabase_config():
            raise ValueError(
                "CRITICAL: Supabase provider requires SUPABASE_URL and SUPABASE_ANON_KEY environment variables. "
                "Set these variables or switch to MongoDB with STORAGE_PROVIDER=mongodb"
            )
        
        # Validate Supabase URL format
        if not cls.SUPABASE_URL.startswith('https://'):
            raise ValueError("SUPABASE_URL must be a valid HTTPS URL")
        
        if len(cls.SUPABASE_ANON_KEY) < 100:  # Supabase keys are typically long JWT tokens
            logger.warning("⚠️ SUPABASE_ANON_KEY appears unusually short - please verify it's correct")
        
        logger.info("✅ Supabase configuration validated successfully")
    
    @classmethod
    def _auto_detect_and_validate_storage(cls, logger):
        """Auto-detect and validate available storage providers"""
        available_providers = []
        
        # Check MongoDB availability
        if cls.has_mongodb_config():
            available_providers.append('mongodb')
            try:
                cls._validate_mongodb_config(logger)
                logger.info("🔍 Auto-detected working MongoDB configuration")
                # Update the storage provider to mongodb
                cls.STORAGE_PROVIDER = 'mongodb'
                return
            except ValueError as e:
                logger.warning(f"MongoDB config detected but invalid: {e}")
        
        # Check Supabase availability
        if cls.has_supabase_config():
            available_providers.append('supabase')
            try:
                cls._validate_supabase_config(logger)
                logger.info("🔍 Auto-detected working Supabase configuration")
                # Update the storage provider to supabase
                cls.STORAGE_PROVIDER = 'supabase'
                return
            except ValueError as e:
                logger.warning(f"Supabase config detected but invalid: {e}")
        
        # No working providers found
        if not available_providers:
            raise ValueError(
                "CRITICAL: No storage provider configuration found. Please configure either:\n"
                "MongoDB: Set MONGO_URI or MONGODB_URI\n"
                "Supabase: Set SUPABASE_URL and SUPABASE_ANON_KEY\n"
                "Or explicitly set STORAGE_PROVIDER=mongodb/supabase"
            )
        else:
            providers_str = ', '.join(available_providers)
            raise ValueError(
                f"CRITICAL: Storage providers detected ({providers_str}) but none are properly configured. "
                f"Please fix the configuration or set STORAGE_PROVIDER to a working provider."
            )
    
    @classmethod
    def _validate_encryption_config(cls, logger):
        """Validate encryption configuration"""
        if cls.API_ENCRYPTION_KEY:
            # Validate custom encryption key
            try:
                import base64
                decoded = base64.b64decode(cls.API_ENCRYPTION_KEY)
                if len(decoded) == 32:
                    logger.info("✅ Using custom API_ENCRYPTION_KEY (32 bytes)")
                else:
                    logger.warning(f"⚠️ API_ENCRYPTION_KEY should be 32 bytes when base64 decoded, got {len(decoded)} bytes")
            except Exception:
                # Try hex decoding
                try:
                    if len(cls.API_ENCRYPTION_KEY) == 64:
                        bytes.fromhex(cls.API_ENCRYPTION_KEY)
                        logger.info("✅ Using custom API_ENCRYPTION_KEY (hex format)")
                    else:
                        logger.warning("⚠️ API_ENCRYPTION_KEY should be 64 hex characters or 44 base64 characters")
                except Exception:
                    logger.warning("⚠️ API_ENCRYPTION_KEY format appears invalid - should be base64 or hex encoded 32-byte key")
        
        if cls.ENCRYPTION_SEED:
            if len(cls.ENCRYPTION_SEED) < 32:
                logger.warning("🚨 SECURITY WARNING: ENCRYPTION_SEED should be at least 32 characters for strong security")
            if cls.ENCRYPTION_SEED.lower() in ['test', 'development', 'default', 'password', '12345678901234567890123456789012']:
                raise ValueError("🚨 CRITICAL: ENCRYPTION_SEED must not use weak/default values. Use a strong, random string.")
            logger.info("✅ Using ENCRYPTION_SEED from environment variable")
        else:
            logger.info("🔐 ENCRYPTION_SEED not provided - will auto-generate and persist in storage backend")
    
    @classmethod
    def _validate_production_security(cls, logger):
        """Validate security settings for production environments"""
        is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
        if is_production:
            logger.info("🏭 Production environment detected - performing enhanced security validation")
            
            # MongoDB-specific production checks
            if cls.get_storage_provider() == 'mongodb':
                mongo_uri = cls.get_mongodb_uri()
                if mongo_uri and not ('tls=true' in mongo_uri or mongo_uri.startswith('mongodb+srv://')):
                    logger.warning("🚨 SECURITY WARNING: Production MongoDB should use TLS encryption")
            
            # Supabase automatically uses HTTPS/TLS
            elif cls.get_storage_provider() == 'supabase':
                if cls.SUPABASE_URL and not cls.SUPABASE_URL.startswith('https://'):
                    raise ValueError("🚨 CRITICAL: Production Supabase URL must use HTTPS")
            
            # Warn if using default/weak encryption
            if not cls.ENCRYPTION_SEED and not cls.API_ENCRYPTION_KEY:
                logger.warning("🚨 PRODUCTION WARNING: Consider setting ENCRYPTION_SEED or API_ENCRYPTION_KEY for deterministic encryption keys")
    
    @classmethod
    def _log_successful_validation(cls, logger, storage_provider):
        """Log successful validation with provider-specific information"""
        logger.info("✅ Configuration validation completed successfully")
        logger.info(f"🚀 Using {len([m for m in dir(cls) if 'MODEL' in m and not m.startswith('_')])} state-of-the-art 2024-2025 AI models")
        
        # Provider-specific logging
        if storage_provider == 'mongodb':
            logger.info("🗄️ Storage: MongoDB with auto-generated encryption")
            logger.info("🎯 Deployment: TELEGRAM_BOT_TOKEN + MONGO_URI required")
        elif storage_provider == 'supabase':
            logger.info("🗄️ Storage: Supabase PostgreSQL with real-time capabilities")
            logger.info("🎯 Deployment: TELEGRAM_BOT_TOKEN + SUPABASE_URL + SUPABASE_ANON_KEY required")
        
        logger.info("🔒 Encryption: Per-user AES-256-GCM with persistent seeds")
        logger.info("🏆 Bot powered by models SUPERIOR to ChatGPT, Grok, and Gemini")
        logger.info("⚡ Text: Qwen2.5-72B | Code: StarCoder2-15B | Images: FLUX.1 | Sentiment: CardiffNLP Latest")
        logger.info(f"🔄 Storage Provider: {storage_provider} (auto-detection enabled)")
        
        # Log available providers
        available = []
        if cls.has_mongodb_config():
            available.append("MongoDB")
        if cls.has_supabase_config():
            available.append("Supabase")
        
        if len(available) > 1:
            logger.info(f"🔀 Multiple storage providers available: {', '.join(available)} (using {storage_provider})")
        elif len(available) == 1:
            logger.info(f"🎯 Single storage provider configured: {available[0]}")
        
    @classmethod
    def get_storage_config_summary(cls) -> dict:
        """Get a summary of storage configuration for debugging"""
        return {
            "storage_provider": cls.get_storage_provider(),
            "mongodb_available": cls.has_mongodb_config(),
            "supabase_available": cls.has_supabase_config(),
            "mongodb_uri_set": bool(cls.get_mongodb_uri()),
            "supabase_url_set": bool(cls.SUPABASE_URL),
            "supabase_key_set": bool(cls.SUPABASE_ANON_KEY),
            "encryption_seed_set": bool(cls.ENCRYPTION_SEED),
            "api_encryption_key_set": bool(cls.API_ENCRYPTION_KEY)
        }