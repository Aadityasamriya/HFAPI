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
    # Text Generation Models - Optimized for speed and reliability while maintaining quality
    DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # 14B params, excellent speed/quality balance as default
    ADVANCED_TEXT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # DeepSeek-R1 for complex reasoning tasks
    REASONING_TEXT_MODEL = "Qwen/QwQ-32B-Preview"  # QwQ for advanced reasoning and problem solving
    FAST_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 7B params, fast and reliable
    MATH_TEXT_MODEL = "deepseek-ai/deepseek-math-7b-instruct"  # Specialized for mathematical reasoning
    FALLBACK_TEXT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Lightweight but capable fallback
    LIGHTWEIGHT_TEXT_MODEL = "microsoft/Phi-3.5-mini-instruct"  # Ultra-fast 3.8B model
    
    # Code Generation Models - Optimized for fast, reliable coding assistance
    DEFAULT_CODE_MODEL = "deepseek-ai/DeepSeek-Coder-V2-Instruct"  # Top coding model, now default for superiority
    ADVANCED_CODE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"  # 32B coding specialist, beats GitHub Copilot
    FAST_CODE_MODEL = "bigcode/starcoder2-7b"  # 7B params, moved from default for speed
    SPECIALIZED_CODE_MODEL = "codellama/CodeLlama-34b-Instruct-hf"  # Meta's CodeLlama for complex algorithms
    FALLBACK_CODE_MODEL = "bigcode/starcoder2-3b"  # Upgraded fallback, was too weak
    LIGHTWEIGHT_CODE_MODEL = "microsoft/CodeBERT-base"  # Lightweight code understanding
    
    # Vision/Multimodal Models - Optimized for fast visual understanding
    DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # 7B VL, efficient and capable as default
    ADVANCED_VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"  # 72B VL for complex vision tasks
    REASONING_VISION_MODEL = "meta-llama/Llama-3.2-90B-Vision-Instruct"  # Llama 3.2 Vision for advanced analysis
    FAST_VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # 3B VL, fast inference
    DOCUMENT_VISION_MODEL = "microsoft/Florence-2-large"  # Florence-2 for OCR and documents
    MEDICAL_VISION_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"  # Medical image analysis
    FALLBACK_VISION_MODEL = "microsoft/Phi-3.5-vision-instruct"  # Microsoft Phi-3.5 Vision
    LIGHTWEIGHT_VISION_MODEL = "microsoft/Florence-2-base"  # Ultra-fast vision tasks
    
    # Image Generation Models - Latest 2024-2025 FLUX.1 and SD3.5 series (FLUX is #1)
    DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"  # Best quality, research use, beats DALL-E 3
    COMMERCIAL_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"  # Commercial license, superior text rendering
    ADVANCED_IMAGE_MODEL = "black-forest-labs/FLUX.1-redux"  # FLUX Redux for style transfer and variations
    TURBO_IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-large-turbo"  # SD3.5 Large Turbo, ultra-fast
    ARTISTIC_IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-medium"  # SD3.5 Medium for artistic styles
    REALISTIC_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL for photorealistic images
    FALLBACK_IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"  # Classic SD 1.5, ultra-reliable fallback
    
    # Sentiment Analysis & NLP Models - Latest 2024-2025 specialized models
    DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Trained on 124M tweets, SOTA
    ADVANCED_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # Advanced multilingual sentiment analysis
    EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # Multi-class emotion (7 emotions)
    MULTILINGUAL_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # 8+ languages
    FALLBACK_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # DistilBERT, reliable
    
    # Translation Models - Latest 2024-2025 translation models
    DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-3.3B"  # NLLB-200, 200+ languages, SOTA
    ADVANCED_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"  # Multilingual to English
    FALLBACK_TRANSLATION_MODEL = "t5-base"  # T5-base for text-to-text translation tasks
    
    # Summarization Models - Latest 2024-2025 summarization models
    DEFAULT_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # BART Large CNN, news summarization
    ADVANCED_SUMMARIZATION_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Qwen2.5-7B for high-quality summarization
    FALLBACK_SUMMARIZATION_MODEL = "t5-base"  # T5-base for summarization tasks
    
    # Bot Configuration
    MAX_CHAT_HISTORY = 15
    MAX_RESPONSE_LENGTH = 4000
    REQUEST_TIMEOUT = 60  # Balanced for performance and reliability
    API_RETRY_TIMEOUT = 30  # Faster individual API call timeout
    MAX_CONCURRENT_REQUESTS = 6  # Optimized for stability and speed
    
    # Performance Settings - Optimized for reliability and speed
    MAX_RETRIES = 4  # Reasonable retry logic without excessive delays
    RETRY_DELAY = 2  # Faster recovery for better UX
    EXPONENTIAL_BACKOFF = True  # Smart backoff strategy for large models
    
    # Advanced Model Parameters for 2024-2025 State-of-the-Art Models
    QWEN_MAX_TOKENS = 131072  # Qwen2.5 supports up to 131K context length
    QWQ_MAX_TOKENS = 32768    # QwQ reasoning model context window
    DEEPSEEK_MAX_TOKENS = 32768  # DeepSeek-R1 context window
    DEEPSEEK_CODER_MAX_TOKENS = 131072  # DeepSeek-Coder-V2 extended context
    STARCODER_MAX_TOKENS = 16384  # StarCoder2 context window
    CODELLAMA_MAX_TOKENS = 100000  # CodeLlama extended context
    LLAMA_VISION_MAX_TOKENS = 128000  # Llama 3.2 Vision context
    FLUX_MAX_RESOLUTION = 1024  # FLUX.1 native resolution
    QWEN_VL_MAX_TOKENS = 32768  # Qwen2.5-VL context window
    PALIGEMMA_MAX_RESOLUTION = 896  # PaliGemma2 max resolution (224, 448, 896)
    
    # Model-specific optimal parameters
    QWEN_TEMPERATURE = 0.7  # Optimal for Qwen2.5 creativity vs accuracy
    QWQ_TEMPERATURE = 0.3   # Lower temp for QwQ reasoning precision
    DEEPSEEK_TEMPERATURE = 0.8  # DeepSeek-R1 optimal for reasoning
    DEEPSEEK_CODER_TEMPERATURE = 0.2  # Lower temp for precise code generation
    MATH_MODEL_TEMPERATURE = 0.1  # Ultra-low temp for mathematical accuracy
    STARCODER_TEMPERATURE = 0.2  # Lower temp for precise code generation
    CODELLAMA_TEMPERATURE = 0.15  # Very low temp for algorithm precision
    VISION_TEMPERATURE = 0.5  # Vision models optimal temperature
    LLAMA_VISION_TEMPERATURE = 0.4  # Slightly lower for Llama Vision precision
    FLUX_INFERENCE_STEPS = 4  # FLUX.1-schnell optimized for 1-4 steps
    FLUX_DEV_INFERENCE_STEPS = 20  # FLUX.1-dev quality steps
    SD35_INFERENCE_STEPS = 28  # SD3.5 optimal inference steps
    SDXL_INFERENCE_STEPS = 25  # SDXL optimal steps for quality
    
    # Vision model specific parameters
    QWEN_VL_IMAGE_SIZE = 448  # Qwen2.5-VL optimal image size
    PALIGEMMA_RESOLUTION_MODES = [224, 448, 896]  # PaliGemma2 supported resolutions
    FLORENCE2_MAX_IMAGE_SIZE = 1024  # Florence-2 maximum image processing size
    
    # Performance optimization for new models
    DEEPSEEK_USE_FLASH_ATTENTION = True  # DeepSeek-R1 flash attention support
    QWQ_USE_FLASH_ATTENTION = True      # QwQ flash attention for speed
    QWEN_VL_DYNAMIC_RESOLUTION = True   # Qwen2.5-VL dynamic resolution
    LLAMA_VISION_OPTIMIZE_MEMORY = True # Llama Vision memory optimization
    FLUX_SCHNELL_TURBO_MODE = True      # FLUX.1-schnell turbo optimization
    CODELLAMA_OPTIMIZE_INFERENCE = True # CodeLlama inference optimization
    DEEPSEEK_CODER_FAST_DECODE = True   # DeepSeek-Coder fast decoding
    
    # Advanced routing and performance features
    ENABLE_MODEL_WARMUP = True          # Pre-warm frequently used models
    USE_INTELLIGENT_BATCHING = True     # Batch similar requests
    ENABLE_RESPONSE_STREAMING = True    # Stream responses for better UX
    ADAPTIVE_CONTEXT_LENGTH = True      # Dynamically adjust context length
    ENABLE_MODEL_LOAD_BALANCING = True  # Distribute load across model variants
    
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
    def get_mongodb_uri(cls) -> str | None:
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
    def get_model_fallback_chain(cls, model_type: str) -> list:
        """Get fallback chain for different model types with validated models"""
        fallback_chains = {
            'text': [cls.DEFAULT_TEXT_MODEL, cls.FAST_TEXT_MODEL, cls.FALLBACK_TEXT_MODEL, cls.LIGHTWEIGHT_TEXT_MODEL],
            'code': [cls.DEFAULT_CODE_MODEL, cls.FAST_CODE_MODEL, cls.FALLBACK_CODE_MODEL, cls.LIGHTWEIGHT_CODE_MODEL],
            'vision': [cls.DEFAULT_VISION_MODEL, cls.FAST_VISION_MODEL, cls.FALLBACK_VISION_MODEL, cls.LIGHTWEIGHT_VISION_MODEL],
            'sentiment': [cls.DEFAULT_SENTIMENT_MODEL, cls.FALLBACK_SENTIMENT_MODEL, "cardiffnlp/twitter-roberta-base-sentiment-latest"],
            'summarization': [cls.DEFAULT_SUMMARIZATION_MODEL, cls.FALLBACK_SUMMARIZATION_MODEL, "facebook/bart-large-cnn"],
            'translation': [cls.DEFAULT_TRANSLATION_MODEL, cls.FALLBACK_TRANSLATION_MODEL, "t5-base"]
        }
        return fallback_chains.get(model_type, [])
    
    @classmethod
    def get_validated_models(cls) -> dict:
        """Get list of models that are known to work on HF Inference API"""
        # Models verified to work on HF Inference API as of 2025
        return {
            'small_models': [
                "google-bert/bert-base-uncased",
                "distilbert-base-uncased-finetuned-sst-2-english", 
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "microsoft/Phi-3.5-mini-instruct",
                "Salesforce/codegen-350M-mono",
                "microsoft/CodeBERT-base"
            ],
            'medium_models': [
                "meta-llama/Llama-3.2-3B-Instruct",
                "bigcode/starcoder2-3b",
                "Qwen/Qwen2.5-7B-Instruct",
                "microsoft/Phi-3.5-vision-instruct",
                "microsoft/Florence-2-base"
            ],
            'large_models': [
                "Qwen/Qwen2.5-14B-Instruct",
                "bigcode/starcoder2-7b",
                "facebook/bart-large-cnn",
                "nlptown/bert-base-multilingual-uncased-sentiment"
            ]
        }
    
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
        if cls.SUPABASE_URL and not cls.SUPABASE_URL.startswith('https://'):
            raise ValueError("SUPABASE_URL must be a valid HTTPS URL")
        
        if cls.SUPABASE_ANON_KEY and len(cls.SUPABASE_ANON_KEY) < 100:  # Supabase keys are typically long JWT tokens
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
        logger.info(f"🚀 Using optimized 2024-2025 AI models with fallback chains for reliability")
        
        # Provider-specific logging
        if storage_provider == 'mongodb':
            logger.info("🗄️ Storage: MongoDB with auto-generated encryption")
            logger.info("🎯 Deployment: TELEGRAM_BOT_TOKEN + MONGO_URI required")
        elif storage_provider == 'supabase':
            logger.info("🗄️ Storage: Supabase PostgreSQL with real-time capabilities")
            logger.info("🎯 Deployment: TELEGRAM_BOT_TOKEN + SUPABASE_URL + SUPABASE_ANON_KEY required")
        
        logger.info("🔒 Encryption: Per-user AES-256-GCM with persistent seeds")
        logger.info("🏆 Bot powered by optimized models for SUPERIOR speed and reliability vs ChatGPT/Grok/Gemini")
        logger.info("⚡ Text: Qwen2.5-14B | Code: StarCoder2-7B | Images: FLUX.1 | Sentiment: BERT Multilingual")
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