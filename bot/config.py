"""
Configuration management for the AI Assistant Telegram Bot
Handles environment variables and secure configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Configuration class for environment variables
    
    REQUIRED VARIABLES for deployment:
    - TELEGRAM_BOT_TOKEN: Bot token from @BotFather
    - MONGODB_URI: MongoDB connection string
    
    OPTIONAL VARIABLES:
    - BOT_NAME: Custom bot name (default: AI Assistant Pro)
    - OWNER_ID: Telegram user ID for admin features
    - ENCRYPTION_SEED: Custom encryption seed (auto-generated if not provided)
    """
    
    # ===== REQUIRED CONFIGURATION =====
    # These MUST be set for the bot to function
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Database Configuration - MongoDB (legacy) or Supabase (new)
    MONGODB_URI = os.getenv('MONGODB_URI') or os.getenv('MONGO_URI')  # Support both variable names
    
    # Supabase Configuration for Individual User Databases
    SUPABASE_MGMT_URL = os.getenv('SUPABASE_MGMT_URL') or os.getenv('DATABASE_URL')  # Management database URL
    SUPABASE_USER_BASE_URL = os.getenv('SUPABASE_USER_BASE_URL')  # Base URL for user databases (optional)
    
    # ===== CORE BOT CONFIGURATION =====
    BOT_NAME = os.getenv('BOT_NAME', 'Hugging Face By AadityaLabs AI')
    BOT_DESCRIPTION = "Sophisticated Telegram Bot with Intelligent AI Routing"
    BOT_VERSION = "2025.1.0"
    
    # ===== SECURITY CONFIGURATION =====
    # Optional but recommended for production
    OWNER_ID = int(os.getenv('OWNER_ID', '0')) if os.getenv('OWNER_ID', '').isdigit() else None
    ENCRYPTION_SEED = os.getenv('ENCRYPTION_SEED')  # Auto-generated if not provided
    
    # ===== PERFORMANCE TUNING =====
    # These have sensible defaults but can be customized
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))
    MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '20'))
    MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', '4000'))
    
    # ===== AI MODEL PARAMETERS =====
    # Optional fine-tuning parameters for advanced users
    
    # 2025 STATE-OF-THE-ART Hugging Face Models - SUPERIOR TO CHATGPT/GROK/GEMINI
    # Text Generation Models - Latest breakthrough releases with enhanced capabilities
    DEFAULT_TEXT_MODEL = "deepseek-ai/DeepSeek-R1-0528"  # BREAKTHROUGH: 90.2% math performance, surpasses GPT-4
    FLAGSHIP_TEXT_MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"  # Enhanced thinking capabilities, Jan 2025 release
    ADVANCED_TEXT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # Flagship MoE, competitive with O1/O3-mini
    REASONING_TEXT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 32B distilled R1 with production-ready performance
    EFFICIENT_TEXT_MODEL = "Qwen/Qwen3-32B"  # Latest Qwen3 series, superior architecture
    FAST_TEXT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B"  # 8B R1 distilled for fast inference
    COMPACT_TEXT_MODEL = "Qwen/Qwen3-14B"  # Latest Qwen3 14B with Apache 2.0 license
    MATH_TEXT_MODEL = "deepseek-ai/deepseek-math-7b-instruct"  # Specialized for mathematical reasoning
    FALLBACK_TEXT_MODEL = "Qwen/Qwen3-8B"  # Qwen3-8B as primary fallback
    LIGHTWEIGHT_TEXT_MODEL = "Qwen/Qwen3-4B"  # Qwen3-4B rivals Qwen2.5-72B performance
    
    # Code Generation Models - Latest 2025 coding specialists outperforming GPT-4
    DEFAULT_CODE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Currently best open-source, matches GPT-4o
    ADVANCED_CODE_MODEL = "deepseek-ai/DeepSeek-Coder-V2-Instruct"  # Top coding model for complex algorithms
    TOOL_USE_CODE_MODEL = "Groq/Llama-3-Groq-70B-Tool-Use"  # BREAKTHROUGH: #1 on BFCL, 90.76% accuracy
    FAST_CODE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"  # Balanced performance for general coding
    SPECIALIZED_CODE_MODEL = "codellama/CodeLlama-34b-Instruct-hf"  # Meta's CodeLlama for complex algorithms
    EFFICIENT_CODE_MODEL = "Groq/Llama-3-Groq-8B-Tool-Use"  # 8B tool use model, 89.06% BFCL accuracy
    FALLBACK_CODE_MODEL = "bigcode/starcoder2-7b"  # Reliable StarCoder2 fallback
    LIGHTWEIGHT_CODE_MODEL = "bigcode/starcoder2-3b"  # Fast code completion
    
    # Vision/Multimodal Models - Revolutionary visual understanding with latest breakthroughs
    DEFAULT_VISION_MODEL = "openbmb/MiniCPM-Llama3-V-2_5"  # BREAKTHROUGH: GPT-4V level performance, 700+ OCRBench
    ADVANCED_VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"  # 72B VL for complex vision tasks
    REASONING_VISION_MODEL = "meta-llama/Llama-3.2-90B-Vision-Instruct"  # Llama 3.2 Vision for advanced analysis
    GUI_AUTOMATION_MODEL = "ByteDance-Seed/UI-TARS-7B-DPO"  # BREAKTHROUGH: Native GUI agent, outperforms GPT-4V
    FAST_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"  # 7B VL, efficient and capable
    DOCUMENT_VISION_MODEL = "microsoft/Florence-2-large"  # Florence-2 for OCR and documents
    MEDICAL_VISION_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"  # Medical image analysis
    FALLBACK_VISION_MODEL = "microsoft/Phi-3.5-vision-instruct"  # Microsoft Phi-3.5 Vision
    LIGHTWEIGHT_VISION_MODEL = "openbmb/MiniCPM-Llama3-V-2_5-int4"  # Quantized for 9GB GPU memory
    
    # Image Generation Models - 2025 Breakthrough models outperforming DALL-E 3
    DEFAULT_IMAGE_MODEL = "tencent/HunyuanImage-2.1"  # BREAKTHROUGH: Ultra-high res (2048x2048), RLHF enhanced
    FLAGSHIP_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"  # Superior text rendering, beats DALL-E 3
    COMMERCIAL_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"  # Commercial license, 1-4 steps generation
    ADVANCED_IMAGE_MODEL = "hidreamai/HiDream-I1"  # BREAKTHROUGH: 17B params, outperforms FLUX & DALL-E 3
    EDITING_IMAGE_MODEL = "Qwen/Qwen-Image-Edit"  # Specialized editing with dual-encoding
    PROFESSIONAL_IMAGE_MODEL = "Qwen/Qwen-Image"  # 20B params, superior text rendering and Chinese support
    KONTEXT_IMAGE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"  # In-context editing and control
    TURBO_IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-large-turbo"  # SD3.5 Large Turbo, ultra-fast
    ARTISTIC_IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-medium"  # SD3.5 Medium for artistic styles
    REALISTIC_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL for photorealistic images
    FALLBACK_IMAGE_MODEL = "black-forest-labs/FLUX.1-redux"  # FLUX Redux for style transfer and variations
    
    # Sentiment Analysis & NLP Models - Latest 2024-2025 specialized models
    DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Trained on 124M tweets, SOTA
    ADVANCED_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # Advanced multilingual sentiment analysis
    EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # Multi-class emotion (7 emotions)
    MULTILINGUAL_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # 8+ languages
    FALLBACK_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # DistilBERT, reliable
    
    # Translation Models - Optimized for accuracy and speed
    DEFAULT_TRANSLATION_MODEL = "google/madlad400-3b-mt"  # 3B params, 400+ languages
    ADVANCED_TRANSLATION_MODEL = "google/madlad400-10b-mt"  # 10B params for complex translations
    FAST_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"  # Fast multilingual to English
    SPECIALIZED_TRANSLATION_MODEL = "facebook/m2m100_1.2B"  # Many-to-many translation
    FALLBACK_TRANSLATION_MODEL = "t5-base"  # T5 base for basic translation tasks
    
    # Question Answering Models - For complex Q&A tasks
    DEFAULT_QA_MODEL = "deepset/roberta-base-squad2"  # RoBERTa trained on SQuAD 2.0
    ADVANCED_QA_MODEL = "microsoft/DialoGPT-medium"  # Conversational QA
    FACTUAL_QA_MODEL = "facebook/dpr-question_encoder-single-nq-base"  # Dense passage retrieval
    FALLBACK_QA_MODEL = "distilbert-base-cased-distilled-squad"  # DistilBERT for QA
    
    # Summarization Models - For document and text summarization
    DEFAULT_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # BART for CNN/Daily Mail summaries
    ADVANCED_SUMMARIZATION_MODEL = "google/pegasus-large"  # Pegasus for abstractive summaries
    FAST_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # DistilBART for speed
    FALLBACK_SUMMARIZATION_MODEL = "t5-small"  # T5 small for basic summaries
    
    # Text Classification Models - For content categorization
    DEFAULT_CLASSIFICATION_MODEL = "microsoft/DialoGPT-medium"  # General classification
    TOPIC_CLASSIFICATION_MODEL = "cardiffnlp/tweet-topic-21-multi"  # Topic classification
    INTENT_CLASSIFICATION_MODEL = "facebook/bart-large-mnli"  # Intent classification via NLI
    FALLBACK_CLASSIFICATION_MODEL = "distilbert-base-uncased"  # DistilBERT for classification
    
    # Audio Processing Models (for future voice features)
    DEFAULT_ASR_MODEL = "openai/whisper-small"  # Whisper Small for ASR
    ADVANCED_ASR_MODEL = "openai/whisper-large-v3"  # Whisper Large v3 for high-quality ASR
    FAST_ASR_MODEL = "openai/whisper-tiny"  # Whisper Tiny for fast ASR
    
    # Named Entity Recognition Models
    DEFAULT_NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"  # BERT NER
    MULTILINGUAL_NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Multilingual NER
    FALLBACK_NER_MODEL = "distilbert-base-cased"  # DistilBERT NER
    
    # 2025 BREAKTHROUGH SPECIALIZED MODELS - Superior task-specific performance
    # GUI Automation Models - Revolutionary automation capabilities
    DEFAULT_GUI_MODEL = "ByteDance-Seed/UI-TARS-7B-DPO"  # BREAKTHROUGH: Native GUI agent, DPO trained
    ADVANCED_GUI_MODEL = "ByteDance-Seed/UI-TARS-72B-DPO"  # Large scale GUI automation
    LIGHTWEIGHT_GUI_MODEL = "ByteDance-Seed/UI-TARS-2B-SFT"  # Lightweight GUI tasks
    
    # Tool Use & Function Calling Models - Best-in-class function calling
    DEFAULT_TOOL_MODEL = "Groq/Llama-3-Groq-70B-Tool-Use"  # #1 on BFCL leaderboard (90.76%)
    EFFICIENT_TOOL_MODEL = "Groq/Llama-3-Groq-8B-Tool-Use"  # 8B model with 89.06% BFCL accuracy
    
    # Enhanced Vision Models - Latest breakthroughs in visual understanding
    PREMIUM_VISION_MODEL = "openbmb/MiniCPM-Llama3-V-2_5"  # Beats GPT-4V, 700+ OCRBench score
    QUANTIZED_VISION_MODEL = "openbmb/MiniCPM-Llama3-V-2_5-int4"  # 9GB GPU memory optimized
    
    # 2025 Model Performance Optimizations
    # Enable specific optimizations for each model type
    QWEN_FLASH_ATTENTION = True             # Qwen models flash attention
    DEEPSEEK_OPTIMIZED_INFERENCE = True     # DeepSeek models optimized inference
    STARCODER_FAST_GENERATION = True        # StarCoder fast code generation
    FLUX_SCHNELL_ACCELERATION = True        # FLUX.1-schnell acceleration
    LLAMA_VISION_FAST_PROCESSING = True     # Llama Vision fast processing
    BERT_SENTIMENT_BATCH_OPT = True         # BERT sentiment batch optimization
    T5_TRANSLATION_STREAMING = True         # T5 translation streaming
    BART_SUMMARIZATION_CACHING = True       # BART summarization caching
    WHISPER_AUDIO_OPTIMIZATION = True       # Whisper audio processing optimization
    
    # 2025 Specific Model Optimizations
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
    
    # 2025: MODEL-SPECIFIC TEMPERATURE AND PARAMETER SETTINGS
    # Optimized temperature settings for different model types
    DEEPSEEK_TEMPERATURE = 0.8          # DeepSeek-R1 reasoning optimal temp
    QWEN_TEMPERATURE = 0.7               # Qwen3 series optimal temp
    STARCODER_TEMPERATURE = 0.2          # StarCoder2 precise code generation
    VISION_TEMPERATURE = 0.5             # Vision models optimal temp
    
    # Model-specific token limits
    DEEPSEEK_MAX_TOKENS = 2500           # DeepSeek-R1 extended reasoning
    QWEN_MAX_TOKENS = 2000               # Qwen3 series standard
    CODE_MAX_TOKENS = 1500               # Coding models optimal
    
    # DeepSeek optimizations
    DEEPSEEK_USE_FLASH_ATTENTION = True  # Enable flash attention for DeepSeek
    
    # Vision model settings
    QWEN_VL_IMAGE_SIZE = 448             # Qwen2.5-VL optimal image size
    VISION_MAX_RESOLUTION = 1024         # Standard vision resolution
    
    # Image generation settings
    FLUX_INFERENCE_STEPS = 4             # FLUX.1-schnell optimal steps
    FLUX_MAX_RESOLUTION = 1024           # FLUX optimal resolution
    SD_INFERENCE_STEPS = 20              # Stable Diffusion steps
    
    # Performance optimization flags
    ENABLE_BATCH_PROCESSING = True       # Batch multiple requests
    USE_MODEL_CACHING = True             # Cache frequently used models
    OPTIMIZE_MEMORY_USAGE = True         # Memory optimization
    
    @classmethod
    def get_mongodb_uri(cls) -> str | None:
        """Get MongoDB URI with fallback options"""
        return cls.MONGODB_URI
    
    @classmethod
    def get_supabase_mgmt_url(cls) -> str | None:
        """Get Supabase management database URL with validation"""
        return cls.SUPABASE_MGMT_URL
    
    @classmethod
    def get_supabase_user_base_url(cls) -> str | None:
        """Get Supabase user base database URL, defaults to management URL"""
        return cls.SUPABASE_USER_BASE_URL or cls.SUPABASE_MGMT_URL
    
    @classmethod
    def has_supabase_config(cls) -> bool:
        """Check if Supabase configuration is available"""
        return bool(cls.SUPABASE_MGMT_URL)
    
    @classmethod
    def has_mongodb_config(cls) -> bool:
        """Check if MongoDB configuration is available"""
        return bool(cls.MONGODB_URI)
    
    @classmethod
    def get_preferred_storage_provider(cls) -> str:
        """
        Get preferred storage provider based on available configuration
        
        Returns:
            str: 'supabase' or 'mongodb' based on available config
        """
        if cls.has_supabase_config():
            return 'supabase'
        elif cls.has_mongodb_config():
            return 'mongodb'
        else:
            return 'supabase'  # Default to supabase for new deployments
    
    @classmethod
    def get_model_fallback_chain(cls, model_type: str) -> list:
        """Enhanced fallback chains with latest 2025 models for optimal performance"""
        fallback_chains = {
            'text': [cls.DEFAULT_TEXT_MODEL, cls.FLAGSHIP_TEXT_MODEL, cls.REASONING_TEXT_MODEL, cls.EFFICIENT_TEXT_MODEL, cls.FAST_TEXT_MODEL],
            'reasoning': [cls.DEFAULT_TEXT_MODEL, cls.FLAGSHIP_TEXT_MODEL, cls.REASONING_TEXT_MODEL, cls.ADVANCED_TEXT_MODEL],
            'code': [cls.DEFAULT_CODE_MODEL, cls.ADVANCED_CODE_MODEL, cls.TOOL_USE_CODE_MODEL, cls.FAST_CODE_MODEL, cls.FALLBACK_CODE_MODEL],
            'tool_use': [cls.DEFAULT_TOOL_MODEL, cls.EFFICIENT_TOOL_MODEL, cls.TOOL_USE_CODE_MODEL],
            'vision': [cls.DEFAULT_VISION_MODEL, cls.PREMIUM_VISION_MODEL, cls.ADVANCED_VISION_MODEL, cls.FAST_VISION_MODEL, cls.FALLBACK_VISION_MODEL],
            'gui_automation': [cls.DEFAULT_GUI_MODEL, cls.ADVANCED_GUI_MODEL, cls.LIGHTWEIGHT_GUI_MODEL],
            'image_generation': [cls.DEFAULT_IMAGE_MODEL, cls.FLAGSHIP_IMAGE_MODEL, cls.ADVANCED_IMAGE_MODEL, cls.PROFESSIONAL_IMAGE_MODEL, cls.COMMERCIAL_IMAGE_MODEL],
            'sentiment': [cls.DEFAULT_SENTIMENT_MODEL, cls.ADVANCED_SENTIMENT_MODEL, cls.FALLBACK_SENTIMENT_MODEL],
            'summarization': [cls.DEFAULT_SUMMARIZATION_MODEL, cls.ADVANCED_SUMMARIZATION_MODEL, cls.FALLBACK_SUMMARIZATION_MODEL],
            'translation': [cls.DEFAULT_TRANSLATION_MODEL, cls.ADVANCED_TRANSLATION_MODEL, cls.FALLBACK_TRANSLATION_MODEL]
        }
        return fallback_chains.get(model_type, [])
    
    @classmethod
    def get_model_temperature(cls, model_name: str) -> float:
        """Get optimal temperature for specific model"""
        if 'deepseek' in model_name.lower():
            return cls.DEEPSEEK_TEMPERATURE
        elif 'qwen' in model_name.lower():
            return cls.QWEN_TEMPERATURE
        elif 'starcoder' in model_name.lower() or 'coder' in model_name.lower():
            return cls.STARCODER_TEMPERATURE
        elif any(vision_term in model_name.lower() for vision_term in ['vision', 'vl', 'florence', 'minicp']):
            return cls.VISION_TEMPERATURE
        else:
            return 0.7  # Default temperature
    
    @classmethod
    def get_model_max_tokens(cls, model_name: str) -> int:
        """Get optimal max tokens for specific model"""
        if 'deepseek' in model_name.lower():
            return cls.DEEPSEEK_MAX_TOKENS
        elif 'qwen' in model_name.lower():
            return cls.QWEN_MAX_TOKENS
        elif any(code_term in model_name.lower() for code_term in ['coder', 'starcoder', 'code']):
            return cls.CODE_MAX_TOKENS
        else:
            return 1500  # Default max tokens
    
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
        """Validate basic configuration requirements"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Always require Telegram bot token
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("CRITICAL: TELEGRAM_BOT_TOKEN environment variable is required")
        
        # Validate Telegram bot token format
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_BOT_TOKEN.count(':') == 1:
            from bot.core.model_caller import SecureLogger
            SecureLogger(logger).warning("Telegram bot token format may be invalid")
        
        # Database validation - require either MongoDB or Supabase configuration
        mongo_uri = cls.get_mongodb_uri()
        supabase_url = cls.get_supabase_mgmt_url()
        
        if not mongo_uri and not supabase_url:
            raise ValueError(
                "CRITICAL: Database configuration is required. "
                "Set either MONGODB_URI or SUPABASE_MGMT_URL environment variable."
            )
        
        # Validate MongoDB URI format if provided
        if mongo_uri:
            if not (mongo_uri.startswith('mongodb://') or mongo_uri.startswith('mongodb+srv://')):
                raise ValueError("MONGODB_URI must be a valid MongoDB connection string (mongodb:// or mongodb+srv://)")
            logger.info("✅ MongoDB configuration detected")
        
        # Validate Supabase URL format if provided
        if supabase_url:
            if not (supabase_url.startswith('postgresql://') or supabase_url.startswith('postgres://')):
                raise ValueError("SUPABASE_MGMT_URL must be a valid PostgreSQL connection string (postgresql:// or postgres://)")
            logger.info("✅ Supabase configuration detected")
        
        # Log preferred provider
        preferred_provider = cls.get_preferred_storage_provider()
        logger.info(f"🎯 Preferred storage provider: {preferred_provider}")
        
        logger.info("✅ Configuration validated successfully")
        return True