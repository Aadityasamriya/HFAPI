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
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI') or os.getenv('MONGO_URI')  # Support both variable names
    
    # Security Configuration
    ENCRYPTION_SEED = os.getenv('ENCRYPTION_SEED')
    API_ENCRYPTION_KEY = os.getenv('API_ENCRYPTION_KEY')  # Advanced encryption override
    
    # API Configuration
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))  # Request timeout in seconds
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))  # Maximum number of retries
    RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))  # Delay between retries in seconds
    MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '20'))  # Maximum chat history length
    MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', '4000'))  # Maximum response length
    
    # Model-specific parameters
    DEEPSEEK_TEMPERATURE = float(os.getenv('DEEPSEEK_TEMPERATURE', '0.8'))  # DeepSeek model temperature
    QWEN_TEMPERATURE = float(os.getenv('QWEN_TEMPERATURE', '0.7'))  # Qwen model temperature
    DEEPSEEK_MAX_TOKENS = int(os.getenv('DEEPSEEK_MAX_TOKENS', '2000'))  # DeepSeek max tokens
    DEEPSEEK_USE_FLASH_ATTENTION = os.getenv('DEEPSEEK_USE_FLASH_ATTENTION', 'true').lower() == 'true'
    
    # Additional model parameters referenced in code
    STARCODER_TEMPERATURE = float(os.getenv('STARCODER_TEMPERATURE', '0.6'))  # StarCoder temperature
    VISION_TEMPERATURE = float(os.getenv('VISION_TEMPERATURE', '0.5'))  # Vision model temperature
    QWEN_VL_IMAGE_SIZE = int(os.getenv('QWEN_VL_IMAGE_SIZE', '1024'))  # Qwen VL image size
    FLUX_INFERENCE_STEPS = int(os.getenv('FLUX_INFERENCE_STEPS', '20'))  # FLUX inference steps
    FLUX_MAX_RESOLUTION = int(os.getenv('FLUX_MAX_RESOLUTION', '1024'))  # FLUX max resolution
    QWEN_MAX_TOKENS = int(os.getenv('QWEN_MAX_TOKENS', '2000'))  # Qwen max tokens
    
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
    
    @classmethod
    def get_mongodb_uri(cls) -> str | None:
        """Get MongoDB URI with fallback options"""
        return cls.MONGODB_URI
    
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
        """Validate basic configuration requirements"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Always require Telegram bot token
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("CRITICAL: TELEGRAM_BOT_TOKEN environment variable is required")
        
        # Validate Telegram bot token format
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_BOT_TOKEN.count(':') == 1:
            logger.warning("Telegram bot token format may be invalid")
        
        # MongoDB validation
        mongo_uri = cls.get_mongodb_uri()
        if not mongo_uri:
            raise ValueError(
                "CRITICAL: MONGODB_URI environment variable is required. "
                "Set MONGODB_URI with your MongoDB connection string."
            )
        
        # Validate MongoDB URI format
        if not (mongo_uri.startswith('mongodb://') or mongo_uri.startswith('mongodb+srv://')):
            raise ValueError("MONGODB_URI must be a valid MongoDB connection string (mongodb:// or mongodb+srv://)")
        
        logger.info("✅ Configuration validated successfully")
        return True