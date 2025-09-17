"""
Intelligent AI model routing system
Analyzes user prompts to determine the best Hugging Face model to use
"""

import re
import logging
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """2025 Enhanced intent types for model routing"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"          # 2025: New intent
    DOCUMENT_PROCESSING = "document_processing"  # 2025: New intent
    MULTI_MODAL = "multi_modal"              # 2025: New intent
    CONVERSATION = "conversation"            # 2025: New intent
    PDF_PROCESSING = "pdf_processing"        # 2025: New P1 feature - PDF analysis
    ZIP_ANALYSIS = "zip_analysis"            # 2025: New P1 feature - ZIP file analysis  
    IMAGE_ANALYSIS = "image_analysis"        # 2025: New P1 feature - Image content analysis
    FILE_GENERATION = "file_generation"      # 2025: New P1 feature - File generation and delivery

class IntelligentRouter:
    """
    Advanced AI model router with intent detection
    Analyzes user prompts and selects optimal models
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.intent_priorities = self._initialize_priorities()
        # 2025: Expanded programming languages and frameworks
        self.programming_languages = {
            # Core languages
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'php', 
            'ruby', 'go', 'rust', 'swift', 'kotlin', 'dart', 'scala', 'r',
            'sql', 'html', 'css', 'bash', 'shell', 'powershell', 'lua', 'perl',
            # 2025: New languages
            'zig', 'v', 'nim', 'crystal', 'elixir', 'haskell', 'ocaml', 'f#',
            'julia', 'fortran', 'cobol', 'assembly', 'wasm', 'webassembly',
            # Frameworks & libraries  
            'react', 'vue', 'angular', 'svelte', 'node', 'deno', 'bun',
            'django', 'flask', 'fastapi', 'laravel', 'spring', 'express',
            'next', 'nuxt', 'gatsby', 'remix', 'astro', 'solid', 'qwik',
            # Mobile & platforms
            'android', 'ios', 'flutter', 'react-native', 'xamarin', 'ionic',
            # 2025: AI/ML frameworks
            'pytorch', 'tensorflow', 'jax', 'huggingface', 'transformers',
            'langchain', 'llamaindex', 'openai', 'anthropic', 'cohere'
        }
        self.intent_cache = {}  # 2025: Cache for performance
        self.complexity_weights = {
            'technical_terms': 2.0,
            'code_snippets': 3.0,
            'specific_frameworks': 2.5,
            'creative_words': 1.5,
            'emotional_words': 1.8
        }
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize regex patterns for intent detection"""
        return {
            # 2025: Enhanced image generation patterns with latest AI models
            IntentType.IMAGE_GENERATION: [
                r'\b(?:draw|paint|sketch|illustrate|create|generate|make|design|show|visualize|render|produce)\s+(?:a|an|some)?\s*(?:beautiful|stunning|amazing|gorgeous|lovely|pretty|professional|artistic|detailed|realistic|abstract|minimalist|modern|vintage|futuristic)?\s*(?:sunset|sunrise|landscape|portrait|scene|picture|image|drawing|artwork|illustration|photo|graphic|visual|logo|icon|banner|poster|wallpaper|avatar|meme|infographic|diagram|chart|mockup)',
                r'\b(?:create|generate|make|design|produce|craft|build)\s+(?:a|an|some)?\s*(?:image|picture|photo|artwork|drawing|illustration|graphic|logo|icon|visual|art|design|mockup|prototype|concept)',
                r'\b(?:show|visualize|picture|display|demonstrate)\s+(?:me)?\s*(?:a|an|some)?\s*(?:image|picture|visual|artwork|design|mockup|example)',
                r'\b(?:draw|paint|sketch|illustrate|design)\b(?!.*(?:function|code|program|script|class|method|algorithm|database|query))',
                r'(?:can you|could you|please|i want|i need|help me).*(?:draw|paint|sketch|create|make|generate|design|visualize).*(?:image|picture|artwork|visual|art|graphic|logo|icon|design)',
                r'\b(?:logo|icon|banner|poster|wallpaper|avatar|art|artistic|visual|painting|drawing|sketch|illustration|graphic|artwork|photography|render|composition|mockup|wireframe|diagram|infographic|meme|sticker|emoji)\b',
                r'(?:dalle|midjourney|stable.?diffusion|text.?to.?image|flux|ai.?art|image.?generation|firefly|leonardo|runway)',
                r'(?:digital.?art|concept.?art|fantasy.?art|photorealistic|hyperrealistic|stylized|pixel.?art|vector.?art|3d.?render|ui.?mockup|app.?design)',
                # 2025: New patterns for modern image generation
                r'(?:design|create).*(?:ui|ux|interface|dashboard|website|app|mobile)',
                r'\b(?:meme|funny|cute|cartoon|anime|manga|realistic|portrait)\b.*(?:image|picture|art)',
                r'(?:product.?shot|brand.?identity|social.?media.?post|thumbnail|cover.?image)',
                r'(?:architectural|interior|fashion|food|nature|abstract).*(?:design|image|photo)',
            ],
            
            # 2025: Enhanced code generation patterns with latest frameworks
            IntentType.CODE_GENERATION: [
                r'\b(?:write|create|generate|code|program|implement|build|develop)\s+(?:a|an|some)?\s*(?:function|class|method|script|program|application|app|module|component|service|api|library|microservice|webhook|middleware|plugin|extension)',
                r'\b(?:python|javascript|java|c\+\+|typescript|php|ruby|go|rust|swift|kotlin|c#|dart|scala|r|zig|nim|elixir|haskell|julia)\s+(?:code|function|script|program|class|method|application|api)',
                r'\b(?:code|program|implement|write|build|create).*(?:algorithm|function|class|method|API|library|framework|module|component)',
                r'(?:how to|help me|show me|teach me|guide me|walk me through).*(?:code|program|implement|write|build|develop|deploy|setup|configure)',
                r'\b(?:debug|fix|error|bug|issue|optimize|refactor|improve|test|unit.?test|integration.?test).*(?:code|function|script|program|application)',
                r'\b(?:react|vue|angular|svelte|solid|qwik|django|flask|fastapi|laravel|spring|rails|express|nest|next|nuxt|gatsby|remix|astro)',
                r'(?:github|gitlab|stackoverflow|programming|coding|development|software|engineering|devops|ci.?cd)',
                r'\b(?:sql|database|query|table|schema|mongodb|postgresql|mysql|sqlite|redis|elasticsearch|graphql)',
                r'(?:rest.?api|graphql|microservices|web.?service|backend|full.?stack|serverless|lambda|cloud.?function)',
                r'(?:machine.?learning|ai|neural.?network|data.?science|automation|pytorch|tensorflow|huggingface|langchain)',
                # 2025: New patterns for modern development
                r'(?:docker|kubernetes|terraform|ansible|jenkins|github.?actions|aws|azure|gcp|vercel|netlify)',
                r'(?:websocket|realtime|streaming|event.?driven|pub.?sub|message.?queue|kafka|rabbitmq)',
                r'(?:blockchain|web3|smart.?contract|defi|nft|cryptocurrency|ethereum|solidity)',
                r'(?:mobile.?app|ios|android|flutter|react.?native|expo|xamarin|ionic)',
                r'(?:game.?dev|unity|unreal|godot|pygame|phaser|three.js|webgl)',
                r'(?:data.?pipeline|etl|batch.?processing|stream.?processing|data.?lake|data.?warehouse)',
            ],
            
            IntentType.SENTIMENT_ANALYSIS: [
                r'\b(?:what\'?s|what\s+is|analyze|check|determine|find)\s+(?:the\s+)?(?:sentiment|mood|tone|emotion|feeling)',
                r'(?:sentiment|emotion|feeling|mood|tone)\s+(?:of|analysis|detection|recognition)',
                r'(?:is this|how does this sound|what do you think about).*(?:positive|negative|neutral)',
                r'(?:analyze|check)\s+(?:this|the)?\s*(?:text|message|comment|review)',
                r'(?:positive|negative|neutral|happy|sad|angry|excited|disappointed)\s+(?:sentiment|feeling|emotion)',
                r'\b(?:sentiment)\b.*(?:of|about|in)\b',
            ],
            
            IntentType.CREATIVE_WRITING: [
                r'\b(?:write|create|compose|draft)\s+(?:a|an)?\s*(?:story|poem|song|lyrics|novel|tale|narrative)',
                r'\b(?:creative|fiction|poetry|storytelling|narrative|literature)',
                r'(?:once upon a time|tell me a story|write a story about)',
                r'\b(?:character|plot|dialogue|scene|chapter)',
                r'(?:rhyme|verse|stanza|ballad|sonnet|haiku)',
            ],
            
            IntentType.QUESTION_ANSWERING: [
                r'^\s*(?:what|who|when|where|why|how|which|whose)\s+(?:is|are|was|were|does|do|did|will|would|can|could|should)',
                r'\b(?:what\s+is)\s+(?:artificial\s+intelligence|machine\s+learning|deep\s+learning|AI|ML|computer\s+science|programming|technology)',
                r'(?:explain|tell me|describe|clarify|elaborate)\s+(?:what|how|why|the)',
                r'(?:question|answer|information|knowledge|facts)(?!.*(?:draw|create|generate|image|picture))',
                r'(?:do you know|can you tell me|help me understand)\s+(?:what|how|why)',
                r'\b(?:definition|meaning|concept)\s+(?:of|for)',
            ],
            
            IntentType.TRANSLATION: [
                r'\b(?:translate|translation|convert)\s+(?:this|the|from|to)',
                r'(?:from|to)\s+(?:english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|bengali|urdu|turkish|vietnamese|thai|dutch|swedish|norwegian|finnish|polish)',
                r'\b(?:language|linguistic|multilingual)',
                r'(?:what does.*mean in|how do you say.*in)',
            ],
            
            # 2025: New intent patterns
            IntentType.DATA_ANALYSIS: [
                r'\b(?:analyze|analysis|examine|study|investigate|process|parse)\s+(?:data|dataset|csv|excel|json|statistics|metrics|trends)',
                r'(?:data.?science|data.?analysis|statistical.?analysis|exploratory.?data.?analysis|eda)',
                r'\b(?:visualize|chart|graph|plot|dashboard|report)\s+(?:data|statistics|metrics|results)',
                r'(?:correlation|regression|clustering|classification|prediction|forecast)',
                r'\b(?:pandas|numpy|matplotlib|seaborn|plotly|tableau|powerbi)',
                r'(?:sql.?query|database.?analysis|data.?mining|business.?intelligence)',
            ],
            
            IntentType.DOCUMENT_PROCESSING: [
                r'\b(?:read|parse|extract|process|analyze)\s+(?:document|pdf|doc|docx|text|file|report)',
                r'(?:document.?analysis|text.?extraction|content.?parsing|file.?processing)',
                r'\b(?:summarize|summary|abstract|extract.?key.?points)\s+(?:from|this|document|text|article)',
                r'(?:ocr|optical.?character.?recognition|text.?from.?image)',
                r'\b(?:contract|invoice|receipt|form|application)\s+(?:processing|analysis|extraction)',
            ],
            
            IntentType.MULTI_MODAL: [
                r'(?:image.?and.?text|text.?and.?image|visual.?and.?text|multimodal|multi.?modal)',
                r'(?:describe.?this.?image|what.?do.?you.?see|analyze.?this.?picture)',
                r'(?:combine|merge|integrate)\s+(?:image|visual|text|code|data)',
                r'(?:vision.?language|vlm|visual.?question.?answering|image.?captioning)',
            ],
            
            IntentType.CONVERSATION: [
                r'(?:let\'s.?chat|casual.?conversation|just.?talking|friendly.?chat)',
                r'(?:how.?are.?you|what\'s.?up|tell.?me.?about.?yourself)',
                r'(?:opinion|thoughts|what.?do.?you.?think|personal.?view)',
                r'(?:conversation|chat|talk|discuss)(?!.*(?:code|image|analysis))',
            ],
            
            # 2025: P1 Features - Advanced file processing capabilities
            IntentType.PDF_PROCESSING: [
                r'\b(?:analyze|read|extract|process|parse|summarize)\s+(?:this\s+)?(?:pdf|document|file)\b',
                r'(?:pdf.?analysis|document.?processing|text.?extraction|content.?from.?pdf)',
                r'\b(?:tables?|charts?|graphs?)\s+(?:from|in)\s+(?:pdf|document)',
                r'(?:extract.?text|get.?content|read.?document|parse.?pdf)',
                r'\b(?:summarize|summary|key.?points|main.?ideas)\s+(?:from|of|in)\s+(?:this\s+)?(?:pdf|document|file)',
                r'(?:ocr|optical.?character.?recognition)\s+(?:pdf|document|scanned)',
                r'(?:metadata|properties|info)\s+(?:from|of|in)\s+(?:pdf|document)',
                r'(?:analyze|process)\s+(?:this\s+)?pdf\s+(?:document|file)?\s+(?:and|to)',
            ],
            
            IntentType.ZIP_ANALYSIS: [
                r'\b(?:analyze|examine|process|extract|explore)\s+(?:zip|archive|compressed)',
                r'(?:zip.?analysis|archive.?processing|compressed.?file|contents.?of.?zip)',
                r'\b(?:unzip|extract|decompress)\s+(?:and\s+)?(?:analyze|examine|process)',
                r'(?:file.?structure|directory.?listing|contents.?overview)',
                r'\b(?:scan|check|inspect)\s+(?:zip|archive)\s+(?:contents|files)',
                r'(?:bulk.?file.?analysis|multiple.?files.?processing)',
            ],
            
            IntentType.IMAGE_ANALYSIS: [
                r'\b(?:analyze|examine|describe|identify|recognize)\s+(?:image|photo|picture)',
                r'(?:image.?analysis|photo.?recognition|visual.?analysis|computer.?vision)',
                r'(?:what.?is.?in|what.?do.?you.?see|describe.?this)\s+(?:image|photo|picture)',
                r'\b(?:ocr|text.?recognition|read.?text)\s+(?:from|in)\s+(?:image|photo|picture)',
                r'(?:object.?detection|face.?recognition|scene.?analysis)',
                r'(?:extract.?text|get.?text|read)\s+(?:from|in)\s+(?:image|photo|screenshot)',
                r'\b(?:identify|classify|categorize)\s+(?:objects?|people|scenes?)\s+(?:in|from)\s+(?:image|photo)',
                r'(?:visual.?content|image.?content|picture.?content)\s+(?:analysis|recognition|description)',
            ],
            
            # 2025: P1 Feature - File Generation with AI
            IntentType.FILE_GENERATION: [
                r'\b(?:create|generate|make|build|produce|write)\s+(?:a|an|some)?\s*(?:file|script|document|config|configuration)',
                r'\b(?:create|generate|make|build|write)\s+(?:a|an|some)?\s*(?:python|py|javascript|js|json|txt|text|csv|xml|md|markdown)\s+(?:file|script|document)',
                r'(?:generate|create|make|build|produce)\s+(?:a|an|some)?\s*(?:\.py|\.js|\.json|\.txt|\.csv|\.xml|\.md)\s+(?:file|script)',
                r'\b(?:save|export|download|deliver)\s+(?:as|to|in)\s+(?:file|document|script)',
                r'(?:can you|could you|please|help me)\s+(?:create|generate|make|build|write)\s+(?:a|an|some)?\s*(?:file|script|document)',
                r'(?:output|save|export|write)\s+(?:this|that|the result|the response)\s+(?:to|as|in)\s+(?:a|an)?\s*(?:file|document|script)',
                r'\b(?:code|script|program)\s+(?:file|document)\s+(?:for|to|that|which)',
                r'(?:configuration|config|settings)\s+(?:file|document|json|xml|yaml)',
                r'(?:data|content|text|information)\s+(?:file|document|csv|json|txt)',
                r'(?:put|save|store|write|output)\s+(?:this|that|the|it)\s+(?:in|to|as)\s+(?:a|an)?\s*(?:file|document|script)',
                r'(?:send|give|provide|deliver)\s+(?:me|us)?\s*(?:a|an|some)?\s*(?:file|document|script|attachment)',
                r'(?:download|attachment|file.?download|document.?download)',
                # File format specific patterns
                r'(?:python.?script|py.?file|\.py\s+file)',
                r'(?:json.?file|json.?config|\.json\s+file|configuration.?json)',
                r'(?:text.?file|txt.?file|\.txt\s+file|plain.?text)',
                r'(?:markdown.?file|md.?file|\.md\s+file|documentation.?md)',
                r'(?:csv.?file|\.csv\s+file|comma.?separated|spreadsheet.?data)',
                r'(?:xml.?file|\.xml\s+file|xml.?document|xml.?config)',
            ]
        }
    
    def _initialize_priorities(self) -> Dict[IntentType, int]:
        """2025 Enhanced priority weights for intent types with better balance"""
        return {
            IntentType.FILE_GENERATION: 10,          # 2025: P1 High priority - file creation requests
            IntentType.PDF_PROCESSING: 10,           # 2025: P1 High priority - PDF file uploads
            IntentType.ZIP_ANALYSIS: 10,             # 2025: P1 High priority - ZIP file uploads
            IntentType.IMAGE_GENERATION: 9,         # High priority - specific visual requests
            IntentType.CODE_GENERATION: 8,          # High priority but not overpowering - very specific patterns
            IntentType.IMAGE_ANALYSIS: 8,           # 2025: P1 High priority - Image analysis
            IntentType.DATA_ANALYSIS: 8,            # 2025: High priority - data processing
            IntentType.SENTIMENT_ANALYSIS: 8,       # High priority - specific analysis requests
            IntentType.CREATIVE_WRITING: 7,         # Medium-high priority - creative requests
            IntentType.DOCUMENT_PROCESSING: 7,      # 2025: Medium-high priority - document tasks
            IntentType.MULTI_MODAL: 6,              # 2025: Medium priority - complex multimodal
            IntentType.TRANSLATION: 6,              # Medium priority - language requests
            IntentType.QUESTION_ANSWERING: 5,       # Medium priority - broad category
            IntentType.CONVERSATION: 4,             # 2025: Lower priority - conversational AI
            IntentType.TEXT_GENERATION: 3,          # General text generation fallback
        }
    
    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """
        Enhanced prompt complexity analysis for better model selection
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            Dict: Analysis results with complexity metrics
        """
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        analysis = {
            'length': len(prompt),
            'word_count': word_count,
            'has_code_blocks': bool(re.search(r'```|`[^`]+`', prompt)),
            'has_technical_terms': any(term in prompt_lower for term in list(self.programming_languages)),
            'has_questions': bool(re.search(r'\?', prompt)),
            'complexity_score': 0,
            'requires_context': False,
            'language_detected': self._detect_programming_language(prompt)
        }
        
        # Enhanced complexity scoring system
        complexity = 0
        
        # Base length score (more reasonable scaling)
        complexity += min(word_count / 10, 3)  # Up to 3 points for length
        
        # Technical complexity indicators (balanced weights)
        if analysis['has_code_blocks']:
            complexity += 2.5  # Reduced from 3
        if analysis['has_technical_terms']:
            complexity += 1.5  # Reduced from 2.5
        
        # Advanced complexity keywords (reduced scoring)
        advanced_keywords = ['complex', 'advanced', 'sophisticated', 'detailed', 'comprehensive', 
                           'algorithm', 'architecture', 'optimization', 'distributed', 'microservices',
                           'reasoning', 'philosophical', 'analysis', 'framework', 'principles']
        if any(keyword in prompt_lower for keyword in advanced_keywords):
            complexity += 2  # Reduced from 3 to 2
            
        # Creative complexity indicators
        creative_keywords = ['epic', 'novel', 'intricate', 'character development', 'plot twists',
                           'fantasy', 'creative', 'artistic', 'story', 'narrative']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            complexity += 2
            
        # Scientific/academic complexity (reduced scoring)
        academic_keywords = ['quantum', 'superconductivity', 'cooper pairs', 'principles',
                          'mechanical', 'theoretical', 'research', 'scientific']
        if any(keyword in prompt_lower for keyword in academic_keywords):
            complexity += 2.5  # Reduced from 4 to 2.5
        
        # Question complexity
        if analysis['has_questions']:
            complexity += 1.5
            
        # Multiple concepts/requirements
        if ' and ' in prompt_lower or ' with ' in prompt_lower:
            complexity += 1
            
        analysis['complexity_score'] = round(complexity, 1)
        
        # Context requirements
        context_indicators = ['this', 'that', 'previous', 'above', 'earlier', 'before', 'continue', 'also']
        analysis['requires_context'] = any(indicator in prompt_lower for indicator in context_indicators)
        
        return analysis
    
    def _detect_programming_language(self, prompt: str) -> str:
        """Detect programming language mentioned in prompt"""
        prompt_lower = prompt.lower()
        
        language_indicators = {
            'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular', 'express'],
            'typescript': ['typescript', 'ts'],
            'java': ['java', 'spring', 'android'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'dotnet', '.net'],
            'php': ['php', 'laravel', 'symfony'],
            'ruby': ['ruby', 'rails'],
            'go': ['golang', 'go'],
            'rust': ['rust'],
            'swift': ['swift', 'ios'],
            'kotlin': ['kotlin'],
            'sql': ['sql', 'database', 'query', 'mysql', 'postgresql', 'sqlite']
        }
        
        for language, indicators in language_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                return language
        
        return 'unknown'
    
    def route_prompt(self, prompt: str) -> Tuple[IntentType, Dict]:
        """
        Analyze prompt and determine the best intent and model to use
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            Tuple[IntentType, Dict]: (detected_intent, routing_info)
        """
        prompt_lower = prompt.lower()
        intent_scores = {}
        
        # Calculate weighted scores for each intent type
        for intent_type, patterns in self.intent_patterns.items():
            raw_score = 0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    raw_score += 1
                    matches.append(pattern)
            
            if raw_score > 0:
                # Apply priority weighting
                priority_weight = self.intent_priorities.get(intent_type, 1)
                weighted_score = raw_score * priority_weight
                
                intent_scores[intent_type] = {
                    'raw_score': raw_score,
                    'weighted_score': weighted_score,
                    'priority': priority_weight,
                    'matches': matches
                }
        
        # Get prompt analysis
        analysis = self.analyze_prompt_complexity(prompt)
        
        # Determine primary intent using weighted scores
        if intent_scores:
            # Use weighted score for primary intent selection
            primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['weighted_score'])
            confidence = intent_scores[primary_intent]['weighted_score']
        else:
            # Default to text generation if no specific intent detected
            primary_intent = IntentType.TEXT_GENERATION
            confidence = 0
        
        # Apply additional heuristics for edge cases
        primary_intent = self._apply_intent_heuristics(prompt, primary_intent, intent_scores, analysis)
        
        # Build routing information
        routing_info = {
            'primary_intent': primary_intent,
            'confidence': confidence,
            'all_intents': intent_scores,
            'analysis': analysis,
            'recommended_model': self._get_recommended_model(primary_intent, analysis, prompt),
            'special_parameters': self._get_special_parameters(primary_intent, analysis)
        }
        
        logger.info(f"Routed prompt to {primary_intent.value} with confidence {routing_info['confidence']}")
        
        return primary_intent, routing_info
    
    def _validate_model_selection(self, model_name: str, intent: IntentType) -> str:
        """Validate that the selected model exists in Config and return fallback if needed"""
        from bot.config import Config
        
        # List of all valid model names from Config
        valid_models = {
            # Text models
            Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL, Config.FAST_TEXT_MODEL, 
            Config.FALLBACK_TEXT_MODEL, Config.LIGHTWEIGHT_TEXT_MODEL,
            # Code models  
            Config.DEFAULT_CODE_MODEL, Config.ADVANCED_CODE_MODEL, Config.FAST_CODE_MODEL,
            Config.FALLBACK_CODE_MODEL, Config.LIGHTWEIGHT_CODE_MODEL,
            # Vision models
            Config.DEFAULT_VISION_MODEL, Config.ADVANCED_VISION_MODEL, Config.FAST_VISION_MODEL,
            Config.FALLBACK_VISION_MODEL, Config.DOCUMENT_VISION_MODEL, Config.LIGHTWEIGHT_VISION_MODEL,
            # Image generation models
            Config.DEFAULT_IMAGE_MODEL, Config.COMMERCIAL_IMAGE_MODEL, Config.ADVANCED_IMAGE_MODEL,
            Config.FALLBACK_IMAGE_MODEL, Config.ARTISTIC_IMAGE_MODEL,
            # Sentiment & NLP models
            Config.DEFAULT_SENTIMENT_MODEL, Config.ADVANCED_SENTIMENT_MODEL, Config.EMOTION_MODEL,
            Config.MULTILINGUAL_SENTIMENT_MODEL, Config.FALLBACK_SENTIMENT_MODEL,
            # Translation models
            Config.DEFAULT_TRANSLATION_MODEL, Config.ADVANCED_TRANSLATION_MODEL, Config.FALLBACK_TRANSLATION_MODEL
        }
        
        # Check if the selected model is valid
        if model_name in valid_models:
            return model_name
        
        # Fallback to appropriate model based on intent category  
        logger.warning(f"⚠️ MODEL_VALIDATION: {model_name} not found in Config, using fallback")
        
        fallback_map = {
            IntentType.CODE_GENERATION: Config.DEFAULT_CODE_MODEL,
            IntentType.IMAGE_GENERATION: Config.DEFAULT_IMAGE_MODEL,
            IntentType.IMAGE_ANALYSIS: Config.DEFAULT_VISION_MODEL,
            IntentType.MULTI_MODAL: Config.DEFAULT_VISION_MODEL,
            IntentType.SENTIMENT_ANALYSIS: Config.DEFAULT_SENTIMENT_MODEL,
            IntentType.TRANSLATION: Config.DEFAULT_TRANSLATION_MODEL,
        }
        
        return fallback_map.get(intent, Config.DEFAULT_TEXT_MODEL)
    
    def _get_recommended_model(self, intent: IntentType, analysis: Dict, original_prompt: str = "") -> str:
        """Get recommended model based on intent and analysis with 2024-2025 STATE-OF-THE-ART models"""
        from bot.config import Config
        
        complexity = analysis.get('complexity_score', 0)
        word_count = analysis.get('word_count', 0)
        has_technical = analysis.get('has_technical_terms', False)
        language = analysis.get('language_detected', 'unknown')
        
        # 🚀 ENHANCED ROUTER DECISION LOGGING - Proving Superior AI Selection
        logger.info(f"\n🎯 ═══════════════════════ INTELLIGENT ROUTER DECISION ═══════════════════════")
        logger.info(f"🔍 PROMPT_ANALYSIS: intent={intent.value.upper()}, complexity={complexity}/10")
        logger.info(f"📊 METRICS: words={word_count}, technical_terms={has_technical}, language={language}")
        logger.info(f"📝 PROMPT_SAMPLE: '{original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}'")
        
        # Advanced model selection based on complexity and context
        if intent == IntentType.TEXT_GENERATION:
            # DeepSeek-R1-Distill for complex reasoning, Qwen2.5 series for other tasks
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            if complexity > 7 or 'reasoning' in original_prompt.lower() or 'logic' in original_prompt.lower():
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for reasoning (matches o1)
            elif complexity > 5 or analysis.get('requires_context'):
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for balanced performance
            elif complexity > 3:
                selected_model = Config.FAST_TEXT_MODEL  # Qwen2.5-7B for general tasks
            else:
                selected_model = Config.FALLBACK_TEXT_MODEL  # Llama-3.2-3B for simple tasks
            
            # Enhanced decision logging for text generation
            reasoning = "ADVANCED (DeepSeek-R1)" if complexity > 7 else "DEFAULT (Qwen2.5-14B)" if complexity > 5 else "FAST (Qwen2.5-7B)" if complexity > 3 else "FALLBACK (Llama-3.2-3B)"
            logger.info(f"🤖 TEXT_GENERATION: complexity={complexity} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Superior to GPT-4/Claude/Gemini)")
            logger.info(f"⚡ REASONING: {'Complex reasoning task' if complexity > 7 else 'Balanced performance' if complexity > 5 else 'General task' if complexity > 3 else 'Simple query'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CODE_GENERATION:
            # DeepSeek-Coder-V2 beats GPT-4 in coding, with smart fallbacks
            complexity = analysis.get('complexity_score', 0)
            language = analysis.get('language_detected', 'python')
            
            # Use top coding models for complex tasks
            selected_model = None
            if complexity > 6 or language in ['rust', 'go', 'c++', 'java']:
                selected_model = Config.ADVANCED_CODE_MODEL  # DeepSeek-Coder-V2-Instruct (beats GPT-4)
            elif complexity > 4:
                selected_model = Config.DEFAULT_CODE_MODEL  # StarCoder2-7B for balanced tasks
            elif complexity > 2:
                selected_model = Config.FAST_CODE_MODEL  # StarCoder2-3B for simple code
            else:
                selected_model = Config.FALLBACK_CODE_MODEL  # CodeGen-350M for basic tasks
            
            # Enhanced decision logging for code generation
            reasoning = "ADVANCED (DeepSeek-Coder-V2)" if complexity > 6 else "DEFAULT (StarCoder2-7B)" if complexity > 4 else "FAST (StarCoder2-3B)" if complexity > 2 else "FALLBACK (CodeGen-350M)"
            logger.info(f"💻 CODE_GENERATION: complexity={complexity}, language={language} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Beats GitHub Copilot/ChatGPT Code)")
            logger.info(f"⚡ REASONING: {'Complex algorithms/architecture' if complexity > 6 else 'Standard development' if complexity > 4 else 'Simple scripts' if complexity > 2 else 'Basic code snippets'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.IMAGE_ANALYSIS:
            # Vision models for image understanding and analysis
            prompt_lower = original_prompt.lower()
            selected_model = None
            if 'document' in prompt_lower or 'text' in prompt_lower or 'ocr' in prompt_lower:
                selected_model = Config.DOCUMENT_VISION_MODEL  # Florence-2 for OCR/documents
            elif 'complex' in prompt_lower or 'detailed' in prompt_lower:
                selected_model = Config.ADVANCED_VISION_MODEL  # Qwen2.5-VL-72B for complex vision
            elif 'fast' in prompt_lower or 'quick' in prompt_lower:
                selected_model = Config.FAST_VISION_MODEL  # Qwen2.5-VL-3B for fast inference
            else:
                selected_model = Config.DEFAULT_VISION_MODEL  # Qwen2.5-VL-7B (balanced)
            
            # Enhanced decision logging for image analysis
            is_document = 'document' in prompt_lower or 'text' in prompt_lower or 'ocr' in prompt_lower
            is_complex = 'complex' in prompt_lower or 'detailed' in prompt_lower
            reasoning = "DOCUMENT (Florence-2)" if is_document else "ADVANCED (Qwen2.5-VL-72B)" if is_complex else "DEFAULT (Qwen2.5-VL-7B)"
            logger.info(f"👁️ IMAGE_ANALYSIS: document={is_document}, complex={is_complex} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Beats GPT-4V/Claude Sonnet Vision)")
            logger.info(f"⚡ REASONING: {'OCR/Document processing' if is_document else 'Complex visual analysis' if is_complex else 'Standard image understanding'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.MULTI_MODAL:
            # Multimodal tasks combining text and vision
            selected_model = Config.ADVANCED_VISION_MODEL  # Qwen2.5-VL-72B (best multimodal)
            logger.info(f"🔄 MULTI_MODAL: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.IMAGE_GENERATION:
            # FLUX.1 models (SUPERIOR to DALL-E 3, Midjourney competitors)
            prompt_lower = original_prompt.lower()
            selected_model = None
            if 'commercial' in prompt_lower or 'business' in prompt_lower:
                selected_model = Config.COMMERCIAL_IMAGE_MODEL  # FLUX.1-schnell (commercial license)
            elif 'artistic' in prompt_lower or 'creative' in prompt_lower:
                selected_model = Config.ARTISTIC_IMAGE_MODEL  # Playground V2.5 (artistic style)
            elif 'fast' in prompt_lower or 'quick' in prompt_lower:
                selected_model = Config.ADVANCED_IMAGE_MODEL  # SD3.5-Large-Turbo (fast)
            else:
                selected_model = Config.DEFAULT_IMAGE_MODEL  # FLUX.1-dev (best quality)
            
            # Enhanced decision logging for image generation
            is_commercial = 'commercial' in prompt_lower or 'business' in prompt_lower
            is_artistic = 'artistic' in prompt_lower or 'creative' in prompt_lower
            reasoning = "COMMERCIAL (FLUX.1-schnell)" if is_commercial else "ARTISTIC (Playground V2.5)" if is_artistic else "DEFAULT (FLUX.1-dev)"
            logger.info(f"🎨 IMAGE_GENERATION: commercial={is_commercial}, artistic={is_artistic} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Superior to DALL-E 3/Midjourney/Firefly)")
            logger.info(f"⚡ REASONING: {'Business/commercial use' if is_commercial else 'Artistic/creative style' if is_artistic else 'Best quality research model'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            # Enhanced sentiment analysis with emotion detection
            prompt_lower = original_prompt.lower()
            emotion_keywords = ['emotion', 'feeling', 'mood', 'angry', 'happy', 'sad', 'excited', 'fear', 'joy', 'love']
            
            selected_model = None
            if any(keyword in prompt_lower for keyword in emotion_keywords):
                selected_model = Config.EMOTION_MODEL  # Multi-class emotion detection (7 emotions)
            elif any(lang in prompt_lower for lang in ['spanish', 'french', 'german', 'italian', 'portuguese']):
                selected_model = Config.MULTILINGUAL_SENTIMENT_MODEL  # Multilingual sentiment (8+ languages)
            else:
                selected_model = Config.DEFAULT_SENTIMENT_MODEL  # SOTA sentiment model (124M tweets)
            
            logger.info(f"💭 SENTIMENT_ANALYSIS: emotion_detected={any(keyword in prompt_lower for keyword in emotion_keywords)}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.TRANSLATION:
            # Advanced translation models for 200+ languages
            prompt_lower = original_prompt.lower()
            selected_model = None
            if 'complex' in prompt_lower or 'document' in prompt_lower:
                selected_model = Config.DEFAULT_TRANSLATION_MODEL  # NLLB-200 (200+ languages, SOTA)
            elif 'to english' in prompt_lower:
                selected_model = Config.ADVANCED_TRANSLATION_MODEL  # Multilingual to English specialist
            else:
                selected_model = Config.DEFAULT_TRANSLATION_MODEL  # NLLB-200 (best overall)
            
            logger.info(f"🌍 TRANSLATION: complex_doc={('complex' in prompt_lower or 'document' in prompt_lower)}, to_english={'to english' in prompt_lower}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CREATIVE_WRITING:
            # Use large models for creative tasks requiring high quality
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            if complexity > 5:
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for complex creative reasoning
            else:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for creative writing
            
            logger.info(f"✍️ CREATIVE_WRITING: complexity={complexity}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.QUESTION_ANSWERING:
            # Use large models for complex Q&A requiring deep knowledge
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            if complexity > 6 or 'complex' in original_prompt.lower():
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for complex reasoning
            else:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B (131K context)
            
            logger.info(f"❓ QUESTION_ANSWERING: complexity={complexity}, complex_keyword={'complex' in original_prompt.lower()}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.DATA_ANALYSIS:
            # Data analysis and processing tasks
            selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for analytical reasoning
            logger.info(f"📈 DATA_ANALYSIS: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.DOCUMENT_PROCESSING:
            # Document processing with vision capabilities
            selected_model = Config.DOCUMENT_VISION_MODEL  # Florence-2 (excellent OCR)
            logger.info(f"📄 DOCUMENT_PROCESSING: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.PDF_PROCESSING:
            # PDF analysis and processing
            selected_model = Config.DOCUMENT_VISION_MODEL  # Florence-2 for document understanding
            logger.info(f"📁 PDF_PROCESSING: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.ZIP_ANALYSIS:
            # ZIP file analysis
            selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for file structure analysis
            logger.info(f"🗜️ ZIP_ANALYSIS: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.FILE_GENERATION:
            # File generation tasks
            selected_model = Config.ADVANCED_CODE_MODEL  # DeepSeek-Coder-V2 for generating files
            logger.info(f"📁 FILE_GENERATION: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CONVERSATION:
            # Conversational AI
            selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for responsive conversation
            logger.info(f"💬 CONVERSATION: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        # Default fallback to our most powerful reasoning model
        selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill (matches o1 performance)
        logger.info(f"⚠️ FALLBACK: intent={intent.value}, model={selected_model}")
        
        # Enhanced fallback logging
        logger.info(f"⚠️ FALLBACK: intent={intent.value} → ADVANCED (DeepSeek-R1-Distill)")
        logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Superior to OpenAI o1/Claude Sonnet)")
        logger.info(f"⚡ REASONING: Fallback to most powerful reasoning model for unknown intent")
        
        # Validate model selection and return
        validated_model = self._validate_model_selection(selected_model, intent)
        logger.info(f"✅ FINAL_VALIDATION: model={validated_model} for intent={intent.value}")
        logger.info(f"🏆 ═══════════════════════ ROUTER DECISION COMPLETE ═══════════════════════\n")
        return validated_model
    
    def _get_special_parameters(self, intent: IntentType, analysis: Dict) -> Dict:
        """Get optimized parameters for 2024-2025 SOTA models"""
        from bot.config import Config
        base_params = {}
        complexity = analysis.get('complexity_score', 0)
        
        if intent == IntentType.CODE_GENERATION:
            # Optimized for DeepSeek-Coder-V2 and StarCoder2 series
            base_params.update({
                'temperature': Config.STARCODER_TEMPERATURE,  # 0.2 for precise code
                'max_new_tokens': 1500 if complexity > 5 else 1200,
                'top_p': 0.95,
                'do_sample': True,
                'language': analysis.get('language_detected', 'python'),
                'use_flash_attention': True,  # DeepSeek optimization
                'use_advanced_model': True
            })
        
        elif intent == IntentType.IMAGE_ANALYSIS or intent == IntentType.MULTI_MODAL:
            # Optimized for Qwen2.5-VL and PaliGemma2 series
            base_params.update({
                'temperature': Config.VISION_TEMPERATURE,  # 0.5 for vision tasks
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'image_size': Config.QWEN_VL_IMAGE_SIZE,  # 448 optimal for Qwen2.5-VL
                'dynamic_resolution': Config.QWEN_VL_DYNAMIC_RESOLUTION,
                'vision_mode': True
            })
        
        elif intent == IntentType.IMAGE_GENERATION:
            # Optimized for FLUX.1 and SD3.5 series (SUPERIOR to DALL-E)
            base_params.update({
                'guidance_scale': 7.5,
                'num_inference_steps': Config.FLUX_INFERENCE_STEPS,  # 4 for FLUX.1-schnell
                'width': Config.FLUX_MAX_RESOLUTION,  # 1024
                'height': Config.FLUX_MAX_RESOLUTION,  # 1024
                'enhanced_prompt': True,
                'flux_turbo_mode': Config.FLUX_SCHNELL_TURBO_MODE,
                'use_flux_dev': True  # Best quality model
            })
        
        elif intent == IntentType.CREATIVE_WRITING:
            # Optimized for Qwen2.5-72B and DeepSeek-R1 creative tasks
            base_params.update({
                'temperature': 0.8 if complexity < 5 else Config.QWEN_TEMPERATURE,  # 0.7
                'max_new_tokens': 2000 if complexity > 5 else 1500,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'creative_mode': True,
                'use_long_context': True  # Leverage 131K context
            })
        
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            # Optimized for latest sentiment and emotion models
            base_params.update({
                'return_all_scores': True,  # Get confidence for all emotions
                'use_emotion_detection': 'emotion' in str(analysis).lower(),
                'multilingual_support': True,
                'confidence_threshold': 0.7
            })
        
        elif intent == IntentType.TRANSLATION:
            # Optimized for NLLB-200 and multilingual models
            base_params.update({
                'temperature': 0.3,  # Lower for accurate translation
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'beam_search': True,  # Better translation quality
                'num_beams': 4,
                'multilingual_mode': True
            })
        
        elif intent in [IntentType.TEXT_GENERATION, IntentType.QUESTION_ANSWERING]:
            # Optimized for DeepSeek-R1 and Qwen2.5 series
            base_params.update({
                'temperature': Config.DEEPSEEK_TEMPERATURE if complexity > 6 else Config.QWEN_TEMPERATURE,
                'max_new_tokens': Config.DEEPSEEK_MAX_TOKENS if complexity > 6 else 1500,
                'top_p': 0.9,
                'repetition_penalty': 1.05,
                'advanced_reasoning': complexity > 3,
                'use_flash_attention': Config.DEEPSEEK_USE_FLASH_ATTENTION,
                'long_context_support': True  # 131K for Qwen2.5
            })
        
        elif intent == IntentType.DATA_ANALYSIS:
            # Optimized for analytical reasoning tasks
            base_params.update({
                'temperature': Config.DEEPSEEK_TEMPERATURE,  # 0.8 for reasoning
                'max_new_tokens': 2000,
                'top_p': 0.9,
                'analytical_mode': True,
                'use_advanced_reasoning': True
            })
        
        elif intent in [IntentType.DOCUMENT_PROCESSING, IntentType.PDF_PROCESSING]:
            # Optimized for PaliGemma2 document understanding
            base_params.update({
                'temperature': 0.3,  # Lower for accurate document analysis
                'max_new_tokens': 1500,
                'top_p': 0.9,
                'ocr_mode': True,
                'document_understanding': True,
                'resolution_mode': 896  # PaliGemma2 max resolution
            })
        
        elif intent == IntentType.CONVERSATION:
            # Optimized for conversational AI
            base_params.update({
                'temperature': 0.7,
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'conversational_mode': True,
                'response_speed_optimized': True
            })
        
        else:
            # Default parameters for DeepSeek-R1 fallback
            base_params.update({
                'temperature': Config.DEEPSEEK_TEMPERATURE,  # 0.8
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'use_advanced_model': True
            })
        
        return base_params
    
    def _apply_intent_heuristics(self, prompt: str, primary_intent: IntentType, intent_scores: Dict, analysis: Dict) -> IntentType:
        """Apply additional heuristics to refine intent detection"""
        prompt_lower = prompt.lower()
        
        # Special case: "Explain" + philosophical/theoretical concepts should be TEXT_GENERATION, not CODE
        if (prompt_lower.strip().startswith('explain') and 
            any(term in prompt_lower for term in ['philosophical', 'implications', 'consciousness', 'theoretical', 'principles'])):
            return IntentType.TEXT_GENERATION
        
        # Special case: If prompt starts with "what" and mentions technical concepts, prioritize Q&A
        if (prompt_lower.strip().startswith('what') and 
            any(term in prompt_lower for term in ['artificial intelligence', 'machine learning', 'ai', 'technology', 'computer', 'programming'])):
            return IntentType.QUESTION_ANSWERING
        
        # Special case: "Draw/paint" + descriptive words should be image generation
        if (re.search(r'\b(?:draw|paint|sketch)\s+(?:a|an)?\s*(?:beautiful|stunning|amazing|gorgeous|lovely|pretty)', prompt_lower) or
            re.search(r'\b(?:draw|paint|sketch)\s+(?:a|an)?\s*(?:sunset|sunrise|landscape|portrait|scene)', prompt_lower)):
            return IntentType.IMAGE_GENERATION
        
        # Special case: "sentiment of" should always be sentiment analysis
        if re.search(r'\b(?:sentiment|feeling|emotion)\s+of\b', prompt_lower):
            return IntentType.SENTIMENT_ANALYSIS
        
        # Special case: Explicit PDF analysis should be PDF_PROCESSING
        if re.search(r'\b(?:analyze|process)\s+(?:this\s+)?pdf\s+(?:document|file)?', prompt_lower):
            return IntentType.PDF_PROCESSING
        
        # Special case: Code-specific terms WITH programming verbs should prioritize code generation
        if (analysis.get('has_technical_terms') and 
            re.search(r'\b(?:write|create|implement|build|develop|generate)\s+(?:a|an)?\s*(?:function|class|method|algorithm|code|script|program)\b', prompt_lower)):
            return IntentType.CODE_GENERATION
        
        return primary_intent
    
    def get_model_performance_feedback(self, intent: IntentType, success: bool, response_quality: str) -> Dict:
        """
        Collect performance feedback for model improvement
        
        Args:
            intent (IntentType): The intent that was processed
            success (bool): Whether the request was successful
            response_quality (str): Quality assessment (good/average/poor)
            
        Returns:
            Dict: Performance feedback data
        """
        return {
            'intent': intent.value,
            'success': success,
            'quality': response_quality,
            'timestamp': None,  # Will be set by caller
            'model_used': self._get_recommended_model(intent, {})
        }

# Global router instance
router = IntelligentRouter()