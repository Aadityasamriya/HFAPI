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
                r'\b(?:write|create|generate|code|program|implement|build|develop|design|craft|architect)\s+(?:a|an|some)?\s*(?:function|class|method|script|program|application|app|module|component|service|api|library|microservice|webhook|middleware|plugin|extension)',
                r'\b(?:python|javascript|java|c\+\+|typescript|php|ruby|go|rust|swift|kotlin|c#|dart|scala|r|zig|nim|elixir|haskell|julia)\s+(?:code|function|script|program|class|method|application|api)',
                r'\b(?:algorithm|function|class|method|API|library|framework|module|component|microservice|backend|frontend|fullstack|devops|cicd|pipeline)',
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
                r'\b(?:analyze|read|extract|process|parse|summarize)\s+(?:pdf|document|file)',
                r'(?:pdf.?analysis|document.?processing|text.?extraction|content.?from.?pdf)',
                r'\b(?:tables?|charts?|graphs?)\s+(?:from|in)\s+(?:pdf|document)',
                r'(?:extract.?text|get.?content|read.?document|parse.?pdf)',
                r'\b(?:summarize|summary|key.?points|main.?ideas)\s+(?:from|of|in)\s+(?:pdf|document|file)',
                r'(?:ocr|optical.?character.?recognition)\s+(?:pdf|document|scanned)',
                r'(?:metadata|properties|info)\s+(?:from|of|in)\s+(?:pdf|document)',
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
            ]
        }
    
    def _initialize_priorities(self) -> Dict[IntentType, int]:
        """2025 Enhanced priority weights for intent types"""
        return {
            IntentType.CODE_GENERATION: 10,         # Highest priority - very specific patterns
            IntentType.IMAGE_GENERATION: 9,         # High priority - specific visual requests
            IntentType.PDF_PROCESSING: 9,           # 2025: P1 High priority - PDF file uploads
            IntentType.ZIP_ANALYSIS: 9,             # 2025: P1 High priority - ZIP file uploads
            IntentType.IMAGE_ANALYSIS: 8,           # 2025: P1 High priority - Image analysis
            IntentType.DATA_ANALYSIS: 8,            # 2025: High priority - data processing
            IntentType.SENTIMENT_ANALYSIS: 8,       # High priority - specific analysis requests
            IntentType.DOCUMENT_PROCESSING: 7,      # 2025: Medium-high priority - document tasks
            IntentType.CREATIVE_WRITING: 7,         # Medium-high priority - creative requests
            IntentType.MULTI_MODAL: 6,              # 2025: Medium priority - complex multimodal
            IntentType.TRANSLATION: 6,              # Medium priority - language requests
            IntentType.CONVERSATION: 5,             # 2025: Medium priority - conversational AI
            IntentType.QUESTION_ANSWERING: 4,       # Lower priority - broad category
            IntentType.TEXT_GENERATION: 1,          # Lowest priority - fallback
        }
    
    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt characteristics for better model selection
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            Dict: Analysis results with complexity metrics
        """
        analysis = {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'has_code_blocks': bool(re.search(r'```|`[^`]+`', prompt)),
            'has_technical_terms': any(term in prompt.lower() for term in list(self.programming_languages)),
            'has_questions': bool(re.search(r'\?', prompt)),
            'complexity_score': 0,
            'requires_context': False,
            'language_detected': self._detect_programming_language(prompt)
        }
        
        # Calculate complexity score
        analysis['complexity_score'] += min(analysis['word_count'] / 10, 5)  # Length factor
        analysis['complexity_score'] += 2 if analysis['has_code_blocks'] else 0
        analysis['complexity_score'] += 2 if analysis['has_technical_terms'] else 0
        analysis['complexity_score'] += 1 if analysis['has_questions'] else 0
        
        # Determine if context is needed
        context_indicators = ['this', 'that', 'previous', 'above', 'earlier', 'before', 'continue', 'also']
        analysis['requires_context'] = any(indicator in prompt.lower() for indicator in context_indicators)
        
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
    
    def _get_recommended_model(self, intent: IntentType, analysis: Dict, original_prompt: str = "") -> str:
        """Get recommended model based on intent and analysis with 2024-2025 STATE-OF-THE-ART models"""
        from bot.config import Config
        
        # Advanced model selection based on complexity and context
        if intent == IntentType.TEXT_GENERATION:
            # Use smaller model by default for serverless compatibility, larger only for very complex requests
            if analysis.get('complexity_score', 0) > 5 or analysis.get('requires_context'):
                return Config.DEFAULT_TEXT_MODEL  # Qwen2.5-72B for highly complex tasks
            else:
                return Config.ADVANCED_TEXT_MODEL  # Qwen2.5-7B for general use (fast & reliable)
        
        elif intent == IntentType.CODE_GENERATION:
            # Use smaller model by default for serverless compatibility
            if analysis.get('complexity_score', 0) > 6:
                return Config.DEFAULT_CODE_MODEL  # StarCoder2-15B for complex code
            else:
                return Config.ADVANCED_CODE_MODEL  # StarCoder2-7B for simpler code
        
        elif intent == IntentType.IMAGE_GENERATION:
            # Use FLUX.1 models (SUPERIOR to Stable Diffusion, better text rendering)
            return Config.DEFAULT_IMAGE_MODEL  # FLUX.1-schnell (commercial license, 4-step generation)
        
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            # Check if emotion detection is needed - use the original prompt parameter
            emotion_keywords = ['emotion', 'feeling', 'mood', 'angry', 'happy', 'sad', 'excited']
            prompt_text = original_prompt.lower()
            if any(keyword in prompt_text for keyword in emotion_keywords):
                return Config.EMOTION_MODEL  # Multi-class emotion detection
            else:
                return Config.DEFAULT_SENTIMENT_MODEL  # Latest sentiment model (124M tweets trained)
        
        elif intent == IntentType.CREATIVE_WRITING:
            # Use large model for creative tasks requiring high quality
            return Config.DEFAULT_TEXT_MODEL  # Qwen2.5-72B for superior creative writing
        
        elif intent == IntentType.QUESTION_ANSWERING:
            # Use large model for complex Q&A requiring deep knowledge
            return Config.DEFAULT_TEXT_MODEL  # Qwen2.5-72B for superior knowledge (131K context)
        
        elif intent == IntentType.TRANSLATION:
            # Use large multilingual model for accurate translation
            return Config.DEFAULT_TEXT_MODEL  # Qwen2.5-72B supports 29+ languages with superior accuracy
        
        # Default fallback to our most powerful model
        return Config.DEFAULT_TEXT_MODEL  # Qwen2.5-72B (SUPERIOR to ChatGPT/Grok/Gemini)
    
    def _get_special_parameters(self, intent: IntentType, analysis: Dict) -> Dict:
        """Get optimized parameters for 2024-2025 models"""
        base_params = {}
        
        if intent == IntentType.CODE_GENERATION:
            # Optimized for StarCoder2-15B
            base_params.update({
                'temperature': 0.2,  # Lower for more precise code
                'max_new_tokens': 1200,  # More tokens for complex code
                'top_p': 0.95,
                'do_sample': True,
                'language': analysis.get('language_detected', 'python'),
                'use_advanced_model': True
            })
        elif intent == IntentType.CREATIVE_WRITING:
            # Optimized for Qwen2.5-7B creative tasks
            base_params.update({
                'temperature': 0.8,
                'max_new_tokens': 1500,  # More tokens for stories
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'creative_mode': True
            })
        elif intent == IntentType.IMAGE_GENERATION:
            # Optimized for FLUX.1-schnell (faster variant)
            base_params.update({
                'guidance_scale': 7.5,
                'num_inference_steps': 4,  # FLUX.1-schnell optimized for 4 steps
                'width': 1024,
                'height': 1024,
                'enhanced_prompt': True,
                'flux_mode': True
            })
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            base_params.update({
                'return_all_scores': True,  # Get confidence for all emotions
                'use_emotion_detection': 'emotion' in str(analysis).lower()
            })
        elif intent in [IntentType.TEXT_GENERATION, IntentType.QUESTION_ANSWERING]:
            # Optimized for Llama-3.2 and Qwen2.5
            complexity = analysis.get('complexity_score', 0)
            base_params.update({
                'temperature': 0.7 if complexity < 3 else 0.6,  # Lower temp for complex tasks
                'max_new_tokens': 1000 if complexity < 3 else 1500,
                'top_p': 0.9,
                'repetition_penalty': 1.05,
                'advanced_reasoning': complexity > 3
            })
        else:
            base_params.update({
                'temperature': 0.7,
                'max_new_tokens': 1000,
                'top_p': 0.9
            })
        
        return base_params
    
    def _apply_intent_heuristics(self, prompt: str, primary_intent: IntentType, intent_scores: Dict, analysis: Dict) -> IntentType:
        """Apply additional heuristics to refine intent detection"""
        prompt_lower = prompt.lower()
        
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
        
        # Special case: Code-specific terms should prioritize code generation
        if (analysis.get('has_technical_terms') and 
            re.search(r'\b(?:function|class|method|algorithm|code|script|program)\b', prompt_lower)):
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