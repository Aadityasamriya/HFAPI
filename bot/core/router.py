"""
Intelligent AI model routing system
Analyzes user prompts to determine the best Hugging Face model to use
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Available intent types for model routing"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"

class IntelligentRouter:
    """
    Advanced AI model router with intent detection
    Analyzes user prompts and selects optimal models
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.intent_priorities = self._initialize_priorities()
        self.programming_languages = {
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'php', 
            'ruby', 'go', 'rust', 'swift', 'kotlin', 'dart', 'scala', 'r',
            'sql', 'html', 'css', 'react', 'vue', 'angular', 'node', 'django',
            'flask', 'laravel', 'spring', 'android', 'ios'
        }
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize regex patterns for intent detection"""
        return {
            IntentType.IMAGE_GENERATION: [
                r'\b(?:draw|paint|sketch|illustrate)\s+(?:a|an)?\s*(?:beautiful|stunning|amazing|gorgeous|lovely|pretty)?\s*(?:sunset|sunrise|landscape|portrait|scene|picture|image|drawing|artwork|illustration)',
                r'\b(?:create|generate|make|design)\s+(?:a|an)?\s*(?:image|picture|photo|artwork|drawing|illustration|graphic|logo|icon|visual)',
                r'\b(?:show|visualize|picture)\s+(?:me)?\s*(?:a|an)?\s*(?:image|picture)',
                r'\b(?:draw|paint|sketch|illustrate)\b(?!.*(?:function|code|program|script|class|method))',
                r'(?:can you|could you|please).*(?:draw|paint|sketch|create|make).*(?:image|picture|artwork|visual)',
                r'\b(?:logo|icon|banner|poster|wallpaper|avatar|art|artistic|visual|painting|drawing|sketch|illustration|graphic)\b',
                r'(?:dalle|midjourney|stable.?diffusion|text.?to.?image)',
            ],
            
            IntentType.CODE_GENERATION: [
                r'\b(?:write|create|generate|code|program|implement|build|develop)\s+(?:a|an)?\s*(?:function|class|method|script|program|application|app)',
                r'\b(?:python|javascript|java|c\+\+|typescript|php|ruby|go|rust|swift|kotlin)\s+(?:code|function|script|program)',
                r'\b(?:algorithm|function|class|method|API|library|framework)',
                r'(?:how to|help me).*(?:code|program|implement|write)',
                r'\b(?:debug|fix|error|bug|issue).*(?:code|function|script)',
                r'\b(?:react|vue|angular|django|flask|express|laravel)',
                r'(?:github|stackoverflow|programming|coding|development)',
                r'\b(?:sql|database|query|table|schema)',
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
                r'(?:from|to)\s+(?:english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian)',
                r'\b(?:language|linguistic|multilingual)',
                r'(?:what does.*mean in|how do you say.*in)',
            ]
        }
    
    def _initialize_priorities(self) -> Dict[IntentType, int]:
        """Initialize priority weights for intent types"""
        return {
            IntentType.CODE_GENERATION: 10,      # Highest priority - very specific patterns
            IntentType.IMAGE_GENERATION: 9,      # High priority - specific visual requests
            IntentType.SENTIMENT_ANALYSIS: 8,    # High priority - specific analysis requests
            IntentType.CREATIVE_WRITING: 7,      # Medium-high priority - creative requests
            IntentType.TRANSLATION: 6,           # Medium priority - language requests
            IntentType.QUESTION_ANSWERING: 5,    # Lower priority - broad category
            IntentType.TEXT_GENERATION: 1,       # Lowest priority - fallback
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
            'recommended_model': self._get_recommended_model(primary_intent, analysis),
            'special_parameters': self._get_special_parameters(primary_intent, analysis)
        }
        
        logger.info(f"Routed prompt to {primary_intent.value} with confidence {routing_info['confidence']}")
        
        return primary_intent, routing_info
    
    def _get_recommended_model(self, intent: IntentType, analysis: Dict) -> str:
        """Get recommended model based on intent and analysis"""
        from bot.config import Config
        
        model_mapping = {
            IntentType.TEXT_GENERATION: Config.DEFAULT_TEXT_MODEL,
            IntentType.CODE_GENERATION: Config.DEFAULT_CODE_MODEL,
            IntentType.IMAGE_GENERATION: Config.DEFAULT_IMAGE_MODEL,
            IntentType.CREATIVE_WRITING: "microsoft/DialoGPT-large",
            IntentType.QUESTION_ANSWERING: Config.DEFAULT_TEXT_MODEL,
            IntentType.SENTIMENT_ANALYSIS: "cardiffnlp/twitter-roberta-base-sentiment-latest",
            IntentType.TRANSLATION: "Helsinki-NLP/opus-mt-en-de"  # Example translation model
        }
        
        return model_mapping.get(intent, Config.DEFAULT_TEXT_MODEL)
    
    def _get_special_parameters(self, intent: IntentType, analysis: Dict) -> Dict:
        """Get special parameters based on intent and analysis"""
        base_params = {}
        
        if intent == IntentType.CODE_GENERATION:
            base_params.update({
                'temperature': 0.3,
                'max_new_tokens': 800,
                'language': analysis.get('language_detected', 'python')
            })
        elif intent == IntentType.CREATIVE_WRITING:
            base_params.update({
                'temperature': 0.8,
                'max_new_tokens': 1200,
                'top_p': 0.9
            })
        elif intent == IntentType.IMAGE_GENERATION:
            base_params.update({
                'guidance_scale': 7.5,
                'num_inference_steps': 50,
                'enhanced_prompt': True
            })
        else:
            base_params.update({
                'temperature': 0.7,
                'max_new_tokens': 1000
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