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
                r'\b(?:draw|create|generate|make|design|paint|sketch|illustrate)\s+(?:a|an)?\s*(?:image|picture|photo|artwork|drawing|illustration|graphic|logo|icon)',
                r'\b(?:show|visualize|picture)\s+(?:me)?\s*(?:a|an)?\s*(?:image|picture)',
                r'\b(?:art|artistic|visual|painting|drawing|sketch|illustration|graphic)',
                r'\b(?:generate|create).*(?:visual|image|picture|artwork)',
                r'(?:can you|could you|please).*(?:draw|create|make).*(?:image|picture|artwork)',
                r'\b(?:logo|icon|banner|poster|wallpaper|avatar)',
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
                r'\b(?:analyze|check|determine|find)\s+(?:the\s+)?(?:sentiment|mood|tone|emotion|feeling)',
                r'(?:is this|how does this sound|what do you think about).*(?:positive|negative|neutral)',
                r'\b(?:sentiment|emotion|feeling|mood|tone)\s+(?:analysis|detection|recognition)',
                r'(?:positive|negative|neutral|happy|sad|angry|excited|disappointed)',
            ],
            
            IntentType.CREATIVE_WRITING: [
                r'\b(?:write|create|compose|draft)\s+(?:a|an)?\s*(?:story|poem|song|lyrics|novel|tale|narrative)',
                r'\b(?:creative|fiction|poetry|storytelling|narrative|literature)',
                r'(?:once upon a time|tell me a story|write a story about)',
                r'\b(?:character|plot|dialogue|scene|chapter)',
                r'(?:rhyme|verse|stanza|ballad|sonnet|haiku)',
            ],
            
            IntentType.QUESTION_ANSWERING: [
                r'^\s*(?:what|who|when|where|why|how|which|whose)\s+',
                r'(?:explain|tell me|describe|clarify|elaborate)',
                r'(?:question|answer|information|knowledge|facts)',
                r'(?:do you know|can you tell me|help me understand)',
            ],
            
            IntentType.TRANSLATION: [
                r'\b(?:translate|translation|convert)\s+(?:this|the|from|to)',
                r'(?:from|to)\s+(?:english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian)',
                r'\b(?:language|linguistic|multilingual)',
                r'(?:what does.*mean in|how do you say.*in)',
            ]
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
        
        # Calculate scores for each intent type
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    score += 1
                    matches.append(pattern)
            
            if score > 0:
                intent_scores[intent_type] = {
                    'score': score,
                    'matches': matches
                }
        
        # Get prompt analysis
        analysis = self.analyze_prompt_complexity(prompt)
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['score'])
        else:
            # Default to text generation if no specific intent detected
            primary_intent = IntentType.TEXT_GENERATION
        
        # Build routing information
        routing_info = {
            'primary_intent': primary_intent,
            'confidence': intent_scores.get(primary_intent, {}).get('score', 0),
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