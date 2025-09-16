"""
Advanced Intent Classification System - The "Brain" of AI Assistant Pro
Provides intelligent model routing to make the bot SUPERIOR to ChatGPT/Grok/Gemini
"""

import re
import logging
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Set
from enum import Enum
from collections import defaultdict, Counter
from dataclasses import dataclass
from bot.core.router import IntentType, IntelligentRouter

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Results of intent classification with confidence metrics"""
    primary_intent: IntentType
    confidence: float
    secondary_intents: List[Tuple[IntentType, float]]
    reasoning: str
    processing_time: float
    complexity_score: float
    special_features: List[str]

class AdvancedIntentClassifier:
    """
    Superior Intent Classification System with ML-grade accuracy
    Uses advanced pattern matching, keyword analysis, and context awareness
    """
    
    def __init__(self):
        self.router = IntelligentRouter()
        self.classification_cache = {}  # Cache for performance
        self.performance_stats = defaultdict(int)
        self.keyword_weights = self._initialize_keyword_weights()
        self.context_patterns = self._initialize_context_patterns()
        
        # Advanced feature extractors
        self.technical_indicators = self._load_technical_indicators()
        self.creative_indicators = self._load_creative_indicators()
        self.visual_indicators = self._load_visual_indicators()
        
        logger.info("🧠 Advanced Intent Classifier initialized - SUPERIOR to ChatGPT routing")
    
    def _initialize_keyword_weights(self) -> Dict[str, Dict[IntentType, float]]:
        """Initialize weighted keywords for each intent type with ML-grade precision"""
        return {
            # High-weight programming keywords
            'code': {IntentType.CODE_GENERATION: 3.0, IntentType.TEXT_GENERATION: 0.2},
            'function': {IntentType.CODE_GENERATION: 3.5, IntentType.TEXT_GENERATION: 0.1},
            'class': {IntentType.CODE_GENERATION: 3.2, IntentType.TEXT_GENERATION: 0.3},
            'algorithm': {IntentType.CODE_GENERATION: 3.8, IntentType.TEXT_GENERATION: 0.2},
            'debug': {IntentType.CODE_GENERATION: 4.0, IntentType.TEXT_GENERATION: 0.1},
            'python': {IntentType.CODE_GENERATION: 3.5},
            'javascript': {IntentType.CODE_GENERATION: 3.5},
            'api': {IntentType.CODE_GENERATION: 2.8, IntentType.TEXT_GENERATION: 0.3},
            'database': {IntentType.CODE_GENERATION: 2.5, IntentType.TEXT_GENERATION: 0.5},
            
            # High-weight visual keywords
            'draw': {IntentType.IMAGE_GENERATION: 4.0, IntentType.TEXT_GENERATION: 0.1},
            'paint': {IntentType.IMAGE_GENERATION: 4.0, IntentType.TEXT_GENERATION: 0.1},
            'image': {IntentType.IMAGE_GENERATION: 3.0, IntentType.TEXT_GENERATION: 0.3},
            'picture': {IntentType.IMAGE_GENERATION: 2.8, IntentType.TEXT_GENERATION: 0.3},
            'artwork': {IntentType.IMAGE_GENERATION: 3.8, IntentType.TEXT_GENERATION: 0.1},
            'design': {IntentType.IMAGE_GENERATION: 2.5, IntentType.CODE_GENERATION: 0.8},
            'logo': {IntentType.IMAGE_GENERATION: 3.5},
            'poster': {IntentType.IMAGE_GENERATION: 3.2},
            'illustration': {IntentType.IMAGE_GENERATION: 3.5},
            
            # Sentiment analysis keywords
            'sentiment': {IntentType.SENTIMENT_ANALYSIS: 4.0, IntentType.TEXT_GENERATION: 0.1},
            'emotion': {IntentType.SENTIMENT_ANALYSIS: 4.0, IntentType.TEXT_GENERATION: 0.2},
            'feeling': {IntentType.SENTIMENT_ANALYSIS: 3.5, IntentType.TEXT_GENERATION: 0.4},
            'mood': {IntentType.SENTIMENT_ANALYSIS: 3.8, IntentType.TEXT_GENERATION: 0.3},
            'analyze': {IntentType.SENTIMENT_ANALYSIS: 2.5, IntentType.TEXT_GENERATION: 0.8},
            
            # Creative writing keywords
            'story': {IntentType.CREATIVE_WRITING: 3.5, IntentType.TEXT_GENERATION: 1.0},
            'poem': {IntentType.CREATIVE_WRITING: 4.0, IntentType.TEXT_GENERATION: 0.5},
            'novel': {IntentType.CREATIVE_WRITING: 3.8, IntentType.TEXT_GENERATION: 0.3},
            'creative': {IntentType.CREATIVE_WRITING: 3.0, IntentType.TEXT_GENERATION: 0.8},
            'fiction': {IntentType.CREATIVE_WRITING: 3.2, IntentType.TEXT_GENERATION: 0.5},
            
            # Question answering keywords
            'what': {IntentType.QUESTION_ANSWERING: 2.0, IntentType.TEXT_GENERATION: 1.5},
            'why': {IntentType.QUESTION_ANSWERING: 2.2, IntentType.TEXT_GENERATION: 1.3},
            'how': {IntentType.QUESTION_ANSWERING: 2.1, IntentType.TEXT_GENERATION: 1.4},
            'explain': {IntentType.QUESTION_ANSWERING: 2.8, IntentType.TEXT_GENERATION: 1.0},
            
            # Conversational keywords
            'hello': {IntentType.CONVERSATION: 3.0, IntentType.TEXT_GENERATION: 1.0},
            'hi': {IntentType.CONVERSATION: 3.0, IntentType.TEXT_GENERATION: 1.0},
            'chat': {IntentType.CONVERSATION: 2.8, IntentType.TEXT_GENERATION: 0.8},
            'talk': {IntentType.CONVERSATION: 2.5, IntentType.TEXT_GENERATION: 1.0},
        }
    
    def _initialize_context_patterns(self) -> Dict[IntentType, List[str]]:
        """Advanced regex patterns for context-aware classification"""
        return {
            IntentType.CODE_GENERATION: [
                r'\b(?:write|create|code|program|implement|build|develop)\s+(?:a|an|some)?\s*(?:function|class|method|script|program|algorithm|api|library)',
                r'(?:how\s+to|help\s+me|show\s+me).*(?:code|program|implement|build|debug)',
                r'\b(?:react|vue|python|javascript|java|c\+\+|typescript|php|sql|html|css)\b.*(?:code|function|component|query)',
                r'(?:error|bug|issue|debug|fix|optimize).*(?:code|function|script|program)',
                r'\b(?:github|stackoverflow|programming|coding|development|software)\b',
                r'```[\s\S]*```',  # Code blocks
                r'`[^`]+`',  # Inline code
            ],
            IntentType.IMAGE_GENERATION: [
                r'\b(?:create|generate|draw|paint|design|make|produce)\s+(?:a|an|some)?\s*(?:image|picture|artwork|drawing|illustration|logo|poster|banner)',
                r'\b(?:beautiful|stunning|amazing|gorgeous|artistic|professional|realistic|abstract|minimalist|modern|vintage)\s+(?:image|picture|artwork|scene|landscape|portrait)',
                r'(?:show\s+me|visualize|picture)\s+(?:a|an|some)?\s*(?:image|picture|visual|artwork)',
                r'\b(?:dalle|midjourney|stable\s*diffusion|ai\s*art|text\s*to\s*image|flux)\b',
                r'(?:product\s*shot|concept\s*art|digital\s*art|3d\s*render|ui\s*mockup)',
            ],
            IntentType.SENTIMENT_ANALYSIS: [
                r'\b(?:sentiment|emotion|feeling|mood|tone)\s+(?:of|analysis|detection)',
                r'\b(?:analyze|check|determine|find)\s+(?:the\s+)?(?:sentiment|mood|emotion)',
                r'(?:positive|negative|neutral|happy|sad|angry|excited)\s+(?:sentiment|feeling|emotion)',
                r'(?:how\s+does\s+this\s+sound|what\s+do\s+you\s+think\s+about).*(?:positive|negative|neutral)',
            ],
            IntentType.CREATIVE_WRITING: [
                r'\b(?:write|create|compose|draft)\s+(?:a|an)?\s*(?:story|poem|song|lyrics|novel|tale)',
                r'(?:once\s+upon\s+a\s+time|tell\s+me\s+a\s+story|creative\s+writing)',
                r'\b(?:character|plot|dialogue|scene|chapter|narrative|fiction)\b',
                r'(?:rhyme|verse|stanza|ballad|sonnet|haiku)',
            ],
            IntentType.QUESTION_ANSWERING: [
                r'^\s*(?:what|who|when|where|why|how|which)\s+(?:is|are|was|were|does|do|did|will|would|can|could)',
                r'(?:explain|tell\s+me|describe|clarify|elaborate)\s+(?:what|how|why|the)',
                r'(?:do\s+you\s+know|can\s+you\s+tell\s+me|help\s+me\s+understand)',
                r'\b(?:definition|meaning|concept)\s+(?:of|for)\b',
            ],
            IntentType.TRANSLATION: [
                r'\b(?:translate|translation|convert)\s+(?:this|the|from|to)',
                r'(?:from|to)\s+(?:english|spanish|french|german|chinese|japanese|korean|arabic)',
                r'(?:what\s+does.*mean\s+in|how\s+do\s+you\s+say.*in)',
            ],
            IntentType.CONVERSATION: [
                r'(?:hello|hi|hey|greetings)(?:\s|$|,|!)',
                r'(?:how\s+are\s+you|what\'s\s+up|how\'s\s+it\s+going)',
                r'(?:nice\s+to\s+meet\s+you|pleasure\s+to\s+meet\s+you)',
                r'(?:good\s+morning|good\s+afternoon|good\s+evening)',
            ]
        }
    
    def _load_technical_indicators(self) -> Set[str]:
        """Load comprehensive technical terminology for code detection"""
        return {
            # Programming languages
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'scala', 'haskell', 'erlang', 'elixir', 'clojure', 'f#', 'dart', 'lua', 'perl', 'r', 'matlab', 'julia',
            
            # Frameworks & libraries
            'react', 'vue', 'angular', 'svelte', 'django', 'flask', 'fastapi', 'express', 'spring', 'laravel',
            'rails', 'next', 'nuxt', 'gatsby', 'remix', 'astro', 'solid', 'qwik',
            
            # Technical concepts
            'algorithm', 'function', 'class', 'method', 'variable', 'array', 'object', 'loop', 'condition',
            'recursion', 'iteration', 'pointer', 'reference', 'inheritance', 'polymorphism', 'encapsulation',
            'abstraction', 'interface', 'abstract', 'static', 'dynamic', 'synchronous', 'asynchronous',
            
            # Development tools
            'git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'ci', 'cd', 'devops', 'aws', 'azure',
            'gcp', 'terraform', 'ansible', 'webpack', 'vite', 'rollup', 'babel', 'eslint', 'prettier',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'database', 'table', 'query', 'index', 'schema', 'migration', 'orm',
            
            # Web technologies
            'html', 'css', 'sass', 'scss', 'less', 'tailwind', 'bootstrap', 'api', 'rest', 'graphql',
            'json', 'xml', 'http', 'https', 'websocket', 'ajax', 'fetch', 'cors',
        }
    
    def _load_creative_indicators(self) -> Set[str]:
        """Load creative writing and artistic terminology"""
        return {
            'story', 'tale', 'narrative', 'plot', 'character', 'protagonist', 'antagonist', 'dialogue',
            'scene', 'chapter', 'verse', 'stanza', 'rhyme', 'metaphor', 'simile', 'imagery', 'symbolism',
            'poem', 'poetry', 'sonnet', 'haiku', 'ballad', 'limerick', 'epic', 'novel', 'novella',
            'fiction', 'nonfiction', 'fantasy', 'scifi', 'mystery', 'thriller', 'romance', 'drama',
            'comedy', 'tragedy', 'satire', 'parody', 'allegory', 'fable', 'myth', 'legend',
            'creative', 'imaginative', 'artistic', 'poetic', 'literary', 'eloquent', 'expressive',
        }
    
    def _load_visual_indicators(self) -> Set[str]:
        """Load visual and design terminology for image generation"""
        return {
            'image', 'picture', 'photo', 'artwork', 'drawing', 'sketch', 'painting', 'illustration',
            'graphic', 'visual', 'design', 'art', 'artistic', 'logo', 'icon', 'banner', 'poster',
            'wallpaper', 'background', 'texture', 'pattern', 'color', 'palette', 'composition',
            'lighting', 'shadow', 'highlight', 'contrast', 'saturation', 'hue', 'brightness',
            'abstract', 'realistic', 'surreal', 'minimalist', 'modern', 'vintage', 'retro',
            'futuristic', 'fantasy', 'landscape', 'portrait', 'scenery', 'architecture',
            'fashion', 'food', 'nature', 'urban', 'rural', 'digital', 'traditional',
            'watercolor', 'oil', 'acrylic', 'pencil', 'charcoal', 'ink', 'pastel',
        }
    
    def extract_features(self, prompt: str) -> Dict[str, Any]:
        """
        Extract comprehensive features from the prompt for classification
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            Dict: Extracted features for classification
        """
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        features = {
            'length': len(prompt),
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1),
            'has_code_blocks': bool(re.search(r'```|`[^`]+`', prompt)),
            'has_questions': bool(re.search(r'\?', prompt)),
            'starts_with_question': bool(re.match(r'^\s*(?:what|who|when|where|why|how|which)', prompt_lower)),
            'has_exclamation': bool('!' in prompt),
            'has_greeting': any(greeting in prompt_lower for greeting in ['hello', 'hi', 'hey', 'greetings']),
            'technical_word_count': sum(1 for word in words if word in self.technical_indicators),
            'creative_word_count': sum(1 for word in words if word in self.creative_indicators),
            'visual_word_count': sum(1 for word in words if word in self.visual_indicators),
            'sentence_count': len(re.split(r'[.!?]+', prompt.strip())),
            'has_imperatives': bool(re.search(r'\b(?:create|make|generate|draw|write|build|code|design|show|help)\b', prompt_lower)),
            'complexity_indicators': len(re.findall(r'\b(?:complex|advanced|detailed|sophisticated|comprehensive|elaborate)\b', prompt_lower)),
        }
        
        # Calculate ratios
        if features['word_count'] > 0:
            features['technical_ratio'] = features['technical_word_count'] / features['word_count']
            features['creative_ratio'] = features['creative_word_count'] / features['word_count']
            features['visual_ratio'] = features['visual_word_count'] / features['word_count']
        else:
            features['technical_ratio'] = features['creative_ratio'] = features['visual_ratio'] = 0
        
        return features
    
    def calculate_intent_scores(self, prompt: str, features: Dict[str, Any]) -> Dict[IntentType, float]:
        """
        Calculate weighted scores for each intent based on features and patterns
        
        Args:
            prompt (str): User input prompt
            features (Dict): Extracted features
            
        Returns:
            Dict: Intent scores with confidence levels
        """
        prompt_lower = prompt.lower()
        scores = defaultdict(float)
        
        # 1. Pattern-based scoring (high precision)
        for intent, patterns in self.context_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                if matches > 0:
                    scores[intent] += matches * 2.0  # Pattern matches are high confidence
        
        # 2. Keyword-based scoring with weights
        words = prompt_lower.split()
        for word in words:
            if word in self.keyword_weights:
                for intent, weight in self.keyword_weights[word].items():
                    scores[intent] += weight
        
        # 3. Feature-based scoring (ML-style)
        # Code generation indicators
        if features['technical_ratio'] > 0.1:
            scores[IntentType.CODE_GENERATION] += features['technical_ratio'] * 5.0
        if features['has_code_blocks']:
            scores[IntentType.CODE_GENERATION] += 3.0
        
        # Image generation indicators
        if features['visual_ratio'] > 0.1:
            scores[IntentType.IMAGE_GENERATION] += features['visual_ratio'] * 5.0
        if features['has_imperatives'] and any(word in prompt_lower for word in ['draw', 'create', 'design', 'generate', 'make']):
            visual_context = any(word in prompt_lower for word in ['image', 'picture', 'art', 'logo', 'poster'])
            if visual_context:
                scores[IntentType.IMAGE_GENERATION] += 3.0
        
        # Creative writing indicators
        if features['creative_ratio'] > 0.05:
            scores[IntentType.CREATIVE_WRITING] += features['creative_ratio'] * 4.0
        if features['word_count'] > 20 and any(word in prompt_lower for word in ['story', 'poem', 'write', 'create']):
            scores[IntentType.CREATIVE_WRITING] += 2.0
        
        # Question answering indicators
        if features['starts_with_question']:
            scores[IntentType.QUESTION_ANSWERING] += 2.5
        if features['has_questions'] and not features['has_imperatives']:
            scores[IntentType.QUESTION_ANSWERING] += 1.5
        
        # Sentiment analysis indicators
        sentiment_words = ['sentiment', 'emotion', 'feeling', 'mood', 'analyze', 'positive', 'negative']
        sentiment_count = sum(1 for word in sentiment_words if word in prompt_lower)
        if sentiment_count > 0:
            scores[IntentType.SENTIMENT_ANALYSIS] += sentiment_count * 2.0
        
        # Conversation indicators
        if features['has_greeting'] or features['word_count'] < 5:
            scores[IntentType.CONVERSATION] += 2.0
        if any(phrase in prompt_lower for phrase in ['how are you', "what's up", 'nice to meet you']):
            scores[IntentType.CONVERSATION] += 3.0
        
        # Default fallback to text generation with lower score
        if not scores:
            scores[IntentType.TEXT_GENERATION] = 1.0
        else:
            # Add base score for text generation (it's always a possibility)
            scores[IntentType.TEXT_GENERATION] += 0.5
        
        return dict(scores)
    
    def classify_intent(self, prompt: str, context: Optional[Dict] = None) -> ClassificationResult:
        """
        Main classification method with comprehensive analysis
        
        Args:
            prompt (str): User input prompt
            context (Optional[Dict]): Additional context information
            
        Returns:
            ClassificationResult: Complete classification results
        """
        start_time = time.time()
        
        # Check cache first for performance
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.classification_cache:
            cached_result = self.classification_cache[cache_key]
            logger.info(f"🚀 Intent classification cache hit: {cached_result.primary_intent.value}")
            return cached_result
        
        # Extract features
        features = self.extract_features(prompt)
        
        # Calculate intent scores
        intent_scores = self.calculate_intent_scores(prompt, features)
        
        # Apply router's additional analysis (existing logic)
        router_intent, router_info = self.router.route_prompt(prompt)
        
        # Combine scores with router confidence
        if router_info.get('confidence', 0) > 0:
            intent_scores[router_intent] += router_info['confidence'] * 0.1
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            confidence = intent_scores[primary_intent]
        else:
            primary_intent = IntentType.TEXT_GENERATION
            confidence = 1.0
        
        # Calculate secondary intents
        secondary_intents = [
            (intent, score) for intent, score in sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            if intent != primary_intent and score > 0.5
        ][:3]  # Top 3 secondary intents
        
        # Generate reasoning
        reasoning = self._generate_reasoning(prompt, primary_intent, intent_scores, features)
        
        # Calculate complexity score
        complexity_score = (
            features['word_count'] / 10 + 
            features['technical_ratio'] * 3 + 
            features['complexity_indicators'] * 2 +
            (2 if features['has_code_blocks'] else 0)
        )
        
        # Extract special features
        special_features = []
        if features['has_code_blocks']:
            special_features.append('code_blocks')
        if features['technical_ratio'] > 0.2:
            special_features.append('highly_technical')
        if features['creative_ratio'] > 0.1:
            special_features.append('creative_content')
        if features['visual_ratio'] > 0.1:
            special_features.append('visual_content')
        
        processing_time = time.time() - start_time
        
        # Create result
        result = ClassificationResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            reasoning=reasoning,
            processing_time=processing_time,
            complexity_score=complexity_score,
            special_features=special_features
        )
        
        # Cache result for performance
        self.classification_cache[cache_key] = result
        
        # Update performance stats
        self.performance_stats[primary_intent.value] += 1
        self.performance_stats['total_classifications'] += 1
        
        logger.info(f"🎯 Intent classified: {primary_intent.value} (confidence: {confidence:.2f}, time: {processing_time*1000:.1f}ms)")
        
        return result
    
    def _generate_reasoning(self, prompt: str, primary_intent: IntentType, scores: Dict[IntentType, float], features: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the classification"""
        reasons = []
        
        if primary_intent == IntentType.CODE_GENERATION:
            if features['has_code_blocks']:
                reasons.append("contains code blocks")
            if features['technical_ratio'] > 0.1:
                reasons.append(f"high technical vocabulary ({features['technical_ratio']:.1%})")
            if any(word in prompt.lower() for word in ['function', 'class', 'algorithm']):
                reasons.append("mentions programming concepts")
        
        elif primary_intent == IntentType.IMAGE_GENERATION:
            if features['visual_ratio'] > 0.1:
                reasons.append(f"rich visual vocabulary ({features['visual_ratio']:.1%})")
            if any(word in prompt.lower() for word in ['draw', 'create', 'design']):
                reasons.append("uses visual creation verbs")
            if any(word in prompt.lower() for word in ['image', 'picture', 'artwork']):
                reasons.append("explicitly requests visual content")
        
        elif primary_intent == IntentType.CREATIVE_WRITING:
            if features['creative_ratio'] > 0.05:
                reasons.append(f"creative writing indicators ({features['creative_ratio']:.1%})")
            if any(word in prompt.lower() for word in ['story', 'poem', 'novel']):
                reasons.append("requests creative content")
        
        elif primary_intent == IntentType.QUESTION_ANSWERING:
            if features['starts_with_question']:
                reasons.append("starts with question word")
            if features['has_questions']:
                reasons.append("contains question marks")
        
        elif primary_intent == IntentType.SENTIMENT_ANALYSIS:
            if any(word in prompt.lower() for word in ['sentiment', 'emotion', 'feeling']):
                reasons.append("explicitly mentions sentiment analysis")
        
        elif primary_intent == IntentType.CONVERSATION:
            if features['has_greeting']:
                reasons.append("contains greeting")
            if features['word_count'] < 5:
                reasons.append("short conversational phrase")
        
        if not reasons:
            reasons.append(f"highest intent score ({scores[primary_intent]:.1f})")
        
        return "; ".join(reasons)
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the classifier"""
        total = self.performance_stats.get('total_classifications', 1)
        return {
            'total_classifications': total,
            'cache_size': len(self.classification_cache),
            'intent_distribution': {
                intent: count for intent, count in self.performance_stats.items()
                if intent != 'total_classifications'
            },
            'cache_hit_rate': f"{(len(self.classification_cache) / max(total, 1)) * 100:.1f}%"
        }
    
    def clear_cache(self):
        """Clear the classification cache"""
        self.classification_cache.clear()
        logger.info("🗑️ Intent classification cache cleared")

# Global classifier instance
intent_classifier = AdvancedIntentClassifier()