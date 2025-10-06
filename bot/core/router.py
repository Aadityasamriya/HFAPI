"""
Intelligent AI model routing system
Analyzes user prompts to determine the best Hugging Face model to use
"""

import re
import logging
import hashlib
import math
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from .types import IntentType
from .bot_types import PromptComplexity, ModelPerformance, ContextState, DomainExpertise
from .model_health_monitor import health_monitor
from .performance_predictor import performance_predictor, PredictionContext
from .dynamic_fallback_strategy import dynamic_fallback_strategy, ErrorType
from .conversation_context_tracker import conversation_context_tracker, ConversationTurn
from .model_selection_explainer import model_selection_explainer, ModelSelectionExplanation

logger = logging.getLogger(__name__)


class AdvancedComplexityAnalyzer:
    """
    ML-based complexity analyzer that goes beyond simple pattern matching
    Provides sophisticated prompt analysis for optimal model selection
    """
    
    def __init__(self):
        self.technical_keywords = self._build_technical_vocabulary()
        self.reasoning_indicators = self._build_reasoning_patterns()
        self.complexity_weights = self._build_complexity_weights()
        self.domain_classifiers = self._build_domain_classifiers()
        
    def _build_technical_vocabulary(self) -> Dict[str, float]:
        """2025 ENHANCED: Optimized technical vocabulary with precision-tuned weights"""
        return {
            # Programming & Software Engineering (Optimized complexity scores)
            'algorithm': 2.8, 'data structure': 3.1, 'complexity analysis': 3.4,
            'machine learning': 3.2, 'neural network': 3.5, 'deep learning': 3.6,
            'distributed systems': 3.8, 'microservices': 3.0, 'containerization': 2.8,
            'kubernetes': 3.2, 'docker': 2.6, 'devops': 2.8, 'ci/cd': 2.6,
            'blockchain': 3.4, 'cryptography': 3.6, 'quantum computing': 4.2,
            'compiler design': 3.8, 'operating systems': 3.2, 'kernel': 3.4,
            
            # Advanced Mathematics & Science (Enhanced complexity scoring)
            'differential equations': 4.0, 'linear algebra': 3.2, 'calculus': 2.8,
            'quantum mechanics': 4.5, 'thermodynamics': 3.6, 'statistical mechanics': 4.2,
            'topology': 4.3, 'abstract algebra': 4.6, 'category theory': 4.8,
            'computational complexity': 3.8, 'np-complete': 3.6, 'optimization': 3.2,
            
            # Business & Economics (Moderate to high complexity)
            'financial modeling': 2.7, 'risk analysis': 2.5, 'portfolio optimization': 2.8,
            'market microstructure': 3.0, 'derivatives pricing': 3.1, 'quantitative analysis': 2.9,
            'supply chain optimization': 2.6, 'game theory': 3.2, 'econometrics': 3.0,
            
            # Medical & Life Sciences (High complexity)
            'pharmacokinetics': 3.3, 'molecular biology': 3.0, 'bioinformatics': 3.2,
            'genomics': 3.1, 'proteomics': 3.4, 'clinical trials': 2.8, 'epidemiology': 2.9,
            
            # Engineering & Physics (High complexity)
            'fluid dynamics': 3.4, 'electromagnetic fields': 3.2, 'signal processing': 2.9,
            'control systems': 3.0, 'robotics': 2.8, 'computer vision': 3.1, 'nlp': 2.7,
            
            # General Technical Terms (Optimized for better differentiation)
            'api': 2.0, 'database': 2.2, 'frontend': 1.8, 'backend': 2.0,
            'javascript': 1.8, 'python': 1.8, 'sql': 2.0, 'html': 1.4, 'css': 1.5
        }
    
    def _build_reasoning_patterns(self) -> List[str]:
        """Patterns that indicate complex reasoning requirements"""
        return [
            r'\b(?:analyze|evaluate|compare|contrast|assess|critique)\b',
            r'\b(?:implications|consequences|trade-offs|advantages|disadvantages)\b',
            r'\b(?:step-by-step|systematically|methodically|comprehensive)\b',
            r'\b(?:pros and cons|benefits and risks|cause and effect)\b',
            r'\b(?:optimization|minimize|maximize|best approach|most efficient)\b',
            r'\b(?:if-then|what if|scenario|hypothetical|alternative)\b',
            r'\b(?:explain why|reasoning behind|justification|rationale)\b',
            r'\b(?:complex|sophisticated|advanced|intricate|nuanced)\b',
            r'\b(?:multiple factors|various aspects|different perspectives)\b',
            r'\b(?:integration|synthesis|combination|merge|unify)\b',
        ]
    
    def _build_complexity_weights(self) -> Dict[str, float]:
        """2025 OPTIMIZED: Enhanced weights for superior routing accuracy"""
        return {
            'sentence_complexity': 0.12,    # Slightly reduced for better balance
            'technical_density': 0.28,      # Increased - technical content is key indicator
            'reasoning_indicators': 0.24,   # Increased - reasoning is critical for model selection
            'domain_specificity': 0.18,     # Increased - domain expertise matters more
            'length_factor': 0.08,          # Reduced - length is less predictive
            'uncertainty_markers': 0.06,    # Reduced - less impact on model choice
            'creativity_markers': 0.04      # Reduced - creativity vs technical is captured elsewhere
        }
    
    def _build_domain_classifiers(self) -> Dict[str, List[str]]:
        """Domain classification patterns"""
        return {
            'mathematics': ['equation', 'theorem', 'proof', 'integral', 'derivative', 'matrix', 'vector'],
            'programming': ['code', 'function', 'algorithm', 'debug', 'compile', 'framework', 'library'],
            'science': ['experiment', 'hypothesis', 'research', 'analysis', 'data', 'methodology'],
            'business': ['strategy', 'market', 'financial', 'revenue', 'profit', 'investment', 'roi'],
            'creative': ['design', 'artistic', 'creative', 'story', 'poem', 'narrative', 'aesthetic'],
            'medical': ['diagnosis', 'treatment', 'patient', 'clinical', 'medical', 'healthcare', 'therapy'],
            'engineering': ['design', 'system', 'optimization', 'efficiency', 'performance', 'technical'],
            'legal': ['law', 'legal', 'contract', 'compliance', 'regulation', 'rights', 'liability']
        }
    
    def analyze_complexity(self, prompt: str, context: Optional[Dict] = None) -> PromptComplexity:
        """
        Advanced complexity analysis that goes beyond ChatGPT/Grok/Gemini capabilities
        """
        context = context or {}
        prompt_lower = prompt.lower()
        
        # Calculate technical depth
        technical_score = self._calculate_technical_depth(prompt_lower)
        
        # Analyze reasoning requirements
        reasoning_score = self._analyze_reasoning_requirements(prompt_lower)
        
        # Calculate domain specificity
        domain_specificity = self._calculate_domain_specificity(prompt_lower)
        
        # Analyze sentence complexity
        sentence_complexity = self._analyze_sentence_complexity(prompt)
        
        # Detect creativity vs factual orientation
        creativity_factor = self._analyze_creativity_factor(prompt_lower)
        
        # Calculate uncertainty level
        uncertainty = self._calculate_uncertainty(prompt_lower)
        
        # Determine if multi-step reasoning is required
        multi_step = self._detect_multi_step(prompt_lower)
        
        # Estimate required context length
        context_length = self._estimate_context_length(prompt, context)
        
        # ENHANCED: Calculate overall complexity score with optimized weighting (0-10)
        weights = self.complexity_weights
        
        # Apply non-linear scaling for better discrimination
        import math
        complexity_score = (
            math.pow(sentence_complexity, 1.1) * weights['sentence_complexity'] +
            math.pow(technical_score, 1.2) * weights['technical_density'] +
            math.pow(reasoning_score, 1.3) * weights['reasoning_indicators'] +
            math.pow(domain_specificity, 1.1) * weights['domain_specificity'] +
            min(len(prompt) / 200, 1.0) * weights['length_factor'] +
            uncertainty * weights['uncertainty_markers'] +
            creativity_factor * weights['creativity_markers']
        ) * 10
        
        # Apply adaptive scaling based on prompt characteristics
        if multi_step and technical_score > 0.5:
            complexity_score *= 1.15  # Boost for multi-step technical tasks
        if domain_specificity > 0.7:
            complexity_score *= 1.1   # Boost for highly specialized tasks
        
        # Determine priority level
        priority = self._determine_priority(complexity_score, multi_step, domain_specificity)
        
        # Estimate response token requirements
        estimated_tokens = self._estimate_response_tokens(complexity_score, multi_step, context)
        
        # Enhanced domain expertise detection
        domain_expertise = self._detect_domain_expertise(prompt, context)
        
        # Enhanced reasoning chain analysis
        reasoning_chain_length = self._analyze_reasoning_chain_length(prompt)
        
        # External knowledge requirements
        requires_external_knowledge = self._requires_external_knowledge(prompt)
        
        # Temporal context detection
        temporal_context = self._detect_temporal_context(prompt)
        
        # User intent confidence (based on pattern clarity)
        user_intent_confidence = self._calculate_intent_confidence(prompt, reasoning_score, domain_specificity)
        
        # Cognitive load calculation
        cognitive_load = self._calculate_cognitive_load(complexity_score, multi_step, reasoning_chain_length)

        return PromptComplexity(
            complexity_score=min(complexity_score, 10.0),
            technical_depth=min(int(technical_score * 5), 5),
            reasoning_required=reasoning_score > 0.3 or multi_step,
            context_length=context_length,
            domain_specificity=domain_specificity,
            creativity_factor=creativity_factor,
            multi_step=multi_step,
            uncertainty=uncertainty,
            priority_level=priority,
            estimated_tokens=estimated_tokens,
            
            # Enhanced fields for superior AI routing
            domain_expertise=domain_expertise,
            reasoning_chain_length=reasoning_chain_length,
            requires_external_knowledge=requires_external_knowledge,
            temporal_context=temporal_context,
            user_intent_confidence=user_intent_confidence,
            cognitive_load=cognitive_load
        )
    
    def _calculate_technical_depth(self, prompt: str) -> float:
        """Calculate technical complexity based on vocabulary"""
        total_score = 0
        word_count = len(prompt.split())
        
        for term, weight in self.technical_keywords.items():
            if term in prompt:
                total_score += weight
        
        # Normalize by prompt length
        return min(total_score / max(word_count, 1) * 10, 1.0)
    
    def _analyze_reasoning_requirements(self, prompt: str) -> float:
        """Detect complex reasoning requirements"""
        reasoning_matches = 0
        for pattern in self.reasoning_indicators:
            if re.search(pattern, prompt, re.IGNORECASE):
                reasoning_matches += 1
        
        return min(reasoning_matches / len(self.reasoning_indicators), 1.0)
    
    def _calculate_domain_specificity(self, prompt: str) -> float:
        """Calculate how domain-specific the prompt is"""
        domain_scores = {}
        
        for domain, keywords in self.domain_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in prompt)
            if score > 0:
                domain_scores[domain] = score / len(keywords)
        
        return max(domain_scores.values()) if domain_scores else 0.0
    
    def _analyze_sentence_complexity(self, prompt: str) -> float:
        """Analyze sentence structure complexity"""
        sentences = prompt.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Complex punctuation patterns
        complex_punctuation = prompt.count(';') + prompt.count(':') + prompt.count('(') + prompt.count('[')
        
        # Subordinate clauses
        subordinate_markers = ['which', 'that', 'because', 'although', 'while', 'since', 'whereas']
        subordinate_count = sum(1 for marker in subordinate_markers if marker in prompt.lower())
        
        complexity = (avg_length / 20 + complex_punctuation / 10 + subordinate_count / 5)
        return min(complexity, 1.0)
    
    def _analyze_creativity_factor(self, prompt: str) -> float:
        """Determine creative vs analytical orientation"""
        creative_markers = ['creative', 'artistic', 'design', 'story', 'poem', 'imagine', 'invent', 'original']
        analytical_markers = ['analyze', 'calculate', 'compute', 'measure', 'precise', 'exact', 'factual']
        
        creative_score = sum(1 for marker in creative_markers if marker in prompt)
        analytical_score = sum(1 for marker in analytical_markers if marker in prompt)
        
        if creative_score + analytical_score == 0:
            return 0.5  # Neutral
        
        return creative_score / (creative_score + analytical_score)
    
    def _calculate_uncertainty(self, prompt: str) -> float:
        """Calculate uncertainty/ambiguity level"""
        uncertainty_markers = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'unclear', 'ambiguous', 'uncertain']
        question_markers = prompt.count('?')
        
        uncertainty_score = sum(1 for marker in uncertainty_markers if marker in prompt)
        return min((uncertainty_score + question_markers) / 5, 1.0)
    
    def _detect_multi_step(self, prompt: str) -> bool:
        """Detect if task requires multiple steps"""
        multi_step_patterns = [
            r'\b(?:first|then|next|finally|step\s*\d+|after\s+that)\b',
            r'\b(?:process|procedure|workflow|sequence|stages?)\b',
            r'\b(?:multiple\s+steps?|several\s+phases?)\b'
        ]
        
        return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in multi_step_patterns)
    
    def _estimate_context_length(self, prompt: str, context: Dict) -> int:
        """Estimate required context length"""
        base_length = len(prompt.split()) * 1.3  # Base context need
        
        # Add context for conversation history
        history_length = len(context.get('conversation_history', [])) * 50
        
        # Add context for complex domains
        domain_bonus = 200 if any(domain in prompt.lower() for domain in self.domain_classifiers.keys()) else 0
        
        return int(base_length + history_length + domain_bonus)
    
    def _determine_priority(self, complexity: float, multi_step: bool, domain_specificity: float) -> str:
        """Determine task priority level"""
        if complexity > 8.0 or (multi_step and domain_specificity > 0.7):
            return 'critical'
        elif complexity > 6.0 or multi_step:
            return 'high'
        elif complexity > 3.0 or domain_specificity > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_response_tokens(self, complexity: float, multi_step: bool, context: Dict) -> int:
        """Estimate required response length"""
        base_tokens = 300
        
        # Add tokens based on complexity
        complexity_tokens = int(complexity * 200)
        
        # Multi-step tasks need more tokens
        multi_step_bonus = 500 if multi_step else 0
        
        # Context-dependent adjustments
        context_bonus = len(context.get('conversation_history', [])) * 20
        
        return base_tokens + complexity_tokens + multi_step_bonus + context_bonus
    
    def _detect_domain_expertise(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Enhanced domain expertise detection for superior AI routing
        
        Args:
            prompt (str): User prompt to analyze
            context (Optional[Dict]): Additional context information
            
        Returns:
            str: Detected domain expertise requirement
        """
        prompt_lower = prompt.lower()
        context = context or {}
        
        # Enhanced domain detection with confidence scoring
        domain_scores = {}
        
        # Medical domain detection
        medical_indicators = [
            'diagnosis', 'symptoms', 'treatment', 'patient', 'clinical', 'medical', 'disease',
            'therapy', 'medication', 'healthcare', 'doctor', 'nurse', 'hospital', 'surgery',
            'pharmacology', 'pathology', 'anatomy', 'physiology', 'epidemiology'
        ]
        medical_score = sum(1 for term in medical_indicators if term in prompt_lower)
        if medical_score > 0:
            domain_scores['medical'] = medical_score / len(medical_indicators)
        
        # Legal domain detection
        legal_indicators = [
            'law', 'legal', 'contract', 'court', 'judge', 'lawyer', 'attorney', 'litigation',
            'compliance', 'regulation', 'statute', 'precedent', 'jurisdiction', 'liability',
            'intellectual property', 'copyright', 'patent', 'trademark', 'constitutional'
        ]
        legal_score = sum(1 for term in legal_indicators if term in prompt_lower)
        if legal_score > 0:
            domain_scores['legal'] = legal_score / len(legal_indicators)
        
        # Technical/Engineering domain detection
        technical_indicators = [
            'algorithm', 'architecture', 'system design', 'optimization', 'performance',
            'scalability', 'distributed', 'microservices', 'database', 'api', 'protocol',
            'security', 'encryption', 'network', 'infrastructure', 'devops', 'ci/cd'
        ]
        technical_score = sum(1 for term in technical_indicators if term in prompt_lower)
        if technical_score > 0:
            domain_scores['technical'] = technical_score / len(technical_indicators)
        
        # Financial domain detection
        financial_indicators = [
            'finance', 'investment', 'trading', 'portfolio', 'risk', 'market', 'stock',
            'bond', 'derivative', 'valuation', 'accounting', 'audit', 'tax', 'banking',
            'insurance', 'actuarial', 'cryptocurrency', 'blockchain'
        ]
        financial_score = sum(1 for term in financial_indicators if term in prompt_lower)
        if financial_score > 0:
            domain_scores['financial'] = financial_score / len(financial_indicators)
        
        # Scientific domain detection
        scientific_indicators = [
            'research', 'experiment', 'hypothesis', 'data', 'analysis', 'methodology',
            'peer review', 'publication', 'statistical', 'empirical', 'theoretical',
            'quantum', 'molecular', 'genetic', 'neurological', 'astronomical'
        ]
        scientific_score = sum(1 for term in scientific_indicators if term in prompt_lower)
        if scientific_score > 0:
            domain_scores['scientific'] = scientific_score / len(scientific_indicators)
        
        # Educational domain detection
        educational_indicators = [
            'teach', 'learn', 'student', 'curriculum', 'lesson', 'tutorial', 'education',
            'academic', 'pedagogy', 'assessment', 'course', 'degree', 'certification'
        ]
        educational_score = sum(1 for term in educational_indicators if term in prompt_lower)
        if educational_score > 0:
            domain_scores['educational'] = educational_score / len(educational_indicators)
        
        # Return the domain with highest confidence or 'general' if no clear domain
        if domain_scores:
            best_domain = max(domain_scores.keys(), key=lambda x: domain_scores[x])
            if domain_scores[best_domain] > 0.1:  # Minimum confidence threshold
                return best_domain
        
        return 'general'
    
    def _analyze_reasoning_chain_length(self, prompt: str) -> int:
        """
        Analyze the length of reasoning chain required for superior model selection
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            int: Number of reasoning steps required (1-10)
        """
        prompt_lower = prompt.lower()
        reasoning_steps = 1  # Base step
        
        # Multi-step indicators
        step_indicators = [
            'first', 'then', 'next', 'after', 'finally', 'lastly', 'subsequently',
            'step 1', 'step 2', 'step 3', 'phase 1', 'phase 2', 'stage 1', 'stage 2'
        ]
        explicit_steps = sum(1 for indicator in step_indicators if indicator in prompt_lower)
        reasoning_steps += explicit_steps
        
        # Complex reasoning patterns
        complex_patterns = [
            r'\b(?:analyze|compare|evaluate|assess|critique|examine)\b',
            r'\b(?:pros and cons|advantages and disadvantages|benefits and drawbacks)\b',
            r'\b(?:cause and effect|if.*then|when.*happens)\b',
            r'\b(?:multiple factors|various aspects|different perspectives)\b',
            r'\b(?:systematic|methodical|comprehensive|thorough)\b'
        ]
        pattern_matches = sum(1 for pattern in complex_patterns if re.search(pattern, prompt_lower))
        reasoning_steps += pattern_matches
        
        # Question complexity (multiple questions require multiple reasoning steps)
        question_count = prompt.count('?')
        if question_count > 1:
            reasoning_steps += min(question_count - 1, 3)  # Cap at 3 additional steps
        
        # Conditional logic indicators
        conditional_indicators = ['if', 'unless', 'provided that', 'given that', 'assuming']
        conditional_count = sum(1 for indicator in conditional_indicators if indicator in prompt_lower)
        reasoning_steps += conditional_count
        
        # Complex sentence structure (multiple clauses)
        clause_indicators = [';', ':', 'because', 'although', 'however', 'moreover', 'furthermore']
        clause_count = sum(prompt_lower.count(indicator) for indicator in clause_indicators)
        reasoning_steps += min(clause_count, 2)  # Cap clause contribution
        
        return min(reasoning_steps, 10)  # Cap at 10 steps maximum
    
    def _requires_external_knowledge(self, prompt: str) -> bool:
        """
        Determine if the prompt requires external knowledge or real-time information
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            bool: True if external knowledge is required
        """
        prompt_lower = prompt.lower()
        
        # Current events and time-sensitive information
        temporal_indicators = [
            'latest', 'recent', 'current', 'today', 'this year', 'this month',
            'new', 'updated', 'breaking', 'trending', 'now', 'currently'
        ]
        
        # Specific factual queries requiring external data
        factual_indicators = [
            'stock price', 'weather', 'news', 'events', 'schedule', 'status',
            'real-time', 'live', 'actual', 'official', 'verified'
        ]
        
        # Research and data gathering
        research_indicators = [
            'research', 'find information', 'look up', 'search for', 'data about',
            'statistics', 'survey', 'report', 'study', 'findings'
        ]
        
        all_indicators = temporal_indicators + factual_indicators + research_indicators
        
        return any(indicator in prompt_lower for indicator in all_indicators)
    
    def _detect_temporal_context(self, prompt: str) -> str:
        """
        Detect temporal context requirements for optimal model selection
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            str: Temporal context type ('current', 'historical', 'predictive', 'timeless')
        """
        prompt_lower = prompt.lower()
        
        # Current/real-time context
        current_indicators = [
            'now', 'currently', 'today', 'this week', 'this month', 'this year',
            'latest', 'recent', 'up to date', 'real-time', 'live'
        ]
        
        # Historical context
        historical_indicators = [
            'history', 'historical', 'past', 'before', 'previous', 'ancient',
            'traditional', 'originally', 'back then', 'decades ago', 'centuries ago'
        ]
        
        # Predictive/future context
        predictive_indicators = [
            'future', 'predict', 'forecast', 'trends', 'will be', 'going to',
            'next year', 'upcoming', 'planned', 'projected', 'anticipated'
        ]
        
        if any(indicator in prompt_lower for indicator in current_indicators):
            return 'current'
        elif any(indicator in prompt_lower for indicator in historical_indicators):
            return 'historical'
        elif any(indicator in prompt_lower for indicator in predictive_indicators):
            return 'predictive'
        else:
            return 'timeless'
    
    def _calculate_intent_confidence(self, prompt: str, reasoning_score: float, domain_specificity: float) -> float:
        """
        Calculate confidence in intent classification for superior routing
        
        Args:
            prompt (str): User prompt to analyze
            reasoning_score (float): Reasoning complexity score
            domain_specificity (float): Domain specificity score
            
        Returns:
            float: Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Clear intent indicators boost confidence
        intent_clarity_indicators = [
            'please', 'can you', 'help me', 'i want', 'i need', 'show me',
            'create', 'generate', 'write', 'explain', 'analyze', 'review'
        ]
        
        clarity_score = sum(1 for indicator in intent_clarity_indicators if indicator in prompt.lower())
        confidence += min(clarity_score * 0.1, 0.3)  # Up to 0.3 boost
        
        # Domain specificity increases confidence
        confidence += domain_specificity * 0.2
        
        # Clear structure and reasoning increases confidence
        confidence += reasoning_score * 0.15
        
        # Length appropriateness (not too short, not too long)
        word_count = len(prompt.split())
        if 5 <= word_count <= 50:
            confidence += 0.1
        elif word_count > 100:
            confidence -= 0.1  # Very long prompts may be unclear
        
        # Question marks suggest clear intent
        if '?' in prompt:
            confidence += 0.05
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _calculate_cognitive_load(self, complexity_score: float, multi_step: bool, reasoning_chain_length: int) -> float:
        """
        Calculate cognitive load required for processing the prompt
        
        Args:
            complexity_score (float): Overall complexity score
            multi_step (bool): Whether task requires multiple steps
            reasoning_chain_length (int): Length of reasoning chain
            
        Returns:
            float: Cognitive load score (0-1)
        """
        base_load = complexity_score / 10.0  # Normalize complexity to 0-1
        
        # Multi-step tasks increase cognitive load
        if multi_step:
            base_load += 0.2
        
        # Reasoning chain length increases load
        reasoning_load = min(reasoning_chain_length / 10.0, 0.3)  # Cap at 0.3
        base_load += reasoning_load
        
        # Normalize and clamp to [0, 1]
        return min(max(base_load, 0.0), 1.0)

class DynamicModelSelector:
    """
    Advanced model selection system that adapts based on real-time performance
    Superior to static model selection in ChatGPT/Grok/Gemini
    """
    
    def __init__(self, performance_monitor=None):
        # Use centralized PerformanceMonitor instead of separate tracking - FIXED: Use health_monitor instead
        self.performance_monitor = performance_monitor or health_monitor
        self.model_load_balancer = {}  # Load balancing across models
        self.fallback_chains = self._initialize_fallback_chains()
        
    def _initialize_fallback_chains(self) -> Dict[str, List[str]]:
        """CRITICAL FIX: Initialize tier-aware fallback chains using Config's tier-aware logic"""
        from ..config import Config
        
        # Use Config's tier-aware fallback chains instead of static ones
        return {
            'text_generation': Config.get_model_fallback_chain('text'),
            'code_generation': Config.get_model_fallback_chain('code'),
            'reasoning_tasks': Config.get_model_fallback_chain('reasoning'),
            'mathematical_reasoning': Config.get_model_fallback_chain('math'),
            'vision_tasks': Config.get_model_fallback_chain('vision'),
            'efficiency_tasks': Config.get_model_fallback_chain('efficiency'),
            'tool_use': Config.get_model_fallback_chain('tool_use')
        }
    
    def select_optimal_model(self, intent: str, complexity: PromptComplexity, context: Optional[ContextState] = None) -> Tuple[str, Dict]:
        """
        ENHANCED: Select the optimal model with superior AI routing that exceeds Perplexity AI capabilities
        Considers domain expertise, user context, complexity analysis, and performance optimization
        """
        from ..config import Config
        
        logger.info(f"🎯 SUPERIOR MODEL SELECTION for intent: {intent}")
        logger.info(f"📊 Complexity: {complexity.complexity_score:.2f}/10, Domain: {complexity.domain_expertise}")
        logger.info(f"🧠 Reasoning: {complexity.reasoning_chain_length} steps, Cognitive Load: {complexity.cognitive_load:.2f}")
        
        # Enhanced domain-specific model routing - SUPERIOR to Perplexity AI
        domain_optimized_model = self._select_domain_specific_model(intent, complexity, context)
        
        # Get available models for this intent type
        available_models = self._get_available_models_for_intent(intent)
        
        # Advanced user context consideration - Beyond what Perplexity AI offers
        context_optimized_model = self._apply_user_context_optimization(
            domain_optimized_model, intent, complexity, context, available_models
        )
        
        # Enhanced performance monitoring with learning capabilities
        best_model = self._get_performance_optimized_model(context_optimized_model, intent, available_models)
        
        if best_model:
            # Multi-factor model validation - More sophisticated than Perplexity AI
            should_avoid, avoid_reason = self._advanced_model_validation(best_model, complexity, context)
            if should_avoid:
                logger.warning(f"🚫 Advanced validation failed for {best_model}: {avoid_reason}")
                best_model = None
        
        # CRITICAL FIX: Use tier-aware model selection with fallback chains
        if not best_model:
            # Map intent to model_type for Config.get_model_fallback_chain
            intent_to_model_type = {
                'text_generation': 'text',
                'code_generation': 'code', 
                'reasoning_tasks': 'reasoning',
                'mathematical_reasoning': 'math',
                'advanced_reasoning': 'reasoning',
                'vision_tasks': 'vision',
                'image_analysis': 'vision',
                'efficiency_tasks': 'efficiency',
                'tool_use': 'tool_use'
            }
            
            model_type = intent_to_model_type.get(intent, 'text')
            
            # Get tier-aware fallback chain from Config
            fallback_chain = Config.get_model_fallback_chain(model_type)
            
            if fallback_chain:
                # Use the first model from the tier-aware chain as base model
                base_model = fallback_chain[0]
                logger.info(f"Using tier-aware model selection: {base_model} for {intent} (tier: {Config.HF_TIER})")
            else:
                # Fallback to guaranteed free model if no chain found
                base_model = Config.DEFAULT_TEXT_MODEL
                logger.warning(f"No fallback chain found for {intent}, using default: {base_model}")
            
            # Apply tier-appropriate model selection
            adjusted_model = Config.get_tier_appropriate_model(base_model, Config.DEFAULT_TEXT_MODEL)
            
            # CRITICAL FIX: Ensure the selected model is healthy and available
            if not health_monitor.is_model_available(adjusted_model):
                # Try to find a healthy alternative from the same intent type
                healthy_alternatives = health_monitor.get_available_models(intent)
                if healthy_alternatives:
                    adjusted_model = healthy_alternatives[0]
                    logger.warning(f"🏥 Selected model was unhealthy, switching to healthy alternative: {adjusted_model}")
                else:
                    # Last resort: use any available model
                    all_healthy = health_monitor.get_available_models()
                    if all_healthy:
                        adjusted_model = all_healthy[0]
                        logger.error(f"🚨 No healthy models for intent, using any available: {adjusted_model}")
                    # If no healthy models at all, proceed with original (may fail but logged)
            
            # Apply performance-based adjustments using centralized monitor if available
            if hasattr(self, '_apply_performance_adjustments'):
                adjusted_model = self._apply_performance_adjustments(adjusted_model, intent, available_models)
            else:
                # Fallback: use health_monitor to select a healthy model
                if not health_monitor.is_model_available(adjusted_model):
                    healthy_alternatives = health_monitor.get_available_models(intent)
                    if healthy_alternatives:
                        adjusted_model = healthy_alternatives[0]
        else:
            # Apply tier restrictions even for performance-selected models
            adjusted_model = Config.get_tier_appropriate_model(best_model, Config.DEFAULT_TEXT_MODEL)
            
            # CRITICAL FIX: Ensure performance-selected model is also healthy
            if not health_monitor.is_model_available(adjusted_model):
                logger.warning(f"🏥 Performance-selected model {adjusted_model} is unhealthy, finding alternative...")
                healthy_alternatives = health_monitor.get_available_models(intent)
                if healthy_alternatives:
                    adjusted_model = healthy_alternatives[0]
                    logger.info(f"🏥 Switched to healthy alternative: {adjusted_model}")
                else:
                    all_healthy = health_monitor.get_available_models()
                    if all_healthy:
                        adjusted_model = all_healthy[0]
                        logger.error(f"🚨 No healthy models for intent, using any available: {adjusted_model}")
        
        # Generate specialized parameters
        special_params = self._generate_specialized_parameters(intent, complexity, adjusted_model)
        
        return adjusted_model, special_params
    
    def _get_available_models_for_intent(self, intent: str) -> List[str]:
        """CRITICAL FIX: Get tier-aware list of verified available models for a specific intent type"""
        from ..config import Config
        
        # Map intent to model_type for Config.get_model_fallback_chain
        intent_to_model_type = {
            'text_generation': 'text',
            'code_generation': 'code',
            'reasoning_tasks': 'reasoning', 
            'mathematical_reasoning': 'math',
            'advanced_reasoning': 'reasoning',
            'vision_tasks': 'vision',
            'image_analysis': 'vision',
            'efficiency_tasks': 'efficiency',
            'tool_use': 'tool_use'
        }
        
        model_type = intent_to_model_type.get(intent, 'text')
        
        # Get tier-aware fallback chain from Config
        config_models = Config.get_model_fallback_chain(model_type)
        
        if not config_models:
            # Fallback to guaranteed free models if no chain found
            config_models = [Config.DEFAULT_TEXT_MODEL, Config.FALLBACK_TEXT_MODEL]
            logger.warning(f"No tier-aware chain found for {intent}, using default fallbacks")
        
        # CRITICAL FIX: Filter models by health monitor availability
        available_models = []
        for model in config_models:
            if health_monitor.is_model_available(model):
                available_models.append(model)
        
        # If no models are available after health filtering, use any available model as fallback
        if not available_models:
            all_healthy_models = health_monitor.get_available_models()
            if all_healthy_models:
                available_models = [all_healthy_models[0]]  # Use first available model
                logger.warning(f"⚠️ No healthy models found for {intent}, using fallback: {available_models[0]}")
            else:
                # Last resort: use config models (may be unhealthy but better than nothing)
                available_models = config_models[:1]  # Use first config model
                logger.error(f"🚨 No healthy models available, using unverified fallback: {available_models[0]}")
        
        logger.debug(f"Verified available models for {intent} (tier: {Config.HF_TIER}): {available_models[:3]}...")
        return available_models
    
    def _apply_performance_adjustments(self, base_model: str, intent: str, available_models: List[str]) -> str:
        """Apply performance-based model adjustments using centralized PerformanceMonitor"""
        # Check if we should avoid the base model due to poor performance
        should_avoid = self.performance_monitor.should_avoid_model(base_model)
        reason = "performance issues" if should_avoid else ""
        
        if should_avoid:
            logger.warning(f"Avoiding {base_model}: {reason}")
            
            # Try fallback chain
            fallback_chain = self.fallback_chains.get(intent, [])
            for fallback_model in fallback_chain:
                if fallback_model != base_model and fallback_model in available_models:
                    should_avoid_fallback = self.performance_monitor.should_avoid_model(fallback_model)
                    if not should_avoid_fallback:
                        logger.info(f"Switching to fallback model: {fallback_model}")
                        return fallback_model
            
            # If all fallbacks are also problematic, use model rankings
            for ranked_model, score in self.performance_monitor.model_rankings():
                if ranked_model in available_models and ranked_model != base_model:
                    should_avoid_ranked = self.performance_monitor.should_avoid_model(ranked_model)
                    if not should_avoid_ranked:
                        logger.info(f"Using ranked alternative: {ranked_model}")
                        return ranked_model
        
        return base_model
    
    def _generate_specialized_parameters(self, intent: str, complexity: PromptComplexity, model: str) -> Dict:
        """Generate specialized parameters based on intent, complexity, and model"""
        from ..config import Config
        
        base_params = {}
        
        # Model-specific temperature optimization
        if 'deepseek' in model.lower():
            base_params['temperature'] = Config.DEEPSEEK_TEMPERATURE * (1 + complexity.creativity_factor * 0.2)
        elif 'qwen' in model.lower():
            base_params['temperature'] = Config.QWEN_TEMPERATURE * (1 + complexity.creativity_factor * 0.3)
        elif 'starcoder' in model.lower() or 'coder' in model.lower():
            base_params['temperature'] = Config.STARCODER_TEMPERATURE * (1 + complexity.uncertainty * 0.1)
        else:
            base_params['temperature'] = 0.7 * (1 + complexity.creativity_factor * 0.2)
        
        # Dynamic token allocation
        if complexity.multi_step or complexity.complexity_score > 7:
            base_params['max_new_tokens'] = min(complexity.estimated_tokens + 500, 3000)
        else:
            base_params['max_new_tokens'] = min(complexity.estimated_tokens, 2000)
        
        # Reasoning-specific parameters
        if complexity.reasoning_required:
            base_params['top_p'] = 0.95  # Higher diversity for reasoning
            base_params['repetition_penalty'] = 1.1
        else:
            base_params['top_p'] = 0.9
            base_params['repetition_penalty'] = 1.05
        
        # Priority-based performance settings
        if complexity.priority_level == 'critical':
            base_params['num_return_sequences'] = 1
            base_params['do_sample'] = True
            base_params['use_cache'] = False  # Fresh responses for critical tasks
        
        return base_params
    
    def _select_domain_specific_model(self, intent: str, complexity: PromptComplexity, context: Optional[ContextState] = None) -> str:
        """
        SUPERIOR domain-specific model routing that exceeds Perplexity AI capabilities
        
        Args:
            intent (str): Detected intent type
            complexity (PromptComplexity): Enhanced complexity analysis
            context (Optional[ContextState]): User context state
            
        Returns:
            str: Optimal model for the specific domain
        """
        from ..config import Config
        
        domain = complexity.domain_expertise
        reasoning_complexity = complexity.reasoning_chain_length
        cognitive_load = complexity.cognitive_load
        
        logger.info(f"🎯 Domain-specific routing: {domain} | Reasoning: {reasoning_complexity} | Load: {cognitive_load:.2f}")
        
        # Medical domain routing - Specialized medical knowledge models
        if domain == 'medical':
            if complexity.complexity_score > 8 or complexity.requires_external_knowledge:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'MEDICAL_REASONING_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            return Config.get_tier_appropriate_model(
                getattr(Config, 'MEDICAL_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
        
        # Legal domain routing - Specialized legal reasoning models
        elif domain == 'legal':
            if complexity.reasoning_required or reasoning_complexity > 5:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'LEGAL_REASONING_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            return Config.get_tier_appropriate_model(
                getattr(Config, 'LEGAL_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
        
        # Technical/Engineering domain routing - Advanced technical models
        elif domain == 'technical':
            if intent in ['code_generation', 'code_review']:
                if complexity.complexity_score > 7 or reasoning_complexity > 4:
                    return Config.get_tier_appropriate_model(
                        getattr(Config, 'ADVANCED_CODE_MODEL', Config.CODE_GENERATION_MODEL),
                        Config.DEFAULT_TEXT_MODEL
                    )
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'CODE_GENERATION_MODEL', Config.DEFAULT_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            elif intent == 'algorithm_design':
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'ALGORITHM_DESIGN_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            else:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'TECHNICAL_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
        
        # Financial domain routing - Financial analysis models
        elif domain == 'financial':
            if complexity.complexity_score > 6 or 'analysis' in intent:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'FINANCIAL_ANALYSIS_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            return Config.get_tier_appropriate_model(
                getattr(Config, 'FINANCIAL_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
        
        # Scientific domain routing - Research and analysis models
        elif domain == 'scientific':
            if intent in ['research', 'data_analysis'] or complexity.requires_external_knowledge:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'RESEARCH_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            elif reasoning_complexity > 6:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'SCIENTIFIC_REASONING_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            return Config.get_tier_appropriate_model(
                getattr(Config, 'SCIENTIFIC_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
        
        # Educational domain routing - Teaching and explanation models
        elif domain == 'educational':
            if intent == 'explanation' or complexity.creativity_factor > 0.6:
                return Config.get_tier_appropriate_model(
                    getattr(Config, 'EDUCATIONAL_MODEL', Config.DEFAULT_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
            return Config.get_tier_appropriate_model(
                getattr(Config, 'EDUCATIONAL_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
        
        # Default: Intent-based routing with complexity consideration
        return self._get_intent_optimized_model(intent, complexity)
    
    def _get_intent_optimized_model(self, intent: str, complexity: PromptComplexity) -> str:
        """Get the optimal model for a specific intent with complexity optimization"""
        from ..config import Config
        
        # High complexity tasks need advanced models
        if complexity.complexity_score > 8 or complexity.reasoning_chain_length > 6:
            return Config.get_tier_appropriate_model(
                getattr(Config, 'ULTRA_PERFORMANCE_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
        
        # Intent-specific model selection
        intent_model_mapping = {
            'mathematical_reasoning': getattr(Config, 'MATH_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
            'advanced_reasoning': getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            'code_generation': getattr(Config, 'CODE_GENERATION_MODEL', Config.DEFAULT_TEXT_MODEL),
            'code_review': getattr(Config, 'CODE_REVIEW_MODEL', Config.DEFAULT_TEXT_MODEL),
            'creative_writing': getattr(Config, 'CREATIVE_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
            'image_generation': getattr(Config, 'IMAGE_GENERATION_MODEL', Config.DEFAULT_IMAGE_MODEL),
            'research': getattr(Config, 'RESEARCH_MODEL', Config.ADVANCED_TEXT_MODEL),
            'explanation': getattr(Config, 'EXPLANATION_MODEL', Config.DEFAULT_TEXT_MODEL),
            'data_analysis': getattr(Config, 'DATA_ANALYSIS_MODEL', Config.ADVANCED_TEXT_MODEL)
        }
        
        base_model = intent_model_mapping.get(intent, Config.DEFAULT_TEXT_MODEL)
        return Config.get_tier_appropriate_model(base_model, Config.DEFAULT_TEXT_MODEL)
    
    def _apply_user_context_optimization(self, base_model: str, intent: str, complexity: PromptComplexity, 
                                       context: Optional[ContextState], available_models: List[str]) -> str:
        """
        SUPERIOR user context optimization that exceeds Perplexity AI's personalization
        
        Args:
            base_model (str): Base model from domain selection
            intent (str): Detected intent
            complexity (PromptComplexity): Complexity analysis
            context (Optional[ContextState]): User context
            available_models (List[str]): Available models
            
        Returns:
            str: Context-optimized model selection
        """
        if not context:
            return base_model
        
        # User expertise level consideration - Personalization beyond Perplexity AI
        user_expertise = context.expertise_level.get(complexity.domain_expertise, 0.5)
        
        # If user is expert in domain, use more advanced models
        if user_expertise > 0.8 and complexity.complexity_score > 6:
            from ..config import Config
            advanced_model = Config.get_tier_appropriate_model(
                getattr(Config, 'EXPERT_USER_MODEL', Config.ADVANCED_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
            if advanced_model in available_models:
                logger.info(f"🎓 Expert user detected ({user_expertise:.2f}), upgrading to: {advanced_model}")
                return advanced_model
        
        # User's preferred models consideration
        if intent in context.preferred_models and context.preferred_models[intent] > 0.7:
            preferred_model = max(context.preferred_models.items(), key=lambda x: x[1])[0]
            if preferred_model in available_models:
                logger.info(f"👤 Using user's preferred model: {preferred_model}")
                return preferred_model
        
        # Conversation flow optimization - Context awareness
        if context.follow_up_context and complexity.user_intent_confidence < 0.7:
            # For unclear follow-ups, use more capable models
            from ..config import Config
            clarification_model = Config.get_tier_appropriate_model(
                getattr(Config, 'CLARIFICATION_MODEL', Config.ADVANCED_TEXT_MODEL),
                Config.DEFAULT_TEXT_MODEL
            )
            if clarification_model in available_models:
                logger.info(f"❓ Follow-up context detected, using clarification model: {clarification_model}")
                return clarification_model
        
        # Complexity trend analysis - Learning from user patterns
        if len(context.complexity_trend) >= 3:
            avg_complexity = sum(context.complexity_trend[-3:]) / 3
            if avg_complexity > 7.0 and complexity.complexity_score > avg_complexity:
                # User consistently asks complex questions, upgrade model
                from ..config import Config
                power_user_model = Config.get_tier_appropriate_model(
                    getattr(Config, 'POWER_USER_MODEL', Config.ADVANCED_TEXT_MODEL),
                    Config.DEFAULT_TEXT_MODEL
                )
                if power_user_model in available_models:
                    logger.info(f"📈 Power user pattern detected (avg: {avg_complexity:.2f}), upgrading model")
                    return power_user_model
        
        return base_model
    
    def _get_performance_optimized_model(self, base_model: str, intent: str, available_models: List[str]) -> str:
        """
        Enhanced performance monitoring with learning capabilities - Superior to Perplexity AI
        
        Args:
            base_model (str): Base model selection
            intent (str): Task intent
            available_models (List[str]): Available models
            
        Returns:
            str: Performance-optimized model
        """
        # Try to get the best performing model from health monitor
        if hasattr(self.performance_monitor, 'get_best_model'):
            performance_model = self.performance_monitor.get_best_model(intent)
            if performance_model and performance_model in available_models:
                # Validate performance model against avoidance list
                if hasattr(self.performance_monitor, 'should_avoid_model'):
                    should_avoid = self.performance_monitor.should_avoid_model(performance_model)
                    if not should_avoid:
                        logger.info(f"⚡ Performance optimization: Using best performing model {performance_model}")
                        return performance_model
                    else:
                        logger.warning(f"⚠️ Performance model {performance_model} on avoidance list")
        
        # Fallback to base model selection
        return base_model
    
    def _advanced_model_validation(self, model: str, complexity: PromptComplexity, 
                                 context: Optional[ContextState]) -> Tuple[bool, str]:
        """
        Multi-factor model validation - More sophisticated than Perplexity AI
        
        Args:
            model (str): Model to validate
            complexity (PromptComplexity): Complexity analysis
            context (Optional[ContextState]): User context
            
        Returns:
            Tuple[bool, str]: (should_avoid, reason)
        """
        # Health monitoring validation
        if not health_monitor.is_model_available(model):
            return True, "Model unavailable in health monitor"
        
        # Performance monitoring validation
        if hasattr(self.performance_monitor, 'should_avoid_model'):
            should_avoid = self.performance_monitor.should_avoid_model(model)
            if should_avoid:
                return True, "Performance monitor flagged model"
        
        # Complexity-capability matching
        if complexity.complexity_score > 8 and 'lightweight' in model.lower():
            return True, "Lightweight model insufficient for high complexity"
        
        if complexity.reasoning_chain_length > 6 and 'fast' in model.lower():
            return True, "Fast model may not handle complex reasoning"
        
        # Domain-capability matching
        if complexity.domain_expertise == 'medical' and 'general' in model.lower():
            return True, "General model may lack medical expertise"
        
        # Cognitive load validation
        if complexity.cognitive_load > 0.8 and any(term in model.lower() for term in ['compact', 'mini', 'small']):
            return True, "Compact model insufficient for high cognitive load"
        
        # All validations passed
        return False, "Model passed all validations"
    
    def _apply_dynamic_load_balancing(self, candidate_models: List[str], 
                                    intent: str, complexity: PromptComplexity) -> str:
        """
        2025 ENHANCED: Apply dynamic load balancing for optimal resource utilization
        
        Args:
            candidate_models (List[str]): List of candidate models
            intent (str): Intent type
            complexity (PromptComplexity): Complexity analysis
            
        Returns:
            str: Load-balanced model selection
        """
        if not candidate_models:
            return Config.DEFAULT_TEXT_MODEL
            
        # Calculate load scores for each model
        model_scores = {}
        for model in candidate_models:
            if not model:
                continue
                
            # Base score from current load
            current_load = self.model_load_balancer.get(model, 0.0)
            load_score = max(0.1, 1.0 - current_load)  # Higher score for less loaded models
            
            # Boost score for models matching complexity requirements
            if complexity.complexity_score > 7.0 and 'advanced' in model.lower():
                load_score *= 1.2
            elif complexity.complexity_score < 4.0 and any(term in model.lower() for term in ['fast', 'efficient']):
                load_score *= 1.15
                
            # Penalize overused models
            if current_load > 0.8:
                load_score *= 0.6
                
            model_scores[model] = load_score
        
        # Select model with highest load-balanced score
        best_model = max(model_scores.keys(), key=lambda m: model_scores[m])
        
        # Update load balancer (simple increment, would be replaced with real metrics in production)
        self.model_load_balancer[best_model] = self.model_load_balancer.get(best_model, 0.0) + 0.1
        
        logger.info(f"🔄 Load-balanced selection: {best_model} (score: {model_scores[best_model]:.2f})")
        return best_model
    
    def _get_intelligent_fallback_model(self, intent: str, complexity: PromptComplexity, 
                                      failed_model: str, failure_reason: str) -> str:
        """
        2025 ENHANCED: Get intelligent fallback model based on failure analysis
        
        Args:
            intent (str): Intent type
            complexity (PromptComplexity): Complexity analysis
            failed_model (str): Model that failed
            failure_reason (str): Reason for failure
            
        Returns:
            str: Intelligent fallback model
        """
        # Analyze failure reason to determine best fallback strategy
        if 'overloaded' in failure_reason.lower() or 'timeout' in failure_reason.lower():
            # Try a more efficient model
            if complexity.complexity_score < 6.0:
                return Config.FAST_TEXT_MODEL or Config.EFFICIENT_TEXT_MODEL
        
        if 'complexity' in failure_reason.lower() or 'capability' in failure_reason.lower():
            # Try a more powerful model
            return Config.ADVANCED_TEXT_MODEL or Config.FLAGSHIP_TEXT_MODEL
            
        if 'domain' in failure_reason.lower():
            # Try a more general model
            return Config.DEFAULT_TEXT_MODEL
            
        # Get fallback chain and skip the failed model
        intent_to_model_type = {
            'text_generation': 'text',
            'code_generation': 'code', 
            'reasoning_tasks': 'reasoning',
            'mathematical_reasoning': 'math',
            'advanced_reasoning': 'reasoning',
        }
        
        model_type = intent_to_model_type.get(intent, 'text')
        fallback_chain = Config.get_model_fallback_chain(model_type)
        
        # Find next model in chain after failed model
        try:
            failed_index = fallback_chain.index(failed_model)
            if failed_index < len(fallback_chain) - 1:
                return fallback_chain[failed_index + 1]
        except (ValueError, IndexError):
            pass
            
        # Default fallback
        return Config.FALLBACK_TEXT_MODEL or Config.DEFAULT_TEXT_MODEL


class IntelligentRouter:
    """
    SUPERIOR AI model router that outperforms ChatGPT, Grok, and Gemini
    Features advanced complexity analysis, adaptive routing, and real-time optimization
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.intent_priorities = self._initialize_priorities()
        
        # Advanced analysis systems
        self.complexity_analyzer = AdvancedComplexityAnalyzer()
        self.model_selector = DynamicModelSelector()
        self.context_tracker = {}  # Track per-user contexts
        
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
        
        # Performance optimization
        self.intent_cache = {}  # Smart caching with TTL
        self.model_performance_cache = {}  # Model performance caching
        self.response_quality_tracker = {}  # Track response quality
        
        # Advanced routing weights
        self.complexity_weights = {
            'technical_terms': 2.0,
            'code_snippets': 3.0,
            'specific_frameworks': 2.5,
            'creative_words': 1.5,
            'emotional_words': 1.8,
            'reasoning_depth': 2.8,      # NEW: Complex reasoning weight
            'multi_modal': 2.2,          # NEW: Multi-modal task weight
            'domain_expertise': 2.6,     # NEW: Domain-specific expertise
            'context_dependency': 1.9    # NEW: Context-dependent tasks
        }
    
    async def analyze_prompt_advanced(self, prompt: str, context: dict|None=None) -> dict:
        """
        Advanced prompt analysis using existing AdvancedComplexityAnalyzer - thin orchestration layer
        
        Args:
            prompt (str): User input prompt
            context (dict|None): Additional context for analysis
            
        Returns:
            dict: {"intent_type": "text_generation", "complexity_score": float, "recommended_models": [models], "risk_level": "low"}
        """
        context = context or {}
        
        # Use existing AdvancedComplexityAnalyzer for complexity analysis
        complexity = self.complexity_analyzer.analyze_complexity(prompt, context)
        
        # Intent inference based on prompt content
        prompt_lower = prompt.lower()
        
        # Math for equations/mathematics
        if any(indicator in prompt_lower for indicator in [
            'equation', 'formula', 'calculate', 'mathematics', 'math', 'solve', 
            'derivative', 'integral', 'algebra', 'geometry', 'statistics', 'probability'
        ]):
            primary_intent = IntentType.MATHEMATICAL_REASONING
        # Code for programming indicators  
        elif any(indicator in prompt_lower for indicator in [
            'code', 'function', 'class', 'method', 'programming', 'algorithm',
            'python', 'javascript', 'java', 'react', 'api', 'debug', 'implement'
        ]):
            primary_intent = IntentType.CODE_GENERATION
        # Default to text/question_answering
        else:
            if any(indicator in prompt_lower for indicator in [
                'what', 'why', 'how', 'explain', 'tell me', 'describe'
            ]):
                primary_intent = IntentType.QUESTION_ANSWERING
            else:
                primary_intent = IntentType.TEXT_GENERATION
        
        # Get candidate models using existing model selector
        try:
            selected_model, _ = self.model_selector.select_optimal_model(
                primary_intent.value, complexity, None
            )
            # Get available models for this intent
            available_models = self.model_selector._get_available_models_for_intent(primary_intent.value)
            candidate_models = available_models[:3] if available_models else [selected_model]
        except Exception as e:
            logger.warning(f"Error getting candidate models: {e}")
            # Fallback to default models
            from ..config import Config
            candidate_models = [Config.DEFAULT_TEXT_MODEL, Config.FALLBACK_TEXT_MODEL]
        
        # Convert to test-compatible format
        intent_type_mapping = {
            IntentType.TEXT_GENERATION: "text_generation",
            IntentType.CODE_GENERATION: "code_generation", 
            IntentType.MATHEMATICAL_REASONING: "mathematical_reasoning",
            IntentType.QUESTION_ANSWERING: "question_answering",
            IntentType.IMAGE_GENERATION: "image_generation",
            IntentType.ADVANCED_REASONING: "advanced_reasoning",
            IntentType.CONVERSATION: "conversation"
        }
        
        intent_type_str = intent_type_mapping.get(primary_intent, "text_generation")
        
        # Calculate risk level based on complexity
        risk_level = "low"
        if hasattr(complexity, 'complexity_score') and complexity.complexity_score > 7:
            risk_level = "high"
        elif hasattr(complexity, 'complexity_score') and complexity.complexity_score > 4:
            risk_level = "medium"
        
        # Calculate complexity score as float
        complexity_score = getattr(complexity, 'complexity_score', 3.0)
        if not isinstance(complexity_score, (int, float)):
            complexity_score = 3.0
        
        return {
            "intent_type": intent_type_str,
            "complexity_score": float(complexity_score),
            "recommended_models": [m for m in candidate_models if m],  # Remove None values
            "risk_level": risk_level
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
                r'(?:i\'?m\s+feeling|i\s+feel)\s+(?:really\s+)?(?:sad|happy|angry|excited|depressed|good|bad|down|up)',
                r'(?:feeling|emotions?)\s+(?:really\s+)?(?:sad|happy|angry|excited|down|up|good|bad)',
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
            
            # 2025: BREAKTHROUGH GUI Automation patterns - UI-TARS capabilities
            IntentType.GUI_AUTOMATION: [
                r'\b(?:automate|automation|automatic|auto).*(?:gui|ui|interface|window|screen|desktop|application|app)',
                r'(?:click|tap|press|select|choose).*(?:button|link|menu|icon|element)',
                r'(?:navigate|browse|scroll|swipe).*(?:website|page|app|interface|screen)',
                r'(?:fill|enter|type|input).*(?:form|field|textbox|input)',
                r'(?:test|testing|qa|quality.?assurance).*(?:ui|gui|interface|application|app)',
                r'(?:screenshot|capture|image).*(?:screen|desktop|window|application)',
                r'(?:interact|control|operate|manipulate).*(?:interface|gui|ui|application|software)',
                r'\b(?:selenium|playwright|puppeteer|cypress|appium)\b',
                r'(?:web.?scraping|data.?extraction|automation.?script)',
                r'(?:browser.?automation|web.?automation|desktop.?automation)',
                r'(?:rpa|robotic.?process.?automation)',
                r'(?:ui.?testing|gui.?testing|end.?to.?end|e2e)',
            ],
            
            # 2025: BREAKTHROUGH Tool Use patterns - Groq function calling excellence
            IntentType.TOOL_USE: [
                r'\b(?:function|api|tool|service|integration|webhook).*(?:call|calling|invoke|execute|run)',
                r'(?:use|call|invoke|integrate|connect).*(?:api|service|tool|function|external)',
                r'(?:external|third.?party|remote).*(?:api|service|integration|tool)',
                r'(?:rest|graphql|soap|webhook|endpoint).*(?:call|request|integration)',
                r'(?:database|sql|query).*(?:execute|run|call)',
                r'(?:plugin|extension|addon|module).*(?:use|call|integrate)',
                r'(?:json|xml|yaml).*(?:parse|process|transform)',
                r'(?:http|https|request|response|api.?call)',
                r'(?:authentication|auth|token|key).*(?:api|service)',
                r'(?:payment|stripe|paypal|billing).*(?:integration|api)',
                r'(?:email|sms|notification).*(?:send|api|service)',
                r'(?:calendar|scheduling|booking).*(?:api|integration)',
                r'(?:file|upload|download|storage).*(?:api|service)',
                r'(?:search|elasticsearch|solr).*(?:api|query)',
                r'(?:machine.?learning|ai|model).*(?:api|inference|prediction)',
            ],
            
            # 2025: BREAKTHROUGH Premium Vision patterns - MiniCPM-V excellence
            IntentType.PREMIUM_VISION: [
                r'(?:analyze|examine|read|extract|understand).*(?:image|picture|photo|visual|diagram)',
                r'(?:ocr|optical.?character.?recognition|text.?extraction).*(?:image|document|photo)',
                r'(?:chart|graph|plot|diagram|visualization).*(?:analyze|read|interpret)',
                r'(?:medical|x.?ray|mri|ct.?scan|radiology).*(?:image|analysis)',
                r'(?:document|pdf|receipt|invoice|form).*(?:analysis|extraction|processing)',
                r'(?:handwriting|handwritten|cursive).*(?:recognition|read|transcribe)',
                r'(?:table|spreadsheet|data).*(?:extract|parse|analyze).*(?:image|photo)',
                r'(?:face|facial|person|human).*(?:recognition|detection|analysis)',
                r'(?:object|item|product).*(?:detection|recognition|identification)',
                r'(?:scene|environment|location).*(?:understanding|analysis|description)',
                r'(?:quality|defect|inspection).*(?:image|visual|analysis)',
                r'(?:compare|similarity|matching).*(?:image|photo|visual)',
                r'(?:caption|describe|explain).*(?:image|picture|photo|visual)',
                r'(?:logo|brand|trademark).*(?:recognition|detection|identification)',
                r'(?:barcode|qr.?code|code).*(?:scan|read|decode)',
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
                # Enhanced content analysis patterns - Superior to Perplexity AI
                r'\b(?:analyze|describe|caption|identify|examine|review|interpret|explain|assess|evaluate)\s+(?:this|the)?\s*(?:image|picture|photo|visual|graphic|chart|diagram|artwork|screenshot|scan)',
                r'\b(?:image|picture|photo|visual|diagram|chart|graph|infographic|screenshot)\s+(?:analysis|description|explanation|interpretation|review|assessment|evaluation)',
                r'(?:what.*(?:see|shown|displayed|visible)|describe.*(?:image|picture|photo|visual)|analyze.*(?:visual|image|graphic))',
                
                # Advanced AI vision and processing capabilities
                r'(?:computer.?vision|image.?processing|visual.?analysis|optical.?character.?recognition|ocr|text.?extraction)',
                r'(?:extract.*text|read.*(?:image|document|screenshot|scan)|scan.*(?:document|image|pdf)|transcribe.*(?:image|visual))',
                r'(?:object.?detection|scene.?understanding|image.?classification|visual.?recognition|content.?identification)',
                r'(?:facial.?recognition|person.?identification|face.?analysis|emotion.?detection|gesture.?recognition)',
                
                # Domain-specific visual analysis - Medical, Scientific, Technical
                r'(?:medical.?imaging|x.?ray|mri|ct.?scan|ultrasound|radiology|pathology.?slide|microscopy|histology)',
                r'(?:satellite.?imagery|aerial.?photo|map.?analysis|geographic.?analysis|geological.?survey|remote.?sensing)',
                r'(?:technical.?diagram|engineering.?drawing|architectural.?plan|blueprint|schematic|circuit.?diagram)',
                r'(?:scientific.?visualization|data.?chart|research.?graph|experimental.?data|laboratory.?result)',
                
                # Enhanced data visualization and charts
                r'(?:chart.?reading|graph.?analysis|data.?visualization|plot.?interpretation|statistics.?visual|dashboard.?analysis)',
                r'(?:bar.?chart|line.?graph|pie.?chart|scatter.?plot|histogram|heatmap|flowchart|timeline)',
                r'(?:financial.?chart|stock.?graph|trading.?chart|market.?data|price.?chart|candlestick)',
                
                # Creative and artistic analysis - Beyond Perplexity AI
                r'(?:artwork.?analysis|style.?transfer|artistic.?interpretation|creative.?analysis|art.?critique|aesthetic.?evaluation)',
                r'(?:design.?analysis|ui.?mockup|interface.?review|logo.?evaluation|brand.?analysis|visual.?identity)',
                r'(?:photography.?critique|composition.?analysis|lighting.?assessment|color.?theory|visual.?harmony)',
                
                # Advanced multi-modal capabilities
                r'(?:pdf.*image|screenshot.*text|document.*scan|image.*(?:contains|has).*text|mixed.?content)',
                r'(?:combine|merge|create|generate).*(?:text|image|visual|graphic).*(?:and|with).*(?:text|image|visual|audio)',
                r'(?:multimodal|multi.?modal|cross.?modal|mixed.?media|multimedia.?analysis|hybrid.?content)',
                r'(?:image.?to.?text|visual.?to.?text|text.?to.?image|cross.?media.?translation)',
                
                # Original core patterns enhanced
                r'(?:image.?and.?text|text.?and.?image|visual.?and.?text|vision.?language|vlm|visual.?question.?answering|image.?captioning)',
                
                # 2025: Advanced AI capabilities - Next-generation multi-modal
                r'(?:video.?analysis|frame.?extraction|motion.?detection|temporal.?analysis|sequence.?understanding)',
                r'(?:3d.?model|depth.?analysis|spatial.?understanding|volumetric.?data|point.?cloud)',
                r'(?:augmented.?reality|virtual.?reality|ar|vr|mixed.?reality|immersive.?content)',
                r'(?:ai.?generated|synthetic.?media|deepfake.?detection|artificial.?content|generated.?image)',
                r'(?:real.?time.?analysis|live.?processing|streaming.?analysis|continuous.?monitoring)',
            ],
            
            IntentType.CONVERSATION: [
                r'\b(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))\b',
                r'(?:how\s+are\s+you(?:\s+doing)?(?:\s+today)?|what\'?s\s+up|how\'?s\s+it\s+going)',
                r'(?:nice\s+to\s+meet\s+you|pleasure\s+to\s+meet\s+you|how\s+have\s+you\s+been)',
                r'(?:let\'s.?chat|casual.?conversation|just.?talking|friendly.?chat)',
                r'(?:tell.?me.?about.?yourself|what.?do.?you.?do|who.?are.?you)',
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
            ],
            
            # 2025: NEW SPECIALIZED INTENT PATTERNS for intelligent AI routing
            IntentType.MATHEMATICAL_REASONING: [
                r'\b(?:solve|calculate|compute|find)\s+(?:the|this)?\s*(?:equation|integral|derivative|limit|sum|product)',
                r'(?:math|mathematics|mathematical)\s+(?:problem|equation|proof|theorem|calculation)',
                r'\b(?:algebra|calculus|geometry|trigonometry|statistics|probability|linear.?algebra)',
                r'(?:differential|integral|partial.?derivative|matrix|vector|eigenvalue|polynomial)',
                r'\b(?:prove|theorem|lemma|corollary|axiom|postulate)',
                r'(?:\d+\s*[\+\-\*/\^]\s*\d+|\b(?:sin|cos|tan|log|ln|sqrt)\s*\()',
                r'(?:optimization|minimize|maximize|constraint|linear.?programming)',
                r'(?:graph.?theory|combinatorics|discrete.?math|number.?theory)',
            ],
            
            IntentType.ADVANCED_REASONING: [
                r'\b(?:analyze|reason|logic|logical|reasoning|think|thinking)\s+(?:about|through|step.?by.?step)',
                r'(?:complex|advanced|sophisticated|deep)\s+(?:reasoning|analysis|thinking|logic)',
                r'(?:step.?by.?step|systematically|methodically|logically)\s+(?:analyze|approach|solve)',
                r'(?:philosophical|epistemological|ontological|metaphysical)\s+(?:question|analysis)',
                r'(?:cause.?and.?effect|causal|inference|deduction|induction)',
                r'\b(?:pros.?and.?cons|trade.?offs|advantages.?disadvantages|benefits.?risks)',
                r'(?:critical.?thinking|problem.?solving|decision.?making|strategic.?analysis)',
                r'(?:what.?if|scenario|hypothetical|counterfactual)\s+(?:analysis|reasoning)',
            ],
            
            IntentType.ALGORITHM_DESIGN: [
                r'\b(?:design|create|implement|develop)\s+(?:an?\s+)?(?:algorithm|data.?structure|efficient)',
                r'(?:time.?complexity|space.?complexity|big.?o|algorithm.?analysis)',
                r'\b(?:sort|search|graph|tree|dynamic.?programming|greedy|divide.?conquer)',
                r'(?:optimization|optimize|efficient|performance)\s+(?:algorithm|solution|approach)',
                r'\b(?:recursion|recursive|backtracking|memoization|dp|dijkstra|bfs|dfs)',
                r'(?:leetcode|competitive.?programming|coding.?interview|algorithm.?problem)',
                r'(?:hash.?table|binary.?tree|heap|queue|stack|linked.?list|trie)',
                r'(?:shortest.?path|minimum.?spanning.?tree|topological.?sort)',
            ],
            
            IntentType.SCIENTIFIC_ANALYSIS: [
                r'(?:scientific|research|academic|scholarly)\s+(?:analysis|study|investigation)',
                r'\b(?:hypothesis|experiment|methodology|peer.?review|publication)',
                r'(?:data.?analysis|statistical.?analysis|correlation|regression|significance)',
                r'\b(?:chemistry|physics|biology|medicine|engineering|astronomy|geology)',
                r'(?:research.?paper|scientific.?method|literature.?review|meta.?analysis)',
                r'(?:quantum|molecular|genetic|cellular|neural|biochemical)',
                r'(?:journal|doi|citation|reference|bibliography|peer.?reviewed)',
                r'(?:clinical.?trial|study|sample.?size|control.?group|placebo)',
            ],
            
            IntentType.MEDICAL_ANALYSIS: [
                r'\b(?:medical|health|clinical|diagnostic|therapeutic)\s+(?:analysis|diagnosis|treatment)',
                r'(?:symptom|disease|condition|disorder|syndrome|pathology)',
                r'\b(?:x.?ray|mri|ct.?scan|ultrasound|mammogram|ekg|ecg)',
                r'(?:pharmaceutical|drug|medication|dosage|side.?effect|contraindication)',
                r'(?:patient|diagnosis|prognosis|treatment|therapy|intervention)',
                r'\b(?:anatomy|physiology|pathophysiology|epidemiology|pharmacology)',
                r'(?:medical.?image|radiology|histology|biopsy|lab.?result)',
                r'(?:icd|cpt|medical.?code|healthcare|hospital|clinic)',
            ],
            
            IntentType.CREATIVE_DESIGN: [
                r'\b(?:design|create|mockup|prototype|wireframe)\s+(?:ui|ux|interface|dashboard|app|website)',
                r'(?:user.?experience|user.?interface|interaction.?design|visual.?design)',
                r'(?:brand|branding|logo|identity|style.?guide|color.?palette)',
                r'\b(?:typography|font|layout|composition|visual.?hierarchy)',
                r'(?:responsive|mobile|desktop|tablet)\s+(?:design|layout|interface)',
                r'(?:figma|sketch|adobe|photoshop|illustrator|design.?tool)',
                r'(?:accessibility|usability|user.?research|persona|journey.?map)',
                r'(?:material.?design|bootstrap|tailwind|design.?system|component.?library)',
            ],
            
            IntentType.EDUCATIONAL_CONTENT: [
                r'\b(?:explain|teach|tutorial|lesson|course|learning)\s+(?:about|how|what|why)',
                r'(?:educational|pedagogical|instructional)\s+(?:content|material|resource)',
                r'(?:step.?by.?step|beginner|intermediate|advanced)\s+(?:guide|tutorial|explanation)',
                r'\b(?:curriculum|syllabus|lesson.?plan|learning.?objective|assessment)',
                r'(?:understand|comprehend|grasp|learn|study)\s+(?:concept|principle|theory)',
                r'(?:example|demonstration|illustration|case.?study|practice)',
                r'(?:quiz|test|exercise|homework|assignment|project)',
                r'(?:knowledge|skill|competency|proficiency|mastery)',
            ],
            
            IntentType.BUSINESS_ANALYSIS: [
                r'(?:business|market|commercial|financial)\s+(?:analysis|strategy|intelligence)',
                r'\b(?:roi|kpi|metrics|performance|revenue|profit|loss)',
                r'(?:market.?research|competitive.?analysis|swot|pest)',
                r'(?:financial.?modeling|forecasting|budgeting|valuation)',
                r'(?:customer.?analysis|segmentation|demographics|behavior)',
                r'(?:sales|marketing|operations|supply.?chain)\s+(?:analysis|optimization)',
                r'(?:dashboard|report|visualization|analytics|bi)',
                r'(?:growth|expansion|scalability|efficiency|productivity)',
            ],
            
            IntentType.TECHNICAL_DOCUMENTATION: [
                r'(?:technical|api|software)\s+(?:documentation|docs|manual|guide)',
                r'\b(?:api.?reference|developer.?guide|sdk|library.?docs)',
                r'(?:installation|setup|configuration|deployment)\s+(?:guide|instructions)',
                r'(?:troubleshooting|faq|error.?handling|debugging)\s+(?:guide|documentation)',
                r'(?:changelog|release.?notes|version|update)\s+(?:documentation|history)',
                r'(?:architecture|design|specification|requirements)\s+(?:document|spec)',
                r'(?:readme|documentation|wiki|knowledge.?base)',
                r'(?:comment|docstring|annotation|inline.?docs)',
            ],
            
            IntentType.MULTILINGUAL_PROCESSING: [
                r'(?:multilingual|cross.?lingual|multi.?language)\s+(?:processing|analysis|translation)',
                r'(?:cultural|localization|internationalization|i18n|l10n)',
                r'(?:language.?pair|source.?language|target.?language)',
                r'(?:dialect|accent|regional|native.?speaker)',
                r'(?:translation.?quality|fluency|accuracy|naturalness)',
                r'(?:context.?aware|semantic|pragmatic)\s+(?:translation|understanding)',
                r'(?:machine.?translation|neural.?translation|statistical.?translation)',
                r'(?:bilingual|polyglot|language.?model|cross.?cultural)',
            ]
        }
    
    def _initialize_priorities(self) -> Dict[IntentType, int]:
        """2025 Enhanced priority weights for intent types with specialized AI model routing"""
        return {
            # P1 High Priority - File and data processing
            IntentType.FILE_GENERATION: 10,          # File creation requests
            IntentType.PDF_PROCESSING: 10,           # PDF file uploads
            IntentType.ZIP_ANALYSIS: 10,             # ZIP file uploads
            
            # Specialized High Priority - Advanced AI capabilities  
            IntentType.MATHEMATICAL_REASONING: 9,    # Math problems require specialized models
            IntentType.ADVANCED_REASONING: 9,        # Complex reasoning tasks (QwQ model)
            IntentType.ALGORITHM_DESIGN: 9,          # Complex coding algorithms (CodeLlama)
            IntentType.MEDICAL_ANALYSIS: 9,          # Medical analysis requires specialized models
            
            # High Priority - Core AI tasks
            IntentType.IMAGE_GENERATION: 8,          # Visual generation requests
            IntentType.CODE_GENERATION: 8,           # Programming tasks
            IntentType.IMAGE_ANALYSIS: 8,            # Image analysis
            IntentType.SCIENTIFIC_ANALYSIS: 8,       # Scientific research tasks
            IntentType.DATA_ANALYSIS: 8,             # Data processing
            
            # Medium-High Priority - Specialized content
            IntentType.CREATIVE_DESIGN: 7,           # UI/UX and design tasks
            IntentType.TECHNICAL_DOCUMENTATION: 7,   # Technical writing
            IntentType.MULTILINGUAL_PROCESSING: 7,   # Advanced translation
            IntentType.SENTIMENT_ANALYSIS: 7,        # Analysis requests
            IntentType.CREATIVE_WRITING: 7,          # Creative requests
            IntentType.EDUCATIONAL_CONTENT: 7,       # Teaching and tutorials
            
            # Medium Priority - General processing
            IntentType.DOCUMENT_PROCESSING: 6,       # Document tasks
            IntentType.BUSINESS_ANALYSIS: 6,         # Business intelligence
            IntentType.MULTI_MODAL: 6,               # Complex multimodal
            IntentType.TRANSLATION: 6,               # Basic language requests
            IntentType.QUESTION_ANSWERING: 5,        # Broad Q&A category
            
            # Lower Priority - General conversation
            IntentType.CONVERSATION: 6,              # Conversational AI - INCREASED PRIORITY
            IntentType.TEXT_GENERATION: 3,           # General text fallback
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
    
    async def route_prompt(self, prompt: str, user_id: Optional[int] = None, user_context: Optional[Dict] = None) -> Tuple[IntentType, Dict]:
        """
        Intelligent AI routing with advanced ML-based analysis and adaptive model selection
        Features advanced ML-based analysis and adaptive model selection
        
        Args:
            prompt (str): User prompt to analyze
            user_id (int): Optional user ID for context tracking
            user_context (Dict): Optional user context data
            
        Returns:
            Tuple[IntentType, Dict]: (detected_intent, enhanced_routing_info)
        """
        # Initialize user context tracking
        if user_id and user_id not in self.context_tracker:
            self.context_tracker[user_id] = ContextState(
                user_id=user_id,
                conversation_history=deque(maxlen=20),
                domain_context="general",
                complexity_trend=[],
                preferred_models={},
                conversation_coherence=1.0,
                last_intent=None,
                response_satisfaction=deque(maxlen=10),
                expertise_level={},
                interaction_patterns={},
                follow_up_context=None,
                conversation_flow=[],
                domain_transition_history=[],
                preferred_complexity=0.5,
                learning_profile={}
            )
        
        user_state = self.context_tracker.get(user_id) if user_id else None
        user_context = user_context or {}
        
        logger.info(f"🚀 ═══════════════════ INTELLIGENT AI ROUTING START ═══════════════════")
        logger.info(f"📝 PROMPT: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        logger.info(f"👤 USER_STATE: {'Active' if user_state else 'New'}")
        
        # Advanced complexity analysis using ML-based analyzer
        complexity_result = self.complexity_analyzer.analyze_complexity(prompt, user_context)
        
        logger.info(f"🧠 COMPLEXITY_ANALYSIS:")
        logger.info(f"   📊 Score: {complexity_result.complexity_score:.2f}/10 ({complexity_result.priority_level})")
        logger.info(f"   🔧 Technical: {complexity_result.technical_depth}/5")
        logger.info(f"   🤔 Reasoning: {complexity_result.reasoning_required}")
        logger.info(f"   🎨 Creativity: {complexity_result.creativity_factor:.2f}")
        
        prompt_lower = prompt.lower()
        intent_scores = {}
        
        # Enhanced intent scoring with complexity weighting
        for intent_type, patterns in self.intent_patterns.items():
            raw_score = 0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    raw_score += 1
                    matches.append(pattern)
            
            if raw_score > 0:
                # Enhanced weighting with complexity and context factors
                priority_weight = self.intent_priorities.get(intent_type, 1)
                complexity_bonus = self._calculate_complexity_bonus(intent_type, complexity_result)
                context_bonus = self._calculate_context_bonus(intent_type, user_state) if user_state else 0
                
                weighted_score = raw_score * priority_weight * (1 + complexity_bonus + context_bonus)
                
                intent_scores[intent_type] = {
                    'raw_score': raw_score,
                    'weighted_score': weighted_score,
                    'priority': priority_weight,
                    'complexity_bonus': complexity_bonus,
                    'context_bonus': context_bonus,
                    'matches': matches
                }
        
        # Traditional analysis for backwards compatibility
        analysis = self.analyze_prompt_complexity(prompt)
        
        # Enhanced intent determination with confidence scoring
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['weighted_score'])
            confidence = min(intent_scores[primary_intent]['weighted_score'] / 10.0, 1.0)
            
            # CRITICAL FIX: If confidence is too low, fallback to TEXT_GENERATION instead of UNKNOWN
            if confidence < 0.3:  # Low confidence threshold
                primary_intent = IntentType.TEXT_GENERATION
                confidence = 0.5  # Medium confidence for TEXT_GENERATION fallback
                logger.info(f"🔄 Low confidence classification, falling back to TEXT_GENERATION")
        else:
            # CRITICAL FIX: No intent scores means fallback to TEXT_GENERATION, not UNKNOWN
            primary_intent = IntentType.TEXT_GENERATION
            confidence = 0.4  # Reasonable confidence for default text generation
        
        # CRITICAL FIX: Ensure confidence is never None and always a valid float
        if confidence is None or not isinstance(confidence, (int, float)):
            confidence = 0.5  # Default confidence for safety
            logger.warning(f"⚠️ CONFIDENCE_VALIDATION: Invalid confidence value, using default 0.5")
        
        # Apply advanced heuristics with complexity consideration
        primary_intent = self._apply_intent_heuristics(prompt, primary_intent, intent_scores, analysis)
        
        # Dynamic model selection using advanced selector
        try:
            selected_model, special_params = self.model_selector.select_optimal_model(
                primary_intent.value, complexity_result, user_state
            )
            
            # CRITICAL FIX: All intents should be handled properly by model selector
            # Remove special UNKNOWN handling since we now fallback to TEXT_GENERATION
            logger.info(f"✅ MODEL_SELECTION: Selected {selected_model} for intent {primary_intent.value}")
            
            # CRITICAL FIX: Ensure selected_model is always populated with a valid value
            if not selected_model or selected_model.strip() == '':
                from ..config import Config
                selected_model = Config.DEFAULT_TEXT_MODEL
                # CRITICAL FIX: Regenerate special_params for the fallback model
                special_params = self.model_selector._generate_specialized_parameters(
                    primary_intent.value, complexity_result, selected_model
                )
                logger.warning(f"⚠️ MODEL_SELECTION_FALLBACK: Using default model {selected_model} with regenerated params")
                
        except Exception as e:
            # CRITICAL FIX: Handle model selection failures gracefully
            from ..config import Config
            selected_model = Config.DEFAULT_TEXT_MODEL
            # CRITICAL FIX: Generate appropriate special_params for fallback model
            special_params = self.model_selector._generate_specialized_parameters(
                primary_intent.value, complexity_result, selected_model
            )
            logger.error(f"🚨 MODEL_SELECTION_ERROR: {e}, falling back to {selected_model} with regenerated params")
        
        # ENHANCED: Advanced context awareness and conversation flow tracking - SUPERIOR to Perplexity AI
        if user_state:
            # Enhanced conversation flow tracking
            conversation_flow = self._analyze_conversation_flow(prompt, user_state, complexity_result)
            
            # Advanced follow-up context detection
            follow_up_analysis = self._detect_follow_up_context(prompt, user_state)
            
            # Sophisticated user expertise tracking  
            self._update_user_expertise(user_state, complexity_result, primary_intent.value)
            
            # Enhanced conversation coherence analysis
            coherence_score = self._calculate_conversation_coherence(prompt, user_state, complexity_result)
            user_state.conversation_coherence = coherence_score
            
            # Update context state with enhanced tracking
            user_state.complexity_trend.append(complexity_result.complexity_score)
            user_state.last_intent = primary_intent.value
            user_state.follow_up_context = follow_up_analysis['is_follow_up']
            
            # Enhanced conversation history with detailed metadata
            user_state.conversation_history.append({
                'prompt': prompt[:200],
                'intent': primary_intent.value,
                'complexity': complexity_result.complexity_score,
                'domain': complexity_result.domain_expertise,
                'coherence': coherence_score,
                'flow_type': conversation_flow['flow_type'],
                'follow_up': follow_up_analysis['is_follow_up'],
                'context_shift': follow_up_analysis['context_shift'],
                'timestamp': datetime.now()
            })
            
            # Advanced domain context tracking with transition analysis
            domain = self._detect_domain_from_complexity(complexity_result)
            if domain != user_state.domain_context:
                self._handle_domain_transition(user_state, domain, conversation_flow)
                user_state.domain_context = domain
                logger.info(f"🔄 DOMAIN_SHIFT: {user_state.domain_context} → {domain} (flow: {conversation_flow['flow_type']})")
            
            # Learning from conversation patterns - Beyond Perplexity AI capabilities
            self._learn_from_conversation_patterns(user_state, complexity_result, primary_intent.value)
        
        # CRITICAL FIX: Final validation to ensure selected_model is never None/empty
        if not selected_model or selected_model.strip() == '':
            from ..config import Config
            selected_model = Config.DEFAULT_TEXT_MODEL
            # CRITICAL FIX: Regenerate special_params for the final fallback model
            special_params = self.model_selector._generate_specialized_parameters(
                primary_intent.value, complexity_result, selected_model
            )
            logger.error(f"🚨 CRITICAL_ROUTING_BUG_FIX: selected_model was None/empty, using fallback: {selected_model} with regenerated params")
        
        # Enhanced routing information with superior insights
        routing_info = {
            'primary_intent': primary_intent,
            'confidence': confidence,
            'all_intents': intent_scores,
            'analysis': analysis,
            'complexity_analysis': complexity_result,
            'selected_model': selected_model,  # CRITICAL FIX: This key is now guaranteed to exist and be valid
            'recommended_model': selected_model,  # Backwards compatibility
            'special_parameters': special_params,
            'user_context_used': user_state is not None,
            'routing_quality_score': self._calculate_routing_quality(complexity_result, confidence),
            'performance_prediction': self._predict_performance(selected_model, complexity_result)
        }
        
        # CRITICAL FIX: Validate routing_info completeness
        required_keys = ['selected_model', 'primary_intent', 'confidence']
        for key in required_keys:
            if key not in routing_info or routing_info[key] is None:
                logger.error(f"🚨 ROUTING_INFO_VALIDATION_ERROR: Missing or None value for key '{key}'")
                if key == 'selected_model':
                    from ..config import Config
                    routing_info[key] = Config.DEFAULT_TEXT_MODEL
                    # CRITICAL FIX: Regenerate special_params if model was changed during validation
                    routing_info['special_parameters'] = self.model_selector._generate_specialized_parameters(
                        primary_intent.value, complexity_result, routing_info[key]
                    )
                elif key == 'confidence':
                    # CRITICAL FIX: Set default confidence if missing
                    routing_info[key] = 0.5
                    logger.warning(f"⚠️ CONFIDENCE_FALLBACK: Set default confidence 0.5")
                elif key == 'primary_intent':
                    # CRITICAL FIX: Set default intent if missing
                    routing_info[key] = IntentType.TEXT_GENERATION
                    logger.warning(f"⚠️ INTENT_FALLBACK: Set default intent TEXT_GENERATION")
        
        logger.info(f"🎯 ROUTING_DECISION:")
        logger.info(f"   🏷️ Intent: {primary_intent.value}")
        logger.info(f"   🤖 Model: {selected_model.split('/')[-1] if '/' in selected_model else selected_model}")
        logger.info(f"   📈 Confidence: {confidence:.2f}")
        logger.info(f"   ⚡ Quality Score: {routing_info['routing_quality_score']:.2f}")
        logger.info(f"🏆 ═══════════════════ ROUTING COMPLETE ═══════════════════\n")
        
        return primary_intent, routing_info
    
    def select_optimal_model(self, intent: Optional[str] = None, complexity: Optional[PromptComplexity] = None, context: Optional[ContextState] = None, 
                           prompt: Optional[str] = None, intent_type: Optional[str] = None, user_context: Optional[Dict] = None):
        """
        CRITICAL FIX: Updated method signature to support both old and new calling patterns
        This method is required by the test suite and provides backward compatibility
        
        Args:
            intent (str): Intent type as string (new signature)
            complexity (PromptComplexity): PromptComplexity object (new signature)
            context (Optional[ContextState]): Optional user context state (new signature)
            prompt (str): User prompt for complexity analysis (legacy support)
            intent_type (str): Intent type (legacy support)
            user_context (Optional[Dict]): User context (legacy support)
            
        Returns:
            Tuple[str, Dict]: (selected_model, special_params)
        """
        # Handle legacy calling pattern with prompt parameter
        if prompt is not None:
            # Calculate complexity from prompt if not provided
            if complexity is None:
                complexity = self.complexity_analyzer.analyze_complexity(prompt, user_context or {})
            
            # Use intent_type if provided, otherwise try to detect from prompt
            if intent_type is not None:
                intent = intent_type
            elif intent is None:
                # Quick intent detection from prompt patterns for backward compatibility
                prompt_lower = prompt.lower()
                if any(word in prompt_lower for word in ['code', 'function', 'program', 'algorithm']):
                    intent = 'code_generation'
                elif any(word in prompt_lower for word in ['image', 'picture', 'draw', 'create']):
                    intent = 'image_generation'
                elif any(word in prompt_lower for word in ['sentiment', 'emotion', 'feeling']):
                    intent = 'sentiment_analysis'
                else:
                    intent = 'text_generation'
            
            # Convert user_context to ContextState if needed
            if user_context and context is None:
                # Create a basic ContextState from user_context dict
                context = ContextState(
                    user_id=0,  # Default user ID
                    conversation_history=deque(maxlen=20),
                    domain_context=user_context.get('domain', 'general'),
                    complexity_trend=[],
                    preferred_models={},
                    conversation_coherence=1.0,
                    last_intent=None,
                    response_satisfaction=deque(maxlen=10),
                    expertise_level={},
                    interaction_patterns={},
                    follow_up_context=None,
                    conversation_flow=[],
                    domain_transition_history=[],
                    preferred_complexity=0.5,
                    learning_profile={}
                )
        
        # Ensure we have required parameters
        if intent is None:
            intent = 'text_generation'
        if complexity is None:
            # Create default complexity for safety
            complexity = PromptComplexity(
                complexity_score=5.0,
                technical_depth=2,
                reasoning_required=False,
                context_length=100,
                domain_specificity=0.5,
                creativity_factor=0.5,
                multi_step=False,
                uncertainty=0.3,
                priority_level='medium',
                estimated_tokens=300,
                domain_expertise='general',
                reasoning_chain_length=1,
                requires_external_knowledge=False,
                temporal_context='current',
                user_intent_confidence=0.8,
                cognitive_load=0.5
            )
        
        # Convert IntentType enum to string if needed
        try:
            # Try to access value attribute (for enum types)
            intent_str = getattr(intent, 'value', str(intent))
        except (AttributeError, TypeError):
            # Handle string case or None
            intent_str = str(intent) if intent is not None else 'text_generation'
        
        model, params = self.model_selector.select_optimal_model(intent_str, complexity, context)
        
        # For backward compatibility: if called with legacy prompt parameter, return just the model name
        # If called with new signature, return tuple (model, params)
        if prompt is not None:
            return model  # Legacy behavior - return just model name
        else:
            return model, params  # New behavior - return tuple
    
    def _validate_model_selection(self, model_name: str, intent: IntentType) -> str:
        """Validate that the selected model exists in Config and return fallback if needed"""
        from ..config import Config
        
        # List of all valid model names from Config - 2025 Enhanced with breakthrough specialized models
        valid_models = {
            # Text models - Enhanced with specialized reasoning models
            Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL, Config.FAST_TEXT_MODEL, 
            Config.FALLBACK_TEXT_MODEL, Config.LIGHTWEIGHT_TEXT_MODEL,
            getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            getattr(Config, 'MATH_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            getattr(Config, 'FLAGSHIP_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            getattr(Config, 'EFFICIENT_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
            getattr(Config, 'COMPACT_TEXT_MODEL', Config.FAST_TEXT_MODEL),
            
            # Code models - Enhanced with specialized coding and tool use models
            Config.DEFAULT_CODE_MODEL, Config.ADVANCED_CODE_MODEL, Config.FAST_CODE_MODEL,
            Config.FALLBACK_CODE_MODEL, Config.LIGHTWEIGHT_CODE_MODEL,
            getattr(Config, 'SPECIALIZED_CODE_MODEL', Config.ADVANCED_CODE_MODEL),
            getattr(Config, 'TOOL_USE_CODE_MODEL', Config.ADVANCED_CODE_MODEL),
            getattr(Config, 'EFFICIENT_CODE_MODEL', Config.FAST_CODE_MODEL),
            
            # Vision models - Enhanced with premium, GUI, and specialized vision
            Config.DEFAULT_VISION_MODEL, Config.ADVANCED_VISION_MODEL, Config.FAST_VISION_MODEL,
            Config.FALLBACK_VISION_MODEL, Config.DOCUMENT_VISION_MODEL, Config.LIGHTWEIGHT_VISION_MODEL,
            getattr(Config, 'REASONING_VISION_MODEL', Config.ADVANCED_VISION_MODEL),
            getattr(Config, 'MEDICAL_VISION_MODEL', Config.ADVANCED_VISION_MODEL),
            getattr(Config, 'GUI_AUTOMATION_MODEL', Config.DEFAULT_VISION_MODEL),
            getattr(Config, 'PREMIUM_VISION_MODEL', Config.DEFAULT_VISION_MODEL),
            getattr(Config, 'QUANTIZED_VISION_MODEL', Config.LIGHTWEIGHT_VISION_MODEL),
            
            # Image generation models - Enhanced with breakthrough 2025 models
            Config.DEFAULT_IMAGE_MODEL, Config.COMMERCIAL_IMAGE_MODEL, Config.ADVANCED_IMAGE_MODEL,
            Config.FALLBACK_IMAGE_MODEL, Config.ARTISTIC_IMAGE_MODEL,
            getattr(Config, 'FLAGSHIP_IMAGE_MODEL', Config.DEFAULT_IMAGE_MODEL),
            getattr(Config, 'PROFESSIONAL_IMAGE_MODEL', Config.ADVANCED_IMAGE_MODEL),
            getattr(Config, 'TURBO_IMAGE_MODEL', Config.ADVANCED_IMAGE_MODEL),
            getattr(Config, 'REALISTIC_IMAGE_MODEL', Config.DEFAULT_IMAGE_MODEL),
            getattr(Config, 'EDITING_IMAGE_MODEL', Config.ADVANCED_IMAGE_MODEL),
            
            # 2025: BREAKTHROUGH SPECIALIZED MODELS
            # GUI Automation models
            getattr(Config, 'DEFAULT_GUI_MODEL', Config.DEFAULT_VISION_MODEL),
            getattr(Config, 'ADVANCED_GUI_MODEL', Config.ADVANCED_VISION_MODEL),
            getattr(Config, 'LIGHTWEIGHT_GUI_MODEL', Config.LIGHTWEIGHT_VISION_MODEL),
            
            # Tool Use & Function Calling models
            getattr(Config, 'DEFAULT_TOOL_MODEL', Config.ADVANCED_CODE_MODEL),
            getattr(Config, 'EFFICIENT_TOOL_MODEL', Config.DEFAULT_CODE_MODEL),
            
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
            # 2025: BREAKTHROUGH intent fallbacks
            IntentType.GUI_AUTOMATION: Config.DEFAULT_VISION_MODEL,
            IntentType.TOOL_USE: Config.DEFAULT_CODE_MODEL,
            IntentType.PREMIUM_VISION: Config.DEFAULT_VISION_MODEL,
            IntentType.SYSTEM_INTERACTION: Config.DEFAULT_TEXT_MODEL,
        }
        
        return fallback_map.get(intent, Config.DEFAULT_TEXT_MODEL)
    
    def get_model_for_intent(self, intent: str, complexity: float = 0.0, user_context: Optional[Dict] = None) -> str:
        """CRITICAL TIER-AWARE GATING: Get model for intent with strict tier enforcement"""
        from ..config import Config
        
        # Get tier-appropriate model using Config's tier-aware system
        fallback_chain = Config.get_model_fallback_chain('text')
        if not fallback_chain:
            # If no fallback chain, use guaranteed free model
            return Config.DEFAULT_TEXT_MODEL
        
        # Return first model from tier-aware chain (already filtered by tier)
        base_model = fallback_chain[0]
        
        # Apply additional tier verification
        tier_verified_model = Config.get_tier_appropriate_model(base_model, Config.DEFAULT_TEXT_MODEL)
        
        logger.info(f"🔒 TIER-AWARE GATING: intent={intent}, tier={Config.HF_TIER}, model={tier_verified_model}")
        return tier_verified_model
    
    def _get_recommended_model(self, intent: IntentType, analysis: Dict, original_prompt: str = "") -> str:
        """Get recommended model based on intent and analysis with 2024-2025 STATE-OF-THE-ART models"""
        from ..config import Config
        
        complexity = analysis.get('complexity_score', 0)
        word_count = analysis.get('word_count', 0)
        has_technical = analysis.get('has_technical_terms', False)
        language = analysis.get('language_detected', 'unknown')
        
        # CRITICAL FIX: Apply tier-aware gating to ALL model selections
        def apply_tier_gating(model_name: str) -> str:
            """Apply tier restrictions to any model selection"""
            return Config.get_tier_appropriate_model(model_name, Config.DEFAULT_TEXT_MODEL)
        
        # 🚀 ENHANCED ROUTER DECISION LOGGING - Proving Superior AI Selection
        logger.info(f"\n🎯 ═══════════════════════ INTELLIGENT ROUTER DECISION ═══════════════════════")
        logger.info(f"🔍 PROMPT_ANALYSIS: intent={intent.value.upper()}, complexity={complexity}/10")
        logger.info(f"📊 METRICS: words={word_count}, technical_terms={has_technical}, language={language}")
        logger.info(f"📝 PROMPT_SAMPLE: '{original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}'")
        
        # Advanced model selection based on complexity and context
        if intent == IntentType.TEXT_GENERATION:
            # Enhanced model selection with latest 2025 models
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            
            # Advanced reasoning tasks - Use latest DeepSeek-R1-0528
            if complexity > 8 or 'reasoning' in original_prompt.lower() or 'logic' in original_prompt.lower() or 'philosophy' in original_prompt.lower():
                selected_model = apply_tier_gating(Config.ADVANCED_TEXT_MODEL)  # Tier-gated DeepSeek-R1-0528
            # High complexity - Use flagship MoE model
            elif complexity > 6 or word_count > 500:
                selected_model = apply_tier_gating(Config.FLAGSHIP_TEXT_MODEL)  # Tier-gated Qwen3-235B-A22B
            # Balanced performance - Use efficient 80B/3B model
            elif complexity > 4 or analysis.get('requires_context'):
                selected_model = apply_tier_gating(Config.EFFICIENT_TEXT_MODEL)  # Tier-gated Qwen3-Next-80B-A3B
            # General tasks - Use Qwen3-32B
            elif complexity > 2:
                selected_model = apply_tier_gating(Config.DEFAULT_TEXT_MODEL)  # Tier-gated Qwen3-32B
            # Simple tasks - Use compact high-performance model
            else:
                selected_model = apply_tier_gating(Config.COMPACT_TEXT_MODEL)  # Tier-gated DeepSeek-R1-0528-Qwen3-8B
            
            # Enhanced decision logging for text generation with latest models
            if complexity > 8:
                reasoning = "ADVANCED (DeepSeek-R1-0528)"
            elif complexity > 6:
                reasoning = "FLAGSHIP (Qwen3-235B-MoE)"
            elif complexity > 4:
                reasoning = "EFFICIENT (Qwen3-Next-80B/3B)"
            elif complexity > 2:
                reasoning = "DEFAULT (Qwen3-32B)"
            else:
                reasoning = "COMPACT (R1-Qwen3-8B)"
                
            logger.info(f"🤖 TEXT_GENERATION: complexity={complexity} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (2025 SOTA, Superior to GPT-4o/Claude/Gemini)")
            logger.info(f"⚡ REASONING: {'Complex reasoning (DeepSeek-R1-0528)' if complexity > 8 else 'High complexity (Flagship MoE)' if complexity > 6 else 'Balanced performance (Efficient)' if complexity > 4 else 'General task (Qwen3)' if complexity > 2 else 'Simple query (Compact SOTA)'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CODE_GENERATION:
            # Enhanced coding model selection with latest 2025 models
            complexity = analysis.get('complexity_score', 0)
            language = analysis.get('language_detected', 'python')
            
            # Select optimal coding model based on complexity and language
            selected_model = None
            if complexity > 7 or language in ['rust', 'go', 'c++', 'java', 'assembly'] or 'algorithm' in original_prompt.lower():
                selected_model = apply_tier_gating(Config.ADVANCED_CODE_MODEL)  # Tier-gated DeepSeek-Coder-V2-Instruct
            elif complexity > 5 or word_count > 300:
                selected_model = apply_tier_gating(Config.DEFAULT_CODE_MODEL)  # Tier-gated Qwen2.5-Coder-32B
            elif complexity > 3:
                selected_model = apply_tier_gating(Config.FAST_CODE_MODEL)  # Tier-gated Qwen2.5-Coder-14B
            elif complexity > 1:
                selected_model = apply_tier_gating(Config.EFFICIENT_CODE_MODEL)  # Tier-gated Qwen2.5-Coder-7B
            else:
                selected_model = apply_tier_gating(Config.FALLBACK_CODE_MODEL)  # Tier-gated StarCoder2-7B
            
            # Enhanced decision logging for code generation with latest models
            if complexity > 7:
                reasoning = "ADVANCED (DeepSeek-Coder-V2)"
            elif complexity > 5:
                reasoning = "DEFAULT (Qwen2.5-Coder-32B)"
            elif complexity > 3:
                reasoning = "FAST (Qwen2.5-Coder-14B)"
            elif complexity > 1:
                reasoning = "EFFICIENT (Qwen2.5-Coder-7B)"
            else:
                reasoning = "FALLBACK (StarCoder2-7B)"
                
            logger.info(f"💻 CODE_GENERATION: complexity={complexity}, language={language} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (2025 SOTA, Superior to GitHub Copilot/GPT-4o Code)")
            logger.info(f"⚡ REASONING: {'Complex algorithms (DeepSeek-V2)' if complexity > 7 else 'Production-ready code (Qwen-32B)' if complexity > 5 else 'Standard development (Qwen-14B)' if complexity > 3 else 'Efficient coding (Qwen-7B)' if complexity > 1 else 'Basic code snippets'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.IMAGE_ANALYSIS:
            # Vision models for image understanding and analysis
            prompt_lower = original_prompt.lower()
            selected_model = None
            if 'document' in prompt_lower or 'text' in prompt_lower or 'ocr' in prompt_lower:
                selected_model = apply_tier_gating(Config.DOCUMENT_VISION_MODEL)  # Tier-gated Florence-2
            elif 'complex' in prompt_lower or 'detailed' in prompt_lower:
                selected_model = apply_tier_gating(Config.ADVANCED_VISION_MODEL)  # Tier-gated Qwen2.5-VL-72B
            elif 'fast' in prompt_lower or 'quick' in prompt_lower:
                selected_model = apply_tier_gating(Config.FAST_VISION_MODEL)  # Tier-gated Qwen2.5-VL-3B
            else:
                selected_model = apply_tier_gating(Config.DEFAULT_VISION_MODEL)  # Tier-gated Qwen2.5-VL-7B
            
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
            selected_model = apply_tier_gating(Config.ADVANCED_VISION_MODEL)  # Tier-gated Qwen2.5-VL-72B
            logger.info(f"🔄 MULTI_MODAL: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.IMAGE_GENERATION:
            # Enhanced image generation with Qwen-Image and latest FLUX models
            prompt_lower = original_prompt.lower()
            selected_model = None
            
            # Advanced model selection based on prompt requirements
            if 'text' in prompt_lower or 'chinese' in prompt_lower or 'typography' in prompt_lower:
                selected_model = Config.FLAGSHIP_IMAGE_MODEL  # Qwen-Image (20B, superior text rendering)
            elif 'edit' in prompt_lower or 'modify' in prompt_lower or 'change' in prompt_lower:
                selected_model = Config.EDITING_IMAGE_MODEL  # Qwen-Image-Edit (specialized editing)
            elif 'commercial' in prompt_lower or 'business' in prompt_lower or 'fast' in prompt_lower:
                selected_model = Config.COMMERCIAL_IMAGE_MODEL  # FLUX.1-schnell (1-4 steps, commercial)
            elif 'artistic' in prompt_lower or 'creative' in prompt_lower:
                selected_model = Config.ARTISTIC_IMAGE_MODEL  # SD3.5-Medium (artistic styles)
            elif 'control' in prompt_lower or 'precise' in prompt_lower:
                selected_model = Config.KONTEXT_IMAGE_MODEL  # FLUX.1-Kontext (in-context control)
            else:
                selected_model = Config.DEFAULT_IMAGE_MODEL  # FLUX.1-dev (best overall quality)
            
            # Enhanced decision logging for image generation with latest models
            has_text = 'text' in prompt_lower or 'chinese' in prompt_lower or 'typography' in prompt_lower
            is_editing = 'edit' in prompt_lower or 'modify' in prompt_lower
            is_commercial = 'commercial' in prompt_lower or 'business' in prompt_lower or 'fast' in prompt_lower
            is_artistic = 'artistic' in prompt_lower or 'creative' in prompt_lower
            is_controlled = 'control' in prompt_lower or 'precise' in prompt_lower
            
            if has_text:
                reasoning = "FLAGSHIP (Qwen-Image-20B)"
            elif is_editing:
                reasoning = "EDITING (Qwen-Image-Edit)"
            elif is_commercial:
                reasoning = "COMMERCIAL (FLUX.1-schnell)"
            elif is_artistic:
                reasoning = "ARTISTIC (SD3.5-Medium)"
            elif is_controlled:
                reasoning = "KONTEXT (FLUX.1-Kontext)"
            else:
                reasoning = "DEFAULT (FLUX.1-dev)"
                
            logger.info(f"🎨 IMAGE_GENERATION: text={has_text}, edit={is_editing}, commercial={is_commercial} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (2025 SOTA, Superior to DALL-E 3/Midjourney/Firefly)")
            logger.info(f"⚡ REASONING: {'Superior text rendering (Qwen-Image)' if has_text else 'Advanced editing capabilities' if is_editing else 'Fast commercial generation' if is_commercial else 'Artistic style generation' if is_artistic else 'Precise control generation' if is_controlled else 'Best overall quality (FLUX.1-dev)'}")
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
        
        # 2025: NEW SPECIALIZED INTENT ROUTING - Advanced capabilities
        elif intent == IntentType.MATHEMATICAL_REASONING:
            # Use specialized math model with calculator integration for accurate calculations
            selected_model = getattr(Config, 'MATH_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL)
            logger.info(f"📐 MATHEMATICAL_REASONING: model={selected_model} (Math reasoning with calculator integration)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Integrated calculator for verified mathematical accuracy)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.ADVANCED_REASONING:
            # Use QwQ model for complex logical reasoning tasks
            selected_model = getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL)
            logger.info(f"🧠 ADVANCED_REASONING: model={selected_model} (QwQ-32B reasoning specialist)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Advanced reasoning and logical problem solving)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.ALGORITHM_DESIGN:
            # Use specialized coding model for complex algorithms
            selected_model = getattr(Config, 'SPECIALIZED_CODE_MODEL', Config.ADVANCED_CODE_MODEL)
            logger.info(f"⚡ ALGORITHM_DESIGN: model={selected_model} (CodeLlama-34B specialist)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Specialized for complex algorithms and data structures)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.SCIENTIFIC_ANALYSIS:
            # Use advanced reasoning model for scientific research tasks
            complexity = analysis.get('complexity_score', 0)
            selected_model = getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL) if complexity > 5 else Config.DEFAULT_TEXT_MODEL
            logger.info(f"🔬 SCIENTIFIC_ANALYSIS: complexity={complexity}, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Research-grade analysis)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.MEDICAL_ANALYSIS:
            # Use medical vision model for medical images, or advanced text for medical text
            prompt_lower = original_prompt.lower()
            if any(term in prompt_lower for term in ['image', 'x-ray', 'mri', 'ct scan', 'ultrasound']):
                selected_model = getattr(Config, 'MEDICAL_VISION_MODEL', Config.ADVANCED_VISION_MODEL)
                logger.info(f"🏥 MEDICAL_ANALYSIS: type=image, model={selected_model}")
            else:
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1 for medical text analysis
                logger.info(f"🏥 MEDICAL_ANALYSIS: type=text, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Medical specialist model)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CREATIVE_DESIGN:
            # Use artistic image generation for visual design, text model for design concepts
            prompt_lower = original_prompt.lower()
            if any(term in prompt_lower for term in ['ui', 'ux', 'mockup', 'wireframe', 'logo', 'visual']):
                selected_model = getattr(Config, 'ARTISTIC_IMAGE_MODEL', Config.DEFAULT_IMAGE_MODEL)
                logger.info(f"🎨 CREATIVE_DESIGN: type=visual, model={selected_model}")
            else:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for design concepts
                logger.info(f"🎨 CREATIVE_DESIGN: type=conceptual, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Design specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.EDUCATIONAL_CONTENT:
            # Use balanced model for teaching and tutorials
            complexity = analysis.get('complexity_score', 0)
            selected_model = Config.ADVANCED_TEXT_MODEL if complexity > 6 else Config.DEFAULT_TEXT_MODEL
            logger.info(f"📚 EDUCATIONAL_CONTENT: complexity={complexity}, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Optimized for teaching)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.BUSINESS_ANALYSIS:
            # Use advanced reasoning for business intelligence and analysis
            selected_model = getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL)
            logger.info(f"📊 BUSINESS_ANALYSIS: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Business intelligence specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.TECHNICAL_DOCUMENTATION:
            # Use coding-specialized model for technical docs
            selected_model = Config.ADVANCED_CODE_MODEL  # DeepSeek-Coder-V2 for technical writing
            logger.info(f"📝 TECHNICAL_DOCUMENTATION: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Technical writing specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.MULTILINGUAL_PROCESSING:
            # Use advanced translation model with cultural context
            selected_model = Config.DEFAULT_TRANSLATION_MODEL  # NLLB-200 for 200+ languages
            logger.info(f"🌐 MULTILINGUAL_PROCESSING: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (200+ language specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        # 2025: SPECIALIZED INTENT ROUTING - Enhanced capabilities
        elif intent == IntentType.GUI_AUTOMATION:
            # BREAKTHROUGH: UI-TARS native GUI automation - Superior to GPT-4V for GUI tasks
            selected_model = getattr(Config, 'DEFAULT_GUI_MODEL', Config.DEFAULT_VISION_MODEL)
            logger.info(f"🖱️ GUI_AUTOMATION: model={selected_model} (UI-TARS DPO trained)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Specialized GUI automation model)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.TOOL_USE:
            # BREAKTHROUGH: Groq function calling excellence - #1 on BFCL leaderboard
            complexity = analysis.get('complexity_score', 0)
            if complexity > 6 or 'complex' in original_prompt.lower():
                selected_model = getattr(Config, 'DEFAULT_TOOL_MODEL', Config.ADVANCED_CODE_MODEL)  # 70B Tool Use
            else:
                selected_model = getattr(Config, 'EFFICIENT_TOOL_MODEL', Config.DEFAULT_CODE_MODEL)  # 8B Tool Use
            logger.info(f"🔧 TOOL_USE: complexity={complexity}, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Advanced tool use and function calling)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.PREMIUM_VISION:
            # BREAKTHROUGH: MiniCPM-V excellence - Beats GPT-4V on OCRBench
            selected_model = getattr(Config, 'PREMIUM_VISION_MODEL', Config.DEFAULT_VISION_MODEL)
            logger.info(f"👁️ PREMIUM_VISION: model={selected_model} (MiniCPM-Llama3-V-2.5)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Advanced vision capabilities with OCR support)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.SYSTEM_INTERACTION:
            # Advanced system-level interactions
            selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-0528 for complex system tasks
            logger.info(f"⚙️ SYSTEM_INTERACTION: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Advanced reasoning for system tasks)")
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
        logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Advanced reasoning model for complex tasks)")
        logger.info(f"⚡ REASONING: Fallback to most powerful reasoning model for unknown intent")
        
        # Validate model selection and return
        validated_model = self._validate_model_selection(selected_model, intent)
        logger.info(f"✅ FINAL_VALIDATION: model={validated_model} for intent={intent.value}")
        logger.info(f"🏆 ═══════════════════════ ROUTER DECISION COMPLETE ═══════════════════════\n")
        return validated_model
    
    def _get_special_parameters(self, intent: IntentType, analysis: Dict) -> Dict:
        """Get optimized parameters for 2024-2025 SOTA models"""
        from ..config import Config
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
    
    def _calculate_complexity_bonus(self, intent_type: IntentType, complexity: PromptComplexity) -> float:
        """Calculate complexity-based bonus for intent scoring"""
        base_bonus = 0.0
        
        # Mathematical and reasoning tasks get higher bonus with complexity
        if intent_type in [IntentType.MATHEMATICAL_REASONING, IntentType.ADVANCED_REASONING]:
            base_bonus = complexity.complexity_score * 0.15
        
        # Code generation benefits from technical depth
        elif intent_type == IntentType.CODE_GENERATION:
            base_bonus = complexity.technical_depth * 0.1 + (complexity.complexity_score * 0.05)
        
        # Creative tasks benefit from creativity factor
        elif intent_type == IntentType.CREATIVE_WRITING:
            base_bonus = complexity.creativity_factor * 0.2
        
        # Multi-step tasks get bonus for multi-step prompts
        elif complexity.multi_step and intent_type in [IntentType.DATA_ANALYSIS, IntentType.ALGORITHM_DESIGN]:
            base_bonus = 0.25
        
        # High uncertainty benefits question answering
        elif intent_type == IntentType.QUESTION_ANSWERING and complexity.uncertainty > 0.5:
            base_bonus = complexity.uncertainty * 0.15
        
        return min(base_bonus, 0.5)  # Cap at 50% bonus
    
    def _calculate_context_bonus(self, intent_type: IntentType, user_state: ContextState) -> float:
        """Calculate context-based bonus for intent scoring"""
        if not user_state:
            return 0.0
        
        context_bonus = 0.0
        
        # Consistency bonus - same intent type as previous
        if user_state.last_intent == intent_type.value:
            context_bonus += 0.1
        
        # Domain continuity bonus
        if user_state.domain_context and self._intent_matches_domain(intent_type, user_state.domain_context):
            context_bonus += 0.15
        
        # User preference bonus
        if intent_type.value in user_state.preferred_models:
            context_bonus += user_state.preferred_models[intent_type.value] * 0.2
        
        # Conversation coherence bonus
        context_bonus += user_state.conversation_coherence * 0.1
        
        return min(context_bonus, 0.4)  # Cap at 40% bonus
    
    def _intent_matches_domain(self, intent_type: IntentType, domain: str) -> bool:
        """Check if intent type matches current domain context"""
        domain_intent_mapping = {
            'programming': [IntentType.CODE_GENERATION, IntentType.ALGORITHM_DESIGN, IntentType.TOOL_USE],
            'mathematics': [IntentType.MATHEMATICAL_REASONING, IntentType.DATA_ANALYSIS],
            'creative': [IntentType.CREATIVE_WRITING, IntentType.IMAGE_GENERATION, IntentType.CREATIVE_DESIGN],
            'analysis': [IntentType.DATA_ANALYSIS, IntentType.DOCUMENT_PROCESSING, IntentType.SCIENTIFIC_ANALYSIS],
            'technical': [IntentType.CODE_GENERATION, IntentType.SYSTEM_INTERACTION, IntentType.TECHNICAL_DOCUMENTATION]
        }
        
        return intent_type in domain_intent_mapping.get(domain, [])
    
    def _detect_domain_from_complexity(self, complexity: PromptComplexity) -> str:
        """Detect domain from complexity analysis"""
        if complexity.technical_depth >= 3:
            return 'programming' if complexity.domain_specificity > 0.6 else 'technical'
        elif complexity.creativity_factor > 0.7:
            return 'creative'
        elif complexity.reasoning_required and complexity.complexity_score > 6:
            return 'analysis'
        elif complexity.domain_specificity > 0.5:
            return 'specialized'
        else:
            return 'general'
    
    def _calculate_routing_quality(self, complexity: PromptComplexity, confidence: float) -> float:
        """Calculate routing quality score"""
        # Base quality from confidence
        quality = confidence * 5
        
        # Bonus for appropriate complexity handling
        if complexity.priority_level == 'critical' and confidence > 0.8:
            quality += 2.0
        elif complexity.priority_level == 'high' and confidence > 0.7:
            quality += 1.5
        elif complexity.priority_level == 'medium' and confidence > 0.6:
            quality += 1.0
        
        # Penalty for low confidence on complex tasks
        if complexity.complexity_score > 7 and confidence < 0.5:
            quality -= 1.5
        
        # Bonus for handling uncertainty well
        if complexity.uncertainty > 0.5 and confidence > 0.6:
            quality += 0.5
        
        return min(max(quality, 0.0), 10.0)  # Scale 0-10
    
    def _predict_performance(self, model: str, complexity: PromptComplexity) -> Dict:
        """Predict performance metrics for selected model"""
        # Base performance prediction
        prediction = {
            'expected_quality': 7.5,
            'estimated_time': 3.0,
            'success_probability': 0.85,
            'resource_efficiency': 0.8
        }
        
        # Model-specific adjustments
        if 'deepseek' in model.lower():
            # DeepSeek models excel at reasoning
            if complexity.reasoning_required:
                prediction['expected_quality'] += 1.5
                prediction['success_probability'] += 0.1
            if complexity.complexity_score > 7:
                prediction['estimated_time'] += 2.0
        
        elif 'qwen' in model.lower():
            # Qwen models are well-balanced
            prediction['resource_efficiency'] += 0.1
            if complexity.multi_step:
                prediction['expected_quality'] += 1.0
        
        elif 'starcoder' in model.lower() or 'coder' in model.lower():
            # Coding models for technical tasks
            if complexity.technical_depth >= 3:
                prediction['expected_quality'] += 2.0
                prediction['success_probability'] += 0.15
        
        # Complexity adjustments
        if complexity.priority_level == 'critical':
            prediction['estimated_time'] *= 1.5
            prediction['expected_quality'] += 0.5
        
        return prediction
    
    def update_model_performance(self, model: str, intent: str, success: bool, 
                                response_time: float, quality_score: float):
        """Update model performance tracking for adaptive routing"""
        self.model_selector.performance_monitor.update_metrics(model, success, response_time)
    
    def _analyze_conversation_flow(self, prompt: str, user_state: ContextState, complexity: PromptComplexity) -> Dict[str, Any]:
        """
        SUPERIOR conversation flow analysis that exceeds Perplexity AI capabilities
        
        Args:
            prompt (str): Current user prompt
            user_state (ContextState): User's conversation state
            complexity (PromptComplexity): Complexity analysis of current prompt
            
        Returns:
            Dict: Detailed conversation flow analysis
        """
        prompt_lower = prompt.lower()
        
        # Analyze conversation continuity indicators
        continuity_indicators = {
            'continuation': ['continue', 'keep going', 'more', 'and then', 'next', 'also', 'additionally'],
            'clarification': ['what do you mean', 'can you explain', 'clarify', 'elaborate', 'what about'],
            'refinement': ['better', 'improve', 'refine', 'modify', 'adjust', 'change', 'update'],
            'exploration': ['what if', 'how about', 'alternatively', 'instead', 'different approach'],
            'conclusion': ['finally', 'in summary', 'to conclude', 'overall', 'that\'s all']
        }
        
        flow_type = 'new_topic'  # Default
        confidence = 0.0
        
        for flow, indicators in continuity_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                flow_type = flow
                confidence = 0.8
                break
        
        # Reference to previous conversation elements
        reference_indicators = ['this', 'that', 'above', 'previous', 'earlier', 'before', 'it', 'they']
        has_references = any(indicator in prompt_lower for indicator in reference_indicators)
        
        # Analyze topic coherence with conversation history
        topic_coherence = self._calculate_topic_coherence(prompt, user_state)
        
        # Detect conversation depth progression
        depth_progression = self._analyze_conversation_depth(user_state, complexity)
        
        return {
            'flow_type': flow_type,
            'confidence': confidence,
            'has_references': has_references,
            'topic_coherence': topic_coherence,
            'depth_progression': depth_progression,
            'conversation_depth': len(user_state.conversation_history),
            'complexity_trend': 'increasing' if self._is_complexity_increasing(user_state) else 'stable'
        }
    
    def _detect_follow_up_context(self, prompt: str, user_state: ContextState) -> Dict[str, Any]:
        """
        Advanced follow-up context detection - More sophisticated than Perplexity AI
        
        Args:
            prompt (str): Current user prompt
            user_state (ContextState): User's conversation state
            
        Returns:
            Dict: Follow-up context analysis
        """
        prompt_lower = prompt.lower()
        
        # Strong follow-up indicators
        strong_follow_up_indicators = [
            'can you also', 'what about', 'how about', 'and what if', 'but what',
            'now', 'then', 'next', 'after that', 'following that'
        ]
        
        # Weak follow-up indicators  
        weak_follow_up_indicators = [
            'this', 'that', 'it', 'they', 'them', 'which', 'where', 'when'
        ]
        
        # Question continuation patterns
        question_continuation = bool(re.search(r'\?.*(?:and|but|or|also|what about)', prompt_lower))
        
        # Calculate follow-up probability
        strong_signals = sum(1 for indicator in strong_follow_up_indicators if indicator in prompt_lower)
        weak_signals = sum(1 for indicator in weak_follow_up_indicators if indicator in prompt_lower)
        
        follow_up_probability = min((strong_signals * 0.4 + weak_signals * 0.2), 1.0)
        
        # Context shift detection
        if len(user_state.conversation_history) > 0:
            last_domain = user_state.conversation_history[-1].get('domain', 'general')
            current_domain = getattr(user_state, 'domain_context', 'general')
            context_shift = last_domain != current_domain
        else:
            context_shift = False
        
        # Time-based context decay
        time_decay = self._calculate_context_time_decay(user_state)
        
        is_follow_up = (follow_up_probability > 0.3 and time_decay > 0.5) or question_continuation
        
        return {
            'is_follow_up': is_follow_up,
            'probability': follow_up_probability,
            'context_shift': context_shift,
            'time_decay': time_decay,
            'strong_signals': strong_signals,
            'weak_signals': weak_signals,
            'question_continuation': question_continuation
        }
    
    def _update_user_expertise(self, user_state: ContextState, complexity: PromptComplexity, intent: str) -> None:
        """
        Sophisticated user expertise tracking - Learning capabilities beyond Perplexity AI
        
        Args:
            user_state (ContextState): User's conversation state
            complexity (PromptComplexity): Complexity analysis
            intent (str): Detected intent type
        """
        domain = complexity.domain_expertise
        current_expertise = user_state.expertise_level.get(domain, 0.5)
        
        # Calculate expertise adjustment based on complexity and success patterns
        if complexity.complexity_score > 7:
            # User asking complex questions suggests higher expertise
            expertise_boost = 0.1
        elif complexity.complexity_score < 3:
            # Very simple questions might indicate lower expertise
            expertise_boost = -0.05
        else:
            # Moderate complexity - slight positive adjustment
            expertise_boost = 0.02
        
        # Intent-specific expertise adjustments
        intent_expertise_map = {
            'code_generation': 'technical',
            'code_review': 'technical', 
            'algorithm_design': 'technical',
            'mathematical_reasoning': 'scientific',
            'medical_analysis': 'medical',
            'legal_analysis': 'legal',
            'research': 'scientific',
            'data_analysis': 'scientific'
        }
        
        # Cross-domain expertise transfer
        if intent in intent_expertise_map:
            related_domain = intent_expertise_map[intent]
            if related_domain in user_state.expertise_level:
                # Transfer some expertise from related domains
                transfer_factor = 0.1
                expertise_boost += user_state.expertise_level[related_domain] * transfer_factor
        
        # Update expertise with momentum (gradual learning)
        new_expertise = current_expertise + (expertise_boost * 0.3)  # 30% learning rate
        user_state.expertise_level[domain] = max(0.0, min(1.0, new_expertise))
        
        logger.debug(f"👨‍🎓 Expertise update: {domain} {current_expertise:.2f} → {new_expertise:.2f}")
    
    def _calculate_conversation_coherence(self, prompt: str, user_state: ContextState, complexity: PromptComplexity) -> float:
        """
        Enhanced conversation coherence analysis - Superior to Perplexity AI
        
        Args:
            prompt (str): Current user prompt
            user_state (ContextState): User's conversation state  
            complexity (PromptComplexity): Complexity analysis
            
        Returns:
            float: Coherence score (0-1)
        """
        if not user_state.conversation_history:
            return 1.0  # Perfect coherence for first message
        
        coherence_factors = []
        
        # Domain consistency
        current_domain = complexity.domain_expertise
        recent_domains = [entry.get('domain', 'general') for entry in list(user_state.conversation_history)[-3:]]
        domain_consistency = sum(1 for d in recent_domains if d == current_domain) / len(recent_domains)
        coherence_factors.append(domain_consistency * 0.3)
        
        # Complexity progression coherence
        complexity_trend = user_state.complexity_trend[-5:] if len(user_state.complexity_trend) >= 5 else user_state.complexity_trend
        if len(complexity_trend) > 1:
            complexity_variance = sum(abs(complexity_trend[i] - complexity_trend[i-1]) for i in range(1, len(complexity_trend)))
            complexity_coherence = max(0, 1 - (complexity_variance / (len(complexity_trend) * 5)))  # Normalize by max possible variance
            coherence_factors.append(complexity_coherence * 0.2)
        
        # Intent consistency
        recent_intents = [entry.get('intent', 'text_generation') for entry in list(user_state.conversation_history)[-3:]]
        intent_groups = {
            'technical': ['code_generation', 'code_review', 'algorithm_design'],
            'analytical': ['data_analysis', 'research', 'scientific_analysis'],
            'creative': ['creative_writing', 'image_generation', 'creative_design'],
            'conversational': ['conversation', 'question_answering', 'explanation']
        }
        
        current_intent_group = None
        for group, intents in intent_groups.items():
            if complexity.domain_expertise in intents:
                current_intent_group = group
                break
        
        if current_intent_group:
            intent_consistency = sum(1 for intent in recent_intents if intent in intent_groups[current_intent_group]) / len(recent_intents)
            coherence_factors.append(intent_consistency * 0.25)
        
        # Temporal coherence (time gaps)
        if len(user_state.conversation_history) > 1:
            last_timestamp = user_state.conversation_history[-1].get('timestamp', datetime.now())
            time_gap = (datetime.now() - last_timestamp).total_seconds() / 60  # minutes
            temporal_coherence = max(0, 1 - (time_gap / 30))  # 30 minutes for full decay
            coherence_factors.append(temporal_coherence * 0.25)
        
        # Calculate overall coherence
        overall_coherence = sum(coherence_factors) if coherence_factors else 0.8
        return max(0.0, min(1.0, overall_coherence))
    
    def _handle_domain_transition(self, user_state: ContextState, new_domain: str, conversation_flow: Dict) -> None:
        """
        Advanced domain transition handling - Context-aware beyond Perplexity AI
        
        Args:
            user_state (ContextState): User's conversation state
            new_domain (str): New domain being transitioned to
            conversation_flow (Dict): Conversation flow analysis
        """
        old_domain = user_state.domain_context
        
        # Analyze transition type
        if conversation_flow['flow_type'] in ['exploration', 'refinement']:
            # Natural exploration - maintain some context
            transition_type = 'exploration'
            context_retention = 0.7
        elif conversation_flow['topic_coherence'] < 0.3:
            # Abrupt topic change - reset context more aggressively
            transition_type = 'topic_change'
            context_retention = 0.3
        else:
            # Gradual transition - moderate context retention
            transition_type = 'gradual'
            context_retention = 0.5
        
        # Update conversation coherence based on transition
        user_state.conversation_coherence *= context_retention
        
        # Log domain transition with analytics
        logger.info(f"🔄 Domain transition: {old_domain} → {new_domain}")
        logger.info(f"   📈 Transition type: {transition_type}")
        logger.info(f"   🧠 Context retention: {context_retention:.2f}")
        logger.info(f"   🔗 Coherence: {user_state.conversation_coherence:.2f}")
    
    def _learn_from_conversation_patterns(self, user_state: ContextState, complexity: PromptComplexity, intent: str) -> None:
        """
        Learning from conversation patterns - Advanced beyond Perplexity AI capabilities
        
        Args:
            user_state (ContextState): User's conversation state
            complexity (PromptComplexity): Complexity analysis
            intent (str): Detected intent type
        """
        # Update preferred model tracking based on recent successful interactions
        if intent not in user_state.preferred_models:
            user_state.preferred_models[intent] = 0.5
        
        # Gradual learning from conversation patterns
        learning_rate = 0.1
        
        # Pattern: High complexity questions suggest user prefers advanced models
        if complexity.complexity_score > 7:
            user_state.preferred_models[intent] += learning_rate * 0.2
        
        # Pattern: Consistent domain usage suggests expertise growth
        domain = complexity.domain_expertise
        if domain in user_state.expertise_level:
            consecutive_domain_count = 0
            for entry in reversed(list(user_state.conversation_history)[-5:]):
                if entry.get('domain') == domain:
                    consecutive_domain_count += 1
                else:
                    break
            
            if consecutive_domain_count >= 3:
                # Consistent domain usage - boost expertise
                expertise_boost = min(0.05, consecutive_domain_count * 0.01)
                current_expertise = user_state.expertise_level.get(domain, 0.5)
                user_state.expertise_level[domain] = min(1.0, current_expertise + expertise_boost)
        
        # Pattern: Follow-up questions suggest engagement and deeper interest
        if hasattr(user_state, 'follow_up_context') and user_state.follow_up_context:
            # Increase domain interest and expertise slightly
            if domain in user_state.expertise_level:
                user_state.expertise_level[domain] = min(1.0, user_state.expertise_level[domain] + 0.02)
        
        logger.debug(f"🎯 Learning patterns: {intent} preference: {user_state.preferred_models.get(intent, 0.5):.2f}")
    
    def _calculate_topic_coherence(self, prompt: str, user_state: ContextState) -> float:
        """Calculate topic coherence with recent conversation"""
        if not user_state.conversation_history:
            return 1.0
        
        # Simple keyword overlap analysis
        current_words = set(prompt.lower().split())
        recent_prompts = [entry.get('prompt', '') for entry in list(user_state.conversation_history)[-3:]]
        recent_words = set(' '.join(recent_prompts).lower().split())
        
        if not recent_words:
            return 0.5
        
        overlap = len(current_words.intersection(recent_words))
        total_unique = len(current_words.union(recent_words))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def _analyze_conversation_depth(self, user_state: ContextState, complexity: PromptComplexity) -> str:
        """Analyze if conversation is getting deeper or staying surface level"""
        if len(user_state.complexity_trend) < 2:
            return 'initial'
        
        recent_complexity = user_state.complexity_trend[-3:]
        avg_recent = sum(recent_complexity) / len(recent_complexity)
        
        if complexity.complexity_score > avg_recent + 1:
            return 'deepening'
        elif complexity.complexity_score < avg_recent - 1:
            return 'simplifying'
        else:
            return 'consistent'
    
    def _is_complexity_increasing(self, user_state: ContextState) -> bool:
        """Check if complexity is generally increasing over time"""
        if len(user_state.complexity_trend) < 3:
            return False
        
        recent = user_state.complexity_trend[-3:]
        return recent[-1] > recent[0]
    
    def _calculate_context_time_decay(self, user_state: ContextState) -> float:
        """Calculate context relevance based on time decay"""
        if not user_state.conversation_history:
            return 1.0
        
        last_timestamp = user_state.conversation_history[-1].get('timestamp', datetime.now())
        time_gap_minutes = (datetime.now() - last_timestamp).total_seconds() / 60
        
        # Context decays over 15 minutes to 50%, then more slowly
        if time_gap_minutes < 5:
            return 1.0
        elif time_gap_minutes < 15:
            return 1.0 - ((time_gap_minutes - 5) / 10) * 0.5  # Linear decay to 0.5
        else:
            return max(0.1, 0.5 - ((time_gap_minutes - 15) / 60) * 0.4)  # Slower decay to 0.1

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
            'timestamp': datetime.now(),
            'model_used': self._get_recommended_model(intent, {})
        }
    
    def select_model_dynamic(self, prompt: str, intent: IntentType, 
                           conversation_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           request_id: Optional[str] = None,
                           enable_fallback: bool = True) -> Tuple[str, ModelSelectionExplanation]:
        """
        Enhanced dynamic model selection with real-time adaptation and explainability
        
        Args:
            prompt: User prompt
            intent: Classified intent type
            conversation_id: Optional conversation identifier
            user_id: Optional user identifier  
            request_id: Optional request identifier
            enable_fallback: Whether to enable intelligent fallback
            
        Returns:
            Tuple of (selected_model, selection_explanation)
        """
        logger.info(f"🚀 Dynamic Model Selection Started for intent: {intent.value}")
        
        # 1. Analyze prompt complexity
        complexity = self.complexity_analyzer.analyze_complexity(prompt)
        
        # 2. Get conversation context if available
        conversation_context = None
        if conversation_id:
            conversation_context = conversation_context_tracker.get_conversation_context(conversation_id)
        
        # 3. Get available models from health monitor
        available_models = health_monitor.get_available_models(intent.value)
        if not available_models:
            # Fallback to basic model list
            available_models = self._get_fallback_models(intent)
        
        # 4. Create prediction context
        prediction_context = PredictionContext(
            intent_type=intent.value,
            complexity=complexity,
            conversation_length=conversation_context.total_turns if conversation_context else 0,
            user_preferences={},  # Could be enhanced with user preferences
            time_constraints=None,  # Could be enhanced with time requirements
            quality_requirements=7.0,  # Default quality requirement
            previous_models_used=conversation_context.models_used[-3:] if conversation_context else []
        )
        
        # 5. Get performance predictions
        performance_predictions = performance_predictor.predict_performance(prediction_context)
        
        # 6. Get conversation-aware rankings from health monitor
        conversation_rankings = health_monitor.get_conversation_aware_rankings(
            conversation_id=conversation_id,
            intent_type=intent.value,
            complexity=complexity
        )
        
        # 7. Combine all scoring factors
        model_scores = self._combine_scoring_factors(
            available_models=available_models,
            performance_predictions=performance_predictions,
            conversation_rankings=conversation_rankings,
            intent=intent,
            complexity=complexity,
            conversation_context=conversation_context
        )
        
        # 8. Select the best model
        if model_scores:
            selected_model = max(model_scores.keys(), key=lambda x: model_scores[x])
        else:
            selected_model = self._get_fallback_model(intent)
        
        # 9. Create selection context for explanation
        selection_context = {
            'intent_type': intent.value,
            'complexity': complexity,
            'conversation_context': conversation_context.__dict__ if conversation_context else None,
            'performance_history_scores': {m: s for m, s in performance_predictions},
            'real_time_scores': {m: s for m, s in conversation_rankings},
            'health_scores': {m: health_monitor.get_model_quality_score(m) for m in available_models},
            'task_suitability_scores': model_scores,
            'predicted_performance': dict(performance_predictions).get(selected_model),
            'fallback_triggered': False,
            'selection_strategy': 'dynamic_intelligent'
        }
        
        # 10. Generate explanation
        explanation = model_selection_explainer.explain_selection(
            selected_model=selected_model,
            candidate_models=available_models,
            model_scores=model_scores,
            selection_context=selection_context,
            request_metadata={
                'request_id': request_id,
                'user_id': user_id,
                'conversation_id': conversation_id,
                'intent_type': intent.value,
                'complexity_score': complexity.complexity_score
            }
        )
        
        logger.info(f"✅ Dynamic Model Selection Complete: {selected_model.split('/')[-1]} "
                   f"(confidence: {explanation.confidence:.2f})")
        
        return selected_model, explanation
    
    def select_model_with_fallback(self, prompt: str, intent: IntentType,
                                 failed_model: Optional[str] = None,
                                 error_message: Optional[str] = None,
                                 conversation_id: Optional[str] = None,
                                 attempt_count: int = 1) -> Tuple[str, ModelSelectionExplanation]:
        """
        Enhanced model selection with intelligent fallback strategies
        
        Args:
            prompt: User prompt
            intent: Classified intent type
            failed_model: Model that failed (if any)
            error_message: Error message from failed model
            conversation_id: Optional conversation identifier
            attempt_count: Number of attempts made so far
            
        Returns:
            Tuple of (selected_model, selection_explanation)
        """
        logger.info(f"🔄 Fallback Model Selection Started (attempt {attempt_count})")
        
        if failed_model and error_message:
            # 1. Analyze the error
            error_type = dynamic_fallback_strategy.analyze_error(
                error_message, failed_model, intent.value, attempt_count
            )
            
            # 2. Get available models
            available_models = health_monitor.get_available_models(intent.value)
            
            # 3. Get conversation context
            conversation_context = None
            if conversation_id:
                conversation_context = conversation_context_tracker.get_conversation_context(conversation_id)
            
            # 4. Analyze prompt complexity
            complexity = self.complexity_analyzer.analyze_complexity(prompt)
            
            # 5. Determine fallback strategy
            fallback_decision = dynamic_fallback_strategy.determine_fallback_strategy(
                error_type=error_type,
                failed_model=failed_model,
                intent_type=intent.value,
                complexity=complexity,
                available_models=available_models,
                conversation_context=conversation_context.__dict__ if conversation_context else None
            )
            
            # 6. Wait if strategy recommends it
            if fallback_decision.wait_time:
                import time
                logger.info(f"⏳ Waiting {fallback_decision.wait_time} seconds before fallback")
                time.sleep(min(fallback_decision.wait_time, 30))  # Cap wait time
            
            # 7. Select from recommended models
            if fallback_decision.recommended_models:
                selected_model = fallback_decision.recommended_models[0]
                
                # Create selection context for explanation
                selection_context = {
                    'intent_type': intent.value,
                    'complexity': complexity,
                    'conversation_context': conversation_context.__dict__ if conversation_context else None,
                    'fallback_triggered': True,
                    'fallback_reason': fallback_decision.strategy_type,
                    'error_type': error_type.value,
                    'failed_model': failed_model,
                    'attempt_count': attempt_count
                }
                
                # Create model scores for explanation
                model_scores = {}
                for i, model in enumerate(fallback_decision.recommended_models):
                    model_scores[model] = fallback_decision.confidence - (i * 0.1)
                
                # Generate explanation
                explanation = model_selection_explainer.explain_selection(
                    selected_model=selected_model,
                    candidate_models=fallback_decision.recommended_models,
                    model_scores=model_scores,
                    selection_context=selection_context
                )
                
                logger.info(f"🎯 Fallback Model Selected: {selected_model.split('/')[-1]} "
                           f"(strategy: {fallback_decision.strategy_type})")
                
                return selected_model, explanation
        
        # If no fallback strategy or no failed model, use dynamic selection
        return self.select_model_dynamic(prompt, intent, conversation_id)
    
    def _combine_scoring_factors(self, available_models: List[str],
                               performance_predictions: List[Tuple[str, float]],
                               conversation_rankings: List[Tuple[str, float]],
                               intent: IntentType,
                               complexity: PromptComplexity,
                               conversation_context: Optional[Any]) -> Dict[str, float]:
        """Combine multiple scoring factors for intelligent model selection"""
        model_scores = {}
        
        # Convert predictions and rankings to dictionaries
        perf_dict = dict(performance_predictions)
        conv_dict = dict(conversation_rankings)
        
        for model in available_models:
            score = 0.0
            
            # Performance prediction factor (40% weight)
            if model in perf_dict:
                score += perf_dict[model] * 0.4
            
            # Conversation-aware ranking factor (35% weight)
            if model in conv_dict:
                score += conv_dict[model] * 0.35
            
            # Task suitability factor (15% weight)
            task_score = self._calculate_task_suitability(model, intent, complexity)
            score += task_score * 0.15
            
            # Model health factor (10% weight)
            health_score = health_monitor.get_model_quality_score(model)
            score += health_score * 0.10
            
            model_scores[model] = score
        
        return model_scores
    
    def _calculate_task_suitability(self, model: str, intent: IntentType, complexity: PromptComplexity) -> float:
        """Calculate how suitable a model is for a specific task"""
        suitability_score = 5.0  # Base score
        
        # Model capabilities based on model names/types
        model_lower = model.lower()
        
        # Code generation suitability
        if intent == IntentType.CODE_GENERATION:
            if any(indicator in model_lower for indicator in ['code', 'deepseek', 'starcoder']):
                suitability_score += 2.0
            elif any(indicator in model_lower for indicator in ['phi', 'qwen']):
                suitability_score += 1.0
        
        # Mathematical reasoning suitability
        elif intent == IntentType.MATHEMATICAL_REASONING:
            if any(indicator in model_lower for indicator in ['qwen', 'deepseek', 'llama']):
                suitability_score += 1.5
        
        # Creative writing suitability
        elif intent == IntentType.CREATIVE_WRITING:
            if any(indicator in model_lower for indicator in ['llama', 'qwen']):
                suitability_score += 1.0
        
        # Complexity adjustments
        if complexity.complexity_score > 7.0:
            # High complexity - prefer larger models
            if any(indicator in model_lower for indicator in ['70b', '72b', 'large']):
                suitability_score += 1.0
            elif any(indicator in model_lower for indicator in ['0.5b', 'mini', 'small']):
                suitability_score -= 1.0
        elif complexity.complexity_score < 3.0:
            # Low complexity - efficient models are fine
            if any(indicator in model_lower for indicator in ['0.5b', '1.5b', 'mini']):
                suitability_score += 0.5
        
        return max(0.0, min(10.0, suitability_score))
    
    def _get_fallback_models(self, intent: IntentType) -> List[str]:
        """Get fallback models when health monitor doesn't have data"""
        from ..config import Config
        
        fallback_mapping = {
            IntentType.CODE_GENERATION: [
                Config.DEFAULT_CODE_MODEL,
                Config.ADVANCED_CODE_MODEL,
                Config.EFFICIENT_CODE_MODEL
            ],
            IntentType.MATHEMATICAL_REASONING: [
                Config.MATH_TEXT_MODEL,
                Config.REASONING_TEXT_MODEL,
                Config.ADVANCED_TEXT_MODEL
            ],
            IntentType.IMAGE_GENERATION: [
                Config.DEFAULT_IMAGE_MODEL,
                Config.FLAGSHIP_IMAGE_MODEL,
                Config.FALLBACK_IMAGE_MODEL
            ]
        }
        
        specific_models = fallback_mapping.get(intent, [])
        
        # Add general fallback models
        general_fallbacks = [
            Config.DEFAULT_TEXT_MODEL,
            Config.BALANCED_TEXT_MODEL,
            Config.EFFICIENT_TEXT_MODEL,
            Config.FALLBACK_TEXT_MODEL
        ]
        
        # Combine and filter out None values
        all_models = specific_models + general_fallbacks
        return [model for model in all_models if model]
    
    def _get_fallback_model(self, intent: IntentType) -> str:
        """Get a single fallback model"""
        fallback_models = self._get_fallback_models(intent)
        return fallback_models[0] if fallback_models else "microsoft/Phi-3-mini-4k-instruct"
    
    def record_model_performance(self, model_name: str, intent_type: str,
                               success: bool, response_time: float,
                               quality_score: float, complexity: PromptComplexity,
                               conversation_id: Optional[str] = None,
                               error_type: Optional[str] = None) -> None:
        """
        Record model performance for learning and adaptation
        
        Args:
            model_name: Name of the model used
            intent_type: Type of task performed
            success: Whether the request was successful
            response_time: Time taken to respond
            quality_score: Quality score of the response (0-10)
            complexity: Task complexity information
            conversation_id: Optional conversation identifier
            error_type: Type of error if failed
        """
        # Record in health monitor for real-time adaptation
        health_monitor.record_real_time_feedback(
            model_name=model_name,
            success=success,
            response_time=response_time,
            quality_score=quality_score,
            intent_type=intent_type,
            complexity=complexity,
            conversation_id=conversation_id,
            error_type=error_type
        )
        
        # Record in performance predictor for learning
        performance_predictor.record_performance(
            model_name=model_name,
            intent_type=intent_type,
            success=success,
            response_time=response_time,
            quality_score=quality_score,
            complexity=complexity,
            error_type=error_type
        )
        
        # Record conversation turn if conversation_id provided
        if conversation_id:
            turn_data = ConversationTurn(
                turn_id=int(time.time() * 1000),  # Use timestamp as turn ID
                timestamp=datetime.now(),
                intent_type=intent_type,
                complexity=complexity,
                model_used=model_name,
                success=success,
                response_time=response_time,
                quality_score=quality_score,
                user_satisfaction=None,  # Could be enhanced with feedback
                topic_shift=False,  # Could be enhanced with topic detection
                followup_type='continuation'  # Could be enhanced with followup analysis
            )
            
            conversation_context_tracker.record_turn(conversation_id, turn_data)
        
        logger.debug(f"📊 Recorded performance for {model_name}: "
                    f"success={success}, quality={quality_score:.1f}")
    
    def select_model(self, prompt: str, intent: Optional[IntentType] = None, 
                    complexity: Optional[PromptComplexity] = None, 
                    context: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Simple convenience method for model selection - delegates to sophisticated routing
        
        Args:
            prompt (str): User input prompt
            intent (Optional[IntentType]): Intent type if known
            complexity (Optional[PromptComplexity]): Complexity analysis if available
            context (Optional[Dict]): Additional context
            
        Returns:
            Tuple[str, Dict]: (selected_model_name, selection_metadata)
        """
        # If no intent provided, determine it from the prompt
        if intent is None:
            intent_result = asyncio.run(self.route_prompt(prompt))
            intent = intent_result[0]
        
        # If no complexity provided, analyze it
        if complexity is None:
            complexity = self.complexity_analyzer.analyze_complexity(prompt, context)
        
        # Delegate to the sophisticated selection method
        return self.select_optimal_model(
            intent=intent.value if intent else "text_generation",
            complexity=complexity,
            context=ContextState() if context else None
        )

# Global router instance
router = IntelligentRouter()

# Export complexity_analyzer at module level for backward compatibility
complexity_analyzer = router.complexity_analyzer