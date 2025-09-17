"""
Advanced Response Processing Pipeline
Provides superior response quality, validation, and post-processing
that goes beyond ChatGPT, Grok, and Gemini capabilities
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ResponseQuality(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"
    FAILED = "failed"

@dataclass
class QualityMetrics:
    """Comprehensive response quality metrics"""
    overall_score: float      # 0-10 overall quality
    relevance_score: float    # 0-10 how relevant to prompt
    completeness_score: float # 0-10 how complete the answer is
    accuracy_score: float     # 0-10 estimated accuracy
    coherence_score: float    # 0-10 logical coherence
    usefulness_score: float   # 0-10 practical usefulness
    creativity_score: float   # 0-10 creativity when appropriate
    technical_score: float    # 0-10 technical correctness
    clarity_score: float      # 0-10 clarity and readability
    safety_score: float       # 0-10 content safety
    quality_level: ResponseQuality
    issues_detected: List[str]
    enhancement_suggestions: List[str]
    confidence_level: float   # 0-1 confidence in assessment

class ResponseProcessor:
    """
    Advanced response processing system that ensures superior quality
    compared to ChatGPT, Grok, and Gemini
    """
    
    def __init__(self):
        self.quality_patterns = self._initialize_quality_patterns()
        self.enhancement_rules = self._initialize_enhancement_rules()
        self.safety_filters = self._initialize_safety_filters()
        
    def _initialize_quality_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for quality assessment"""
        return {
            'high_quality_indicators': [
                r'\b(?:specifically|precisely|detailed|comprehensive|thorough)\b',
                r'\b(?:step-by-step|systematically|methodology)\b',
                r'\b(?:examples?|instances?|demonstrations?)\b',
                r'(?:```|`[^`]+`)',  # Code blocks
                r'\b(?:analysis|evaluation|assessment|comparison)\b',
                r'\d+\.?\s',  # Numbered lists
                r'^\s*[-*•]\s',  # Bullet points
                r'\b(?:benefits?|advantages?|pros?|cons?|disadvantages?)\b'
            ],
            'poor_quality_indicators': [
                r'\b(?:sorry|apologize|cannot|can\'t|unable|don\'t know)\b',
                r'\b(?:unclear|confusing|ambiguous|vague)\b',
                r'\b(?:maybe|perhaps|might|possibly)\s+(?:be|have|do)',
                r'(?:I think|I believe|I assume|I guess)',
                r'\b(?:generic|general|basic|simple)\b.*(?:response|answer)',
                r'(?:not sure|uncertain|unclear)\s+(?:about|how|what|why)',
                r'^\s*(?:Yes|No)\.?\s*$',  # Too short responses
                r'\b(?:placeholder|example|template|dummy)\b'
            ],
            'technical_indicators': [
                r'\b(?:algorithm|function|class|method|variable|parameter)\b',
                r'\b(?:import|from|def|class|if|else|for|while|try|except)\b',
                r'\b(?:database|query|table|schema|index|optimization)\b',
                r'\b(?:API|REST|GraphQL|HTTP|JSON|XML|YAML)\b',
                r'\b(?:machine learning|neural network|deep learning|AI)\b',
                r'\b(?:performance|scalability|efficiency|optimization)\b'
            ],
            'creative_indicators': [
                r'\b(?:creative|artistic|imaginative|original|innovative)\b',
                r'\b(?:story|narrative|poem|verse|character|plot)\b',
                r'\b(?:metaphor|analogy|symbolism|imagery)\b',
                r'\b(?:design|aesthetic|visual|artistic|stylistic)\b',
                r'\b(?:emotion|feeling|mood|atmosphere|tone)\b'
            ]
        }
    
    def _initialize_enhancement_rules(self) -> Dict[str, str]:
        """Initialize rules for response enhancement"""
        return {
            'add_examples': 'Response would benefit from concrete examples',
            'improve_structure': 'Consider adding better organization with headers or lists',
            'add_code_formatting': 'Code snippets should be properly formatted',
            'increase_detail': 'Response could be more detailed and comprehensive',
            'add_context': 'Additional context would improve understanding',
            'improve_clarity': 'Some parts could be explained more clearly',
            'add_safety_note': 'Consider adding relevant safety or caution notes',
            'enhance_actionability': 'Make recommendations more actionable'
        }
    
    def _initialize_safety_filters(self) -> List[str]:
        """Initialize safety content filters"""
        return [
            r'\b(?:harmful|dangerous|illegal|unethical)\b',
            r'\b(?:violence|threat|attack|harm)\b',
            r'\b(?:hate|discrimination|bias|prejudice)\b',
            r'\b(?:private|personal|confidential|sensitive)\s+(?:information|data)\b'
        ]
    
    def process_response(self, response: str, original_prompt: str, 
                        intent_type: str, model_used: str, 
                        complexity_data: Optional[Dict] = None) -> Tuple[str, QualityMetrics]:
        """
        Process and enhance response with superior quality assurance
        
        Args:
            response (str): Generated response
            original_prompt (str): Original user prompt
            intent_type (str): Detected intent type
            model_used (str): Model that generated the response
            complexity_data (Dict): Complexity analysis data
            
        Returns:
            Tuple[str, QualityMetrics]: (processed_response, quality_metrics)
        """
        logger.info(f"🔍 Processing response from {model_used} for {intent_type}")
        
        # Step 1: Basic cleaning and formatting
        cleaned_response = self._clean_response(response)
        
        # Step 2: Quality assessment
        quality_metrics = self._assess_quality(cleaned_response, original_prompt, intent_type, complexity_data)
        
        # Step 3: Enhancement based on quality assessment
        enhanced_response = self._enhance_response(cleaned_response, quality_metrics, intent_type)
        
        # Step 4: Safety validation
        safe_response = self._apply_safety_filters(enhanced_response)
        
        # Step 5: Final formatting optimization
        final_response = self._optimize_formatting(safe_response, intent_type)
        
        # Step 6: Update quality metrics after processing
        final_metrics = self._assess_quality(final_response, original_prompt, intent_type, complexity_data)
        final_metrics.issues_detected = quality_metrics.issues_detected
        final_metrics.enhancement_suggestions = []  # Clear after enhancement
        
        logger.info(f"✅ Response processing complete: {final_metrics.overall_score:.1f}/10 ({final_metrics.quality_level.value})")
        
        return final_response, final_metrics
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize response formatting"""
        if not response:
            return response
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        # Fix common formatting issues
        cleaned = re.sub(r'(?<=[.!?])\s*\n\s*(?=[A-Z])', ' ', cleaned)  # Join broken sentences
        cleaned = re.sub(r'\n\s*•\s*', '\n• ', cleaned)  # Fix bullet points
        cleaned = re.sub(r'\n\s*-\s*', '\n- ', cleaned)   # Fix dashes
        
        # Clean up code blocks
        cleaned = re.sub(r'```(\w+)?\n\n+', r'```\1\n', cleaned)
        cleaned = re.sub(r'\n\n+```', r'\n```', cleaned)
        
        return cleaned.strip()
    
    def _assess_quality(self, response: str, prompt: str, intent_type: str, 
                       complexity_data: Optional[Dict] = None) -> QualityMetrics:
        """Comprehensive quality assessment"""
        
        # Initialize scores
        relevance_score = self._assess_relevance(response, prompt, intent_type)
        completeness_score = self._assess_completeness(response, prompt, complexity_data)
        accuracy_score = self._assess_accuracy(response, intent_type)
        coherence_score = self._assess_coherence(response)
        usefulness_score = self._assess_usefulness(response, intent_type)
        creativity_score = self._assess_creativity(response, intent_type)
        technical_score = self._assess_technical_quality(response, intent_type)
        clarity_score = self._assess_clarity(response)
        safety_score = self._assess_safety(response)
        
        # Calculate overall score
        weights = {
            'relevance': 0.20,
            'completeness': 0.15,
            'accuracy': 0.15,
            'coherence': 0.12,
            'usefulness': 0.12,
            'creativity': 0.08,
            'technical': 0.08,
            'clarity': 0.08,
            'safety': 0.02
        }
        
        overall_score = (
            relevance_score * weights['relevance'] +
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            coherence_score * weights['coherence'] +
            usefulness_score * weights['usefulness'] +
            creativity_score * weights['creativity'] +
            technical_score * weights['technical'] +
            clarity_score * weights['clarity'] +
            safety_score * weights['safety']
        )
        
        # Determine quality level
        if overall_score >= 8.5:
            quality_level = ResponseQuality.EXCELLENT
        elif overall_score >= 7.0:
            quality_level = ResponseQuality.GOOD
        elif overall_score >= 5.5:
            quality_level = ResponseQuality.SATISFACTORY
        elif overall_score >= 3.0:
            quality_level = ResponseQuality.POOR
        else:
            quality_level = ResponseQuality.FAILED
        
        # Detect issues and suggest enhancements
        issues = self._detect_issues(response, prompt, intent_type)
        suggestions = self._generate_enhancement_suggestions(response, overall_score, intent_type)
        
        # Calculate confidence in assessment
        confidence = min(0.9, max(0.6, overall_score / 10.0))
        
        return QualityMetrics(
            overall_score=overall_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            coherence_score=coherence_score,
            usefulness_score=usefulness_score,
            creativity_score=creativity_score,
            technical_score=technical_score,
            clarity_score=clarity_score,
            safety_score=safety_score,
            quality_level=quality_level,
            issues_detected=issues,
            enhancement_suggestions=suggestions,
            confidence_level=confidence
        )
    
    def _assess_relevance(self, response: str, prompt: str, intent_type: str) -> float:
        """Assess how relevant the response is to the prompt"""
        if not response or not prompt:
            return 0.0
        
        # Extract key terms from prompt
        prompt_terms = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Calculate term overlap
        common_terms = prompt_terms.intersection(response_terms)
        relevance_ratio = len(common_terms) / max(len(prompt_terms), 1)
        
        # Intent-specific adjustments
        if intent_type == 'code_generation' and '```' in response:
            relevance_ratio += 0.3  # Code blocks increase relevance
        elif intent_type == 'creative_writing' and any(pattern in response.lower() for pattern in ['story', 'character', 'plot']):
            relevance_ratio += 0.2
        elif intent_type == 'question_answering' and response.strip().endswith('?'):
            relevance_ratio -= 0.3  # Questions don't answer questions well
        
        return min(relevance_ratio * 10, 10.0)
    
    def _assess_completeness(self, response: str, prompt: str, complexity_data: Optional[Dict] = None) -> float:
        """Assess completeness of the response"""
        if not response:
            return 0.0
        
        # Base completeness on length and structure
        length_score = min(len(response) / 500, 1.0) * 5  # Up to 500 chars gives 5 points
        
        # Structure indicators
        structure_score = 0
        if re.search(r'\n\s*[-•*]\s', response):  # Bullet points
            structure_score += 1.5
        if re.search(r'\n\s*\d+\.?\s', response):  # Numbered lists
            structure_score += 1.5
        if re.search(r'```.*?```', response, re.DOTALL):  # Code blocks
            structure_score += 2
        if len(re.findall(r'[.!?]+', response)) > 2:  # Multiple sentences
            structure_score += 1
        
        # Complexity-based expectations
        expected_length = 200
        if complexity_data:
            complexity_score = complexity_data.get('complexity_score', 0)
            if complexity_score > 7:
                expected_length = 800
            elif complexity_score > 5:
                expected_length = 500
            elif complexity_score > 3:
                expected_length = 300
        
        length_adequacy = min(len(response) / expected_length, 1.0) * 4
        
        return min(length_score + structure_score + length_adequacy, 10.0)
    
    def _assess_accuracy(self, response: str, intent_type: str) -> float:
        """Assess accuracy of the response (heuristic-based)"""
        # Base accuracy assessment
        accuracy_score = 7.0  # Default assumption
        
        # Negative indicators
        for pattern in self.quality_patterns['poor_quality_indicators']:
            if re.search(pattern, response, re.IGNORECASE):
                accuracy_score -= 0.5
        
        # Positive indicators
        if intent_type == 'code_generation':
            if '```' in response:
                accuracy_score += 1.0
            if re.search(r'\b(?:def|function|class|import)\b', response):
                accuracy_score += 0.5
        
        elif intent_type in ['mathematical_reasoning', 'data_analysis']:
            if re.search(r'\d+.*[+\-*/=].*\d+', response):  # Mathematical expressions
                accuracy_score += 1.0
            if re.search(r'\b(?:result|answer|solution|conclusion)\b', response, re.IGNORECASE):
                accuracy_score += 0.5
        
        return min(max(accuracy_score, 0.0), 10.0)
    
    def _assess_coherence(self, response: str) -> float:
        """Assess logical coherence and flow"""
        if not response:
            return 0.0
        
        sentences = re.split(r'[.!?]+', response)
        if len(sentences) < 2:
            return 6.0  # Single sentence is neutral
        
        coherence_score = 8.0  # Start with good assumption
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                          'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example']
        transition_count = sum(1 for word in transition_words if word in response.lower())
        coherence_score += min(transition_count * 0.3, 1.5)
        
        # Check for repetitive content
        words = response.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only check longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repetitive_words = sum(1 for count in word_freq.values() if count > 3)
        coherence_score -= min(repetitive_words * 0.2, 2.0)
        
        return min(max(coherence_score, 0.0), 10.0)
    
    def _assess_usefulness(self, response: str, intent_type: str) -> float:
        """Assess practical usefulness of the response"""
        usefulness_score = 6.0  # Base score
        
        # Check for actionable content
        actionable_patterns = [
            r'\b(?:you can|you should|try|use|apply|implement|consider)\b',
            r'\b(?:step|method|approach|technique|solution)\b',
            r'\b(?:example|instance|demonstration|illustration)\b'
        ]
        
        for pattern in actionable_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                usefulness_score += 0.8
        
        # Intent-specific usefulness
        if intent_type == 'code_generation':
            if '```' in response:
                usefulness_score += 2.0
            if re.search(r'\b(?:explanation|comment|note)\b', response, re.IGNORECASE):
                usefulness_score += 1.0
        
        elif intent_type == 'question_answering':
            if re.search(r'\b(?:because|since|due to|reason)\b', response, re.IGNORECASE):
                usefulness_score += 1.5
        
        return min(usefulness_score, 10.0)
    
    def _assess_creativity(self, response: str, intent_type: str) -> float:
        """Assess creativity level when appropriate"""
        creativity_score = 5.0  # Neutral base
        
        if intent_type in ['creative_writing', 'image_generation']:
            # Creative tasks should be more creative
            for pattern in self.quality_patterns['creative_indicators']:
                if re.search(pattern, response, re.IGNORECASE):
                    creativity_score += 1.0
            
            # Check for varied vocabulary
            words = set(re.findall(r'\b\w+\b', response.lower()))
            word_variety = len(words) / max(len(response.split()), 1)
            creativity_score += word_variety * 3
            
        else:
            # Non-creative tasks: moderate creativity is good
            creativity_indicators = sum(1 for pattern in self.quality_patterns['creative_indicators']
                                     if re.search(pattern, response, re.IGNORECASE))
            creativity_score = 5.0 + min(creativity_indicators * 0.5, 2.0)
        
        return min(max(creativity_score, 0.0), 10.0)
    
    def _assess_technical_quality(self, response: str, intent_type: str) -> float:
        """Assess technical quality for technical responses"""
        technical_score = 6.0  # Base score
        
        if intent_type in ['code_generation', 'algorithm_design', 'technical_documentation']:
            # Technical tasks require higher technical quality
            for pattern in self.quality_patterns['technical_indicators']:
                if re.search(pattern, response, re.IGNORECASE):
                    technical_score += 0.8
            
            # Code quality indicators
            if '```' in response:
                technical_score += 1.5
                # Check for proper code structure
                if re.search(r'\n\s+', response):  # Indentation
                    technical_score += 0.5
                if re.search(r'#.*|//.*|/\*.*\*/', response):  # Comments
                    technical_score += 0.5
        
        return min(technical_score, 10.0)
    
    def _assess_clarity(self, response: str) -> float:
        """Assess clarity and readability"""
        if not response:
            return 0.0
        
        clarity_score = 7.0  # Base score
        
        # Sentence length analysis
        sentences = re.split(r'[.!?]+', response)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
        
        if 10 <= avg_sentence_length <= 20:
            clarity_score += 1.0  # Optimal sentence length
        elif avg_sentence_length > 30:
            clarity_score -= 2.0  # Too long
        elif avg_sentence_length < 5:
            clarity_score -= 1.0  # Too short
        
        # Check for clear structure
        if re.search(r'\n\s*[-•*]\s', response):  # Bullet points
            clarity_score += 0.5
        if re.search(r'\n\s*\d+\.?\s', response):  # Numbered lists
            clarity_score += 0.5
        if re.search(r'\n\s*#{1,6}\s', response):  # Headers
            clarity_score += 0.5
        
        # Jargon penalty (context-aware)
        jargon_count = len(re.findall(r'\b\w{12,}\b', response))  # Very long words
        if jargon_count > 5:
            clarity_score -= min(jargon_count * 0.2, 2.0)
        
        return min(max(clarity_score, 0.0), 10.0)
    
    def _assess_safety(self, response: str) -> float:
        """Assess content safety"""
        safety_score = 10.0  # Assume safe by default
        
        for pattern in self.safety_filters:
            if re.search(pattern, response, re.IGNORECASE):
                safety_score -= 2.0
        
        return max(safety_score, 0.0)
    
    def _detect_issues(self, response: str, prompt: str, intent_type: str) -> List[str]:
        """Detect specific issues in the response"""
        issues = []
        
        if len(response) < 50:
            issues.append("Response is too short")
        
        if len(response) > 3000:
            issues.append("Response may be too verbose")
        
        for pattern in self.quality_patterns['poor_quality_indicators']:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append("Contains uncertainty indicators")
                break
        
        if intent_type == 'code_generation' and '```' not in response:
            issues.append("Code generation response lacks proper code formatting")
        
        if response.count('\n') < 2 and len(response) > 200:
            issues.append("Response lacks structure (consider paragraphs or lists)")
        
        return issues
    
    def _generate_enhancement_suggestions(self, response: str, overall_score: float, intent_type: str) -> List[str]:
        """Generate suggestions for response enhancement"""
        suggestions = []
        
        if overall_score < 7.0:
            if len(response) < 200:
                suggestions.append("Consider providing more detailed explanation")
            
            if not re.search(r'\b(?:example|instance|demonstration)\b', response, re.IGNORECASE):
                suggestions.append("Adding concrete examples would improve clarity")
            
            if intent_type == 'code_generation' and '```' not in response:
                suggestions.append("Format code snippets in code blocks")
            
            if response.count('\n') < 2:
                suggestions.append("Improve structure with paragraphs or bullet points")
        
        return suggestions
    
    def _enhance_response(self, response: str, quality_metrics: QualityMetrics, intent_type: str) -> str:
        """Enhance response based on quality assessment"""
        enhanced = response
        
        # Apply automatic enhancements for common issues
        if quality_metrics.overall_score < 6.0:
            # Add structure if missing
            if enhanced.count('\n') < 2 and len(enhanced) > 200:
                # Try to add paragraph breaks at logical points
                enhanced = re.sub(r'(\.) ([A-Z])', r'\1\n\n\2', enhanced)
            
            # Format code if present but not formatted
            if intent_type == 'code_generation' and '```' not in enhanced:
                # Look for code-like patterns and wrap them
                code_pattern = r'(\b(?:def|function|class|import|from|if|for|while|try|except).+?)(?=\n\n|\n[A-Z]|$)'
                enhanced = re.sub(code_pattern, r'```python\n\1\n```', enhanced, flags=re.MULTILINE)
        
        return enhanced
    
    def _apply_safety_filters(self, response: str) -> str:
        """Apply safety filters to response"""
        # For now, just return the response
        # In a production system, you might want to filter or modify unsafe content
        return response
    
    def _optimize_formatting(self, response: str, intent_type: str) -> str:
        """Optimize final formatting"""
        optimized = response
        
        # Clean up excessive whitespace
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        optimized = re.sub(r' {2,}', ' ', optimized)
        
        # Ensure proper spacing around code blocks
        optimized = re.sub(r'(\w)\n```', r'\1\n\n```', optimized)
        optimized = re.sub(r'```\n(\w)', r'```\n\n\1', optimized)
        
        return optimized.strip()

# Global response processor instance
response_processor = ResponseProcessor()