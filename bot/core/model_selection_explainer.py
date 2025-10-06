"""
Model Selection Explainer for Superior AI Model Selection
Provides enhanced explainability and transparency in model selection decisions
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

from .types import IntentType
from .bot_types import PromptComplexity
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

@dataclass
class SelectionReason:
    """Represents a reason for model selection"""
    factor: str
    weight: float
    value: float
    contribution: float
    explanation: str

@dataclass
class ModelSelectionExplanation:
    """Complete explanation for a model selection decision"""
    selected_model: str
    confidence: float
    timestamp: datetime
    primary_reasons: List[SelectionReason]
    alternative_models: List[Tuple[str, float, str]]  # (model, score, reason_not_selected)
    context_factors: Dict[str, Any]
    selection_strategy: str
    fallback_triggered: bool
    performance_prediction: Optional[float]
    conversation_influence: Optional[str]
    total_score: float
    
    # Metadata for tracking
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    intent_type: Optional[str] = None
    complexity_score: Optional[float] = None

class ModelSelectionExplainer:
    """
    Advanced Model Selection Explainer System
    Provides comprehensive explanations for model selection decisions
    """
    
    def __init__(self, explanations_file: str = "model_selection_explanations.json"):
        self.explanations_file = Path(explanations_file)
        self.explanation_history: List[Dict] = []
        self.explanation_cache: Dict[str, ModelSelectionExplanation] = {}
        
        # Factor weights for explanation importance
        self.factor_weights = {
            'performance_history': 0.25,
            'real_time_performance': 0.20,
            'conversation_context': 0.15,
            'model_health': 0.15,
            'task_suitability': 0.15,
            'fallback_strategy': 0.10
        }
        
        # Load existing explanations
        self._load_explanation_history()
        
        secure_logger.info("🔍 Model Selection Explainer initialized")
    
    def explain_selection(self, 
                         selected_model: str,
                         candidate_models: List[str],
                         model_scores: Dict[str, float],
                         selection_context: Dict[str, Any],
                         request_metadata: Optional[Dict[str, Any]] = None) -> ModelSelectionExplanation:
        """
        Create a comprehensive explanation for model selection
        
        Args:
            selected_model: The model that was selected
            candidate_models: All models that were considered
            model_scores: Scores for each candidate model
            selection_context: Context information used in selection
            request_metadata: Optional metadata about the request
            
        Returns:
            Complete explanation of the selection decision
        """
        current_time = datetime.now()
        metadata = request_metadata or {}
        
        # Generate primary reasons for selection
        primary_reasons = self._analyze_primary_reasons(
            selected_model, model_scores, selection_context
        )
        
        # Analyze why other models weren't selected
        alternative_models = self._analyze_alternatives(
            selected_model, candidate_models, model_scores, selection_context
        )
        
        # Determine confidence level
        confidence = self._calculate_confidence(
            selected_model, model_scores, primary_reasons
        )
        
        # Extract context factors
        context_factors = self._extract_context_factors(selection_context)
        
        # Determine selection strategy
        strategy = self._determine_selection_strategy(selection_context)
        
        # Check if fallback was triggered
        fallback_triggered = selection_context.get('fallback_triggered', False)
        
        # Get performance prediction if available
        performance_prediction = selection_context.get('predicted_performance')
        
        # Get conversation influence if available
        conversation_influence = self._analyze_conversation_influence(selection_context)
        
        explanation = ModelSelectionExplanation(
            selected_model=selected_model,
            confidence=confidence,
            timestamp=current_time,
            primary_reasons=primary_reasons,
            alternative_models=alternative_models,
            context_factors=context_factors,
            selection_strategy=strategy,
            fallback_triggered=fallback_triggered,
            performance_prediction=performance_prediction,
            conversation_influence=conversation_influence,
            total_score=model_scores.get(selected_model, 0.0),
            request_id=metadata.get('request_id'),
            user_id=metadata.get('user_id'),
            conversation_id=metadata.get('conversation_id'),
            intent_type=metadata.get('intent_type'),
            complexity_score=metadata.get('complexity_score')
        )
        
        # Cache the explanation
        if explanation.request_id:
            self.explanation_cache[explanation.request_id] = explanation
        
        # Add to history
        self._record_explanation(explanation)
        
        # Log the explanation
        self._log_explanation(explanation)
        
        return explanation
    
    def _analyze_primary_reasons(self, selected_model: str, 
                                model_scores: Dict[str, float],
                                context: Dict[str, Any]) -> List[SelectionReason]:
        """Analyze the primary reasons why a model was selected"""
        reasons = []
        selected_score = model_scores.get(selected_model, 0.0)
        
        # Performance history reason
        if context.get('performance_history_scores'):
            perf_scores = context['performance_history_scores']
            if selected_model in perf_scores:
                perf_score = perf_scores[selected_model]
                contribution = perf_score * self.factor_weights['performance_history']
                reasons.append(SelectionReason(
                    factor='performance_history',
                    weight=self.factor_weights['performance_history'],
                    value=perf_score,
                    contribution=contribution,
                    explanation=f"Historical performance score of {perf_score:.2f} indicates strong reliability"
                ))
        
        # Real-time performance reason
        if context.get('real_time_scores'):
            rt_scores = context['real_time_scores']
            if selected_model in rt_scores:
                rt_score = rt_scores[selected_model]
                contribution = rt_score * self.factor_weights['real_time_performance']
                reasons.append(SelectionReason(
                    factor='real_time_performance',
                    weight=self.factor_weights['real_time_performance'],
                    value=rt_score,
                    contribution=contribution,
                    explanation=f"Real-time performance score of {rt_score:.2f} shows current effectiveness"
                ))
        
        # Conversation context reason
        if context.get('conversation_context'):
            conv_context = context['conversation_context']
            conv_score = conv_context.get('model_preferences', {}).get(selected_model, 0.5)
            if conv_score > 0.6:  # Above neutral
                contribution = conv_score * self.factor_weights['conversation_context']
                reasons.append(SelectionReason(
                    factor='conversation_context',
                    weight=self.factor_weights['conversation_context'],
                    value=conv_score,
                    contribution=contribution,
                    explanation=f"Conversation context score of {conv_score:.2f} indicates good fit for this dialogue"
                ))
        
        # Model health reason
        if context.get('health_scores'):
            health_scores = context['health_scores']
            if selected_model in health_scores:
                health_score = health_scores[selected_model]
                contribution = health_score * self.factor_weights['model_health']
                reasons.append(SelectionReason(
                    factor='model_health',
                    weight=self.factor_weights['model_health'],
                    value=health_score,
                    contribution=contribution,
                    explanation=f"Model health score of {health_score:.2f} confirms availability and reliability"
                ))
        
        # Task suitability reason
        if context.get('task_suitability_scores'):
            task_scores = context['task_suitability_scores']
            if selected_model in task_scores:
                task_score = task_scores[selected_model]
                contribution = task_score * self.factor_weights['task_suitability']
                reasons.append(SelectionReason(
                    factor='task_suitability',
                    weight=self.factor_weights['task_suitability'],
                    value=task_score,
                    contribution=contribution,
                    explanation=f"Task suitability score of {task_score:.2f} matches intent type requirements"
                ))
        
        # Fallback strategy reason
        if context.get('fallback_triggered'):
            fallback_reason = context.get('fallback_reason', 'emergency_fallback')
            reasons.append(SelectionReason(
                factor='fallback_strategy',
                weight=self.factor_weights['fallback_strategy'],
                value=1.0,
                contribution=1.0 * self.factor_weights['fallback_strategy'],
                explanation=f"Selected via {fallback_reason} strategy due to primary model failure"
            ))
        
        # Sort by contribution
        reasons.sort(key=lambda x: x.contribution, reverse=True)
        
        return reasons[:5]  # Return top 5 reasons
    
    def _analyze_alternatives(self, selected_model: str, candidate_models: List[str],
                            model_scores: Dict[str, float], 
                            context: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        """Analyze why alternative models weren't selected"""
        alternatives = []
        selected_score = model_scores.get(selected_model, 0.0)
        
        # Sort other models by score
        other_models = [(model, score) for model, score in model_scores.items() 
                       if model != selected_model]
        other_models.sort(key=lambda x: x[1], reverse=True)
        
        for model, score in other_models[:5]:  # Top 5 alternatives
            score_diff = selected_score - score
            
            if score_diff < 0.1:
                reason = "Very close score - selected model marginally better"
            elif score_diff < 0.5:
                reason = "Moderately lower score - selected model notably better"
            elif score_diff < 1.0:
                reason = "Significantly lower score - clear performance gap"
            else:
                reason = "Much lower score - substantial performance difference"
            
            # Add specific reasons based on context
            if context.get('health_scores', {}).get(model, 10.0) < 5.0:
                reason += " + poor health status"
            
            if context.get('conversation_context', {}).get('models_used', []):
                recent_models = context['conversation_context']['models_used'][-3:]
                if model in recent_models and not context.get('success_pattern', True):
                    reason += " + recently failed in conversation"
            
            alternatives.append((model, score, reason))
        
        return alternatives
    
    def _calculate_confidence(self, selected_model: str, model_scores: Dict[str, float],
                            primary_reasons: List[SelectionReason]) -> float:
        """Calculate confidence level in the selection decision"""
        base_confidence = 0.5
        
        # Confidence from score margin
        if len(model_scores) > 1:
            scores = list(model_scores.values())
            scores.sort(reverse=True)
            selected_score = model_scores.get(selected_model, 0.0)
            
            if len(scores) >= 2 and scores[0] > 0:
                if selected_score == scores[0]:  # Selected model has highest score
                    score_margin = (scores[0] - scores[1]) / scores[0]
                    base_confidence += score_margin * 0.3
                else:
                    # Selected model doesn't have highest score - lower confidence
                    base_confidence -= 0.2
        
        # Confidence from number and strength of reasons
        reason_confidence = len(primary_reasons) * 0.05
        avg_contribution = sum(r.contribution for r in primary_reasons) / max(1, len(primary_reasons))
        reason_confidence += avg_contribution * 0.2
        
        base_confidence += reason_confidence
        
        # Cap confidence between 0 and 1
        return max(0.0, min(1.0, base_confidence))
    
    def _extract_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize context factors for explanation"""
        factors = {}
        
        # Intent and complexity
        if 'intent_type' in context:
            factors['intent_type'] = context['intent_type']
        if 'complexity' in context:
            factors['complexity_score'] = getattr(context['complexity'], 'complexity_score', None)
        
        # Conversation context
        if 'conversation_context' in context:
            conv_ctx = context['conversation_context']
            factors['conversation_length'] = conv_ctx.get('turn_count', 0)
            factors['conversation_quality'] = conv_ctx.get('conversation_quality', 5.0)
            factors['dominant_intent'] = conv_ctx.get('dominant_intent')
        
        # Performance context
        if 'predicted_performance' in context:
            factors['predicted_performance'] = context['predicted_performance']
        
        # Error context
        if 'error_type' in context:
            factors['error_type'] = context['error_type']
        if 'failed_model' in context:
            factors['failed_model'] = context['failed_model']
        
        # Time constraints
        if 'time_constraints' in context:
            factors['time_constraints'] = context['time_constraints']
        
        return factors
    
    def _determine_selection_strategy(self, context: Dict[str, Any]) -> str:
        """Determine what selection strategy was used"""
        if context.get('fallback_triggered'):
            return context.get('fallback_reason', 'fallback_strategy')
        elif context.get('conversation_context'):
            return 'conversation_aware'
        elif context.get('predicted_performance'):
            return 'performance_prediction'
        elif context.get('real_time_scores'):
            return 'real_time_adaptation'
        else:
            return 'standard_routing'
    
    def _analyze_conversation_influence(self, context: Dict[str, Any]) -> Optional[str]:
        """Analyze how conversation context influenced the decision"""
        if not context.get('conversation_context'):
            return None
        
        conv_ctx = context['conversation_context']
        influences = []
        
        # Turn count influence
        turn_count = conv_ctx.get('turn_count', 0)
        if turn_count > 10:
            influences.append("long conversation - prioritizing consistency")
        elif turn_count <= 2:
            influences.append("conversation opening - prioritizing engagement")
        
        # Quality influence
        quality = conv_ctx.get('conversation_quality', 5.0)
        if quality > 8.0:
            influences.append("high quality conversation - maintaining successful pattern")
        elif quality < 5.0:
            influences.append("low quality conversation - trying different approach")
        
        # Intent progression influence
        if 'intent_progression' in conv_ctx:
            recent_intents = conv_ctx['intent_progression'][-3:]
            if len(set(recent_intents)) == 1:
                influences.append("consistent intent - selecting specialized model")
            else:
                influences.append("varied intents - selecting versatile model")
        
        return "; ".join(influences) if influences else None
    
    def _record_explanation(self, explanation: ModelSelectionExplanation) -> None:
        """Record explanation in history for analysis"""
        explanation_dict = asdict(explanation)
        # Convert datetime to string for JSON serialization
        explanation_dict['timestamp'] = explanation.timestamp.isoformat()
        
        self.explanation_history.append(explanation_dict)
        
        # Keep only recent explanations (last 1000)
        if len(self.explanation_history) > 1000:
            self.explanation_history = self.explanation_history[-1000:]
        
        # Periodically save to disk
        if len(self.explanation_history) % 50 == 0:
            self._save_explanation_history()
    
    def _log_explanation(self, explanation: ModelSelectionExplanation) -> None:
        """Log explanation for debugging and transparency"""
        model_short = explanation.selected_model.split('/')[-1]
        
        # Create concise log message
        primary_factors = []
        for reason in explanation.primary_reasons[:3]:  # Top 3 reasons
            primary_factors.append(f"{reason.factor}({reason.value:.2f})")
        
        log_message = (f"🎯 Selected {model_short} "
                      f"(score={explanation.total_score:.2f}, "
                      f"confidence={explanation.confidence:.2f}) "
                      f"Strategy: {explanation.selection_strategy} "
                      f"Factors: {', '.join(primary_factors)}")
        
        if explanation.fallback_triggered:
            log_message += " [FALLBACK]"
        
        secure_logger.info(log_message)
        
        # Detailed debug log
        debug_details = {
            'selected_model': model_short,
            'total_score': explanation.total_score,
            'confidence': explanation.confidence,
            'strategy': explanation.selection_strategy,
            'primary_reasons': [r.factor for r in explanation.primary_reasons],
            'conversation_influence': explanation.conversation_influence,
            'context_factors': explanation.context_factors
        }
        
        secure_logger.debug(f"📋 Selection details: {json.dumps(debug_details, indent=2)}")
    
    def get_explanation(self, request_id: str) -> Optional[ModelSelectionExplanation]:
        """Get explanation for a specific request"""
        return self.explanation_cache.get(request_id)
    
    def generate_explanation_report(self, explanation: ModelSelectionExplanation) -> str:
        """Generate a human-readable explanation report"""
        model_name = explanation.selected_model.split('/')[-1]
        
        report = f"""
Model Selection Explanation Report
=====================================

Selected Model: {model_name}
Selection Confidence: {explanation.confidence:.1%}
Total Score: {explanation.total_score:.2f}/10
Strategy Used: {explanation.selection_strategy.replace('_', ' ').title()}
Timestamp: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Primary Selection Reasons:
"""
        
        for i, reason in enumerate(explanation.primary_reasons, 1):
            report += f"""
{i}. {reason.factor.replace('_', ' ').title()}
   Score: {reason.value:.2f} (Weight: {reason.weight:.2f})
   Contribution: {reason.contribution:.2f}
   Explanation: {reason.explanation}
"""
        
        if explanation.conversation_influence:
            report += f"""
Conversation Context Influence:
{explanation.conversation_influence}
"""
        
        if explanation.alternative_models:
            report += """
Alternative Models Considered:
"""
            for model, score, reason in explanation.alternative_models[:3]:
                alt_name = model.split('/')[-1]
                report += f"- {alt_name} (Score: {score:.2f}) - {reason}\n"
        
        if explanation.fallback_triggered:
            report += """
Note: This selection was made via fallback strategy due to primary model failure.
"""
        
        return report
    
    def analyze_selection_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze patterns in model selection over a time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        recent_explanations = [
            exp for exp in self.explanation_history
            if datetime.fromisoformat(exp['timestamp']) > cutoff_time
        ]
        
        if not recent_explanations:
            return {'message': 'No recent selections to analyze'}
        
        analysis = {
            'total_selections': len(recent_explanations),
            'model_distribution': self._analyze_model_distribution(recent_explanations),
            'strategy_distribution': self._analyze_strategy_distribution(recent_explanations),
            'confidence_stats': self._analyze_confidence_stats(recent_explanations),
            'factor_importance': self._analyze_factor_importance(recent_explanations),
            'fallback_rate': self._analyze_fallback_rate(recent_explanations),
            'quality_trends': self._analyze_quality_trends(recent_explanations)
        }
        
        return analysis
    
    def _analyze_model_distribution(self, explanations: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of selected models"""
        model_counts = {}
        for exp in explanations:
            model = exp['selected_model'].split('/')[-1]
            model_counts[model] = model_counts.get(model, 0) + 1
        
        return dict(sorted(model_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_strategy_distribution(self, explanations: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of selection strategies"""
        strategy_counts = {}
        for exp in explanations:
            strategy = exp['selection_strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return strategy_counts
    
    def _analyze_confidence_stats(self, explanations: List[Dict]) -> Dict[str, float]:
        """Analyze confidence statistics"""
        confidences = [exp['confidence'] for exp in explanations]
        
        if not confidences:
            return {}
        
        return {
            'average_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'high_confidence_rate': len([c for c in confidences if c > 0.8]) / len(confidences)
        }
    
    def _analyze_factor_importance(self, explanations: List[Dict]) -> Dict[str, float]:
        """Analyze importance of different factors in selection"""
        factor_contributions = {}
        factor_counts = {}
        
        for exp in explanations:
            for reason in exp['primary_reasons']:
                factor = reason['factor']
                contribution = reason['contribution']
                
                if factor not in factor_contributions:
                    factor_contributions[factor] = 0
                    factor_counts[factor] = 0
                
                factor_contributions[factor] += contribution
                factor_counts[factor] += 1
        
        # Calculate average contributions
        avg_contributions = {}
        for factor, total_contrib in factor_contributions.items():
            avg_contributions[factor] = total_contrib / factor_counts[factor]
        
        return dict(sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_fallback_rate(self, explanations: List[Dict]) -> Dict[str, Any]:
        """Analyze fallback usage patterns"""
        fallback_count = len([exp for exp in explanations if exp['fallback_triggered']])
        total_count = len(explanations)
        
        return {
            'fallback_rate': fallback_count / total_count if total_count > 0 else 0,
            'fallback_count': fallback_count,
            'total_selections': total_count
        }
    
    def _analyze_quality_trends(self, explanations: List[Dict]) -> Dict[str, Any]:
        """Analyze quality trends in selections"""
        scores = [exp['total_score'] for exp in explanations]
        confidences = [exp['confidence'] for exp in explanations]
        
        if not scores:
            return {}
        
        return {
            'average_score': sum(scores) / len(scores),
            'score_trend': 'improving' if scores[-1] > scores[0] else 'declining' if len(scores) > 1 else 'stable',
            'average_confidence': sum(confidences) / len(confidences),
            'confidence_trend': 'improving' if confidences[-1] > confidences[0] else 'declining' if len(confidences) > 1 else 'stable'
        }
    
    def _load_explanation_history(self) -> None:
        """Load explanation history from persistent storage"""
        try:
            if self.explanations_file.exists():
                with open(self.explanations_file, 'r') as f:
                    data = json.load(f)
                
                self.explanation_history = data.get('explanation_history', [])
                
                secure_logger.info(f"📚 Loaded {len(self.explanation_history)} explanation records")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to load explanation history: {safe_error}")
    
    def _save_explanation_history(self) -> None:
        """Save explanation history to persistent storage"""
        try:
            data = {
                'explanation_history': self.explanation_history,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.explanations_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            secure_logger.debug(f"💾 Saved explanation history")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to save explanation history: {safe_error}")
    
    def cleanup_old_explanations(self, days_to_keep: int = 7) -> int:
        """Clean up old explanations to prevent unbounded growth"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        old_count = len(self.explanation_history)
        self.explanation_history = [
            exp for exp in self.explanation_history
            if datetime.fromisoformat(exp['timestamp']) > cutoff_date
        ]
        new_count = len(self.explanation_history)
        
        cleaned_count = old_count - new_count
        
        if cleaned_count > 0:
            self._save_explanation_history()
            secure_logger.info(f"🧹 Cleaned up {cleaned_count} old explanation records")
        
        return cleaned_count

# Global instance
model_selection_explainer = ModelSelectionExplainer()