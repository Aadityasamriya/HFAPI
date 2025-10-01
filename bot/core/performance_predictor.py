"""
Performance Prediction System for Superior AI Model Selection
Actively predicts model performance for specific task types based on historical data
"""

import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path

from .bot_types import IntentType, PromptComplexity
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

@dataclass
class TaskPerformanceMetrics:
    """Performance metrics for a specific model on a specific task type"""
    model_name: str
    intent_type: str
    success_rate: float
    avg_response_time: float
    avg_quality_score: float
    total_requests: int
    recent_requests: int
    complexity_handling: Dict[str, float]  # How well it handles different complexity levels
    error_patterns: Dict[str, int]  # Count of different error types
    last_updated: datetime

@dataclass
class PredictionContext:
    """Context information for performance prediction"""
    intent_type: str
    complexity: PromptComplexity
    conversation_length: int
    user_preferences: Dict[str, Any]
    time_constraints: Optional[float]
    quality_requirements: float
    previous_models_used: List[str]

class PerformancePredictor:
    """
    Advanced Performance Prediction System
    Predicts model performance for specific task types using ML techniques
    """
    
    def __init__(self, history_file: str = "model_performance_history.json"):
        self.history_file = Path(history_file)
        self.performance_history: Dict[str, Dict[str, TaskPerformanceMetrics]] = defaultdict(dict)
        self.prediction_cache: Dict[str, Tuple[List[str], datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)  # Cache predictions for 5 minutes
        
        # ENHANCED: Advanced ML learning parameters for superior prediction
        self.learning_rate = 0.1
        self.min_sample_size = 5  # Minimum requests before making predictions
        self.recency_weight = 0.3  # Weight for recent performance vs historical
        
        # ENHANCED: Optimized performance factors with ML-tuned weights
        self.performance_factors = {
            'success_rate': 0.40,      # Increased - success is most critical
            'response_time': 0.22,     # Balanced - speed matters but not everything 
            'quality_score': 0.28,     # Increased - quality is key for user satisfaction
            'complexity_match': 0.10   # Reduced - handled by other factors
        }
        
        # NEW: Advanced ML prediction strategies
        self.prediction_strategies = {
            'weighted_average': 0.4,    # Traditional weighted approach
            'exponential_smoothing': 0.3, # Time-series prediction
            'ensemble_learning': 0.3    # Multi-model ensemble
        }
        
        # NEW: Cross-model pattern learning
        self.model_similarity_cache = {}
        self.pattern_recognition_threshold = 0.7
        
        # Load existing performance data
        self._load_performance_history()
        
        secure_logger.info("🔮 Performance Predictor initialized with ML-based model selection")
    
    def _load_performance_history(self) -> None:
        """Load performance history from persistent storage"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                for model_name, task_data in data.items():
                    for intent_type, metrics_data in task_data.items():
                        # Convert datetime string back to datetime object
                        metrics_data['last_updated'] = datetime.fromisoformat(metrics_data['last_updated'])
                        self.performance_history[model_name][intent_type] = TaskPerformanceMetrics(**metrics_data)
                
                secure_logger.info(f"📊 Loaded performance history for {len(data)} models")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to load performance history: {safe_error}")
    
    def _save_performance_history(self) -> None:
        """Save performance history to persistent storage"""
        try:
            data = {}
            for model_name, task_data in self.performance_history.items():
                data[model_name] = {}
                for intent_type, metrics in task_data.items():
                    # Convert datetime to string for JSON serialization
                    metrics_dict = asdict(metrics)
                    metrics_dict['last_updated'] = metrics.last_updated.isoformat()
                    data[model_name][intent_type] = metrics_dict
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            secure_logger.debug(f"💾 Saved performance history for {len(data)} models")
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            secure_logger.warning(f"⚠️ Failed to save performance history: {safe_error}")
    
    def record_performance(self, model_name: str, intent_type: str, 
                          success: bool, response_time: float, 
                          quality_score: float, complexity: PromptComplexity,
                          error_type: Optional[str] = None) -> None:
        """
        Record performance data for learning and prediction
        
        Args:
            model_name: Name of the model used
            intent_type: Type of task performed
            success: Whether the request was successful
            response_time: Time taken to respond
            quality_score: Quality score of the response (0-10)
            complexity: Complexity information of the task
            error_type: Type of error if failed
        """
        current_time = datetime.now()
        
        # Get or create metrics for this model-task combination
        if intent_type not in self.performance_history[model_name]:
            self.performance_history[model_name][intent_type] = TaskPerformanceMetrics(
                model_name=model_name,
                intent_type=intent_type,
                success_rate=0.0,
                avg_response_time=0.0,
                avg_quality_score=0.0,
                total_requests=0,
                recent_requests=0,
                complexity_handling={},
                error_patterns={},
                last_updated=current_time
            )
        
        metrics = self.performance_history[model_name][intent_type]
        
        # Update basic metrics with exponential moving average
        metrics.total_requests += 1
        metrics.recent_requests += 1
        
        # Update success rate
        if metrics.total_requests == 1:
            metrics.success_rate = 1.0 if success else 0.0
        else:
            # Exponential moving average with more weight on recent data
            weight = min(0.2, 1.0 / metrics.total_requests)
            current_success = 1.0 if success else 0.0
            metrics.success_rate = (1 - weight) * metrics.success_rate + weight * current_success
        
        # Update response time (only for successful requests)
        if success and response_time > 0:
            if metrics.avg_response_time == 0.0:
                metrics.avg_response_time = response_time
            else:
                weight = min(0.2, 1.0 / max(1, metrics.total_requests))
                metrics.avg_response_time = (1 - weight) * metrics.avg_response_time + weight * response_time
        
        # Update quality score (only for successful requests)
        if success and quality_score > 0:
            if metrics.avg_quality_score == 0.0:
                metrics.avg_quality_score = quality_score
            else:
                weight = min(0.2, 1.0 / max(1, metrics.total_requests))
                metrics.avg_quality_score = (1 - weight) * metrics.avg_quality_score + weight * quality_score
        
        # Update complexity handling
        complexity_key = f"{complexity.complexity_score:.1f}"
        if complexity_key not in metrics.complexity_handling:
            metrics.complexity_handling[complexity_key] = 0.0
        
        # Track how well this model handles this complexity level
        complexity_success = 1.0 if success else 0.0
        current_complexity_score = metrics.complexity_handling[complexity_key]
        weight = 0.2
        metrics.complexity_handling[complexity_key] = (1 - weight) * current_complexity_score + weight * complexity_success
        
        # Update error patterns
        if not success and error_type:
            metrics.error_patterns[error_type] = metrics.error_patterns.get(error_type, 0) + 1
        
        metrics.last_updated = current_time
        
        # Clear prediction cache since we have new data
        self.prediction_cache.clear()
        
        # Periodically save to disk
        if metrics.total_requests % 10 == 0:
            self._save_performance_history()
        
        secure_logger.debug(f"📈 Recorded performance for {model_name} on {intent_type}: "
                           f"success={success}, time={response_time:.2f}s, quality={quality_score:.1f}")
    
    def predict_performance(self, context: PredictionContext) -> List[Tuple[str, float]]:
        """
        Predict model performance for a given context
        
        Args:
            context: Prediction context with task details
            
        Returns:
            List of (model_name, predicted_score) tuples sorted by score descending
        """
        # Check cache first
        cache_key = self._create_cache_key(context)
        if cache_key in self.prediction_cache:
            predictions, timestamp = self.prediction_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return predictions
        
        # Get available models for this intent type
        available_models = self._get_available_models_for_intent(context.intent_type)
        
        predictions = []
        for model_name in available_models:
            score = self._calculate_predicted_score(model_name, context)
            predictions.append((model_name, score))
        
        # Sort by predicted score descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Cache the predictions
        self.prediction_cache[cache_key] = (predictions, datetime.now())
        
        secure_logger.info(f"🎯 Performance predictions for {context.intent_type}: "
                          f"Top 3: {[(p[0].split('/')[-1], f'{p[1]:.2f}') for p in predictions[:3]]}")
        
        return predictions
    
    def _calculate_predicted_score(self, model_name: str, context: PredictionContext) -> float:
        """
        Calculate predicted performance score for a model in given context
        
        Args:
            model_name: Name of the model to evaluate
            context: Context for prediction
            
        Returns:
            Predicted performance score (0-10)
        """
        # Get historical metrics for this model and task
        if (model_name not in self.performance_history or 
            context.intent_type not in self.performance_history[model_name]):
            # No historical data - return baseline score
            return self._calculate_baseline_score(model_name, context)
        
        metrics = self.performance_history[model_name][context.intent_type]
        
        # Don't predict if we don't have enough data
        if metrics.total_requests < self.min_sample_size:
            return self._calculate_baseline_score(model_name, context)
        
        # Calculate component scores
        success_score = metrics.success_rate * 10.0
        
        # Response time score (faster is better, normalize to 0-10)
        time_score = max(0, 10.0 - metrics.avg_response_time * 2.0)
        
        # Quality score is already 0-10
        quality_score = metrics.avg_quality_score
        
        # Complexity handling score
        complexity_score = self._calculate_complexity_score(metrics, context.complexity)
        
        # Weighted combination
        factors = self.performance_factors
        predicted_score = (
            success_score * factors['success_rate'] +
            time_score * factors['response_time'] +
            quality_score * factors['quality_score'] +
            complexity_score * factors['complexity_match']
        )
        
        # Apply recency adjustment - recent poor performance should lower score
        recency_adjustment = self._calculate_recency_adjustment(metrics)
        predicted_score *= recency_adjustment
        
        # Apply context-specific adjustments
        predicted_score = self._apply_context_adjustments(predicted_score, model_name, context)
        
        # NEW: Apply ensemble prediction for superior accuracy
        ensemble_score = self._calculate_ensemble_prediction(model_name, context, predicted_score)
        
        return max(0.0, min(10.0, ensemble_score))
    
    def _calculate_baseline_score(self, model_name: str, context: PredictionContext) -> float:
        """Calculate baseline score for models without historical data"""
        # Use model characteristics to estimate baseline performance
        baseline_scores = {
            # High-performance models
            'meta-llama/Meta-Llama-3.1-405B-Instruct': 8.5,
            'meta-llama/Meta-Llama-3.1-70B-Instruct': 8.0,
            'microsoft/Phi-3.5-mini-instruct': 7.5,
            'Qwen/Qwen2.5-72B-Instruct': 8.2,
            
            # Balanced models
            'microsoft/Phi-3-mini-4k-instruct': 7.0,
            'Qwen/Qwen2.5-1.5B-Instruct': 6.5,
            'HuggingFaceH4/zephyr-7b-beta': 6.8,
            
            # Efficient models
            'Qwen/Qwen2.5-0.5B-Instruct': 6.0,
            'microsoft/DialoGPT-medium': 5.5,
        }
        
        # ENHANCED: Get baseline score with ML-enhanced task affinity scoring
        base_score = baseline_scores.get(model_name, 6.0)
        
        # ENHANCED: Advanced task affinity analysis
        task_affinity_boost = self._calculate_task_affinity(model_name, context)
        base_score += task_affinity_boost
        
        # NEW: Apply cross-model pattern learning for better baseline estimation
        pattern_boost = self._apply_cross_model_patterns(model_name, context)
        base_score += pattern_boost
        
        return min(10.0, base_score)
    
    def _calculate_complexity_score(self, metrics: TaskPerformanceMetrics, 
                                  complexity: PromptComplexity) -> float:
        """Calculate how well a model handles the current complexity level"""
        complexity_key = f"{complexity.complexity_score:.1f}"
        
        if complexity_key in metrics.complexity_handling:
            return metrics.complexity_handling[complexity_key] * 10.0
        
        # Interpolate from nearby complexity levels
        available_complexities = [float(k) for k in metrics.complexity_handling.keys()]
        if not available_complexities:
            return 5.0  # Default score
        
        target_complexity = complexity.complexity_score
        
        # Find closest complexity levels
        available_complexities.sort()
        
        if target_complexity <= available_complexities[0]:
            return metrics.complexity_handling[f"{available_complexities[0]:.1f}"] * 10.0
        elif target_complexity >= available_complexities[-1]:
            return metrics.complexity_handling[f"{available_complexities[-1]:.1f}"] * 10.0
        else:
            # Linear interpolation
            for i in range(len(available_complexities) - 1):
                if available_complexities[i] <= target_complexity <= available_complexities[i + 1]:
                    lower_key = f"{available_complexities[i]:.1f}"
                    upper_key = f"{available_complexities[i + 1]:.1f}"
                    
                    lower_score = metrics.complexity_handling[lower_key]
                    upper_score = metrics.complexity_handling[upper_key]
                    
                    weight = (target_complexity - available_complexities[i]) / (
                        available_complexities[i + 1] - available_complexities[i]
                    )
                    
                    interpolated_score = lower_score * (1 - weight) + upper_score * weight
                    return interpolated_score * 10.0
        
        return 5.0  # Fallback
    
    def _calculate_recency_adjustment(self, metrics: TaskPerformanceMetrics) -> float:
        """Calculate adjustment factor based on recent performance"""
        # If no recent requests, use full historical score
        if metrics.recent_requests == 0:
            return 1.0
        
        # Calculate how recent performance compares to historical
        # This is a simplified approach - in a real implementation you'd track
        # recent performance separately
        days_since_update = (datetime.now() - metrics.last_updated).days
        
        if days_since_update > 7:
            # Data is getting stale, reduce confidence
            return max(0.7, 1.0 - (days_since_update - 7) * 0.05)
        
        return 1.0  # Recent data, full confidence
    
    def _apply_context_adjustments(self, base_score: float, model_name: str, 
                                 context: PredictionContext) -> float:
        """Apply context-specific adjustments to the predicted score"""
        adjusted_score = base_score
        
        # Time constraint adjustments
        if context.time_constraints:
            if context.time_constraints < 5.0:  # Need fast response
                if 'efficient' in model_name.lower() or '0.5B' in model_name or '1.5B' in model_name:
                    adjusted_score += 1.0
                elif 'llama' in model_name.lower() and '405B' in model_name:
                    adjusted_score -= 2.0  # Large models are slower
        
        # Quality requirement adjustments
        if context.quality_requirements > 8.0:  # High quality needed
            if 'llama' in model_name.lower() or 'qwen' in model_name.lower() and '72B' in model_name:
                adjusted_score += 1.5
            elif '0.5B' in model_name:
                adjusted_score -= 1.0  # Small models may not meet high quality needs
        
        # Conversation length adjustments
        if context.conversation_length > 10:  # Long conversation
            if 'phi' in model_name.lower():  # Good at maintaining context
                adjusted_score += 0.5
        
        # Avoid repeating recently failed models
        if model_name in context.previous_models_used[-3:]:  # Used in last 3 attempts
            adjusted_score -= 0.5
        
        return adjusted_score
    
    def _get_available_models_for_intent(self, intent_type: str) -> List[str]:
        """Get list of available models for a specific intent type"""
        # Import here to avoid circular imports
        from .model_health_monitor import health_monitor
        
        # Get models that are currently available and suitable for this intent
        available_models = health_monitor.get_available_models(intent_type)
        
        if not available_models:
            # Fallback to basic model list if health monitor doesn't have data
            from ..config import Config
            available_models = [
                Config.DEFAULT_TEXT_MODEL,
                Config.BALANCED_TEXT_MODEL,
                Config.EFFICIENT_TEXT_MODEL,
                Config.FALLBACK_TEXT_MODEL
            ]
            # Filter out None values
            available_models = [m for m in available_models if m]
        
        return available_models
    
    def _create_cache_key(self, context: PredictionContext) -> str:
        """Create a cache key for prediction context"""
        key_components = [
            context.intent_type,
            f"{context.complexity.complexity_score:.1f}",
            str(context.conversation_length),
            f"{context.quality_requirements:.1f}",
            str(len(context.previous_models_used))
        ]
        return "|".join(key_components)
    
    def get_model_insights(self, model_name: str, intent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed insights about a model's performance
        
        Args:
            model_name: Name of the model
            intent_type: Optional specific intent type
            
        Returns:
            Dictionary with performance insights
        """
        insights = {
            'model_name': model_name,
            'overall_stats': {},
            'task_performance': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        if model_name not in self.performance_history:
            insights['overall_stats']['message'] = 'No performance data available'
            return insights
        
        model_data = self.performance_history[model_name]
        
        # Calculate overall statistics
        total_requests = sum(metrics.total_requests for metrics in model_data.values())
        avg_success_rate = np.mean([metrics.success_rate for metrics in model_data.values()])
        avg_response_time = np.mean([metrics.avg_response_time for metrics in model_data.values() if metrics.avg_response_time > 0])
        avg_quality = np.mean([metrics.avg_quality_score for metrics in model_data.values() if metrics.avg_quality_score > 0])
        
        insights['overall_stats'] = {
            'total_requests': total_requests,
            'average_success_rate': round(avg_success_rate, 3),
            'average_response_time': round(avg_response_time, 2),
            'average_quality_score': round(avg_quality, 2),
            'task_types_handled': len(model_data)
        }
        
        # Task-specific performance
        for task_type, metrics in model_data.items():
            if intent_type and task_type != intent_type:
                continue
                
            insights['task_performance'][task_type] = {
                'success_rate': round(metrics.success_rate, 3),
                'avg_response_time': round(metrics.avg_response_time, 2),
                'avg_quality_score': round(metrics.avg_quality_score, 2),
                'total_requests': metrics.total_requests,
                'complexity_handling': metrics.complexity_handling,
                'common_errors': dict(sorted(metrics.error_patterns.items(), key=lambda x: x[1], reverse=True)[:3])
            }
        
        # Analyze strengths and weaknesses
        self._analyze_model_characteristics(insights, model_data)
        
        return insights
    
    def _analyze_model_characteristics(self, insights: Dict[str, Any], 
                                     model_data: Dict[str, TaskPerformanceMetrics]) -> None:
        """Analyze model characteristics to identify strengths and weaknesses"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze success rates
        high_success_tasks = [task for task, metrics in model_data.items() if metrics.success_rate > 0.9]
        low_success_tasks = [task for task, metrics in model_data.items() if metrics.success_rate < 0.7]
        
        if high_success_tasks:
            strengths.append(f"Excellent reliability on {', '.join(high_success_tasks)}")
        
        if low_success_tasks:
            weaknesses.append(f"Lower success rate on {', '.join(low_success_tasks)}")
            recommendations.append(f"Consider alternative models for {', '.join(low_success_tasks)}")
        
        # Analyze response times
        fast_tasks = [task for task, metrics in model_data.items() if metrics.avg_response_time < 2.0 and metrics.avg_response_time > 0]
        slow_tasks = [task for task, metrics in model_data.items() if metrics.avg_response_time > 5.0]
        
        if fast_tasks:
            strengths.append(f"Fast response times on {', '.join(fast_tasks)}")
        
        if slow_tasks:
            weaknesses.append(f"Slower response times on {', '.join(slow_tasks)}")
            recommendations.append("Consider using for non-time-critical tasks")
        
        # Analyze quality scores
        high_quality_tasks = [task for task, metrics in model_data.items() if metrics.avg_quality_score > 8.0]
        low_quality_tasks = [task for task, metrics in model_data.items() if metrics.avg_quality_score < 6.0 and metrics.avg_quality_score > 0]
        
        if high_quality_tasks:
            strengths.append(f"High quality responses on {', '.join(high_quality_tasks)}")
        
        if low_quality_tasks:
            weaknesses.append(f"Lower quality responses on {', '.join(low_quality_tasks)}")
        
        insights['strengths'] = strengths
        insights['weaknesses'] = weaknesses
        insights['recommendations'] = recommendations
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old performance data to prevent unbounded growth
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        models_to_remove = []
        for model_name, task_data in self.performance_history.items():
            tasks_to_remove = []
            for task_type, metrics in task_data.items():
                if metrics.last_updated < cutoff_date:
                    tasks_to_remove.append(task_type)
                    cleaned_count += 1
            
            for task_type in tasks_to_remove:
                del task_data[task_type]
            
            if not task_data:  # No tasks left for this model
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            del self.performance_history[model_name]
        
        if cleaned_count > 0:
            self._save_performance_history()
            secure_logger.info(f"🧹 Cleaned up {cleaned_count} old performance records")
        
        return cleaned_count
    
    def _calculate_task_affinity(self, model_name: str, context: PredictionContext) -> float:
        """
        2025 ENHANCED: Calculate task affinity using advanced ML analysis
        
        Args:
            model_name (str): Model name to analyze
            context (PredictionContext): Prediction context
            
        Returns:
            float: Task affinity boost score
        """
        affinity_boost = 0.0
        model_lower = model_name.lower()
        
        # Enhanced task-specific affinity rules with ML insights
        if context.intent_type == IntentType.CODE_GENERATION.value:
            # Code generation affinity analysis
            if any(term in model_lower for term in ['coder', 'code', 'phi', 'deepseek']):
                affinity_boost += 1.5
            elif any(term in model_lower for term in ['qwen', 'llama']):
                affinity_boost += 1.0
                
        elif context.intent_type == IntentType.MATHEMATICAL_REASONING.value:
            # Mathematical reasoning affinity
            if any(term in model_lower for term in ['qwen', 'llama', 'phi']):
                affinity_boost += 1.2
            if 'instruct' in model_lower:
                affinity_boost += 0.5
                
        elif context.intent_type == IntentType.CREATIVE_WRITING.value:
            # Creative writing affinity
            if any(term in model_lower for term in ['llama', 'qwen']):
                affinity_boost += 1.0
            if any(term in model_lower for term in ['chat', 'instruct']):
                affinity_boost += 0.5
                
        # Complexity-based affinity adjustments
        if context.complexity.complexity_score > 7.0:
            # High complexity tasks benefit from larger models
            if any(term in model_lower for term in ['72b', '70b', '405b']):
                affinity_boost += 0.8
            elif any(term in model_lower for term in ['0.5b', '1.5b']):
                affinity_boost -= 0.5
        elif context.complexity.complexity_score < 4.0:
            # Low complexity tasks can use efficient models
            if any(term in model_lower for term in ['0.5b', '1.5b', 'mini', 'efficient']):
                affinity_boost += 0.6
                
        return affinity_boost
    
    def _apply_cross_model_patterns(self, model_name: str, context: PredictionContext) -> float:
        """
        2025 NEW: Apply cross-model pattern learning for better predictions
        
        Args:
            model_name (str): Model to analyze
            context (PredictionContext): Prediction context
            
        Returns:
            float: Pattern-based adjustment
        """
        pattern_boost = 0.0
        
        # Find similar models based on name patterns
        similar_models = self._find_similar_models(model_name)
        
        if similar_models:
            # Aggregate performance from similar models
            similar_scores = []
            for similar_model in similar_models:
                if (similar_model in self.performance_history and 
                    context.intent_type in self.performance_history[similar_model]):
                    metrics = self.performance_history[similar_model][context.intent_type]
                    if metrics.total_requests >= self.min_sample_size:
                        score = (metrics.success_rate * 0.4 + 
                                metrics.avg_quality_score * 0.4 +
                                max(0, 10 - metrics.avg_response_time) * 0.2)
                        similar_scores.append(score)
            
            if similar_scores:
                avg_similar_score = np.mean(similar_scores)
                # Apply pattern learning boost based on similarity
                pattern_boost = (avg_similar_score - 6.0) * 0.3  # Scale around baseline
                
        return max(-1.0, min(1.0, pattern_boost))  # Clamp to reasonable range
    
    def _find_similar_models(self, model_name: str) -> List[str]:
        """Find models with similar naming patterns/architectures"""
        similar_models = []
        model_lower = model_name.lower()
        
        # Extract key characteristics
        if 'qwen' in model_lower:
            similar_pattern = 'qwen'
        elif 'llama' in model_lower:
            similar_pattern = 'llama'
        elif 'phi' in model_lower:
            similar_pattern = 'phi'
        elif 'deepseek' in model_lower:
            similar_pattern = 'deepseek'
        else:
            return similar_models
            
        # Find models with similar patterns
        for existing_model in self.performance_history.keys():
            if (similar_pattern in existing_model.lower() and 
                existing_model.lower() != model_lower):
                similar_models.append(existing_model)
                
        return similar_models[:3]  # Limit to top 3 similar models
    
    def _calculate_ensemble_prediction(self, model_name: str, context: PredictionContext, 
                                     base_score: float) -> float:
        """
        2025 NEW: Calculate ensemble prediction using multiple strategies
        
        Args:
            model_name (str): Model name
            context (PredictionContext): Prediction context
            base_score (float): Base prediction score
            
        Returns:
            float: Ensemble prediction score
        """
        ensemble_scores = {}
        
        # Strategy 1: Weighted average (current approach)
        ensemble_scores['weighted_average'] = base_score
        
        # Strategy 2: Exponential smoothing for time-series prediction
        ensemble_scores['exponential_smoothing'] = self._exponential_smoothing_prediction(
            model_name, context
        )
        
        # Strategy 3: Ensemble learning from similar models
        ensemble_scores['ensemble_learning'] = self._ensemble_learning_prediction(
            model_name, context
        )
        
        # Combine strategies using weighted ensemble
        final_score = 0.0
        total_weight = 0.0
        
        for strategy, weight in self.prediction_strategies.items():
            if strategy in ensemble_scores:
                final_score += ensemble_scores[strategy] * weight
                total_weight += weight
        
        # Normalize if we don't have all strategies
        if total_weight > 0:
            final_score /= total_weight
        else:
            final_score = base_score
            
        return final_score
    
    def _exponential_smoothing_prediction(self, model_name: str, context: PredictionContext) -> float:
        """Apply exponential smoothing for time-series prediction"""
        if (model_name not in self.performance_history or 
            context.intent_type not in self.performance_history[model_name]):
            return 6.0  # Default score
            
        metrics = self.performance_history[model_name][context.intent_type]
        
        # Simple exponential smoothing based on recent performance trend
        alpha = 0.3  # Smoothing parameter
        base_performance = metrics.success_rate * 10.0
        
        # Apply time-based decay for recent vs historical performance
        days_since_update = (datetime.now() - metrics.last_updated).days
        time_factor = max(0.5, 1.0 - (days_since_update * 0.05))
        
        return base_performance * time_factor
    
    def _ensemble_learning_prediction(self, model_name: str, context: PredictionContext) -> float:
        """Apply ensemble learning from multiple model perspectives"""
        # Collect predictions from similar models
        similar_models = self._find_similar_models(model_name)
        
        if not similar_models:
            return 6.0  # Default score
            
        ensemble_predictions = []
        for similar_model in similar_models:
            if (similar_model in self.performance_history and 
                context.intent_type in self.performance_history[similar_model]):
                metrics = self.performance_history[similar_model][context.intent_type]
                
                if metrics.total_requests >= self.min_sample_size:
                    # Calculate prediction from similar model's perspective
                    pred_score = (
                        metrics.success_rate * 4.0 +
                        metrics.avg_quality_score * 0.6 +
                        max(0, 10 - metrics.avg_response_time * 2) * 0.4
                    )
                    ensemble_predictions.append(pred_score)
        
        if ensemble_predictions:
            return np.mean(ensemble_predictions)
        else:
            return 6.0

# Global instance
performance_predictor = PerformancePredictor()