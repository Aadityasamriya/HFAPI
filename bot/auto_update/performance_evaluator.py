"""
Performance Evaluator
Scores and ranks models based on multiple criteria
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from .model_discovery import ModelCandidate

logger = logging.getLogger(__name__)

@dataclass
class ModelScore:
    """Represents a model's evaluation score"""
    model_id: str
    task: str
    total_score: float
    popularity_score: float
    performance_score: float
    recency_score: float
    compatibility_score: float
    breakdown: Dict[str, float]


class PerformanceEvaluator:
    """Evaluates and ranks models based on multiple criteria"""
    
    # Scoring weights
    WEIGHTS = {
        'popularity': 0.30,      # Downloads + likes
        'performance': 0.35,     # Reported metrics
        'recency': 0.15,         # Last modified date
        'compatibility': 0.20    # Free-tier support, inference endpoints
    }
    
    # Task-specific performance benchmarks (higher is better)
    BENCHMARK_METRICS = {
        'text-generation': ['mmlu', 'hellaswag', 'arc', 'truthfulqa'],
        'code-generation': ['humaneval', 'mbpp', 'pass@1'],
        'conversational': ['mt_bench', 'alpaca_eval'],
        'question-answering': ['squad', 'natural_questions', 'f1'],
        'summarization': ['rouge', 'bleu'],
        'translation': ['bleu', 'chrf']
    }
    
    def __init__(self):
        self.task_benchmarks: Dict[str, List[str]] = {}
        
    def evaluate_models(self, candidates: List[ModelCandidate]) -> List[ModelScore]:
        """
        Evaluate and score a list of model candidates
        
        Args:
            candidates: List of ModelCandidate objects
            
        Returns:
            List of ModelScore objects, sorted by total score
        """
        scores = []
        
        for candidate in candidates:
            try:
                score = self._calculate_score(candidate)
                scores.append(score)
            except Exception as e:
                logger.error(f"Error scoring {candidate.model_id}: {e}")
        
        # Sort by total score (descending)
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        return scores
    
    def _calculate_score(self, candidate: ModelCandidate) -> ModelScore:
        """Calculate comprehensive score for a model"""
        
        # 1. Popularity Score (downloads + likes)
        popularity_score = self._score_popularity(candidate)
        
        # 2. Performance Score (metrics)
        performance_score = self._score_performance(candidate)
        
        # 3. Recency Score (last modified)
        recency_score = self._score_recency(candidate)
        
        # 4. Compatibility Score (free-tier support)
        compatibility_score = self._score_compatibility(candidate)
        
        # Calculate weighted total
        total_score = (
            popularity_score * self.WEIGHTS['popularity'] +
            performance_score * self.WEIGHTS['performance'] +
            recency_score * self.WEIGHTS['recency'] +
            compatibility_score * self.WEIGHTS['compatibility']
        )
        
        breakdown = {
            'popularity': popularity_score,
            'performance': performance_score,
            'recency': recency_score,
            'compatibility': compatibility_score,
            'downloads': candidate.downloads,
            'likes': candidate.likes,
            'metrics_count': len(candidate.metrics)
        }
        
        return ModelScore(
            model_id=candidate.model_id,
            task=candidate.task,
            total_score=total_score,
            popularity_score=popularity_score,
            performance_score=performance_score,
            recency_score=recency_score,
            compatibility_score=compatibility_score,
            breakdown=breakdown
        )
    
    def _score_popularity(self, candidate: ModelCandidate) -> float:
        """Score based on downloads and likes (0-100)"""
        # Normalize downloads (log scale)
        import math
        
        downloads = max(candidate.downloads, 1)
        likes = max(candidate.likes, 0)
        
        # Log scale for downloads (handles wide range)
        download_score = min(math.log10(downloads) * 10, 70)
        
        # Linear scale for likes (bonus)
        like_score = min(likes / 10, 30)
        
        return download_score + like_score
    
    def _score_performance(self, candidate: ModelCandidate) -> float:
        """Score based on reported performance metrics (0-100)"""
        if not candidate.metrics:
            return 50.0  # Neutral score if no metrics
        
        # Check for task-specific benchmarks
        task_benchmarks = self.BENCHMARK_METRICS.get(candidate.task, [])
        
        relevant_scores = []
        for benchmark in task_benchmarks:
            for metric_key, value in candidate.metrics.items():
                if benchmark in metric_key.lower():
                    try:
                        # Normalize metric value (assume 0-100 or 0-1 scale)
                        if isinstance(value, (int, float)):
                            normalized = value if value <= 100 else value * 100
                            relevant_scores.append(normalized)
                    except:
                        pass
        
        if relevant_scores:
            return min(sum(relevant_scores) / len(relevant_scores), 100)
        
        # Fallback: Give bonus for having any metrics
        return 60.0
    
    def _score_recency(self, candidate: ModelCandidate) -> float:
        """Score based on last modified date (0-100)"""
        from datetime import datetime, timedelta
        
        try:
            # Parse last modified date
            if isinstance(candidate.last_modified, str):
                # Try parsing ISO format
                last_modified = datetime.fromisoformat(candidate.last_modified.replace('Z', '+00:00'))
            else:
                last_modified = candidate.last_modified
            
            # Calculate days since last update
            days_old = (datetime.now() - last_modified.replace(tzinfo=None)).days
            
            # Scoring: 100 for <30 days, decay to 0 at 365+ days
            if days_old <= 30:
                return 100
            elif days_old <= 90:
                return 80
            elif days_old <= 180:
                return 60
            elif days_old <= 365:
                return 40
            else:
                return 20
                
        except Exception as e:
            logger.debug(f"Could not parse date for {candidate.model_id}: {e}")
            return 50.0  # Neutral score
    
    def _score_compatibility(self, candidate: ModelCandidate) -> float:
        """Score based on free-tier and inference endpoint support (0-100)"""
        score = 0
        
        # Base score for free license
        if any(lic in candidate.license.lower() for lic in ['apache', 'mit', 'bsd', 'llama', 'gemma']):
            score += 40
        
        # Bonus for multiple inference endpoints
        endpoint_count = len(candidate.inference_endpoints)
        score += min(endpoint_count * 15, 60)  # Up to 60 points for endpoints
        
        return min(score, 100)
    
    def get_top_models_by_task(self, scores: List[ModelScore], task: str, top_n: int = 5) -> List[ModelScore]:
        """Get top N models for a specific task"""
        task_scores = [s for s in scores if s.task == task]
        return sorted(task_scores, key=lambda s: s.total_score, reverse=True)[:top_n]
    
    def generate_evaluation_report(self, scores: List[ModelScore]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if not scores:
            return {'error': 'No models scored'}
        
        # Group by task
        by_task = {}
        for score in scores:
            if score.task not in by_task:
                by_task[score.task] = []
            by_task[score.task].append(score)
        
        report = {
            'total_models_evaluated': len(scores),
            'tasks': {},
            'overall_top_5': [],
            'recommendations': []
        }
        
        # Top models per task
        for task, task_scores in by_task.items():
            sorted_scores = sorted(task_scores, key=lambda s: s.total_score, reverse=True)
            report['tasks'][task] = {
                'count': len(task_scores),
                'top_model': sorted_scores[0].model_id if sorted_scores else None,
                'top_score': sorted_scores[0].total_score if sorted_scores else 0,
                'top_3': [
                    {
                        'model': s.model_id,
                        'score': round(s.total_score, 2),
                        'breakdown': {k: round(v, 2) for k, v in s.breakdown.items() if isinstance(v, float)}
                    }
                    for s in sorted_scores[:3]
                ]
            }
        
        # Overall top 5
        overall_sorted = sorted(scores, key=lambda s: s.total_score, reverse=True)
        report['overall_top_5'] = [
            {
                'model': s.model_id,
                'task': s.task,
                'score': round(s.total_score, 2)
            }
            for s in overall_sorted[:5]
        ]
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(by_task)
        
        return report
    
    def _generate_recommendations(self, by_task: Dict[str, List[ModelScore]]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for task, scores in by_task.items():
            if not scores:
                continue
                
            top_model = max(scores, key=lambda s: s.total_score)
            
            if top_model.total_score > 80:
                recommendations.append(
                    f"✅ {task}: Use '{top_model.model_id}' (excellent score: {top_model.total_score:.1f})"
                )
            elif top_model.total_score > 60:
                recommendations.append(
                    f"⚠️ {task}: '{top_model.model_id}' recommended (good score: {top_model.total_score:.1f})"
                )
            else:
                recommendations.append(
                    f"⚠️ {task}: '{top_model.model_id}' available but consider alternatives (score: {top_model.total_score:.1f})"
                )
        
        return recommendations
