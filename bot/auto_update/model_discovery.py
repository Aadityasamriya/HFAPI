"""
Model Discovery Engine
Automatically discovers and evaluates the best free AI models from Hugging Face Hub
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from huggingface_hub import HfApi, ModelFilter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelCandidate:
    """Represents a discovered model candidate"""
    model_id: str
    task: str
    downloads: int
    likes: int
    last_modified: str
    library: str
    performance_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    license: str = "unknown"
    context_length: Optional[int] = None
    supports_free_tier: bool = True
    inference_endpoints: List[str] = field(default_factory=list)


class ModelDiscoveryEngine:
    """Discovers best free AI models across different tasks"""
    
    # Task categories to search
    TASK_CATEGORIES = [
        'text-generation',
        'text2text-generation',
        'code-generation',
        'conversational',
        'question-answering',
        'summarization',
        'translation',
        'fill-mask',
        'feature-extraction'
    ]
    
    # Preferred model families (known for free tier support)
    PREFERRED_FAMILIES = [
        'qwen', 'deepseek', 'mistral', 'llama', 'phi', 
        'gemma', 'falcon', 'starcoder', 'codellama',
        'mixtral', 'yi', 'solar', 'zephyr'
    ]
    
    # Free-tier friendly licenses
    FREE_LICENSES = [
        'apache-2.0', 'mit', 'bsd', 'cc-by-4.0', 
        'cc-by-sa-4.0', 'openrail', 'llama3', 'gemma'
    ]
    
    def __init__(self):
        self.api = HfApi()
        self.discovered_models: Dict[str, List[ModelCandidate]] = {}
        
    async def discover_models(self, task: str = None, limit: int = 50) -> List[ModelCandidate]:
        """
        Discover top models for a specific task or all tasks
        
        Args:
            task: Specific task to search for (None = all tasks)
            limit: Maximum models to fetch per task
            
        Returns:
            List of ModelCandidate objects
        """
        tasks_to_search = [task] if task else self.TASK_CATEGORIES
        all_candidates = []
        
        for search_task in tasks_to_search:
            logger.info(f"🔍 Discovering models for task: {search_task}")
            
            try:
                # Search with filters (offload to thread to avoid blocking)
                model_filter = ModelFilter(task=search_task)
                
                # Run blocking HF API call in background thread
                models = await asyncio.to_thread(
                    self.api.list_models,
                    filter=model_filter,
                    cardData=True,
                    sort="downloads",
                    direction=-1,
                    limit=limit
                )
                
                task_candidates = []
                for model in models:
                    # Extract model info
                    candidate = self._create_candidate(model, search_task)
                    
                    # Filter for free-tier compatibility
                    if self._is_free_tier_compatible(candidate):
                        task_candidates.append(candidate)
                
                logger.info(f"✅ Found {len(task_candidates)} free-tier models for {search_task}")
                all_candidates.extend(task_candidates)
                
                # Store by task
                self.discovered_models[search_task] = task_candidates
                
            except Exception as e:
                logger.error(f"❌ Error discovering models for {search_task}: {e}")
                
        return all_candidates
    
    def _create_candidate(self, model, task: str) -> ModelCandidate:
        """Create ModelCandidate from HF model object"""
        try:
            # Extract basic info
            model_id = model.modelId if hasattr(model, 'modelId') else model.id
            downloads = getattr(model, 'downloads', 0) or 0
            likes = getattr(model, 'likes', 0) or 0
            last_modified = getattr(model, 'lastModified', str(datetime.now()))
            library = getattr(model, 'library_name', 'unknown')
            
            # Extract license
            license_info = "unknown"
            if hasattr(model, 'card_data') and model.card_data:
                license_info = model.card_data.get('license', 'unknown')
            
            # Extract metrics from model card
            metrics = self._extract_metrics(model)
            
            # Detect inference endpoints
            endpoints = self._detect_inference_support(model_id)
            
            return ModelCandidate(
                model_id=model_id,
                task=task,
                downloads=downloads,
                likes=likes,
                last_modified=str(last_modified),
                library=library,
                metrics=metrics,
                license=license_info,
                inference_endpoints=endpoints
            )
            
        except Exception as e:
            logger.warning(f"Error creating candidate for {model}: {e}")
            return None
    
    def _extract_metrics(self, model) -> Dict[str, Any]:
        """Extract performance metrics from model card"""
        metrics = {}
        
        try:
            if hasattr(model, 'card_data') and model.card_data:
                # Check for model-index (standardized metrics)
                if 'model-index' in model.card_data:
                    for entry in model.card_data['model-index']:
                        if 'results' in entry:
                            for result in entry['results']:
                                dataset = result.get('dataset', {}).get('name', 'unknown')
                                for metric in result.get('metrics', []):
                                    metric_name = metric.get('name') or metric.get('type')
                                    metric_value = metric.get('value')
                                    if metric_name and metric_value is not None:
                                        metrics[f"{dataset}_{metric_name}"] = metric_value
                
                # Extract from base_model or other fields
                if 'base_model' in model.card_data:
                    metrics['base_model'] = model.card_data['base_model']
                    
        except Exception as e:
            logger.debug(f"Could not extract metrics: {e}")
            
        return metrics
    
    def _detect_inference_support(self, model_id: str) -> List[str]:
        """Detect which inference endpoints support this model"""
        endpoints = ['huggingface']  # Always available via HF
        
        # Check model family for known free-tier providers
        model_lower = model_id.lower()
        
        if any(fam in model_lower for fam in ['qwen', 'deepseek', 'llama', 'mistral']):
            endpoints.extend(['groq', 'together-ai', 'openrouter'])
        
        if 'gemma' in model_lower or 'gemini' in model_lower:
            endpoints.append('google-ai-studio')
            
        if 'phi' in model_lower:
            endpoints.append('azure-ai')
            
        return endpoints
    
    def _is_free_tier_compatible(self, candidate: ModelCandidate) -> bool:
        """Check if model is compatible with free tier usage"""
        if not candidate:
            return False
            
        # Check license
        license_ok = any(
            lic in candidate.license.lower() 
            for lic in self.FREE_LICENSES
        ) or candidate.license == "unknown"
        
        # Check model family
        family_ok = any(
            fam in candidate.model_id.lower() 
            for fam in self.PREFERRED_FAMILIES
        )
        
        # Prioritize popular models (likely to work on free tier)
        popularity_ok = candidate.downloads > 100 or candidate.likes > 5
        
        return license_ok and (family_ok or popularity_ok)
    
    async def get_top_models_by_task(self, task: str, top_n: int = 5) -> List[ModelCandidate]:
        """Get top N models for a specific task"""
        if task not in self.discovered_models:
            await self.discover_models(task=task)
            
        models = self.discovered_models.get(task, [])
        
        # Sort by downloads (popularity proxy for reliability)
        sorted_models = sorted(
            models, 
            key=lambda m: (m.downloads, m.likes), 
            reverse=True
        )
        
        return sorted_models[:top_n]
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovered models"""
        summary = {
            'total_models': sum(len(models) for models in self.discovered_models.values()),
            'tasks_covered': len(self.discovered_models),
            'by_task': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for task, models in self.discovered_models.items():
            summary['by_task'][task] = {
                'count': len(models),
                'top_model': models[0].model_id if models else None,
                'top_downloads': models[0].downloads if models else 0
            }
            
        return summary
