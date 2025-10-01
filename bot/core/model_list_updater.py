"""
Dynamic Model List Updater - 2025 Enhanced
Automatically fetches and validates the latest Hugging Face models
Ensures optimal model selection with real-time availability checking
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from ..config import Config
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

@dataclass
class ModelInfo:
    """Information about a model from Hugging Face"""
    model_id: str
    task: str
    library_name: str
    downloads: int
    likes: int
    created_at: str
    last_modified: str
    pipeline_tag: Optional[str] = None
    tags: List[str] = None
    is_verified: bool = False
    performance_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ModelListUpdater:
    """
    2025 ENHANCED: Intelligent Model List Updater
    Automatically discovers, validates, and updates available models for superior AI routing
    """
    
    def __init__(self, cache_duration_hours: int = 6):
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.last_update = None
        self.cached_models = {}
        self.verified_models = set()
        self.failed_models = set()
        self.session = None
        
        # Model categories for intelligent classification
        self.model_categories = {
            'text_generation': [
                'text-generation', 'text2text-generation', 'conversational',
                'fill-mask', 'text-classification'
            ],
            'code_generation': [
                'text-generation', 'text2text-generation'  # Code models use these tasks
            ],
            'image_generation': [
                'text-to-image', 'image-to-image', 'image-generation'
            ],
            'mathematical_reasoning': [
                'text-generation', 'text2text-generation', 'question-answering'
            ],
            'question_answering': [
                'question-answering', 'text-generation', 'conversational'
            ],
            'sentiment_analysis': [
                'text-classification', 'sentiment-analysis'
            ]
        }
        
        # Quality filters for model selection
        self.quality_thresholds = {
            'min_downloads': 1000,      # Minimum downloads for consideration
            'min_likes': 10,            # Minimum likes for quality
            'verified_bonus': 2.0,      # Bonus multiplier for verified models
            'recency_bonus': 1.5,       # Bonus for recently updated models
            'downloads_weight': 0.6,    # Weight for download count in scoring
            'likes_weight': 0.4         # Weight for likes in scoring
        }
        
        secure_logger.info("🔄 Dynamic Model List Updater initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Replit-AI-Router/2025'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _should_update_cache(self) -> bool:
        """Check if cache should be updated"""
        if self.last_update is None:
            return True
        return datetime.now() - self.last_update > self.cache_duration
    
    async def fetch_latest_models(self, 
                                 intent_type: str = "text_generation",
                                 limit: int = 50,
                                 force_update: bool = False) -> List[ModelInfo]:
        """
        Fetch latest models for a specific intent type from Hugging Face
        
        Args:
            intent_type (str): Type of models to fetch
            limit (int): Maximum number of models to fetch
            force_update (bool): Force cache update
            
        Returns:
            List[ModelInfo]: List of model information
        """
        cache_key = f"{intent_type}_{limit}"
        
        # Check cache first
        if not force_update and not self._should_update_cache():
            cached = self.cached_models.get(cache_key)
            if cached:
                secure_logger.info(f"🚀 Using cached models for {intent_type}")
                return cached
        
        try:
            models = await self._fetch_models_from_hf(intent_type, limit)
            self.cached_models[cache_key] = models
            self.last_update = datetime.now()
            
            secure_logger.info(f"✅ Fetched {len(models)} models for {intent_type}")
            return models
            
        except Exception as e:
            secure_logger.error(f"❌ Failed to fetch models: {redact_sensitive_data(str(e))}")
            # Return cached models if available
            return self.cached_models.get(cache_key, [])
    
    async def _fetch_models_from_hf(self, intent_type: str, limit: int) -> List[ModelInfo]:
        """Fetch models from Hugging Face API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        models = []
        tasks = self.model_categories.get(intent_type, ['text-generation'])
        
        for task in tasks:
            try:
                # Build API request
                params = {
                    'pipeline_tag': task,
                    'sort': 'downloads',
                    'direction': -1,
                    'limit': limit // len(tasks),  # Distribute limit across tasks
                    'full': True
                }
                
                url = "https://huggingface.co/api/models"
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for model_data in data:
                            model_info = self._parse_model_data(model_data, intent_type)
                            if model_info and self._passes_quality_filters(model_info):
                                models.append(model_info)
                    else:
                        secure_logger.warning(f"⚠️ HF API returned status {response.status} for task {task}")
                        
            except Exception as e:
                secure_logger.warning(f"⚠️ Failed to fetch models for task {task}: {str(e)}")
                continue
        
        # Sort by performance score and remove duplicates
        unique_models = {m.model_id: m for m in models}.values()
        sorted_models = sorted(unique_models, key=lambda x: x.performance_score, reverse=True)
        
        return sorted_models[:limit]
    
    def _parse_model_data(self, model_data: Dict, intent_type: str) -> Optional[ModelInfo]:
        """Parse model data from HF API response"""
        try:
            model_id = model_data.get('id', '').strip()
            if not model_id:
                return None
            
            # Skip private or gated models
            if model_data.get('private', False) or model_data.get('gated', False):
                return None
            
            # Extract model information
            downloads = model_data.get('downloads', 0)
            likes = model_data.get('likes', 0)
            tags = model_data.get('tags', [])
            pipeline_tag = model_data.get('pipeline_tag')
            library_name = model_data.get('library_name', 'transformers')
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(
                downloads, likes, tags, model_data.get('created_at', ''),
                model_data.get('last_modified', '')
            )
            
            return ModelInfo(
                model_id=model_id,
                task=pipeline_tag or 'text-generation',
                library_name=library_name,
                downloads=downloads,
                likes=likes,
                created_at=model_data.get('created_at', ''),
                last_modified=model_data.get('last_modified', ''),
                pipeline_tag=pipeline_tag,
                tags=tags,
                is_verified='verified' in tags,
                performance_score=performance_score
            )
            
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to parse model data: {str(e)}")
            return None
    
    def _calculate_performance_score(self, downloads: int, likes: int, tags: List[str],
                                   created_at: str, last_modified: str) -> float:
        """Calculate performance score for model ranking"""
        score = 0.0
        
        # Base score from downloads and likes
        download_score = downloads * self.quality_thresholds['downloads_weight']
        likes_score = likes * self.quality_thresholds['likes_weight']
        score = download_score + likes_score
        
        # Verified model bonus
        if 'verified' in tags:
            score *= self.quality_thresholds['verified_bonus']
        
        # Recency bonus for recently updated models
        try:
            if last_modified:
                last_mod_date = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                days_old = (datetime.now() - last_mod_date.replace(tzinfo=None)).days
                if days_old < 30:  # Recently updated
                    score *= self.quality_thresholds['recency_bonus']
        except:
            pass  # Ignore date parsing errors
        
        # Normalize score (log scale for better distribution)
        import math
        return math.log10(max(score, 1)) * 10
    
    def _passes_quality_filters(self, model_info: ModelInfo) -> bool:
        """Check if model passes quality filters"""
        return (
            model_info.downloads >= self.quality_thresholds['min_downloads'] and
            model_info.likes >= self.quality_thresholds['min_likes'] and
            model_info.model_id not in self.failed_models
        )
    
    async def validate_model_availability(self, model_id: str) -> bool:
        """
        Validate that a model is available and working
        
        Args:
            model_id (str): Model ID to validate
            
        Returns:
            bool: True if model is available and working
        """
        if model_id in self.verified_models:
            return True
        
        if model_id in self.failed_models:
            return False
        
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
            
            # Test model availability with a simple inference request
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {}
            
            # Add API key if available
            hf_token = Config.get_hf_token()
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'
            
            # Simple test payload
            payload = {"inputs": "test"}
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    self.verified_models.add(model_id)
                    secure_logger.debug(f"✅ Model {model_id} verified as available")
                    return True
                elif response.status == 503:
                    # Model loading, consider it available but not ready
                    secure_logger.debug(f"🔄 Model {model_id} is loading")
                    return True
                else:
                    self.failed_models.add(model_id)
                    secure_logger.debug(f"❌ Model {model_id} failed validation: {response.status}")
                    return False
                    
        except Exception as e:
            secure_logger.debug(f"❌ Model {model_id} validation error: {str(e)}")
            self.failed_models.add(model_id)
            return False
    
    async def get_recommended_models(self, intent_type: str, count: int = 5) -> List[str]:
        """
        Get recommended models for a specific intent type
        
        Args:
            intent_type (str): Intent type to get models for
            count (int): Number of models to return
            
        Returns:
            List[str]: List of recommended model IDs
        """
        models = await self.fetch_latest_models(intent_type, limit=count * 3)
        
        # Validate models and filter to working ones
        validated_models = []
        
        for model_info in models:
            if await self.validate_model_availability(model_info.model_id):
                validated_models.append(model_info.model_id)
                if len(validated_models) >= count:
                    break
        
        secure_logger.info(f"🎯 Recommended {len(validated_models)} models for {intent_type}")
        return validated_models
    
    async def update_config_models(self, backup_existing: bool = True) -> Dict[str, List[str]]:
        """
        Update config with latest validated models
        
        Args:
            backup_existing (bool): Whether to backup existing models
            
        Returns:
            Dict[str, List[str]]: Updated model mappings
        """
        secure_logger.info("🔄 Starting dynamic model config update...")
        
        try:
            updated_models = {}
            
            # Get recommendations for each intent type
            for intent_type in self.model_categories.keys():
                try:
                    recommended = await self.get_recommended_models(intent_type, count=3)
                    if recommended:
                        updated_models[intent_type] = recommended
                        secure_logger.info(f"✅ Updated {intent_type}: {recommended[:2]}...")
                    else:
                        secure_logger.warning(f"⚠️ No valid models found for {intent_type}")
                except Exception as e:
                    secure_logger.error(f"❌ Failed to update {intent_type}: {str(e)}")
            
            # Save to cache for future use
            self.cached_models['_config_update'] = {
                'timestamp': datetime.now().isoformat(),
                'models': updated_models
            }
            
            secure_logger.info(f"🎉 Dynamic model update completed: {len(updated_models)} categories updated")
            return updated_models
            
        except Exception as e:
            secure_logger.error(f"❌ Model config update failed: {redact_sensitive_data(str(e))}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cached_categories': list(self.cached_models.keys()),
            'verified_models_count': len(self.verified_models),
            'failed_models_count': len(self.failed_models),
            'cache_size_mb': sum(len(str(v)) for v in self.cached_models.values()) / (1024 * 1024)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.cached_models.clear()
        self.verified_models.clear()
        self.failed_models.clear()
        self.last_update = None
        secure_logger.info("🗑️ Model list cache cleared")

# Global updater instance
model_list_updater = ModelListUpdater()