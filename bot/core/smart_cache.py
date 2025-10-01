"""
Advanced Smart Caching System for Superior AI Performance
Provides intelligent caching that outperforms ChatGPT, Grok, and Gemini
Features context-aware caching, quality-based retention, and adaptive expiration
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import OrderedDict
from enum import Enum
import asyncio
import threading

logger = logging.getLogger(__name__)

class CacheQuality(Enum):
    """Cache entry quality levels"""
    PREMIUM = "premium"      # High-quality responses, long retention
    STANDARD = "standard"    # Good responses, normal retention  
    BASIC = "basic"         # Acceptable responses, short retention
    POOR = "poor"           # Low-quality responses, very short retention

@dataclass
class CacheEntry:
    """Advanced cache entry with quality and context tracking"""
    key: str
    content: Any
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    quality_score: float       # 0-10 quality rating
    quality_level: CacheQuality
    user_context: Dict
    intent_type: str
    model_used: str
    response_time: float
    complexity_score: float
    context_hash: str         # Hash of user context for matching
    expiry_time: datetime
    success_rate: float       # How often this cached response was useful
    user_satisfaction: List[float]  # User satisfaction scores
    semantic_tags: List[str]  # Semantic tags for advanced retrieval
    
class SmartCache:
    """
    Intelligent caching system with superior performance optimization
    Features adaptive expiration, quality-based retention, and context awareness
    """
    
    def __init__(self, max_size: int = 10000, cleanup_interval: int = 3600):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval  # seconds
        self.hit_count = 0
        self.miss_count = 0
        self._cache_lock = asyncio.Lock()  # Thread safety for cache operations
        self._cleanup_task = None  # Store reference to cleanup task
        self.quality_thresholds = {
            CacheQuality.PREMIUM: 8.5,   # Premium: 8.5+ score
            CacheQuality.STANDARD: 6.5,  # Standard: 6.5-8.4 score  
            CacheQuality.BASIC: 4.0,     # Basic: 4.0-6.4 score
            CacheQuality.POOR: 0.0       # Poor: <4.0 score
        }
        
        # Expiration times by quality (in seconds)
        self.expiry_durations = {
            CacheQuality.PREMIUM: 7 * 24 * 3600,   # 7 days for premium
            CacheQuality.STANDARD: 3 * 24 * 3600,  # 3 days for standard
            CacheQuality.BASIC: 24 * 3600,         # 1 day for basic
            CacheQuality.POOR: 2 * 3600            # 2 hours for poor
        }
        
        # Context similarity thresholds
        self.context_similarity_threshold = 0.8
        self.semantic_similarity_threshold = 0.75
        
        # Performance tracking
        self.performance_stats = {
            'total_hits': 0,
            'total_misses': 0,
            'quality_distribution': {q.value: 0 for q in CacheQuality},
            'avg_response_time_saved': 0.0,
            'total_time_saved': 0.0,
            'cache_efficiency': 0.0
        }
        
        logger.info(f"🧠 SmartCache initialized: max_size={max_size}, cleanup_interval={cleanup_interval}s")
    
    def start_cleanup_task(self):
        """Start the async cleanup task (call this from an async context)"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._async_periodic_cleanup())
            logger.info("🧹 Started async cache cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop the cleanup task gracefully"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("🛑 Stopped cache cleanup task")
    
    def _generate_cache_key(self, prompt: str, intent_type: str, user_context: Dict, 
                           model_preferences: Optional[Dict] = None) -> str:
        """Generate intelligent cache key with context consideration"""
        # Create base key from prompt
        prompt_normalized = self._normalize_prompt(prompt)
        
        # Include relevant context elements
        context_elements = {
            'intent_type': intent_type,
            'user_domain': user_context.get('domain_context', 'general'),
            'complexity_level': user_context.get('complexity_level', 'medium'),
            'language_preference': user_context.get('language', 'en'),
            'technical_level': user_context.get('technical_level', 'general')
        }
        
        # Add model preferences if provided
        if model_preferences:
            context_elements['model_prefs'] = json.dumps(model_preferences, sort_keys=True)
        
        # Create comprehensive hash
        key_data = {
            'prompt': prompt_normalized,
            'context': context_elements
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for better cache key generation"""
        # Convert to lowercase and remove extra whitespace
        normalized = ' '.join(prompt.lower().strip().split())
        
        # Remove common variations that shouldn't affect caching
        replacements = [
            ('please ', ''),
            ('can you ', ''),
            ('could you ', ''),
            ('would you ', ''),
            ('help me ', ''),
            ('i need ', ''),
            ('i want ', ''),
            ('i would like ', ''),
        ]
        
        for old, new in replacements:
            normalized = normalized.replace(old, new)
        
        return normalized.strip()
    
    def _calculate_context_similarity(self, ctx1: Optional[Dict], ctx2: Optional[Dict]) -> float:
        """Calculate similarity between user contexts"""
        if not ctx1 or not ctx2:
            return 0.0
        
        # Key context elements to compare
        key_elements = ['domain_context', 'technical_level', 'language', 'complexity_level']
        
        matches = 0
        total = 0
        
        for element in key_elements:
            if element in ctx1 and element in ctx2:
                total += 1
                if ctx1[element] == ctx2[element]:
                    matches += 1
            elif element not in ctx1 and element not in ctx2:
                matches += 1  # Both missing counts as match
                total += 1
        
        return matches / max(total, 1)
    
    def _determine_quality_level(self, quality_score: float) -> CacheQuality:
        """Determine cache quality level from quality score"""
        if quality_score >= self.quality_thresholds[CacheQuality.PREMIUM]:
            return CacheQuality.PREMIUM
        elif quality_score >= self.quality_thresholds[CacheQuality.STANDARD]:
            return CacheQuality.STANDARD
        elif quality_score >= self.quality_thresholds[CacheQuality.BASIC]:
            return CacheQuality.BASIC
        else:
            return CacheQuality.POOR
    
    def _extract_semantic_tags(self, prompt: str, intent_type: str, content: Any) -> List[str]:
        """Extract semantic tags for advanced retrieval"""
        tags = []
        
        # Add intent-based tags
        tags.append(f"intent:{intent_type}")
        
        # Extract keywords from prompt
        prompt_lower = prompt.lower()
        
        # Technical keywords
        tech_keywords = ['python', 'javascript', 'react', 'django', 'api', 'database', 
                        'algorithm', 'function', 'class', 'code', 'programming',
                        'machine learning', 'ai', 'data', 'analysis', 'model']
        
        for keyword in tech_keywords:
            if keyword in prompt_lower:
                tags.append(f"tech:{keyword}")
        
        # Domain keywords
        domains = ['business', 'science', 'education', 'creative', 'technical', 'analysis']
        for domain in domains:
            if domain in prompt_lower:
                tags.append(f"domain:{domain}")
        
        # Content-based tags
        if isinstance(content, str):
            content_lower = content.lower()
            if len(content) > 500:
                tags.append("long_response")
            if '```' in content:
                tags.append("contains_code")
            if any(word in content_lower for word in ['example', 'demonstration', 'tutorial']):
                tags.append("educational")
        
        return tags[:10]  # Limit to 10 tags
    
    async def get(self, prompt: str, intent_type: str, user_context: Optional[Dict] = None,
                 model_preferences: Optional[Dict] = None) -> Optional[CacheEntry]:
        """
        Intelligent cache retrieval with context and quality consideration
        
        Args:
            prompt (str): User prompt to look up
            intent_type (str): Intent type for the prompt
            user_context (Dict): User context for similarity matching
            model_preferences (Dict): Model preferences for matching
            
        Returns:
            Optional[CacheEntry]: Cache entry if found and suitable
        """
        user_context = user_context or {}
        
        # Generate primary cache key
        primary_key = self._generate_cache_key(prompt, intent_type, user_context, model_preferences)
        
        async with self._cache_lock:
            # Direct hit check
            if primary_key in self.cache:
                entry = self.cache[primary_key]
                
                # Check if entry is still valid
                if datetime.now() < entry.expiry_time:
                    # Update access statistics
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    
                    # Move to end (LRU)
                    self.cache.move_to_end(primary_key)
                    
                    self.hit_count += 1
                    self.performance_stats['total_hits'] += 1
                    
                    logger.info(f"🎯 CACHE_HIT: Direct hit for {intent_type} (quality: {entry.quality_level.value}, "
                               f"score: {entry.quality_score:.1f}, uses: {entry.access_count})")
                    return entry
                else:
                    # Entry expired, remove it
                    del self.cache[primary_key]
                    logger.info(f"🗑️ CACHE_EXPIRED: Removed expired entry for {intent_type}")
        
        # Semantic/context similarity search for near-misses (also needs lock)
        similar_entry = await self._find_similar_entry(prompt, intent_type, user_context)
        if similar_entry:
            async with self._cache_lock:
                self.hit_count += 1
                self.performance_stats['total_hits'] += 1
            logger.info(f"🎯 CACHE_SIMILAR: Context-similar hit for {intent_type} "
                       f"(quality: {similar_entry.quality_level.value})")
            return similar_entry
        
        # No suitable cache entry found
        async with self._cache_lock:
            self.miss_count += 1
            self.performance_stats['total_misses'] += 1
        logger.info(f"❌ CACHE_MISS: No suitable entry for {intent_type}")
        return None
    
    async def _find_similar_entry(self, prompt: str, intent_type: str, 
                                user_context: Dict) -> Optional[CacheEntry]:
        """Find semantically or contextually similar cache entries"""
        prompt_normalized = self._normalize_prompt(prompt)
        best_entry = None
        best_similarity = 0.0
        
        # Search through recent, high-quality entries
        current_time = datetime.now()
        
        async with self._cache_lock:
            for entry in reversed(list(self.cache.values())):  # Search recent first
                # Skip if expired or wrong intent
                if (current_time >= entry.expiry_time or 
                    entry.intent_type != intent_type or
                    entry.quality_level == CacheQuality.POOR):
                    continue
                
                # Calculate prompt similarity (simple token overlap)
                prompt_similarity = self._calculate_prompt_similarity(prompt_normalized, entry.key)
                
                # Calculate context similarity
                context_similarity = self._calculate_context_similarity(user_context, entry.user_context)
                
                # Combined similarity score
                combined_similarity = (prompt_similarity * 0.7 + context_similarity * 0.3)
                
                # Check if this is the best match so far
                if (combined_similarity > best_similarity and 
                    combined_similarity >= self.semantic_similarity_threshold):
                    best_similarity = combined_similarity
                    best_entry = entry
            
            # Update access stats if found
            if best_entry:
                best_entry.access_count += 1
                best_entry.last_accessed = datetime.now()
                # Move to end (LRU)
                cache_key = best_entry.key
                if cache_key in self.cache:
                    self.cache.move_to_end(cache_key)
        
        return best_entry
    
    def _calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts using token overlap"""
        tokens1 = set(prompt1.split())
        tokens2 = set(prompt2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def store(self, prompt: str, intent_type: str, content: Any, 
                   user_context: Dict, model_used: str, response_time: float,
                   quality_score: float, complexity_score: float = 5.0,
                   model_preferences: Optional[Dict] = None) -> bool:
        """
        Store response in intelligent cache with quality-based retention
        
        Args:
            prompt (str): Original user prompt
            intent_type (str): Intent type detected
            content (Any): Response content to cache
            user_context (Dict): User context data
            model_used (str): Model that generated the response
            response_time (float): Response generation time
            quality_score (float): Quality score (0-10)
            complexity_score (float): Prompt complexity score
            model_preferences (Dict): Model preferences
            
        Returns:
            bool: Whether the content was successfully stored
        """
        user_context = user_context or {}
        
        # Don't cache very poor quality responses
        if quality_score < 2.0:
            logger.info(f"🚫 CACHE_SKIP: Quality too low ({quality_score:.1f}) for {intent_type}")
            return False
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, intent_type, user_context, model_preferences)
        
        # Determine quality level and expiration
        quality_level = self._determine_quality_level(quality_score)
        expiry_duration = self.expiry_durations[quality_level]
        expiry_time = datetime.now() + timedelta(seconds=expiry_duration)
        
        # Extract semantic tags
        semantic_tags = self._extract_semantic_tags(prompt, intent_type, content)
        
        # Create context hash
        context_hash = hashlib.md5(json.dumps(user_context, sort_keys=True).encode()).hexdigest()
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            content=content,
            timestamp=datetime.now(),
            access_count=1,
            last_accessed=datetime.now(),
            quality_score=quality_score,
            quality_level=quality_level,
            user_context=user_context,
            intent_type=intent_type,
            model_used=model_used,
            response_time=response_time,
            complexity_score=complexity_score,
            context_hash=context_hash,
            expiry_time=expiry_time,
            success_rate=1.0,  # Initialize as successful
            user_satisfaction=[quality_score / 10.0],  # Convert to 0-1 scale
            semantic_tags=semantic_tags
        )
        
        async with self._cache_lock:
            # Store in cache
            self.cache[cache_key] = entry
            
            # Update performance stats
            self.performance_stats['quality_distribution'][quality_level.value] += 1
            
            # Ensure cache doesn't exceed max size
            await self._manage_cache_size()
        
        logger.info(f"💾 CACHE_STORE: Stored {intent_type} response "
                   f"(quality: {quality_level.value}, score: {quality_score:.1f}, "
                   f"expires: {expiry_duration/3600:.1f}h)")
        
        return True
    
    async def _manage_cache_size(self):
        """Manage cache size by removing least valuable entries"""
        while len(self.cache) > self.max_size:
            # Find entry to remove (lowest value score)
            min_value_score = float('inf')
            key_to_remove = None
            
            for key, entry in self.cache.items():
                # Calculate value score based on multiple factors
                age_penalty = (datetime.now() - entry.timestamp).total_seconds() / 3600  # hours
                access_bonus = entry.access_count * 0.5
                quality_bonus = entry.quality_score * 0.3
                
                value_score = quality_bonus + access_bonus - (age_penalty * 0.1)
                
                if value_score < min_value_score:
                    min_value_score = value_score
                    key_to_remove = key
            
            if key_to_remove:
                removed_entry = self.cache.pop(key_to_remove)
                logger.info(f"🗑️ CACHE_EVICT: Removed {removed_entry.intent_type} entry "
                           f"(value_score: {min_value_score:.2f})")
    
    async def _async_periodic_cleanup(self):
        """Async periodic cleanup of expired entries"""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = []
                
                async with self._cache_lock:
                    for key, entry in list(self.cache.items()):
                        if current_time >= entry.expiry_time:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        entry = self.cache.pop(key)
                        logger.info(f"🗑️ CACHE_CLEANUP: Removed expired {entry.intent_type} entry")
                    
                    if expired_keys:
                        logger.info(f"🧹 CACHE_CLEANUP: Removed {len(expired_keys)} expired entries")
                    
                    # Update performance stats
                    self._update_performance_stats()
                
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                logger.info("🛑 Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        total_requests = self.hit_count + self.miss_count
        if total_requests > 0:
            hit_rate = self.hit_count / total_requests
            self.performance_stats['cache_efficiency'] = hit_rate
            
            # Estimate time saved (assuming average response time of 3 seconds)
            avg_response_time = 3.0
            time_saved = self.hit_count * avg_response_time * 0.9  # 90% time save on cache hit
            self.performance_stats['total_time_saved'] = time_saved
            self.performance_stats['avg_response_time_saved'] = time_saved / max(self.hit_count, 1)
    
    async def update_user_satisfaction(self, cache_key: str, satisfaction_score: float):
        """Update user satisfaction score for a cached response"""
        async with self._cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.user_satisfaction.append(satisfaction_score)
                
                # Keep only recent satisfaction scores
                if len(entry.user_satisfaction) > 10:
                    entry.user_satisfaction.pop(0)
                
                # Update success rate based on satisfaction
                avg_satisfaction = sum(entry.user_satisfaction) / len(entry.user_satisfaction)
                entry.success_rate = avg_satisfaction
                
                logger.info(f"📊 CACHE_FEEDBACK: Updated satisfaction for {entry.intent_type} "
                           f"(score: {satisfaction_score:.2f}, avg: {avg_satisfaction:.2f})")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        async with self._cache_lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / max(total_requests, 1)
            
            # Quality distribution
            quality_counts = {}
            total_entries = len(self.cache)
            
            for entry in self.cache.values():
                quality = entry.quality_level.value
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Top performers
            top_entries = sorted(
                self.cache.values(), 
                key=lambda e: (e.access_count * e.quality_score), 
                reverse=True
            )[:5]
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'efficiency_score': hit_rate * 100,
                'quality_distribution': {
                    k: {'count': v, 'percentage': (v/total_entries)*100 if total_entries > 0 else 0}
                    for k, v in quality_counts.items()
                },
                'performance_stats': self.performance_stats,
                'top_cached_intents': [e.intent_type for e in top_entries],
                'avg_quality_score': sum(e.quality_score for e in self.cache.values()) / max(len(self.cache), 1),
                'total_time_saved_hours': self.performance_stats['total_time_saved'] / 3600,
                'system_health': 'excellent' if hit_rate > 0.7 else 'good' if hit_rate > 0.5 else 'needs_improvement'
            }
    
    async def clear_low_quality(self, min_quality_score: float = 4.0):
        """Clear all cache entries below a quality threshold"""
        async with self._cache_lock:
            keys_to_remove = []
            
            for key, entry in self.cache.items():
                if entry.quality_score < min_quality_score:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                removed_entry = self.cache.pop(key)
                logger.info(f"🗑️ CACHE_PURGE: Removed low-quality {removed_entry.intent_type} entry "
                           f"(score: {removed_entry.quality_score:.1f})")
            
            logger.info(f"🧹 CACHE_PURGE: Removed {len(keys_to_remove)} low-quality entries")
    
    def is_healthy(self) -> bool:
        """
        Check if the SmartCache system is healthy and functioning properly
        
        Returns:
            bool: True if cache is healthy, False otherwise
        """
        try:
            # Check 1: Basic structure integrity
            if not isinstance(self.cache, OrderedDict):
                logger.warning("🩺 CACHE_HEALTH: Cache structure is not OrderedDict")
                return False
            
            # Check 2: Cache size within reasonable limits
            if len(self.cache) > self.max_size * 1.1:  # Allow 10% overflow for safety
                logger.warning(f"🩺 CACHE_HEALTH: Cache size ({len(self.cache)}) exceeds limit ({self.max_size})")
                return False
            
            # Check 3: Lock is properly initialized
            if not hasattr(self, '_cache_lock') or self._cache_lock is None:
                logger.warning("🩺 CACHE_HEALTH: Cache lock not properly initialized")
                return False
            
            # Check 4: Performance tracking is functional
            if not hasattr(self, 'performance_stats') or not isinstance(self.performance_stats, dict):
                logger.warning("🩺 CACHE_HEALTH: Performance stats not properly initialized")
                return False
            
            # Check 5: Basic performance metrics are reasonable
            total_requests = self.hit_count + self.miss_count
            if total_requests > 0:
                hit_rate = self.hit_count / total_requests
                # Very low hit rates might indicate cache issues, but not necessarily unhealthy
                if hit_rate < 0.0:  # Only check for impossible values
                    logger.warning(f"🩺 CACHE_HEALTH: Impossible hit rate: {hit_rate}")
                    return False
            
            # Check 6: Cleanup task status (if started)
            if self._cleanup_task is not None and self._cleanup_task.done() and not self._cleanup_task.cancelled():
                # If cleanup task completed unexpectedly, check for exceptions
                try:
                    exception = self._cleanup_task.exception()
                    if exception:
                        logger.warning(f"🩺 CACHE_HEALTH: Cleanup task failed: {exception}")
                        return False
                except Exception:
                    # Task might not be done yet, that's fine
                    pass
            
            # All checks passed
            logger.debug("🩺 CACHE_HEALTH: All health checks passed - cache is healthy")
            return True
            
        except Exception as e:
            logger.error(f"🩺 CACHE_HEALTH: Health check failed with exception: {e}")
            return False

# Global smart cache instance
smart_cache = SmartCache()