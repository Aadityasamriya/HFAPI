"""
Enhanced Integration Layer for Superior AI Model Selection
Provides seamless integration between traditional and enhanced model selection systems
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from .bot_types import IntentType, PromptComplexity
from .dynamic_model_selector import dynamic_model_selector, ModelSelectionRequest, ModelSelectionResponse
from .router import router as intelligent_router, AdvancedComplexityAnalyzer
from .model_selection_explainer import model_selection_explainer
from .conversation_context_tracker import conversation_context_tracker
from ..config import Config
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

class EnhancedModelSelectionIntegration:
    """
    Integration layer for enhanced model selection with backward compatibility
    """
    
    def __init__(self):
        self.enhanced_enabled = getattr(Config, 'ENHANCED_MODEL_SELECTION', True)
        self.fallback_to_traditional = getattr(Config, 'FALLBACK_TO_TRADITIONAL', True)
        self.explanation_enabled = getattr(Config, 'MODEL_SELECTION_EXPLANATIONS', True)
        self.conversation_tracking_enabled = getattr(Config, 'CONVERSATION_TRACKING', True)
        
        # Performance tracking
        self.integration_metrics = {
            'enhanced_selections': 0,
            'traditional_fallbacks': 0,
            'selection_errors': 0,
            'avg_selection_time': 0.0
        }
        
        secure_logger.info(f"🔗 Enhanced Model Selection Integration initialized "
                          f"(enhanced: {self.enhanced_enabled}, "
                          f"explanations: {self.explanation_enabled})")
    
    async def select_model_for_request(self, 
                                     prompt: str,
                                     intent_type: str,
                                     user_id: Optional[str] = None,
                                     conversation_id: Optional[str] = None,
                                     request_id: Optional[str] = None,
                                     enable_fallback: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Primary model selection method with intelligent routing
        
        Args:
            prompt: User prompt
            intent_type: Classified intent type
            user_id: User identifier
            conversation_id: Conversation identifier
            request_id: Request identifier
            enable_fallback: Whether to enable fallback strategies
            
        Returns:
            Tuple of (selected_model, selection_metadata)
        """
        selection_start = time.time()
        
        try:
            if self.enhanced_enabled:
                # Use enhanced dynamic model selection
                return await self._enhanced_model_selection(
                    prompt, intent_type, user_id, conversation_id, request_id, enable_fallback
                )
            else:
                # Use traditional model selection
                return await self._traditional_model_selection(
                    prompt, intent_type, user_id, conversation_id, request_id
                )
                
        except Exception as e:
            secure_logger.error(f"❌ Model selection error: {redact_sensitive_data(str(e))}")
            self.integration_metrics['selection_errors'] += 1
            
            if self.fallback_to_traditional:
                secure_logger.info("🔄 Falling back to traditional model selection")
                return await self._traditional_model_selection(
                    prompt, intent_type, user_id, conversation_id, request_id
                )
            else:
                raise
        finally:
            # Update timing metrics
            selection_time = time.time() - selection_start
            self._update_timing_metrics(selection_time)
    
    async def _enhanced_model_selection(self, 
                                      prompt: str,
                                      intent_type: str,
                                      user_id: Optional[str],
                                      conversation_id: Optional[str],
                                      request_id: Optional[str],
                                      enable_fallback: bool) -> Tuple[str, Dict[str, Any]]:
        """Enhanced model selection using DynamicModelSelector"""
        
        # Create model selection request
        request = ModelSelectionRequest(
            prompt=prompt,
            intent_type=intent_type,
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=request_id,
            enable_fallback=enable_fallback,
            enable_learning=True,
            quality_requirements=7.0
        )
        
        # Get enhanced model selection
        response: ModelSelectionResponse = await dynamic_model_selector.select_model(request)
        
        # Update metrics
        self.integration_metrics['enhanced_selections'] += 1
        
        # Create metadata dictionary
        metadata = {
            'selection_method': 'enhanced_dynamic',
            'confidence': response.confidence,
            'selection_strategy': response.selection_strategy,
            'performance_prediction': response.performance_prediction,
            'fallback_models': response.fallback_models,
            'estimated_response_time': response.estimated_response_time,
            'quality_expectations': response.quality_expectations,
            'explanation': response.explanation,
            'selection_metadata': response.selection_metadata,
            'enhanced_features_used': True
        }
        
        secure_logger.info(f"✅ Enhanced model selection: {response.selected_model.split('/')[-1]} "
                          f"(confidence: {response.confidence:.2f}, "
                          f"strategy: {response.selection_strategy})")
        
        return response.selected_model, metadata
    
    async def _traditional_model_selection(self,
                                         prompt: str,
                                         intent_type: str,
                                         user_id: Optional[str],
                                         conversation_id: Optional[str],
                                         request_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Traditional model selection using IntelligentRouter"""
        
        try:
            # Convert intent_type to IntentType enum
            intent_enum = IntentType[intent_type.upper()]
        except (KeyError, AttributeError):
            secure_logger.warning(f"⚠️ Unknown intent type: {intent_type}, using TEXT_GENERATION")
            intent_enum = IntentType.TEXT_GENERATION
        
        # Analyze complexity
        complexity_analyzer = AdvancedComplexityAnalyzer()
        complexity = complexity_analyzer.analyze_complexity(prompt)
        
        # Get model recommendation from traditional router
        selected_model, _ = intelligent_router.select_model(prompt, intent_enum, complexity, {})
        
        # Update metrics
        self.integration_metrics['traditional_fallbacks'] += 1
        
        # Create basic metadata
        metadata = {
            'selection_method': 'traditional_router',
            'confidence': 0.7,  # Default confidence for traditional routing
            'complexity': complexity,
            'intent_enum': intent_enum.value,
            'enhanced_features_used': False
        }
        
        secure_logger.info(f"🔄 Traditional model selection: {selected_model.split('/')[-1]} "
                          f"(intent: {intent_type}, complexity: {complexity.complexity_score:.1f})")
        
        return selected_model, metadata
    
    async def select_fallback_model(self,
                                   original_prompt: str,
                                   intent_type: str,
                                   failed_model: str,
                                   error_message: str,
                                   attempt_count: int = 1,
                                   conversation_id: Optional[str] = None,
                                   user_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Intelligent fallback model selection
        
        Args:
            original_prompt: Original user prompt
            intent_type: Classified intent type
            failed_model: Model that failed
            error_message: Error message from failure
            attempt_count: Number of attempts made
            conversation_id: Conversation identifier
            user_id: User identifier
            
        Returns:
            Tuple of (fallback_model, selection_metadata)
        """
        try:
            if self.enhanced_enabled:
                # Use enhanced fallback strategy
                return await self._enhanced_fallback_selection(
                    original_prompt, intent_type, failed_model, error_message,
                    attempt_count, conversation_id, user_id
                )
            else:
                # Use traditional fallback
                return await self._traditional_fallback_selection(
                    intent_type, failed_model, attempt_count
                )
                
        except Exception as e:
            secure_logger.error(f"❌ Fallback selection error: {redact_sensitive_data(str(e))}")
            
            # Emergency fallback
            return await self._emergency_fallback_selection(intent_type)
    
    async def _enhanced_fallback_selection(self,
                                         original_prompt: str,
                                         intent_type: str,
                                         failed_model: str,
                                         error_message: str,
                                         attempt_count: int,
                                         conversation_id: Optional[str],
                                         user_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Enhanced fallback selection using DynamicModelSelector"""
        
        # Create original request for fallback
        original_request = ModelSelectionRequest(
            prompt=original_prompt,
            intent_type=intent_type,
            user_id=user_id,
            conversation_id=conversation_id,
            enable_fallback=True,
            enable_learning=True
        )
        
        # Get intelligent fallback
        response: ModelSelectionResponse = await dynamic_model_selector.select_model_with_fallback(
            original_request, failed_model, error_message, attempt_count
        )
        
        # Create metadata
        metadata = {
            'selection_method': 'enhanced_fallback',
            'failed_model': failed_model,
            'error_message': redact_sensitive_data(error_message),
            'attempt_count': attempt_count,
            'confidence': response.confidence,
            'selection_strategy': response.selection_strategy,
            'fallback_reasoning': response.selection_metadata.get('fallback_reasoning'),
            'enhanced_features_used': True
        }
        
        secure_logger.info(f"🎯 Enhanced fallback: {response.selected_model.split('/')[-1]} "
                          f"(strategy: {response.selection_strategy}, attempt: {attempt_count})")
        
        return response.selected_model, metadata
    
    async def _traditional_fallback_selection(self,
                                            intent_type: str,
                                            failed_model: str,
                                            attempt_count: int) -> Tuple[str, Dict[str, Any]]:
        """Traditional fallback selection"""
        
        try:
            intent_enum = IntentType[intent_type.upper()]
        except (KeyError, AttributeError):
            intent_enum = IntentType.TEXT_GENERATION
        
        # Get fallback models from traditional router
        fallback_models = intelligent_router._get_fallback_models(intent_enum)
        
        # Remove failed model and select next
        available_fallbacks = [m for m in fallback_models if m != failed_model]
        
        if available_fallbacks:
            selected_model = available_fallbacks[0]
        else:
            # Emergency fallback
            selected_model = "microsoft/Phi-3-mini-4k-instruct"
        
        metadata = {
            'selection_method': 'traditional_fallback',
            'failed_model': failed_model,
            'attempt_count': attempt_count,
            'available_fallbacks': len(available_fallbacks),
            'enhanced_features_used': False
        }
        
        secure_logger.info(f"🔄 Traditional fallback: {selected_model.split('/')[-1]} "
                          f"(attempt: {attempt_count})")
        
        return selected_model, metadata
    
    async def _emergency_fallback_selection(self, intent_type: str) -> Tuple[str, Dict[str, Any]]:
        """Emergency fallback when all else fails"""
        
        emergency_model = "microsoft/Phi-3-mini-4k-instruct"
        metadata = {
            'selection_method': 'emergency_fallback',
            'intent_type': intent_type,
            'enhanced_features_used': False
        }
        
        secure_logger.warning(f"🚨 Emergency fallback: {emergency_model}")
        
        return emergency_model, metadata
    
    async def record_model_performance(self,
                                     model_name: str,
                                     intent_type: str,
                                     success: bool,
                                     response_time: float,
                                     quality_score: float,
                                     request_id: Optional[str] = None,
                                     conversation_id: Optional[str] = None,
                                     user_satisfaction: Optional[float] = None,
                                     error_details: Optional[str] = None) -> None:
        """
        Record model performance for learning and adaptation
        
        Args:
            model_name: Name of the model used
            intent_type: Type of task performed
            success: Whether the request was successful
            response_time: Time taken for response
            quality_score: Quality score (0-10)
            request_id: Request identifier
            conversation_id: Conversation identifier
            user_satisfaction: User satisfaction rating
            error_details: Error details if failed
        """
        try:
            if self.enhanced_enabled and request_id:
                # Record in enhanced system
                await dynamic_model_selector.record_performance(
                    request_id=request_id,
                    success=success,
                    response_time=response_time,
                    quality_score=quality_score,
                    user_satisfaction=user_satisfaction,
                    error_details=error_details
                )
            
            # Also record in traditional router for backward compatibility
            try:
                intent_enum = IntentType[intent_type.upper()]
            except (KeyError, AttributeError):
                intent_enum = IntentType.TEXT_GENERATION
            
            # Get complexity if available, or create default
            complexity = getattr(intelligent_router, '_last_complexity', None)
            if complexity is None:
                complexity = PromptComplexity(
                    complexity_score=5.0,
                    technical_depth=3,
                    reasoning_required=False,
                    context_length=100,
                    domain_specificity=0.5,
                    creativity_factor=0.5,
                    multi_step=False,
                    uncertainty=0.5,
                    priority_level='medium',
                    estimated_tokens=100,
                    domain_expertise='general',
                    reasoning_chain_length=1,
                    requires_external_knowledge=False,
                    temporal_context='current',
                    user_intent_confidence=0.7,
                    cognitive_load=0.5
                )
            
            intelligent_router.record_model_performance(
                model_name=model_name,
                intent_type=intent_type,
                success=success,
                response_time=response_time,
                quality_score=quality_score,
                complexity=complexity,
                conversation_id=conversation_id,
                error_type=error_details
            )
            
            secure_logger.debug(f"📊 Performance recorded for {model_name.split('/')[-1]}: "
                              f"success={success}, quality={quality_score:.1f}")
            
        except Exception as e:
            secure_logger.error(f"❌ Failed to record performance: {redact_sensitive_data(str(e))}")
    
    def _update_timing_metrics(self, selection_time: float) -> None:
        """Update timing metrics"""
        try:
            current_avg = self.integration_metrics.get('avg_selection_time', 0.0)
            total_selections = (self.integration_metrics['enhanced_selections'] + 
                              self.integration_metrics['traditional_fallbacks'])
            
            if total_selections > 0:
                self.integration_metrics['avg_selection_time'] = (
                    (current_avg * (total_selections - 1) + selection_time) / total_selections
                )
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to update timing metrics: {redact_sensitive_data(str(e))}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and metrics"""
        total_selections = (self.integration_metrics['enhanced_selections'] + 
                           self.integration_metrics['traditional_fallbacks'])
        
        return {
            'enhanced_enabled': self.enhanced_enabled,
            'conversation_tracking_enabled': self.conversation_tracking_enabled,
            'explanation_enabled': self.explanation_enabled,
            'total_selections': total_selections,
            'enhanced_usage_rate': (
                self.integration_metrics['enhanced_selections'] / max(1, total_selections)
            ),
            'average_selection_time': self.integration_metrics['avg_selection_time'],
            'error_rate': (
                self.integration_metrics['selection_errors'] / max(1, total_selections)
            ),
            'metrics': self.integration_metrics.copy()
        }
    
    async def get_system_insights(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive system insights"""
        try:
            insights = {
                'integration_status': self.get_integration_status(),
                'timestamp': datetime.now().isoformat(),
                'time_window_hours': time_window_hours
            }
            
            if self.enhanced_enabled:
                # Get insights from enhanced system
                enhanced_insights = await dynamic_model_selector.get_selection_insights(time_window_hours)
                insights['enhanced_system'] = enhanced_insights
            
            return insights
            
        except Exception as e:
            secure_logger.error(f"❌ Failed to get system insights: {redact_sensitive_data(str(e))}")
            return {'error': str(e)}

# Global integration instance
enhanced_integration = EnhancedModelSelectionIntegration()