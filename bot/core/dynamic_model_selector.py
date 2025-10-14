"""
Dynamic Model Selector - Superior AI Model Selection System
Orchestrates all enhanced components for intelligent, adaptive model selection
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .bot_types import IntentType, PromptComplexity, DomainExpertise
from .performance_predictor import performance_predictor, PredictionContext
from .model_health_monitor import health_monitor
from .dynamic_fallback_strategy import dynamic_fallback_strategy, ErrorType
from .conversation_context_tracker import conversation_context_tracker, ConversationTurn
from .model_selection_explainer import model_selection_explainer, ModelSelectionExplanation
from .router import router as intelligent_router
from ..security_utils import redact_sensitive_data, get_secure_logger

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)

@dataclass
class ModelSelectionRequest:
    """Request for model selection with full context"""
    prompt: str
    intent_type: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    time_constraints: Optional[float] = None
    quality_requirements: float = 7.0
    enable_fallback: bool = True
    enable_learning: bool = True
    custom_parameters: Optional[Dict] = None

@dataclass
class ModelSelectionResponse:
    """Complete response from model selection"""
    selected_model: str
    explanation: ModelSelectionExplanation
    confidence: float
    selection_strategy: str
    performance_prediction: Optional[float]
    fallback_models: List[str]
    estimated_response_time: Optional[float]
    quality_expectations: float
    selection_metadata: Dict[str, Any]

class DynamicModelSelector:
    """
    Superior Dynamic Model Selection System
    Orchestrates all enhanced components for intelligent, adaptive model selection
    """
    
    def __init__(self):
        """Initialize the Dynamic Model Selector"""
        self.selection_history: List[Dict] = []
        self.active_conversations: Dict[str, Any] = {}
        self._last_complexity: Optional[PromptComplexity] = None
        
        # Performance tracking
        self.selection_metrics = {
            'total_selections': 0,
            'successful_selections': 0,
            'fallback_triggers': 0,
            'avg_confidence': 0.0,
            'strategies_used': {}
        }
        
        # Adaptation parameters
        self.adaptation_config = {
            'learning_rate': 0.1,
            'confidence_threshold': 0.7,
            'fallback_threshold': 0.3,
            'real_time_weight': 0.4,
            'conversation_weight': 0.3,
            'prediction_weight': 0.3
        }
        
        secure_logger.info("🚀 Dynamic Model Selector initialized - Superior AI model selection system online")
    
    async def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """
        Primary model selection method with complete intelligence
        
        Args:
            request: Model selection request with full context
            
        Returns:
            Complete model selection response with explanation
        """
        selection_start = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        secure_logger.info(f"🎯 Dynamic model selection started for intent: {request.intent_type} "
                          f"(request_id: {request_id[:8]}...)")
        
        try:
            # 1. Initialize conversation tracking if needed
            await self._initialize_conversation_tracking(request)
            
            # 2. Analyze prompt complexity with enhanced analysis
            complexity = await self._analyze_enhanced_complexity(request.prompt, request.intent_type)
            self._last_complexity = complexity  # Store for response time estimation
            
            # 3. Get conversation context and state
            conversation_context = await self._get_conversation_context(request.conversation_id)
            
            # 4. Create comprehensive prediction context
            prediction_context = self._create_prediction_context(
                request, complexity, conversation_context
            )
            
            # 5. Get performance predictions from ML predictor
            performance_predictions = performance_predictor.predict_performance(prediction_context)
            
            # 6. Get real-time health and conversation-aware rankings
            conversation_rankings = health_monitor.get_conversation_aware_rankings(
                conversation_id=request.conversation_id,
                intent_type=request.intent_type,
                complexity=complexity
            )
            
            # 7. Combine all intelligence sources for optimal selection
            model_selection_result = await self._combine_intelligence_sources(
                request=request,
                complexity=complexity,
                conversation_context=conversation_context,
                performance_predictions=performance_predictions,
                conversation_rankings=conversation_rankings,
                prediction_context=prediction_context
            )
            
            # 8. Generate comprehensive explanation
            explanation = await self._generate_comprehensive_explanation(
                request, model_selection_result, complexity, conversation_context
            )
            
            # 9. Create final response
            response = self._create_selection_response(
                request, model_selection_result, explanation, selection_start
            )
            
            # 10. Record selection for learning
            await self._record_selection(request, response)
            
            # 11. Update metrics
            self._update_selection_metrics(response)
            
            secure_logger.info(f"✅ Dynamic model selection complete: {response.selected_model.split('/')[-1]} "
                              f"(confidence: {response.confidence:.2f}, "
                              f"time: {time.time() - selection_start:.2f}s)")
            
            return response
            
        except Exception as e:
            secure_logger.error(f"❌ Dynamic model selection failed: {redact_sensitive_data(str(e))}")
            
            # Fallback to traditional router
            try:
                intent_enum = IntentType[request.intent_type.upper()]
            except (KeyError, AttributeError):
                intent_enum = IntentType.TEXT_GENERATION
            
            fallback_model, _ = intelligent_router.select_model(
                request.prompt, intent_enum, None, {}
            )
            
            # Create default explanation for fallback
            default_explanation = ModelSelectionExplanation(
                selected_model=fallback_model,
                confidence=0.5,
                timestamp=datetime.now(),
                primary_reasons=[],
                alternative_models=[],
                context_factors={},
                selection_strategy='fallback_traditional',
                fallback_triggered=True,
                performance_prediction=None,
                conversation_influence=None,
                total_score=0.5,
                request_id=request.request_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                intent_type=request.intent_type,
                complexity_score=None
            )
            
            # Create minimal response
            return ModelSelectionResponse(
                selected_model=fallback_model,
                explanation=default_explanation,
                confidence=0.5,
                selection_strategy='fallback_traditional',
                performance_prediction=None,
                fallback_models=[],
                estimated_response_time=None,
                quality_expectations=5.0,
                selection_metadata={'error': str(e), 'fallback_used': True}
            )
    
    async def select_model_with_fallback(self, 
                                       original_request: ModelSelectionRequest,
                                       failed_model: str,
                                       error_message: str,
                                       attempt_count: int = 1) -> ModelSelectionResponse:
        """
        Advanced fallback model selection with intelligent error analysis
        
        Args:
            original_request: Original selection request
            failed_model: Model that failed
            error_message: Error message from failure
            attempt_count: Number of attempts made
            
        Returns:
            Fallback model selection response
        """
        secure_logger.info(f"🔄 Intelligent fallback selection started (attempt {attempt_count})")
        
        try:
            # 1. Analyze error with sophisticated pattern recognition
            error_type = dynamic_fallback_strategy.analyze_error(
                error_message, failed_model, original_request.intent_type, attempt_count
            )
            
            # 2. Get enhanced complexity analysis
            try:
                from .router import AdvancedComplexityAnalyzer
                complexity_analyzer = AdvancedComplexityAnalyzer()
                complexity = complexity_analyzer.analyze_complexity(original_request.prompt)
            except Exception:
                # Fallback to default complexity
                estimated_tokens_val = int(len(original_request.prompt.split()) * 1.3)
                complexity = PromptComplexity(
                    complexity_score=5.0,
                    technical_depth=3,
                    reasoning_required=False,
                    context_length=estimated_tokens_val,
                    domain_specificity=0.5,
                    creativity_factor=0.5,
                    multi_step=False,
                    uncertainty=0.5,
                    priority_level='medium',
                    estimated_tokens=estimated_tokens_val,
                    domain_expertise='general',
                    reasoning_chain_length=1,
                    requires_external_knowledge=False,
                    temporal_context='current',
                    user_intent_confidence=0.7,
                    cognitive_load=0.5
                )
            
            # 3. Get conversation context
            conversation_context = await self._get_conversation_context(
                original_request.conversation_id
            )
            
            # 4. Get available models excluding failed ones
            available_models = health_monitor.get_available_models(original_request.intent_type)
            if not available_models:
                available_models = await self._get_fallback_model_list(original_request.intent_type)
            
            # 5. Determine intelligent fallback strategy
            fallback_decision = dynamic_fallback_strategy.determine_fallback_strategy(
                error_type=error_type,
                failed_model=failed_model,
                intent_type=original_request.intent_type,
                complexity=complexity,
                available_models=available_models,
                conversation_context=conversation_context.__dict__ if conversation_context else None
            )
            
            # 6. Execute fallback strategy
            if fallback_decision.recommended_models:
                selected_model = fallback_decision.recommended_models[0]
                
                # Create fallback selection context
                selection_context = {
                    'intent_type': original_request.intent_type,
                    'complexity': complexity,
                    'conversation_context': conversation_context.__dict__ if conversation_context else None,
                    'fallback_triggered': True,
                    'fallback_reason': fallback_decision.strategy_type,
                    'error_type': error_type.value,
                    'failed_model': failed_model,
                    'attempt_count': attempt_count,
                    'fallback_confidence': fallback_decision.confidence
                }
                
                # Generate explanation for fallback
                explanation = model_selection_explainer.explain_selection(
                    selected_model=selected_model,
                    candidate_models=fallback_decision.recommended_models,
                    model_scores={model: fallback_decision.confidence - (i * 0.1) 
                                for i, model in enumerate(fallback_decision.recommended_models)},
                    selection_context=selection_context,
                    request_metadata={
                        'request_id': original_request.request_id,
                        'user_id': original_request.user_id,
                        'conversation_id': original_request.conversation_id,
                        'intent_type': original_request.intent_type,
                        'complexity_score': complexity.complexity_score
                    }
                )
                
                # Create fallback response
                response = ModelSelectionResponse(
                    selected_model=selected_model,
                    explanation=explanation,
                    confidence=fallback_decision.confidence,
                    selection_strategy=f"fallback_{fallback_decision.strategy_type}",
                    performance_prediction=None,
                    fallback_models=fallback_decision.recommended_models[1:],
                    estimated_response_time=None,
                    quality_expectations=max(5.0, fallback_decision.confidence * 8),
                    selection_metadata={
                        'fallback_triggered': True,
                        'error_type': error_type.value,
                        'failed_model': failed_model,
                        'attempt_count': attempt_count,
                        'fallback_reasoning': fallback_decision.reasoning
                    }
                )
                
                # Record fallback for learning
                await self._record_fallback(original_request, response, failed_model, error_type)
                
                secure_logger.info(f"🎯 Intelligent fallback complete: {selected_model.split('/')[-1]} "
                                 f"(strategy: {fallback_decision.strategy_type}, "
                                 f"confidence: {fallback_decision.confidence:.2f})")
                
                return response
            
            else:
                # No fallback recommendations available
                raise Exception("No suitable fallback models available")
                
        except Exception as e:
            secure_logger.error(f"❌ Intelligent fallback failed: {redact_sensitive_data(str(e))}")
            
            # Ultimate fallback to traditional router
            try:
                intent_enum = IntentType[original_request.intent_type.upper()]
            except (KeyError, AttributeError):
                intent_enum = IntentType.TEXT_GENERATION
            
            fallback_model, _ = intelligent_router.select_model(
                original_request.prompt, intent_enum, None, {}
            )
            
            # Create default explanation for emergency fallback
            emergency_explanation = ModelSelectionExplanation(
                selected_model=fallback_model,
                confidence=0.3,
                timestamp=datetime.now(),
                primary_reasons=[],
                alternative_models=[],
                context_factors={},
                selection_strategy='emergency_fallback',
                fallback_triggered=True,
                performance_prediction=None,
                conversation_influence=None,
                total_score=0.3,
                request_id=original_request.request_id,
                user_id=original_request.user_id,
                conversation_id=original_request.conversation_id,
                intent_type=original_request.intent_type,
                complexity_score=None
            )
            
            return ModelSelectionResponse(
                selected_model=fallback_model,
                explanation=emergency_explanation,
                confidence=0.3,
                selection_strategy='emergency_fallback',
                performance_prediction=None,
                fallback_models=[],
                estimated_response_time=None,
                quality_expectations=4.0,
                selection_metadata={
                    'emergency_fallback': True,
                    'error': str(e),
                    'failed_model': failed_model,
                    'attempt_count': attempt_count
                }
            )
    
    async def record_performance(self, request_id: str, success: bool, 
                               response_time: float, quality_score: float,
                               user_satisfaction: Optional[float] = None,
                               error_details: Optional[str] = None) -> None:
        """
        Record performance results for continuous learning
        
        Args:
            request_id: Request identifier
            success: Whether the request was successful
            response_time: Time taken for response
            quality_score: Quality score of response (0-10)
            user_satisfaction: Optional user satisfaction rating
            error_details: Error details if failed
        """
        try:
            # Find the original selection
            original_selection = None
            for selection in self.selection_history:
                if selection.get('request_id') == request_id:
                    original_selection = selection
                    break
            
            if not original_selection:
                secure_logger.warning(f"⚠️ Cannot find selection for request_id: {request_id}")
                return
            
            # Extract details
            model_name = original_selection['selected_model']
            intent_type = original_selection['intent_type']
            conversation_id = original_selection.get('conversation_id')
            complexity = original_selection.get('complexity')
            
            # Ensure complexity is valid or create default
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
            
            # Record in router for centralized learning
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
            
            # Update conversation tracking if applicable
            if conversation_id and success:
                turn_data = ConversationTurn(
                    turn_id=int(time.time() * 1000),
                    timestamp=datetime.now(),
                    intent_type=intent_type,
                    complexity=complexity,
                    model_used=model_name,
                    success=success,
                    response_time=response_time,
                    quality_score=quality_score,
                    user_satisfaction=user_satisfaction,
                    topic_shift=False,  # Could be enhanced
                    followup_type='continuation'  # Could be enhanced
                )
                
                conversation_context_tracker.record_turn(conversation_id, turn_data)
            
            # Update internal metrics
            self.selection_metrics['total_selections'] += 1
            if success:
                self.selection_metrics['successful_selections'] += 1
            
            # Calculate running average confidence
            if 'confidence' in original_selection:
                current_avg = self.selection_metrics.get('avg_confidence', 0.0)
                total = self.selection_metrics['total_selections']
                new_confidence = original_selection['confidence']
                self.selection_metrics['avg_confidence'] = (
                    (current_avg * (total - 1) + new_confidence) / total
                )
            
            secure_logger.debug(f"📊 Performance recorded for {model_name.split('/')[-1]}: "
                              f"success={success}, quality={quality_score:.1f}")
            
        except Exception as e:
            secure_logger.error(f"❌ Failed to record performance: {redact_sensitive_data(str(e))}")
    
    async def get_selection_insights(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get insights about model selection patterns and performance
        
        Args:
            time_window_hours: Time window for analysis
            
        Returns:
            Comprehensive insights dictionary
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Get insights from all components
            health_insights = []  # health_monitor.get_model_rankings() method doesn't exist
            fallback_insights = dynamic_fallback_strategy.get_error_insights()
            explainer_insights = model_selection_explainer.analyze_selection_patterns(time_window_hours)
            conversation_insights = conversation_context_tracker.get_global_insights()
            predictor_insights = {}  # performance_predictor.get_performance_insights() method doesn't exist
            
            # Compile comprehensive insights
            insights = {
                'timeframe': f"Last {time_window_hours} hours",
                'timestamp': datetime.now().isoformat(),
                'selection_metrics': self.selection_metrics.copy(),
                'model_health': {
                    'top_models': health_insights[:10],
                    'total_models_tracked': len(health_insights)
                },
                'fallback_analysis': fallback_insights,
                'selection_explanations': explainer_insights,
                'conversation_patterns': conversation_insights,
                'performance_predictions': predictor_insights,
                'system_performance': {
                    'success_rate': (
                        self.selection_metrics['successful_selections'] / 
                        max(1, self.selection_metrics['total_selections'])
                    ),
                    'avg_confidence': self.selection_metrics['avg_confidence'],
                    'fallback_rate': (
                        self.selection_metrics['fallback_triggers'] /
                        max(1, self.selection_metrics['total_selections'])
                    )
                },
                'recommendations': self._generate_system_recommendations()
            }
            
            return insights
            
        except Exception as e:
            secure_logger.error(f"❌ Failed to get insights: {redact_sensitive_data(str(e))}")
            return {'error': str(e)}
    
    async def _initialize_conversation_tracking(self, request: ModelSelectionRequest) -> None:
        """Initialize conversation tracking if needed"""
        if request.conversation_id and request.conversation_id not in self.active_conversations:
            # Start conversation tracking
            try:
                from .router import AdvancedComplexityAnalyzer
                complexity_analyzer = AdvancedComplexityAnalyzer()
                initial_complexity = complexity_analyzer.analyze_complexity(request.prompt)
            except Exception:
                estimated_tokens_val = int(len(request.prompt.split()) * 1.3)
                initial_complexity = PromptComplexity(
                    complexity_score=5.0,
                    technical_depth=3,
                    reasoning_required=False,
                    context_length=estimated_tokens_val,
                    domain_specificity=0.5,
                    creativity_factor=0.5,
                    multi_step=False,
                    uncertainty=0.5,
                    priority_level='medium',
                    estimated_tokens=estimated_tokens_val,
                    domain_expertise='general',
                    reasoning_chain_length=1,
                    requires_external_knowledge=False,
                    temporal_context='current',
                    user_intent_confidence=0.7,
                    cognitive_load=0.5
                )
            
            conversation_context_tracker.start_conversation(
                conversation_id=request.conversation_id,
                initial_intent=request.intent_type,
                initial_complexity=initial_complexity
            )
            
            self.active_conversations[request.conversation_id] = {
                'started': datetime.now(),
                'user_id': request.user_id,
                'turn_count': 0
            }
    
    async def _analyze_enhanced_complexity(self, prompt: str, intent_type: str) -> PromptComplexity:
        """Analyze prompt complexity with enhanced analysis"""
        try:
            from .router import AdvancedComplexityAnalyzer
            complexity_analyzer = AdvancedComplexityAnalyzer()
            complexity = complexity_analyzer.analyze_complexity(prompt)
            
            # Store for later use in performance recording
            setattr(self, '_last_complexity', complexity)
            
            return complexity
        except Exception as e:
            secure_logger.warning(f"⚠️ Complexity analysis failed: {redact_sensitive_data(str(e))}")
            # Return default complexity
            estimated_tokens_val = int(len(prompt.split()) * 1.3)
            return PromptComplexity(
                complexity_score=5.0,
                technical_depth=3,
                reasoning_required=False,
                context_length=estimated_tokens_val,
                domain_specificity=0.5,
                creativity_factor=0.5,
                multi_step=False,
                uncertainty=0.5,
                priority_level='medium',
                estimated_tokens=estimated_tokens_val,
                domain_expertise='general',
                reasoning_chain_length=1,
                requires_external_knowledge=False,
                temporal_context='current',
                user_intent_confidence=0.7,
                cognitive_load=0.5
            )
    
    async def _get_conversation_context(self, conversation_id: Optional[str]):
        """Get conversation context if available"""
        if conversation_id:
            return conversation_context_tracker.get_conversation_context(conversation_id)
        return None
    
    def _create_prediction_context(self, request: ModelSelectionRequest, 
                                 complexity: PromptComplexity,
                                 conversation_context) -> PredictionContext:
        """Create comprehensive prediction context"""
        return PredictionContext(
            intent_type=request.intent_type,
            complexity=complexity,
            conversation_length=conversation_context.total_turns if conversation_context else 0,
            user_preferences={},  # Could be enhanced with user preference tracking
            time_constraints=request.time_constraints,
            quality_requirements=request.quality_requirements,
            previous_models_used=(
                conversation_context.models_used[-3:] if conversation_context else []
            )
        )
    
    async def _combine_intelligence_sources(self, request: ModelSelectionRequest,
                                          complexity: PromptComplexity,
                                          conversation_context,
                                          performance_predictions: List[Tuple[str, float]],
                                          conversation_rankings: List[Tuple[str, float]],
                                          prediction_context: PredictionContext) -> Dict[str, Any]:
        """Combine all intelligence sources for optimal model selection"""
        
        # Get available models
        available_models = health_monitor.get_available_models(request.intent_type)
        if not available_models:
            available_models = await self._get_fallback_model_list(request.intent_type)
        
        # Convert to dictionaries for easier processing
        perf_dict = dict(performance_predictions)
        conv_dict = dict(conversation_rankings)
        
        # Combine scores with adaptive weighting
        combined_scores = {}
        
        for model in available_models:
            score = 0.0
            score_components = {}
            
            # Performance prediction component
            if model in perf_dict:
                perf_score = perf_dict[model]
                weighted_score = perf_score * self.adaptation_config['prediction_weight']
                score += weighted_score
                score_components['performance_prediction'] = perf_score
            
            # Conversation-aware component
            if model in conv_dict:
                conv_score = conv_dict[model]
                weighted_score = conv_score * self.adaptation_config['conversation_weight']
                score += weighted_score
                score_components['conversation_aware'] = conv_score
            
            # Real-time health component
            health_score = health_monitor.get_model_quality_score(model)
            weighted_score = health_score * self.adaptation_config['real_time_weight']
            score += weighted_score
            score_components['health_score'] = health_score
            
            # Task suitability component (remaining weight)
            remaining_weight = 1.0 - (
                self.adaptation_config['prediction_weight'] +
                self.adaptation_config['conversation_weight'] + 
                self.adaptation_config['real_time_weight']
            )
            task_score = self._calculate_task_suitability(model, request.intent_type, complexity)
            weighted_score = task_score * remaining_weight
            score += weighted_score
            score_components['task_suitability'] = task_score
            
            combined_scores[model] = {
                'total_score': score,
                'components': score_components
            }
        
        # Select best model
        if combined_scores:
            best_model = max(combined_scores.keys(), 
                           key=lambda x: combined_scores[x]['total_score'])
            
            return {
                'selected_model': best_model,
                'all_scores': combined_scores,
                'available_models': available_models,
                'selection_rationale': combined_scores[best_model]
            }
        else:
            # Fallback to traditional router
            try:
                intent_enum = IntentType[request.intent_type.upper()]
            except (KeyError, AttributeError):
                intent_enum = IntentType.TEXT_GENERATION
            
            fallback_model, _ = intelligent_router.select_model(
                request.prompt, intent_enum, None, {}
            )
            return {
                'selected_model': fallback_model,
                'all_scores': {},
                'available_models': [fallback_model],
                'selection_rationale': {'fallback_used': True}
            }
    
    def _calculate_task_suitability(self, model: str, intent_type: str, 
                                  complexity: PromptComplexity) -> float:
        """Calculate task suitability score for a model"""
        # Use the enhanced task suitability from router
        try:
            intent_enum = IntentType[intent_type.upper()]
            return intelligent_router._calculate_task_suitability(model, intent_enum, complexity)
        except:
            return 5.0  # Default neutral score
    
    async def _get_fallback_model_list(self, intent_type: str) -> List[str]:
        """Get fallback model list for intent type"""
        try:
            intent_enum = IntentType[intent_type.upper()]
            return intelligent_router._get_fallback_models(intent_enum)
        except:
            return ["microsoft/Phi-3-mini-4k-instruct"]  # Ultimate fallback
    
    async def _generate_comprehensive_explanation(self, request: ModelSelectionRequest,
                                                model_selection_result: Dict,
                                                complexity: PromptComplexity,
                                                conversation_context) -> ModelSelectionExplanation:
        """Generate comprehensive explanation for model selection"""
        
        selected_model = model_selection_result['selected_model']
        all_scores = model_selection_result['all_scores']
        
        # Create selection context for explanation
        selection_context = {
            'intent_type': request.intent_type,
            'complexity': complexity,
            'conversation_context': conversation_context.__dict__ if conversation_context else None,
            'performance_history_scores': {
                model: scores['components'].get('performance_prediction', 0)
                for model, scores in all_scores.items()
            },
            'real_time_scores': {
                model: scores['components'].get('conversation_aware', 0)
                for model, scores in all_scores.items()
            },
            'health_scores': {
                model: scores['components'].get('health_score', 0)
                for model, scores in all_scores.items()
            },
            'task_suitability_scores': {
                model: scores['total_score']
                for model, scores in all_scores.items()
            },
            'predicted_performance': all_scores.get(selected_model, {}).get(
                'components', {}
            ).get('performance_prediction'),
            'fallback_triggered': False,
            'selection_strategy': 'dynamic_intelligent_combined'
        }
        
        # Generate explanation
        return model_selection_explainer.explain_selection(
            selected_model=selected_model,
            candidate_models=list(all_scores.keys()),
            model_scores={model: scores['total_score'] for model, scores in all_scores.items()},
            selection_context=selection_context,
            request_metadata={
                'request_id': request.request_id,
                'user_id': request.user_id,
                'conversation_id': request.conversation_id,
                'intent_type': request.intent_type,
                'complexity_score': complexity.complexity_score
            }
        )
    
    def _create_selection_response(self, request: ModelSelectionRequest,
                                 model_selection_result: Dict,
                                 explanation: ModelSelectionExplanation,
                                 selection_start: float) -> ModelSelectionResponse:
        """Create final model selection response"""
        
        selected_model = model_selection_result['selected_model']
        all_scores = model_selection_result['all_scores']
        
        # Get fallback models (other high-scoring models)
        sorted_models = sorted(
            all_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        fallback_models = [model for model, _ in sorted_models[1:6]]  # Top 5 alternatives
        
        # Estimate response time based on model characteristics
        estimated_response_time = self._estimate_response_time(selected_model, request)
        
        return ModelSelectionResponse(
            selected_model=selected_model,
            explanation=explanation,
            confidence=explanation.confidence,
            selection_strategy='dynamic_intelligent',
            performance_prediction=all_scores.get(selected_model, {}).get(
                'components', {}
            ).get('performance_prediction'),
            fallback_models=fallback_models,
            estimated_response_time=estimated_response_time,
            quality_expectations=request.quality_requirements,
            selection_metadata={
                'selection_time': time.time() - selection_start,
                'components_used': ['performance_predictor', 'health_monitor', 
                                  'conversation_tracker', 'explainer'],
                'adaptation_config': self.adaptation_config.copy(),
                'model_scores': {model: scores['total_score'] 
                               for model, scores in all_scores.items()}
            }
        )
    
    def _estimate_response_time(self, model: str, request: ModelSelectionRequest) -> Optional[float]:
        """Estimate response time for selected model"""
        try:
            # Basic estimation based on model characteristics
            base_time = 2.0  # Base response time
            
            # Adjust based on model size indicators
            model_lower = model.lower()
            if any(size in model_lower for size in ['70b', '72b', 'large']):
                base_time *= 2.0
            elif any(size in model_lower for size in ['mini', '0.5b', '1.5b']):
                base_time *= 0.5
            
            # Adjust based on task complexity
            if self._last_complexity is not None:
                complexity_factor = self._last_complexity.complexity_score / 5.0
                base_time *= complexity_factor
            
            return base_time
        except:
            return None
    
    async def _record_selection(self, request: ModelSelectionRequest, 
                              response: ModelSelectionResponse) -> None:
        """Record selection for learning and analysis"""
        try:
            selection_record = {
                'timestamp': datetime.now().isoformat(),
                'request_id': request.request_id,
                'user_id': request.user_id,
                'conversation_id': request.conversation_id,
                'intent_type': request.intent_type,
                'selected_model': response.selected_model,
                'confidence': response.confidence,
                'selection_strategy': response.selection_strategy,
                'quality_requirements': request.quality_requirements,
                'complexity': getattr(self, '_last_complexity', None),
                'selection_metadata': response.selection_metadata
            }
            
            self.selection_history.append(selection_record)
            
            # Keep only recent history (last 1000 selections)
            if len(self.selection_history) > 1000:
                self.selection_history = self.selection_history[-1000:]
                
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to record selection: {redact_sensitive_data(str(e))}")
    
    async def _record_fallback(self, request: ModelSelectionRequest,
                             response: ModelSelectionResponse,
                             failed_model: str, error_type) -> None:
        """Record fallback for learning"""
        try:
            self.selection_metrics['fallback_triggers'] += 1
            
            fallback_record = {
                'timestamp': datetime.now().isoformat(),
                'request_id': request.request_id,
                'failed_model': failed_model,
                'error_type': error_type.value if hasattr(error_type, 'value') else str(error_type),
                'fallback_model': response.selected_model,
                'fallback_strategy': response.selection_strategy,
                'confidence': response.confidence
            }
            
            # Could be stored separately for fallback analysis
            secure_logger.debug(f"📊 Fallback recorded: {json.dumps(fallback_record)}")
            
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to record fallback: {redact_sensitive_data(str(e))}")
    
    def _update_selection_metrics(self, response: ModelSelectionResponse) -> None:
        """Update internal selection metrics"""
        try:
            strategy = response.selection_strategy
            if strategy not in self.selection_metrics['strategies_used']:
                self.selection_metrics['strategies_used'][strategy] = 0
            self.selection_metrics['strategies_used'][strategy] += 1
            
        except Exception as e:
            secure_logger.warning(f"⚠️ Failed to update metrics: {redact_sensitive_data(str(e))}")
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system optimization recommendations"""
        recommendations = []
        
        try:
            # Success rate recommendations
            total = self.selection_metrics['total_selections']
            if total > 10:
                success_rate = self.selection_metrics['successful_selections'] / total
                if success_rate < 0.8:
                    recommendations.append(
                        f"Success rate is {success_rate:.2%} - consider adjusting selection criteria"
                    )
            
            # Confidence recommendations
            avg_confidence = self.selection_metrics.get('avg_confidence', 0)
            if avg_confidence < 0.7:
                recommendations.append(
                    f"Average confidence is {avg_confidence:.2f} - review model selection weights"
                )
            
            # Fallback rate recommendations
            if total > 0:
                fallback_rate = self.selection_metrics['fallback_triggers'] / total
                if fallback_rate > 0.3:
                    recommendations.append(
                        f"Fallback rate is {fallback_rate:.2%} - investigate model availability issues"
                    )
            
            # Strategy distribution recommendations
            strategies = self.selection_metrics.get('strategies_used', {})
            if strategies:
                dominant_strategy = max(strategies.keys(), key=lambda x: strategies[x])
                dominant_percentage = strategies[dominant_strategy] / sum(strategies.values())
                if dominant_percentage > 0.8:
                    recommendations.append(
                        f"Strategy '{dominant_strategy}' dominates ({dominant_percentage:.1%}) - "
                        "consider enabling more diverse selection strategies"
                    )
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {redact_sensitive_data(str(e))}")
        
        return recommendations if recommendations else ["System performing optimally"]

# Global instance
dynamic_model_selector = DynamicModelSelector()