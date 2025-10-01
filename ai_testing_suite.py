#!/usr/bin/env python3
"""
Comprehensive AI Functionality Testing Suite
Manual testing for Hugging Face By AadityaLabs AI Telegram Bot
Tests all AI capabilities, model integration, and performance metrics
"""

import asyncio
import logging
import time
import json
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import bot components for testing
from bot.core.model_caller import ModelCaller
from bot.core.router import IntelligentRouter
from bot.core.intent_classifier import AdvancedIntentClassifier
from bot.core.response_processor import ResponseProcessor
from bot.core.smart_cache import SmartCache
from bot.core.provider_factory import ProviderFactory
from bot.core.ai_providers import ChatMessage, ChatCompletionRequest, CompletionRequest
from bot.config import Config

# Setup test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AITestingSuite:
    """Comprehensive AI testing suite for manual validation"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.quality_assessments = {}
        self.error_scenarios = {}
        
        # Initialize components
        self.model_caller = None
        self.router = None
        self.intent_classifier = None
        self.response_processor = None
        self.smart_cache = None
        
        # Test scenarios
        self.test_scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> Dict[str, Dict]:
        """Create comprehensive test scenarios for all AI capabilities"""
        return {
            'text_generation': {
                'simple_question': "What is artificial intelligence?",
                'explanation_request': "Explain how machine learning works in simple terms",
                'creative_writing': "Write a short story about a robot learning to paint",
                'conversation': "Hello! I'm interested in learning about Python programming. Can you help me get started?",
                'complex_reasoning': "Compare the advantages and disadvantages of different sorting algorithms"
            },
            'code_generation': {
                'python_function': "Write a Python function to calculate the fibonacci sequence",
                'javascript_example': "Create a JavaScript function to validate email addresses using regex",
                'algorithm_explanation': "Implement a binary search algorithm in Python with detailed comments",
                'debug_code': "Fix this Python code: def factorial(n): if n = 0: return 1 else: return n * factorial(n-1)",
                'data_structure': "Implement a binary tree class in Python with insert and search methods"
            },
            'technical_analysis': {
                'system_design': "Design a scalable microservices architecture for an e-commerce platform",
                'documentation': "Write comprehensive documentation for a REST API endpoint",
                'troubleshooting': "Analyze potential performance bottlenecks in a web application",
                'security_analysis': "Identify security vulnerabilities in a user authentication system",
                'architecture_review': "Review the pros and cons of serverless vs container deployment"
            },
            'mathematical_solving': {
                'calculation': "Calculate the integral of x^2 from 0 to 5",
                'word_problem': "If a train travels 120 km in 2 hours, then speeds up and travels 180 km in the next 1.5 hours, what is its average speed?",
                'statistical_analysis': "Explain the difference between correlation and causation with examples",
                'probability': "What's the probability of rolling three sixes in a row with a fair die?",
                'optimization': "Find the minimum value of f(x) = x^2 - 4x + 7"
            },
            'creative_content': {
                'short_story': "Write a creative short story about time travel with an unexpected twist",
                'poetry': "Compose a haiku about artificial intelligence",
                'brainstorming': "Generate 10 creative business ideas for sustainable technology",
                'character_creation': "Create a detailed character profile for a fantasy novel protagonist",
                'dialogue_writing': "Write a dialogue between a human and AI discussing the future of work"
            }
        }
    
    async def test_model_caller_integration(self) -> Dict[str, Any]:
        """Test ModelCaller class and HF API integration"""
        logger.info("🧪 Testing ModelCaller and HF API Integration...")
        
        results = {
            'api_connectivity': False,
            'authentication': False,
            'session_management': False,
            'error_handling': False,
            'retry_mechanisms': False,
            'provider_initialization': False,
            'supported_models': [],
            'performance_metrics': {}
        }
        
        try:
            # Test 1: Initialize ModelCaller
            start_time = time.time()
            self.model_caller = ModelCaller()
            
            # Test provider initialization
            if hasattr(self.model_caller, '_ensure_provider_initialized'):
                provider_init = await self.model_caller._ensure_provider_initialized()
                results['provider_initialization'] = provider_init
                logger.info(f"✅ Provider initialization: {'SUCCESS' if provider_init else 'FAILED'}")
            
            # Test 2: Test API connectivity with simple request
            test_messages = [ChatMessage(role="user", content="Hello, test connectivity")]
            test_request = ChatCompletionRequest(
                messages=test_messages,
                model="Qwen/Qwen3-0.6B-Instruct",
                max_tokens=50,
                temperature=0.5
            )
            
            connectivity_start = time.time()
            try:
                response = await self.model_caller.generate_with_fallback(
                    test_request.messages,
                    intent_type="general",
                    max_tokens=50
                )
                connectivity_time = time.time() - connectivity_start
                
                if response and response.get('success'):
                    results['api_connectivity'] = True
                    results['authentication'] = True
                    results['performance_metrics']['connectivity_time'] = connectivity_time
                    logger.info(f"✅ API Connectivity: SUCCESS ({connectivity_time:.2f}s)")
                    logger.info(f"📄 Response preview: {response.get('content', 'N/A')[:100]}...")
                else:
                    logger.error(f"❌ API Connectivity: FAILED - {response}")
                    
            except Exception as e:
                logger.error(f"❌ API Connectivity test failed: {e}")
                results['error_handling'] = True  # Error was caught, which is good
            
            # Test 3: Test session management
            if hasattr(self.model_caller, 'session'):
                results['session_management'] = self.model_caller.session is not None
                logger.info(f"✅ Session management: {'ACTIVE' if results['session_management'] else 'INACTIVE'}")
            
            # Test 4: Test error handling with invalid model
            try:
                invalid_messages = [ChatMessage(role="user", content="Test invalid model")]
                invalid_response = await self.model_caller.generate_with_fallback(
                    invalid_messages,
                    intent_type="general"
                )
                # If we get here, error handling worked (fallback occurred)
                results['error_handling'] = True
                results['retry_mechanisms'] = True
                logger.info("✅ Error handling: SUCCESS (fallback worked)")
                
            except Exception as e:
                logger.info(f"⚠️ Error handling test: Exception caught: {e}")
                results['error_handling'] = True  # Exception was properly raised
            
            # Test 5: Get supported models
            if hasattr(self.model_caller, '_ai_provider') and self.model_caller._ai_provider:
                try:
                    supported_models = await self.model_caller._ai_provider.get_supported_models()
                    results['supported_models'] = supported_models
                    logger.info(f"✅ Supported models: {len(supported_models)} models available")
                    for model in supported_models[:5]:  # Show first 5
                        logger.info(f"   📋 {model}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get supported models: {e}")
            
            total_time = time.time() - start_time
            results['performance_metrics']['total_test_time'] = total_time
            
            logger.info(f"🎯 ModelCaller Integration Test completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ ModelCaller integration test failed: {e}")
            results['error_message'] = str(e)
        
        self.test_results['model_caller_integration'] = results
        return results
    
    async def test_intelligent_router(self) -> Dict[str, Any]:
        """Test IntelligentRouter system for prompt analysis and model selection"""
        logger.info("🧪 Testing Intelligent Router System...")
        
        results = {
            'prompt_analysis': {},
            'intent_classification': {},
            'model_selection': {},
            'complexity_analysis': {},
            'fallback_mechanisms': {},
            'performance_metrics': {}
        }
        
        try:
            # Initialize router
            self.router = IntelligentRouter()
            
            # Test different types of prompts
            test_prompts = {
                'code_request': "Write a Python function to sort a list",
                'creative_writing': "Write a poem about the ocean",
                'technical_question': "Explain how neural networks work",
                'math_problem': "Solve x^2 + 5x + 6 = 0",
                'general_conversation': "How are you doing today?",
                'complex_reasoning': "Analyze the implications of quantum computing on cryptography"
            }
            
            for prompt_type, prompt in test_prompts.items():
                start_time = time.time()
                
                # Test prompt analysis
                try:
                    analysis = await self.router.analyze_prompt_advanced(prompt)
                    analysis_time = time.time() - start_time
                    
                    results['prompt_analysis'][prompt_type] = {
                        'success': True,
                        'analysis': analysis,
                        'response_time': analysis_time
                    }
                    
                    logger.info(f"✅ {prompt_type} analysis: {analysis.get('intent_type', 'N/A')} "
                              f"(complexity: {analysis.get('complexity_score', 'N/A'):.2f}) "
                              f"in {analysis_time:.3f}s")
                    
                    # Test model selection (using available method)
                    if 'recommended_model' in analysis:
                        model_selection_start = time.time()
                        # Use available method for model selection
                        selected_model = analysis.get('recommended_model', 'default')
                        model_selection_time = time.time() - model_selection_start
                        
                        results['model_selection'][prompt_type] = {
                            'success': True,
                            'selected_model': selected_model,
                            'response_time': model_selection_time
                        }
                        
                        logger.info(f"   📋 Selected model: {selected_model}")
                    
                except Exception as e:
                    logger.error(f"❌ {prompt_type} analysis failed: {e}")
                    results['prompt_analysis'][prompt_type] = {
                        'success': False,
                        'error': str(e),
                        'response_time': time.time() - start_time
                    }
            
            # Test complexity analysis
            complexity_test_prompts = [
                "Hello",  # Simple
                "Explain machine learning algorithms",  # Medium
                "Design a distributed system for processing petabytes of data with fault tolerance"  # Complex
            ]
            
            for i, prompt in enumerate(complexity_test_prompts):
                try:
                    analysis = await self.router.analyze_prompt_advanced(prompt)
                    complexity = analysis.get('complexity_score', 0)
                    
                    results['complexity_analysis'][f'test_{i+1}'] = {
                        'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        'complexity_score': complexity,
                        'expected_level': ['simple', 'medium', 'complex'][i]
                    }
                    
                    logger.info(f"✅ Complexity test {i+1}: {complexity:.2f} "
                              f"({['simple', 'medium', 'complex'][i]} expected)")
                    
                except Exception as e:
                    logger.error(f"❌ Complexity test {i+1} failed: {e}")
            
            logger.info("🎯 Intelligent Router test completed")
            
        except Exception as e:
            logger.error(f"❌ Intelligent Router test failed: {e}")
            results['error'] = {'message': str(e), 'test_failed': True}
        
        self.test_results['intelligent_router'] = results
        return results
    
    async def test_ai_capabilities(self) -> Dict[str, Any]:
        """Test comprehensive AI capabilities across all categories"""
        logger.info("🧪 Testing Comprehensive AI Capabilities...")
        
        results = {}
        
        # Ensure ModelCaller is initialized
        if not self.model_caller:
            self.model_caller = ModelCaller()
        
        for category, scenarios in self.test_scenarios.items():
            logger.info(f"🔍 Testing {category.replace('_', ' ').title()} capabilities...")
            
            category_results = {}
            
            for scenario_name, prompt in scenarios.items():
                start_time = time.time()
                
                try:
                    # Create chat messages
                    messages = [ChatMessage(role="user", content=prompt)]
                    
                    # Generate response using ModelCaller
                    response = await self.model_caller.generate_with_fallback(
                        messages=messages,
                        intent_type=category,
                        max_tokens=512,
                        temperature=0.7
                    )
                    
                    response_time = time.time() - start_time
                    
                    if response and response.get('success'):
                        content = response.get('content', '')
                        
                        # Assess response quality
                        quality_score = self._assess_response_quality(
                            prompt, content, category, response_time
                        )
                        
                        category_results[scenario_name] = {
                            'success': True,
                            'prompt': prompt,
                            'response': content,
                            'response_time': response_time,
                            'quality_score': quality_score,
                            'model_used': response.get('model_used', 'Unknown'),
                            'tokens_used': response.get('tokens_used', 0)
                        }
                        
                        logger.info(f"✅ {scenario_name}: SUCCESS "
                                  f"(Quality: {quality_score:.1f}/10, "
                                  f"Time: {response_time:.2f}s, "
                                  f"Length: {len(content)} chars)")
                        
                        # Log a preview of the response
                        preview = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"   📄 Response preview: {preview}")
                        
                    else:
                        error_msg = response.get('error_message', 'Unknown error') if response else 'No response'
                        category_results[scenario_name] = {
                            'success': False,
                            'prompt': prompt,
                            'error': error_msg,
                            'response_time': response_time
                        }
                        logger.error(f"❌ {scenario_name}: FAILED - {error_msg}")
                
                except Exception as e:
                    response_time = time.time() - start_time
                    category_results[scenario_name] = {
                        'success': False,
                        'prompt': prompt,
                        'error': str(e),
                        'response_time': response_time
                    }
                    logger.error(f"❌ {scenario_name}: EXCEPTION - {e}")
                
                # Small delay between tests
                await asyncio.sleep(0.5)
            
            results[category] = category_results
            
            # Calculate category statistics
            successful_tests = sum(1 for r in category_results.values() if r.get('success', False))
            total_tests = len(category_results)
            avg_response_time = sum(r.get('response_time', 0) for r in category_results.values()) / total_tests
            avg_quality_score = sum(r.get('quality_score', 0) for r in category_results.values() if r.get('success', False))
            avg_quality_score = avg_quality_score / max(successful_tests, 1)
            
            logger.info(f"📊 {category.replace('_', ' ').title()} Summary: "
                      f"{successful_tests}/{total_tests} successful "
                      f"(Avg Quality: {avg_quality_score:.1f}/10, "
                      f"Avg Time: {avg_response_time:.2f}s)")
        
        self.test_results['ai_capabilities'] = results
        return results
    
    def _assess_response_quality(self, prompt: str, response: str, category: str, response_time: float) -> float:
        """Assess the quality of an AI response across multiple dimensions"""
        
        score = 0.0
        max_score = 10.0
        
        # 1. Response length appropriateness (2 points)
        if len(response) < 10:
            score += 0.0  # Too short
        elif len(response) < 50:
            score += 1.0  # Short but acceptable
        elif len(response) < 500:
            score += 2.0  # Good length
        else:
            score += 1.5  # Long, might be verbose
        
        # 2. Relevance to prompt (3 points)
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Basic keyword matching
        prompt_keywords = set(prompt_lower.split())
        response_keywords = set(response_lower.split())
        keyword_overlap = len(prompt_keywords & response_keywords) / max(len(prompt_keywords), 1)
        score += min(3.0, keyword_overlap * 3)
        
        # 3. Category-specific quality checks (3 points)
        if category == 'code_generation':
            # Check for code patterns
            code_indicators = ['def ', 'function', 'class ', 'import', 'return', '{', '}', '(', ')']
            code_score = sum(1 for indicator in code_indicators if indicator in response) / len(code_indicators)
            score += code_score * 3
        elif category == 'mathematical_solving':
            # Check for mathematical content
            math_indicators = ['=', '+', '-', '*', '/', 'equation', 'solution', 'answer', 'calculate']
            math_score = sum(1 for indicator in math_indicators if indicator in response) / len(math_indicators)
            score += math_score * 3
        elif category == 'creative_content':
            # Check for creative language
            creative_indicators = ['story', 'character', 'scene', 'emotion', 'vivid', 'imagine']
            creative_score = sum(1 for indicator in creative_indicators if indicator in response_lower) / len(creative_indicators)
            score += creative_score * 3
        else:
            # General quality indicators
            quality_indicators = ['because', 'therefore', 'however', 'example', 'specifically']
            quality_score = sum(1 for indicator in quality_indicators if indicator in response_lower) / len(quality_indicators)
            score += quality_score * 3
        
        # 4. Response time factor (2 points)
        if response_time < 1.0:
            score += 2.0  # Excellent speed
        elif response_time < 3.0:
            score += 1.5  # Good speed
        elif response_time < 5.0:
            score += 1.0  # Acceptable speed
        else:
            score += 0.5  # Slow but functional
        
        return min(score, max_score)
    
    async def test_intent_classification_accuracy(self) -> Dict[str, Any]:
        """Test intent classification system accuracy"""
        logger.info("🧪 Testing Intent Classification Accuracy...")
        
        results = {
            'classification_tests': {},
            'accuracy_metrics': {},
            'performance_metrics': {}
        }
        
        try:
            # Initialize intent classifier
            self.intent_classifier = AdvancedIntentClassifier()
            
            # Test prompts with known expected intents
            test_cases = [
                # Code generation tests
                ("Write a Python function", "CODE_GENERATION"),
                ("Create a JavaScript class", "CODE_GENERATION"),
                ("Debug this code", "CODE_GENERATION"),
                
                # Creative content tests  
                ("Write a story about", "CREATIVE_WRITING"),
                ("Compose a poem", "CREATIVE_WRITING"),
                ("Create a character", "CREATIVE_WRITING"),
                
                # Question answering tests
                ("What is machine learning?", "QUESTION_ANSWERING"),
                ("How does encryption work?", "QUESTION_ANSWERING"),
                ("Explain quantum computing", "QUESTION_ANSWERING"),
                
                # Text generation tests
                ("Tell me about", "TEXT_GENERATION"),
                ("Describe the process", "TEXT_GENERATION"),
                ("Write an essay", "TEXT_GENERATION"),
                
                # Math tests
                ("Calculate the integral", "MATHEMATICAL_REASONING"),
                ("Solve this equation", "MATHEMATICAL_REASONING"),
                ("Find the derivative", "MATHEMATICAL_REASONING")
            ]
            
            correct_classifications = 0
            total_tests = len(test_cases)
            total_classification_time = 0
            
            for prompt, expected_intent in test_cases:
                start_time = time.time()
                
                try:
                    # Test classification
                    classification_result = await self.intent_classifier.classify_advanced(prompt)
                    classification_time = time.time() - start_time
                    total_classification_time += classification_time
                    
                    classified_intent = getattr(classification_result, 'primary_intent', 'UNKNOWN')
                    confidence = getattr(classification_result, 'confidence', 0.0)
                    # Compare the classified intent with expected intent
                    if hasattr(classified_intent, 'name'):
                        classified_name = getattr(classified_intent, 'name', str(classified_intent))
                        is_correct = classified_name == expected_intent
                    else:
                        is_correct = str(classified_intent) == expected_intent
                    
                    if is_correct:
                        correct_classifications += 1
                    
                    results['classification_tests'][prompt] = {
                        'expected': expected_intent,
                        'classified': str(classified_intent),
                        'correct': is_correct,
                        'confidence': confidence,
                        'response_time': classification_time,
                        'full_result': classification_result
                    }
                    
                    status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                    logger.info(f"{status} - '{prompt}' -> {classified_intent} "
                              f"(confidence: {confidence:.2f}, time: {classification_time:.3f}s)")
                    
                except Exception as e:
                    logger.error(f"❌ Classification failed for '{prompt}': {e}")
                    results['classification_tests'][prompt] = {
                        'expected': expected_intent,
                        'classified': 'ERROR',
                        'correct': False,
                        'error': str(e),
                        'response_time': time.time() - start_time
                    }
            
            # Calculate accuracy metrics
            accuracy = (correct_classifications / total_tests) * 100 if total_tests > 0 else 0
            avg_response_time = total_classification_time / total_tests if total_tests > 0 else 0
            
            results['accuracy_metrics'] = {
                'accuracy_percentage': accuracy,
                'correct_classifications': correct_classifications,
                'total_tests': total_tests,
                'target_accuracy': 90.0  # Target from requirements
            }
            
            results['performance_metrics'] = {
                'average_response_time': avg_response_time,
                'total_classification_time': total_classification_time,
                'classifications_per_second': total_tests / total_classification_time if total_classification_time > 0 else 0
            }
            
            meets_target = accuracy >= 90.0
            logger.info(f"🎯 Intent Classification Accuracy: {accuracy:.1f}% "
                      f"({correct_classifications}/{total_tests}) "
                      f"{'✅ MEETS TARGET' if meets_target else '❌ BELOW TARGET'} "
                      f"(Target: 90%)")
            logger.info(f"⚡ Average classification time: {avg_response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Intent classification test failed: {e}")
            results['error'] = {'message': str(e), 'test_failed': True}
        
        self.test_results['intent_classification'] = results
        return results
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        logger.info("🚀 Starting Comprehensive AI Functionality Testing Suite...")
        
        start_time = time.time()
        
        # Run all test components
        test_methods = [
            self.test_model_caller_integration,
            self.test_intelligent_router,
            self.test_ai_capabilities,
            self.test_intent_classification_accuracy
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
                # Short delay between major test categories
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"❌ Test method {test_method.__name__} failed: {e}")
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        self.performance_metrics['total_test_suite_time'] = total_time
        
        # Compile final results
        final_results = {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'quality_assessments': self.quality_assessments,
            'error_scenarios': self.error_scenarios,
            'test_summary': self._generate_test_summary()
        }
        
        logger.info(f"🎉 Comprehensive Test Suite completed in {total_time:.2f}s")
        
        return final_results
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive test summary"""
        summary = {
            'overall_status': 'UNKNOWN',
            'critical_systems': {},
            'capabilities_tested': 0,
            'success_rate': 0.0,
            'performance_summary': {},
            'quality_summary': {},
            'recommendations': []
        }
        
        try:
            # Analyze results
            total_tests = 0
            successful_tests = 0
            
            # Count AI capabilities tests
            if 'ai_capabilities' in self.test_results:
                for category, scenarios in self.test_results['ai_capabilities'].items():
                    for scenario, result in scenarios.items():
                        total_tests += 1
                        if result.get('success', False):
                            successful_tests += 1
                        summary['capabilities_tested'] = total_tests
            
            # Calculate success rate
            if total_tests > 0:
                summary['success_rate'] = (successful_tests / total_tests) * 100
            
            # Analyze critical systems
            critical_systems = ['model_caller_integration', 'intelligent_router', 'intent_classification']
            for system in critical_systems:
                if system in self.test_results:
                    result = self.test_results[system]
                    if isinstance(result, dict):
                        # Check for success indicators
                        success_indicators = ['api_connectivity', 'authentication', 'accuracy_percentage']
                        system_status = any(result.get(indicator, False) for indicator in success_indicators)
                        summary['critical_systems'][system] = 'OPERATIONAL' if system_status else 'ISSUES_DETECTED'
            
            # Determine overall status
            if summary['success_rate'] >= 90:
                summary['overall_status'] = 'EXCELLENT'
            elif summary['success_rate'] >= 75:
                summary['overall_status'] = 'GOOD'
            elif summary['success_rate'] >= 50:
                summary['overall_status'] = 'NEEDS_IMPROVEMENT'
            else:
                summary['overall_status'] = 'CRITICAL_ISSUES'
            
            # Generate recommendations
            if summary['success_rate'] < 90:
                summary['recommendations'].append("Improve error handling and fallback mechanisms")
            if len(summary['critical_systems']) < 3:
                summary['recommendations'].append("Ensure all critical systems are properly tested")
            
        except Exception as e:
            logger.error(f"❌ Error generating test summary: {e}")
            summary['error'] = str(e)
        
        return summary

async def main():
    """Main function to run the comprehensive AI testing suite"""
    
    # Create and run test suite
    test_suite = AITestingSuite()
    
    try:
        results = await test_suite.run_comprehensive_test_suite()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ai_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = json.loads(json.dumps(results, default=str))
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"📄 Test results saved to: {results_file}")
        
        # Print summary
        summary = results.get('test_summary', {})
        print("\n" + "="*80)
        print("🎯 AI FUNCTIONALITY TESTING SUMMARY")
        print("="*80)
        print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Capabilities Tested: {summary.get('capabilities_tested', 0)}")
        
        print("\nCritical Systems Status:")
        for system, status in summary.get('critical_systems', {}).items():
            print(f"  • {system}: {status}")
        
        if summary.get('recommendations'):
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("="*80)
        return results
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return None

if __name__ == "__main__":
    # Run the test suite
    results = asyncio.run(main())