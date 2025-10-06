#!/usr/bin/env python3
"""
Discover Working Free-Tier HuggingFace Models
Systematically test models to find those that actually work on the free tier without 404 errors.
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import bot components for testing
from bot.core.model_caller import ModelCaller
from bot.core.ai_providers import ChatMessage, ChatCompletionRequest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingModelDiscovery:
    """Discover and test working models on HuggingFace free tier"""
    
    def __init__(self):
        self.working_models = {}
        self.failed_models = {}
        
    async def test_model_candidates(self) -> Dict[str, Any]:
        """
        Test carefully selected model candidates that are most likely to work on free tier
        Based on HuggingFace documentation and free tier limitations
        """
        logger.info("🔍 Discovering Working Free-Tier Models...")
        
        # 2025 CHAT-COMPATIBLE MODELS - Updated for Inference Providers API
        # These models work with chat completion format required by the new API
        model_candidates = {
            # TEXT GENERATION - 2025 chat-compatible models
            "text_generation": [
                # Microsoft Phi models - Small but powerful
                "microsoft/Phi-3-mini-4k-instruct",      # 3.8B, proven reliable
                "microsoft/Phi-3.5-mini-instruct",       # Updated version
                "microsoft/Phi-4-mini-instruct",         # Latest Phi model
                
                # Qwen models - Alibaba's chat models
                "Qwen/Qwen2.5-1.5B-Instruct",           # Efficient small model
                "Qwen/Qwen2.5-7B-Instruct",             # Balanced performance
                "Qwen/Qwen2.5-72B-Instruct",            # High performance (if available)
                
                # Meta Llama models
                "meta-llama/Llama-3.1-8B-Instruct",     # Latest Llama
                "meta-llama/Llama-3.3-70B-Instruct",    # Larger Llama (if available)
                
                # DeepSeek models - Reasoning capable
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # Small reasoning
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",    # Medium reasoning
                "deepseek-ai/DeepSeek-V3",                     # Latest DeepSeek
                
                # Google models
                "google/gemma-2-2b-it",                 # Compact instruction model
            ],
            
            # CODE GENERATION - 2025 code-capable chat models
            "code_generation": [
                # Qwen code models
                "Qwen/Qwen2.5-Coder-7B-Instruct",       # Code specialist
                "Qwen/Qwen2.5-Coder-32B-Instruct",      # Advanced code (if available)
                
                # Microsoft Phi for code
                "microsoft/Phi-3-mini-4k-instruct",      # Good for code
                "microsoft/Phi-4-mini-instruct",         # Latest, code-capable
                
                # DeepSeek code variants
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # Reasoning for code
                
                # Meta Llama for code
                "meta-llama/Llama-3.1-8B-Instruct",     # Code capable
            ],
            
            # GENERAL PURPOSE - Multi-task models
            "general_purpose": [
                "microsoft/Phi-3-mini-4k-instruct",      # Versatile
                "Qwen/Qwen2.5-7B-Instruct",             # Multi-task
                "meta-llama/Llama-3.1-8B-Instruct",     # General purpose
                "google/gemma-2-2b-it",                 # Compact general
            ],
            
            # REASONING - Models with strong reasoning
            "reasoning": [
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   # Reasoning focused
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # Efficient reasoning
                "microsoft/Phi-4-mini-instruct",              # Strong reasoning
            ],
            
            # FALLBACK MODELS - Most reliable 2025 options
            "fallback": [
                "microsoft/Phi-3-mini-4k-instruct",      # Most reliable
                "Qwen/Qwen2.5-1.5B-Instruct",           # Efficient backup
                "google/gemma-2-2b-it",                 # Compact fallback
            ]
        }
        
        # Initialize model caller
        model_caller = ModelCaller()
        provider_ready = await model_caller._ensure_provider_initialized()
        
        if not provider_ready or not model_caller._ai_provider:
            logger.error("❌ Failed to initialize AI provider")
            return {"error": "Provider initialization failed"}
        
        logger.info("✅ AI provider initialized successfully")
        
        # Test each category
        for category, models in model_candidates.items():
            logger.info(f"\n🧪 Testing {category.upper()} models...")
            category_working = []
            category_failed = []
            
            for model_id in models:
                logger.info(f"   Testing: {model_id}")
                
                try:
                    # All 2025 models are chat-compatible - use simple test prompt
                    test_prompt = "Hello, how are you?"
                    
                    # Test with chat completion format (2025 Inference Providers API)
                    messages = [ChatMessage(role="user", content=test_prompt)]
                    request = ChatCompletionRequest(
                        messages=messages,
                        model=model_id,
                        max_tokens=20,  # Increased for better response
                        temperature=0.7
                    )
                    
                    start_time = time.time()
                    response = await model_caller._ai_provider.chat_completion(request)
                    response_time = time.time() - start_time
                    
                    if response.success and response.content:
                        logger.info(f"   ✅ {model_id} - WORKING")
                        category_working.append({
                            "model": model_id,
                            "response": response.content[:50] + "..." if len(response.content) > 50 else response.content,
                            "response_time": response_time
                        })
                    else:
                        logger.info(f"   ❌ {model_id} - FAILED: {response.error_message}")
                        category_failed.append({
                            "model": model_id,
                            "error": response.error_message
                        })
                
                except Exception as e:
                    logger.info(f"   ❌ {model_id} - ERROR: {str(e)[:60]}...")
                    category_failed.append({
                        "model": model_id,
                        "error": str(e)[:100]
                    })
                
                # Rate limiting
                await asyncio.sleep(1)
            
            self.working_models[category] = category_working
            self.failed_models[category] = category_failed
            
            logger.info(f"   📊 {category}: {len(category_working)} working, {len(category_failed)} failed")
        
        # Summary
        total_working = sum(len(models) for models in self.working_models.values())
        total_tested = sum(len(models) + len(failed) for models, failed in zip(self.working_models.values(), self.failed_models.values()))
        
        logger.info(f"\n📈 DISCOVERY COMPLETE: {total_working}/{total_tested} models working")
        
        return {
            "working_models": self.working_models,
            "failed_models": self.failed_models,
            "total_working": total_working,
            "total_tested": total_tested,
            "success_rate": (total_working / total_tested * 100) if total_tested > 0 else 0
        }
    
    async def save_results(self, results: Dict[str, Any]):
        """Save discovery results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"working_models_discovery_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "discovery_results": results,
                "working_models_by_category": self.working_models,
                "failed_models_by_category": self.failed_models
            }, f, indent=2)
        
        logger.info(f"💾 Results saved to: {filename}")
        return filename

async def main():
    """Run the working model discovery"""
    discovery = WorkingModelDiscovery()
    
    try:
        results = await discovery.test_model_candidates()
        filename = await discovery.save_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 WORKING MODEL DISCOVERY COMPLETE")
        print("="*60)
        
        if results.get("error"):
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"📊 Total Models Tested: {results['total_tested']}")
        print(f"✅ Working Models: {results['total_working']}")
        print(f"📈 Success Rate: {results['success_rate']:.1f}%")
        print(f"💾 Results saved to: {filename}")
        
        # Show working models by category
        working_models = results["working_models"]
        for category, models in working_models.items():
            if models:
                print(f"\n🔧 {category.upper()}:")
                for model_info in models:
                    print(f"   ✅ {model_info['model']}")
        
        if results['total_working'] == 0:
            print("\n⚠️  NO WORKING MODELS FOUND")
            print("This suggests either:")
            print("- HF_TOKEN is not configured properly")
            print("- Free tier access has been restricted")
            print("- API endpoint has changed")
            print("- Need to use different model format/API")
        
    except Exception as e:
        logger.error(f"❌ Discovery failed: {e}")
        print(f"\n❌ Discovery failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())