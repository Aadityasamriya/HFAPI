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
        
        # Carefully selected candidates based on free tier research
        model_candidates = {
            # TEXT GENERATION - Lightweight models most likely to work on free tier
            "text_generation": [
                "gpt2",                              # Original GPT-2 - free tier standard
                "distilgpt2",                        # Distilled GPT-2 - lightweight
                "microsoft/DialoGPT-small",          # Small conversational model
                "microsoft/DialoGPT-medium",         # Medium conversational model  
                "EleutherAI/gpt-neo-125m",          # Small GPT-Neo
                "EleutherAI/gpt-neo-1.3B",          # Larger GPT-Neo (might work)
                "facebook/blenderbot_small-90M",     # Small BlenderBot
                "microsoft/DialoGPT-large",          # Large conversational (might work)
            ],
            
            # CODE GENERATION - Smaller code models for free tier
            "code_generation": [
                "microsoft/CodeBERT-base",           # CodeBERT base model
                "huggingface/CodeBERTa-small-v1",   # Small CodeBERTa
                "microsoft/codebert-base-mlm",       # Masked language model for code
                "microsoft/graphcodebert-base",      # Graph-based code model
                "EleutherAI/gpt-neo-125m",          # Can be used for simple code
                "gpt2",                              # GPT-2 for basic code tasks
            ],
            
            # QUESTION ANSWERING - Dedicated QA models
            "question_answering": [
                "distilbert-base-cased-distilled-squad", # DistilBERT for QA
                "bert-large-uncased-whole-word-masking-finetuned-squad", # BERT QA
                "deepset/roberta-base-squad2",        # RoBERTa for QA
                "distilbert-base-uncased-distilled-squad", # Efficient QA
            ],
            
            # SUMMARIZATION - Specialized summarization models
            "summarization": [
                "facebook/bart-large-cnn",           # BART for summarization
                "sshleifer/distilbart-cnn-12-6",    # Distilled BART
                "google/pegasus-xsum",               # Pegasus for abstractive summarization
                "facebook/bart-base",                # Base BART model
            ],
            
            # SENTIMENT ANALYSIS - Classification models
            "sentiment": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest", # Twitter sentiment
                "distilbert-base-uncased-finetuned-sst-2-english", # SST-2 sentiment
                "j-hartmann/emotion-english-distilroberta-base",    # Emotion detection
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",   # Multilingual sentiment
            ],
            
            # TEXT CLASSIFICATION - General classification
            "text_classification": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "distilbert-base-uncased-finetuned-sst-2-english",
                "facebook/bart-large-mnli",          # Natural language inference
            ],
            
            # TRANSLATION - Translation models
            "translation": [
                "Helsinki-NLP/opus-mt-en-fr",       # English to French
                "Helsinki-NLP/opus-mt-en-de",       # English to German
                "Helsinki-NLP/opus-mt-fr-en",       # French to English
                "Helsinki-NLP/opus-mt-de-en",       # German to English
            ],
            
            # IMAGE MODELS - Vision models that might work
            "image": [
                "google/vit-base-patch16-224",       # Vision Transformer
                "microsoft/resnet-50",               # ResNet for classification
                "openai/clip-vit-base-patch32",     # CLIP model
                "nlpconnect/vit-gpt2-image-captioning", # Image captioning
            ],
            
            # FALLBACK MODELS - Ultra-reliable fallbacks
            "fallback": [
                "gpt2",                              # Most reliable
                "distilgpt2",                        # Lightweight reliable
                "distilbert-base-uncased",           # BERT-based fallback
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
                    # Simple test prompt appropriate for the model type
                    if category in ["text_generation", "code_generation", "fallback"]:
                        test_prompt = "Hello"
                    elif category == "question_answering":
                        # QA models need context and question format
                        continue  # Skip QA for now as they need special format
                    elif category == "summarization":
                        # Summarization models need text to summarize
                        continue  # Skip summarization for now as they need special format
                    elif category in ["sentiment", "text_classification"]:
                        # Classification models need text input
                        continue  # Skip classification for now as they need special format
                    elif category == "translation":
                        # Translation models need special format
                        continue  # Skip translation for now as they need special format
                    elif category == "image":
                        # Image models need different API
                        continue  # Skip image for now as they need different format
                    else:
                        test_prompt = "Hello"
                    
                    # Test with chat completion format
                    messages = [ChatMessage(role="user", content=test_prompt)]
                    request = ChatCompletionRequest(
                        messages=messages,
                        model=model_id,
                        max_tokens=10,
                        temperature=0.5
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