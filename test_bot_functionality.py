#!/usr/bin/env python3
"""
Test script for AI Assistant Pro Telegram Bot functionality
Tests all core components without requiring actual Telegram/API keys
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import bot components
from bot.config import Config
from bot.core.router import router, IntentType
from bot.core.model_caller import ModelCaller
from bot.database import Database, encryption_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotTester:
    """Comprehensive bot functionality tester"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all functionality tests"""
        print("🧪 Starting AI Assistant Pro Functionality Tests")
        print("=" * 60)
        
        # Test configuration
        await self.test_configuration()
        
        # Test encryption
        await self.test_encryption()
        
        # Test intelligent routing
        await self.test_router_intelligence()
        
        # Test model caller structure
        await self.test_model_caller_structure()
        
        # Test database structure (without actual MongoDB)
        await self.test_database_structure()
        
        # Print summary
        self.print_test_summary()
        
    async def test_configuration(self):
        """Test configuration management"""
        print("\n🔧 Testing Configuration Management...")
        
        try:
            # Test that OWNER_ID is now optional
            original_owner_id = Config.OWNER_ID
            Config.OWNER_ID = None
            
            # Should not raise an error now
            result = Config.validate_config()
            
            # Restore original value
            Config.OWNER_ID = original_owner_id
            
            self.test_results['config_optional_owner'] = True
            print("   ✅ OWNER_ID is now optional - PASSED")
            
        except Exception as e:
            self.test_results['config_optional_owner'] = False
            print(f"   ❌ Configuration test failed: {e}")
            
    async def test_encryption(self):
        """Test encryption/decryption functionality"""
        print("\n🔐 Testing Encryption System...")
        
        try:
            test_data = "hf_test_api_key_12345678901234567890"
            
            # Test encryption
            encrypted = encryption_manager.encrypt(test_data)
            print(f"   🔒 Encrypted data length: {len(encrypted)} chars")
            
            # Test decryption
            decrypted = encryption_manager.decrypt(encrypted)
            
            if decrypted == test_data:
                self.test_results['encryption'] = True
                print("   ✅ Encryption/Decryption - PASSED")
            else:
                self.test_results['encryption'] = False
                print("   ❌ Encryption/Decryption - FAILED: Data mismatch")
                
        except Exception as e:
            self.test_results['encryption'] = False
            print(f"   ❌ Encryption test failed: {e}")
            
    async def test_router_intelligence(self):
        """Test intelligent routing system"""
        print("\n🧠 Testing Intelligent Routing System...")
        
        test_cases = [
            ("Draw a beautiful sunset", IntentType.IMAGE_GENERATION),
            ("Create a Python function to sort data", IntentType.CODE_GENERATION),
            ("Analyze sentiment: I love this product!", IntentType.SENTIMENT_ANALYSIS),
            ("What is machine learning?", IntentType.QUESTION_ANSWERING),
            ("Write a story about dragons", IntentType.CREATIVE_WRITING),
            ("Hello, how are you?", IntentType.TEXT_GENERATION),
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for prompt, expected_intent in test_cases:
            try:
                detected_intent, routing_info = router.route_prompt(prompt)
                
                if detected_intent == expected_intent:
                    passed_tests += 1
                    print(f"   ✅ '{prompt[:30]}...' → {detected_intent.value}")
                else:
                    print(f"   ⚠️  '{prompt[:30]}...' → Expected: {expected_intent.value}, Got: {detected_intent.value}")
                    
            except Exception as e:
                print(f"   ❌ Router failed for '{prompt[:30]}...': {e}")
                
        self.test_results['routing'] = passed_tests / total_tests
        print(f"   📊 Routing Intelligence: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
    async def test_model_caller_structure(self):
        """Test model caller structure and methods"""
        print("\n🤖 Testing Model Caller Structure...")
        
        try:
            # Test async context manager
            async with ModelCaller() as mc:
                # Test that all required methods exist
                required_methods = [
                    'generate_text',
                    'generate_code', 
                    'generate_image',
                    'analyze_sentiment'
                ]
                
                missing_methods = []
                for method in required_methods:
                    if not hasattr(mc, method):
                        missing_methods.append(method)
                        
                if not missing_methods:
                    self.test_results['model_caller'] = True
                    print("   ✅ Model Caller Structure - PASSED")
                    print(f"   📋 All required methods present: {', '.join(required_methods)}")
                else:
                    self.test_results['model_caller'] = False
                    print(f"   ❌ Missing methods: {', '.join(missing_methods)}")
                    
        except Exception as e:
            self.test_results['model_caller'] = False
            print(f"   ❌ Model Caller test failed: {e}")
            
    async def test_database_structure(self):
        """Test database class structure"""
        print("\n🗄️  Testing Database Structure...")
        
        try:
            db = Database()
            
            # Test that all required methods exist
            required_methods = [
                'connect',
                'disconnect',
                'save_user_api_key',
                'get_user_api_key',
                'reset_user_database'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(db, method):
                    missing_methods.append(method)
                    
            if not missing_methods:
                self.test_results['database'] = True
                print("   ✅ Database Structure - PASSED")
                print(f"   📋 All required methods present: {', '.join(required_methods)}")
            else:
                self.test_results['database'] = False
                print(f"   ❌ Missing methods: {', '.join(missing_methods)}")
                
        except Exception as e:
            self.test_results['database'] = False
            print(f"   ❌ Database test failed: {e}")
            
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result is True else "❌ FAILED"
            if isinstance(result, float):
                status = f"📊 {result*100:.1f}% SUCCESS"
            print(f"   {test_name:<25} {status}")
            
        print("-" * 60)
        print(f"Overall Status: {passed_tests}/{total_tests} core tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL CORE FUNCTIONALITY TESTS PASSED!")
            print("✅ Bot is ready for deployment!")
        else:
            print("⚠️  Some tests had issues - check details above")
            
        print("\n🚀 Ready for Telegram Bot Token and MongoDB URI setup!")

async def main():
    """Main test runner"""
    tester = BotTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())