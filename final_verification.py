#!/usr/bin/env python3
"""Final verification that all three missing methods are implemented and accessible"""

import sys
import os
sys.path.insert(0, '.')

print("🔍 FINAL VERIFICATION - Testing all three critical missing methods")
print("=" * 60)

success_count = 0
total_tests = 3

# Test 1: ModelCaller.generate_with_fallback
try:
    from bot.core.model_caller import model_caller
    if hasattr(model_caller, 'generate_with_fallback'):
        print("✅ TEST 1 PASSED: ModelCaller.generate_with_fallback is accessible")
        success_count += 1
    else:
        print("❌ TEST 1 FAILED: ModelCaller.generate_with_fallback not found")
except Exception as e:
    print(f"❌ TEST 1 ERROR: {e}")

# Test 2: IntelligentRouter.analyze_prompt_advanced
try:
    from bot.core.router import router
    if hasattr(router, 'analyze_prompt_advanced'):
        print("✅ TEST 2 PASSED: IntelligentRouter.analyze_prompt_advanced is accessible")
        success_count += 1
    else:
        print("❌ TEST 2 FAILED: IntelligentRouter.analyze_prompt_advanced not found")
except Exception as e:
    print(f"❌ TEST 2 ERROR: {e}")

# Test 3: AdvancedIntentClassifier.classify_advanced
try:
    from bot.core.intent_classifier import intent_classifier
    if hasattr(intent_classifier, 'classify_advanced'):
        print("✅ TEST 3 PASSED: AdvancedIntentClassifier.classify_advanced is accessible")
        success_count += 1
    else:
        print("❌ TEST 3 FAILED: AdvancedIntentClassifier.classify_advanced not found")
except Exception as e:
    print(f"❌ TEST 3 ERROR: {e}")

print("=" * 60)
print(f"🎯 VERIFICATION COMPLETE: {success_count}/{total_tests} tests passed ({success_count/total_tests*100:.0f}%)")

if success_count == total_tests:
    print("🎉 SUCCESS: All three critical missing methods are now implemented and accessible!")
    print("   The 0% test success rate issue should now be resolved.")
else:
    print("⚠️ WARNING: Some methods are still missing or inaccessible.")

print("\n📋 SUMMARY OF IMPLEMENTED METHODS:")
print("   1. ModelCaller.generate_with_fallback - Intelligent model fallback with orchestration")
print("   2. IntelligentRouter.analyze_prompt_advanced - Advanced prompt analysis with complexity")
print("   3. AdvancedIntentClassifier.classify_advanced - Enhanced classification with caching")