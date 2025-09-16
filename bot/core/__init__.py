"""
Core AI routing and model calling functionality

This module contains the intelligent routing system that analyzes user prompts
and selects the optimal AI model from our curated collection of 2024-2025 models.

Components:
- IntelligentRouter: Analyzes prompts and routes to optimal models
- ModelCaller: Handles all Hugging Face API interactions with retry logic
"""

from .router import IntelligentRouter, IntentType, router
from .model_caller import ModelCaller, model_caller

__all__ = [
    'IntelligentRouter', 
    'IntentType',
    'router',
    'ModelCaller',
    'model_caller'
]