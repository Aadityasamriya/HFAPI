"""
AI Assistant Pro - Sophisticated Telegram Bot
Professional AI orchestrator with intelligent model routing

Features:
- Latest 2024-2025 AI models (FLUX.1, StarCoder2-15B, Llama-3.2, Qwen2.5)
- Intelligent model routing and selection
- Secure encrypted API key storage  
- Advanced error handling and retry logic
- Production-ready architecture

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant Pro Team"
__description__ = "Advanced Telegram bot with intelligent AI routing"

# Core components
from . import config, security_utils, storage_manager
from .core import router, model_caller  
from .handlers import command_handlers, message_handlers

__all__ = [
    'config',
    'storage_manager', 
    'security_utils',
    'router',
    'model_caller',
    'command_handlers',
    'message_handlers'
]