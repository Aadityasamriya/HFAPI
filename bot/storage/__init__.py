"""
Storage Provider System for AI Assistant Telegram Bot
Flexible interface for multiple database backends including MongoDB, Supabase, and more
"""

from .base import StorageProvider
from .factory import create_storage_provider

__all__ = ['StorageProvider', 'create_storage_provider']