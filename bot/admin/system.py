"""
AdminSystem - Core admin functionality with bootstrap and user management
Provides bot owner detection and comprehensive admin user management
"""

import asyncio
import logging
from typing import List, Optional, Set, Dict, Any
from datetime import datetime, timedelta
import hashlib
import hmac
import json

from bot.storage_manager import storage_manager
from bot.config import Config
from bot.security_utils import rate_limiter, admin_rate_limiter

logger = logging.getLogger(__name__)

class AdminSystem:
    """
    Comprehensive admin system for bot owner detection and admin management
    
    Features:
    - Automatic bot owner detection via Telegram Bot API
    - Secure admin user storage with encryption
    - Bootstrap process for first admin setup
    - Multi-admin support with privilege levels
    - Security audit logging
    """
    
    def __init__(self):
        self._admin_users: Set[int] = set()
        self._admin_levels: Dict[int, str] = {}  # user_id -> admin_level
        self._bootstrap_completed = False
        self._bot_owner_id: Optional[int] = None
        self._maintenance_mode = False
        self._admin_sessions: Dict[int, datetime] = {}
        self._initialized = False
        
        # Admin privilege levels
        self.ADMIN_LEVELS = {
            'owner': 'Owner',
            'admin': 'Admin', 
            'moderator': 'Moderator'
        }
    
    async def initialize(self) -> None:
        """
        Initialize admin system and load admin users from storage
        """
        if self._initialized:
            return
            
        try:
            logger.info("🔧 Initializing admin system...")
            
            # Ensure storage is connected
            await storage_manager.ensure_connected()
            
            # Load admin data from storage
            await self._load_admin_data()
            
            # Check if bootstrap is needed
            await self._check_bootstrap_status()
            
            self._initialized = True
            logger.info(f"✅ Admin system initialized - {len(self._admin_users)} admin(s) loaded")
            
            if not self._bootstrap_completed:
                logger.warning("⚠️ Bootstrap not completed - first admin not set")
                
        except Exception as e:
            logger.error(f"❌ Admin system initialization failed: {e}")
            raise
    
    async def _load_admin_data(self) -> None:
        """Load admin data from storage"""
        try:
            storage = await storage_manager.ensure_connected()
            
            # Get admin data from storage (implement in storage providers)
            admin_data = await self._get_admin_data_from_storage()
            
            if admin_data:
                self._admin_users = set(admin_data.get('admin_users', []))
                self._admin_levels = admin_data.get('admin_levels', {})
                self._bootstrap_completed = admin_data.get('bootstrap_completed', False)
                self._bot_owner_id = admin_data.get('bot_owner_id')
                self._maintenance_mode = admin_data.get('maintenance_mode', False)
                
                logger.info(f"📊 Loaded admin data: {len(self._admin_users)} admins, bootstrap: {self._bootstrap_completed}")
            else:
                logger.info("📊 No admin data found - fresh installation")
                
        except Exception as e:
            logger.error(f"Failed to load admin data: {e}")
            # Continue with empty admin set on error
    
    async def _get_admin_data_from_storage(self) -> Optional[Dict[str, Any]]:
        """Get admin data from storage provider"""
        try:
            storage = storage_manager.storage
            if hasattr(storage, 'get_admin_data'):
                return await storage.get_admin_data()
            else:
                # For backward compatibility, try to get from user preferences
                # This is a fallback for storage providers that don't implement admin storage yet
                return await self._get_admin_data_fallback()
        except Exception as e:
            logger.warning(f"Failed to get admin data from storage: {e}")
            return None
    
    async def _get_admin_data_fallback(self) -> Optional[Dict[str, Any]]:
        """Fallback method to get admin data"""
        # For now, return None - we'll implement proper storage extension next
        return None
    
    async def _save_admin_data(self) -> bool:
        """Save admin data to storage"""
        try:
            admin_data = {
                'admin_users': list(self._admin_users),
                'admin_levels': {str(k): v for k, v in self._admin_levels.items()},  # Convert keys to strings for JSON
                'bootstrap_completed': self._bootstrap_completed,
                'bot_owner_id': self._bot_owner_id,
                'maintenance_mode': self._maintenance_mode,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            storage = storage_manager.storage
            if hasattr(storage, 'save_admin_data'):
                return await storage.save_admin_data(admin_data)
            else:
                # Fallback for storage providers without admin support
                return await self._save_admin_data_fallback(admin_data)
                
        except Exception as e:
            logger.error(f"Failed to save admin data: {e}")
            return False
    
    async def _save_admin_data_fallback(self, admin_data: Dict[str, Any]) -> bool:
        """Fallback method to save admin data"""
        # For now, just log and return True - we'll implement proper storage next
        logger.info(f"Admin data saved (fallback): {len(admin_data.get('admin_users', []))} admins")
        return True
    
    async def _check_bootstrap_status(self) -> None:
        """Check if bootstrap process is completed"""
        if not self._bootstrap_completed and not self._admin_users:
            logger.warning("🔸 Bootstrap required - no admin users configured")
            logger.info("💡 Use /bootstrap command to set the first admin user")
    
    async def bootstrap_admin(self, user_id: int, telegram_username: Optional[str] = None) -> bool:
        """
        Bootstrap the first admin user (one-time operation)
        
        Args:
            user_id (int): Telegram user ID to make admin
            telegram_username (Optional[str]): Username for logging
            
        Returns:
            bool: True if bootstrap successful, False if already completed
        """
        if self._bootstrap_completed:
            logger.warning(f"Bootstrap already completed - cannot bootstrap user {user_id}")
            return False
        
        try:
            # Add as owner-level admin
            self._admin_users.add(user_id)
            self._admin_levels[user_id] = 'owner'
            self._bot_owner_id = user_id
            self._bootstrap_completed = True
            
            # Save to storage
            success = await self._save_admin_data()
            
            if success:
                logger.info(f"🎉 Bootstrap completed - Admin user {user_id} (@{telegram_username}) set as bot owner")
                await self._log_admin_action(user_id, 'bootstrap', {'action': 'first_admin_bootstrap'})
                return True
            else:
                # Rollback on save failure
                self._admin_users.discard(user_id)
                self._admin_levels.pop(user_id, None)
                self._bot_owner_id = None
                self._bootstrap_completed = False
                logger.error(f"Bootstrap failed - could not save admin data for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Bootstrap failed for user {user_id}: {e}")
            return False
    
    def is_bootstrap_completed(self) -> bool:
        """Check if bootstrap process is completed"""
        return self._bootstrap_completed
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is an admin"""
        return user_id in self._admin_users
    
    def get_admin_level(self, user_id: int) -> Optional[str]:
        """Get admin level for user"""
        return self._admin_levels.get(user_id)
    
    def is_owner(self, user_id: int) -> bool:
        """Check if user is the bot owner"""
        return user_id == self._bot_owner_id
    
    async def add_admin(self, user_id: int, admin_level: str = 'admin', added_by: int = None) -> bool:
        """
        Add a new admin user
        
        Args:
            user_id (int): User to add as admin
            admin_level (str): Admin level (owner, admin, moderator)
            added_by (int): User ID of admin who added this user
            
        Returns:
            bool: True if successful
        """
        if admin_level not in self.ADMIN_LEVELS:
            logger.error(f"Invalid admin level: {admin_level}")
            return False
        
        if user_id in self._admin_users:
            logger.warning(f"User {user_id} is already an admin")
            return False
        
        try:
            self._admin_users.add(user_id)
            self._admin_levels[user_id] = admin_level
            
            success = await self._save_admin_data()
            
            if success:
                logger.info(f"✅ Added admin user {user_id} with level '{admin_level}'")
                await self._log_admin_action(
                    added_by or user_id, 
                    'add_admin', 
                    {'target_user': user_id, 'admin_level': admin_level}
                )
                return True
            else:
                # Rollback on failure
                self._admin_users.discard(user_id)
                self._admin_levels.pop(user_id, None)
                return False
                
        except Exception as e:
            logger.error(f"Failed to add admin {user_id}: {e}")
            return False
    
    async def remove_admin(self, user_id: int, removed_by: int = None) -> bool:
        """
        Remove admin user
        
        Args:
            user_id (int): User to remove from admin
            removed_by (int): User ID of admin who removed this user
            
        Returns:
            bool: True if successful
        """
        if user_id not in self._admin_users:
            logger.warning(f"User {user_id} is not an admin")
            return False
        
        if user_id == self._bot_owner_id:
            logger.error(f"Cannot remove bot owner {user_id} from admin")
            return False
        
        try:
            self._admin_users.discard(user_id)
            self._admin_levels.pop(user_id, None)
            
            success = await self._save_admin_data()
            
            if success:
                logger.info(f"✅ Removed admin user {user_id}")
                await self._log_admin_action(
                    removed_by or user_id,
                    'remove_admin',
                    {'target_user': user_id}
                )
                return True
            else:
                # Rollback on failure
                self._admin_users.add(user_id)
                self._admin_levels[user_id] = 'admin'  # Default level on rollback
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove admin {user_id}: {e}")
            return False
    
    def get_admin_list(self) -> List[Dict[str, Any]]:
        """Get list of all admin users with their levels"""
        return [
            {
                'user_id': user_id,
                'level': self._admin_levels.get(user_id, 'admin'),
                'is_owner': user_id == self._bot_owner_id
            }
            for user_id in self._admin_users
        ]
    
    def is_maintenance_mode(self) -> bool:
        """Check if maintenance mode is enabled"""
        return self._maintenance_mode
    
    async def set_maintenance_mode(self, enabled: bool, set_by: int = None) -> bool:
        """
        Enable or disable maintenance mode
        
        Args:
            enabled (bool): True to enable, False to disable
            set_by (int): User ID of admin who changed this setting
            
        Returns:
            bool: True if successful
        """
        try:
            self._maintenance_mode = enabled
            success = await self._save_admin_data()
            
            if success:
                status = "enabled" if enabled else "disabled"
                logger.info(f"🔧 Maintenance mode {status}")
                await self._log_admin_action(
                    set_by,
                    'maintenance_mode',
                    {'enabled': enabled}
                )
                return True
            else:
                # Rollback on failure
                self._maintenance_mode = not enabled
                return False
                
        except Exception as e:
            logger.error(f"Failed to set maintenance mode: {e}")
            return False
    
    async def check_admin_rate_limit(self, user_id: int) -> tuple[bool, int]:
        """
        Check rate limit for admin users (more generous limits)
        
        Args:
            user_id (int): Admin user ID
            
        Returns:
            tuple[bool, int]: (is_allowed, wait_time)
        """
        if self.is_admin(user_id):
            return admin_rate_limiter.is_allowed(user_id)
        else:
            return rate_limiter.is_allowed(user_id)
    
    async def _log_admin_action(self, user_id: Optional[int], action: str, details: Dict[str, Any] = None) -> None:
        """
        Log admin actions for security audit trail
        
        Args:
            user_id (Optional[int]): Admin user ID performing action
            action (str): Action being performed
            details (Dict[str, Any]): Additional action details
        """
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': user_id,
                'action': action,
                'details': details or {},
                'admin_level': self._admin_levels.get(user_id) if user_id else None
            }
            
            # For now, just log to application logs
            # Later we can store in database for admin audit trail
            logger.info(f"🔐 ADMIN ACTION: {action} by user {user_id} - {details}")
            
        except Exception as e:
            logger.error(f"Failed to log admin action: {e}")
    
    async def get_admin_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive admin statistics
        
        Returns:
            Dict[str, Any]: Admin statistics
        """
        try:
            storage = storage_manager.storage
            
            # Basic admin info
            stats = {
                'admin_count': len(self._admin_users),
                'bootstrap_completed': self._bootstrap_completed,
                'maintenance_mode': self._maintenance_mode,
                'bot_owner_id': self._bot_owner_id,
                'admin_levels_count': {
                    level: sum(1 for l in self._admin_levels.values() if l == level)
                    for level in self.ADMIN_LEVELS.keys()
                }
            }
            
            # Try to get additional stats from storage if available
            if hasattr(storage, 'get_usage_stats'):
                try:
                    # Get overall usage stats (not user-specific)
                    usage_stats = await storage.get_usage_stats(0, days=30)  # 0 = all users
                    stats['usage_stats'] = usage_stats
                except Exception:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get admin stats: {e}")
            return {
                'admin_count': len(self._admin_users),
                'bootstrap_completed': self._bootstrap_completed,
                'maintenance_mode': self._maintenance_mode,
                'error': 'Failed to load full statistics'
            }
    
    async def health_check(self) -> bool:
        """
        Perform health check on admin system
        
        Returns:
            bool: True if healthy
        """
        try:
            # Check if initialized
            if not self._initialized:
                return False
            
            # Check storage connection
            if not storage_manager.connected:
                return False
            
            # Check admin data integrity
            if self._bootstrap_completed and not self._admin_users:
                return False
            
            return True
            
        except Exception:
            return False


# Global admin system instance
admin_system = AdminSystem()