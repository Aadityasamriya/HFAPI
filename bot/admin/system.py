"""
AdminSystem - Core admin functionality with bootstrap and user management
Provides bot owner detection and comprehensive admin user management
"""

import asyncio
import logging
from typing import List, Optional, Set, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
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
        self._admin_session_duration = timedelta(hours=8)  # 8-hour session timeout
        self._failed_admin_attempts: Dict[int, int] = defaultdict(int)
        self._admin_access_history: Dict[int, List[datetime]] = defaultdict(list)
        self._privilege_escalation_attempts: Dict[int, int] = defaultdict(int)
        self._initialized = False
        
        # Admin privilege levels
        self.ADMIN_LEVELS = {
            'owner': 'Owner',
            'admin': 'Admin', 
            'moderator': 'Moderator'
        }
        
        # Support for test mode
        self._test_mode_admins = set()  # Test-only admin users
    
    async def initialize(self) -> None:
        """
        Initialize admin system and load admin users from storage
        """
        if self._initialized:
            return
            
        try:
            logger.info("🔧 Initializing admin system...")
            
            # CRITICAL: Ensure ENCRYPTION_SEED is available before storage operations
            from bot.config import Config
            try:
                Config.ensure_encryption_seed()
                logger.info("🔐 ENCRYPTION_SEED verified for admin system")
            except ValueError as e:
                logger.error(f"❌ Admin system ENCRYPTION_SEED validation failed: {e}")
                raise RuntimeError(f"Admin system requires ENCRYPTION_SEED: {e}")
            
            # Ensure storage is connected
            await storage_manager.ensure_connected()
            
            # Load admin data from storage
            await self._load_admin_data()
            
            # Auto-bootstrap with OWNER_ID if configured and no bootstrap completed
            await self._auto_bootstrap_from_config()
            
            # Check if bootstrap is needed
            await self._check_bootstrap_status()
            
            self._initialized = True
            logger.info(f"✅ Admin system initialized - {len(self._admin_users)} admin(s) loaded")
            
            if not self._bootstrap_completed:
                logger.warning("⚠️ Bootstrap not completed - use /bootstrap command or set OWNER_ID environment variable")
                
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
                # Convert string keys back to integers for admin_levels
                admin_levels_raw = admin_data.get('admin_levels', {})
                self._admin_levels = {int(k): v for k, v in admin_levels_raw.items() if k.isdigit()} if admin_levels_raw else {}
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
            if storage is not None and hasattr(storage, 'get_admin_data'):
                return await storage.get_admin_data()
            else:
                # For backward compatibility, try to get from user preferences
                # This is a fallback for storage providers that don't implement admin storage yet
                return await self._get_admin_data_fallback()
        except Exception as e:
            logger.warning(f"Failed to get admin data from storage: {e}")
            return None
    
    async def _get_admin_data_fallback(self) -> Optional[Dict[str, Any]]:
        """Fallback method to get admin data using user preferences storage"""
        try:
            storage = storage_manager.storage
            if storage is not None and hasattr(storage, 'get_user_preference'):
                # Use a special system user ID for admin data storage
                admin_data_raw = await storage.get_user_preference(999999999, 'admin_system_data')
                if admin_data_raw:
                    import json
                    return json.loads(admin_data_raw)
            return None
        except Exception as e:
            logger.warning(f"Admin fallback get failed: {e}")
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
            if storage is not None and hasattr(storage, 'save_admin_data'):
                return await storage.save_admin_data(admin_data)
            else:
                # Fallback for storage providers without admin support
                return await self._save_admin_data_fallback(admin_data)
                
        except Exception as e:
            logger.error(f"Failed to save admin data: {e}")
            return False
    
    async def _save_admin_data_fallback(self, admin_data: Dict[str, Any]) -> bool:
        """Fallback method to save admin data using user preferences storage"""
        try:
            storage = storage_manager.storage
            if storage is not None and hasattr(storage, 'save_user_preference'):
                # Use a special system user ID for admin data storage
                import json
                admin_data_json = json.dumps(admin_data)
                success = await storage.save_user_preference(999999999, 'admin_system_data', admin_data_json)
                if success:
                    logger.info(f"✅ Admin data saved (fallback): {len(admin_data.get('admin_users', []))} admins")
                    return True
                else:
                    logger.error("❌ Failed to save admin data via fallback method")
                    return False
            else:
                logger.warning("⚠️ No fallback storage method available for admin data")
                return False
        except Exception as e:
            logger.error(f"❌ Admin fallback save failed: {e}")
            return False
    
    async def _auto_bootstrap_from_config(self) -> None:
        """Auto-bootstrap admin user from OWNER_ID environment variable if configured"""
        if self._bootstrap_completed:
            return
            
        if Config.OWNER_ID and Config.OWNER_ID > 0:
            logger.info(f"🔐 Auto-bootstrapping admin system with OWNER_ID: {Config.OWNER_ID}")
            
            try:
                success = await self.bootstrap_admin(Config.OWNER_ID, "Environment OWNER_ID")
                if success:
                    logger.info(f"✅ Successfully auto-bootstrapped owner from OWNER_ID environment variable")
                else:
                    logger.warning(f"⚠️ Failed to auto-bootstrap owner from OWNER_ID environment variable")
            except Exception as e:
                logger.error(f"❌ Error during auto-bootstrap from OWNER_ID: {e}")
    
    async def _check_bootstrap_status(self) -> None:
        """Check if bootstrap process is completed"""
        if not self._bootstrap_completed and not self._admin_users:
            logger.warning("🔸 Bootstrap required - no admin users configured")
            if Config.OWNER_ID:
                logger.info("💡 OWNER_ID is configured but bootstrap failed - check logs above")
            else:
                logger.info("💡 Set OWNER_ID environment variable or use /bootstrap command to set the first admin user")
    
    async def bootstrap_admin(self, user_id: int, telegram_username: Optional[str] = None) -> bool:
        """
        Bootstrap the first admin user (one-time operation)
        SECURITY FIX C7: Atomic check-and-set to prevent race conditions
        
        Args:
            user_id (int): Telegram user ID to make admin
            telegram_username (Optional[str]): Username for logging
            
        Returns:
            bool: True if bootstrap successful, False if already completed
        """
        # SECURITY FIX C7: Atomic check-and-set using asyncio lock
        # Create lock if it doesn't exist (lazy initialization)
        if not hasattr(self, '_bootstrap_lock'):
            self._bootstrap_lock = asyncio.Lock()
        
        # Acquire lock for atomic operation
        async with self._bootstrap_lock:
            # Check again inside lock to prevent race condition
            if self._bootstrap_completed:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.warning(f"🔒 Bootstrap already completed (atomic check) - cannot bootstrap user_hash {user_hash}")
                return False
            
            try:
                # Add as owner-level admin
                self._admin_users.add(user_id)
                self._admin_levels[user_id] = 'owner'
                self._bot_owner_id = user_id
                self._bootstrap_completed = True
                
                # CRITICAL FIX: Initialize admin session for bootstrapped owner
                self._admin_sessions[user_id] = datetime.utcnow()
                logger.info(f"🔐 Admin session initialized for bootstrapped owner")
                
                # Save to storage
                success = await self._save_admin_data()
                
                if success:
                    user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                    logger.info(f"🎉 Bootstrap completed (atomic) - Admin user_hash {user_hash} (@{telegram_username}) set as bot owner")
                    await self._log_admin_action(user_id, 'bootstrap', {'action': 'first_admin_bootstrap'})
                    return True
                else:
                    # Rollback on save failure
                    self._admin_users.discard(user_id)
                    self._admin_levels.pop(user_id, None)
                    self._bot_owner_id = None
                    self._bootstrap_completed = False
                    self._admin_sessions.pop(user_id, None)  # Clean up session on rollback
                    user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                    logger.error(f"Bootstrap failed - could not save admin data for user_hash {user_hash}")
                    return False
                    
            except Exception as e:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.error(f"Bootstrap failed for user_hash {user_hash}: {e}")
                return False
    
    def is_bootstrap_completed(self) -> bool:
        """Check if bootstrap process is completed"""
        return self._bootstrap_completed
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is an admin with session validation"""
        # Support test mode admins
        from bot.config import Config
        if Config.is_test_mode() and user_id in self._test_mode_admins:
            return True
            
        if user_id not in self._admin_users:
            return False
        
        # CRITICAL FIX: For the owner, auto-create session if missing (handles fresh bootstrap)
        if user_id == self._bot_owner_id and user_id not in self._admin_sessions:
            self._admin_sessions[user_id] = datetime.utcnow()
            logger.info(f"🔐 Auto-initialized session for bot owner after bootstrap")
        
        # Check session validity
        if not self._is_session_valid(user_id):
            # For existing admins, try to refresh session automatically
            if user_id in self._admin_users:
                self._admin_sessions[user_id] = datetime.utcnow()
                logger.info(f"🔄 Auto-refreshed admin session for existing admin")
                return True
            else:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.warning(f"SECURITY: Admin session expired for user_hash {user_hash}")
                return False
        
        return True
    
    def get_admin_level(self, user_id: int) -> Optional[str]:
        """Get admin level for user"""
        # Support test mode admins  
        from bot.config import Config
        if Config.is_test_mode() and user_id in self._test_mode_admins:
            return 'owner'  # Test admins get owner level
        return self._admin_levels.get(user_id)
    
    def add_admin_for_testing(self, user_id: int) -> None:
        """Add admin user for testing purposes only"""
        from bot.config import Config
        if Config.is_test_mode():
            self._test_mode_admins.add(user_id)
            logger.info(f"🔐 Test admin added: user_id {user_id}")
        else:
            logger.warning("add_admin_for_testing called outside test mode")
    
    def is_owner(self, user_id: int) -> bool:
        """Check if user is the bot owner"""
        # Support test mode admins as owners
        from bot.config import Config
        if Config.is_test_mode() and user_id in self._test_mode_admins:
            return True
        return user_id == self._bot_owner_id
    
    async def add_admin(self, user_id: int, admin_level: str = 'admin', added_by: Optional[int] = None) -> bool:
        """
        Add a new admin user with enhanced privilege escalation protection
        
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
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.warning(f"User_hash {user_hash} is already an admin")
            return False
        
        # Enhanced privilege escalation protection
        if added_by is not None:
            # Verify the person adding is still an admin with valid session
            if not self.is_admin(added_by):
                self._privilege_escalation_attempts[added_by] += 1
                added_by_hash = hashlib.sha256(f"{added_by}".encode()).hexdigest()[:8]
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.critical(f"SECURITY: Privilege escalation attempt - user_hash {added_by_hash} tried to add admin {user_hash} without valid admin status")
                return False
            
            adder_level = self.get_admin_level(added_by)
            
            # Prevent privilege escalation: can't grant higher or equal privileges
            level_hierarchy = {'moderator': 1, 'admin': 2, 'owner': 3}
            adder_rank = level_hierarchy.get(adder_level, 0) if adder_level is not None else 0
            target_rank = level_hierarchy.get(admin_level, 0)
            
            if target_rank >= adder_rank:
                self._privilege_escalation_attempts[added_by] += 1
                added_by_hash = hashlib.sha256(f"{added_by}".encode()).hexdigest()[:8]
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.critical(f"SECURITY: Privilege escalation attempt - {adder_level} user_hash {added_by_hash} tried to grant {admin_level} level to {user_hash}")
                return False
            
            # Only owners can create other admins
            if admin_level == 'admin' and adder_level != 'owner':
                self._privilege_escalation_attempts[added_by] += 1
                added_by_hash = hashlib.sha256(f"{added_by}".encode()).hexdigest()[:8]
                logger.critical(f"SECURITY: Privilege escalation attempt - Only owners can create admin-level users. {adder_level} user_hash {added_by_hash} denied.")
                return False
        
        try:
            self._admin_users.add(user_id)
            self._admin_levels[user_id] = admin_level
            
            # Initialize session for new admin
            self._admin_sessions[user_id] = datetime.utcnow()
            
            success = await self._save_admin_data()
            
            if success:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                added_by_hash = hashlib.sha256(f"{added_by}".encode()).hexdigest()[:8] if added_by else 'system'
                logger.info(f"✅ Added admin user_hash {user_hash} with level '{admin_level}' by {added_by_hash}")
                await self._log_admin_action(
                    added_by if added_by is not None else user_id, 
                    'add_admin', 
                    {
                        'target_user': user_id, 
                        'admin_level': admin_level,
                        'adder_level': self.get_admin_level(added_by) if added_by else 'system'
                    }
                )
                return True
            else:
                # Rollback on failure
                self._admin_users.discard(user_id)
                self._admin_levels.pop(user_id, None)
                self._admin_sessions.pop(user_id, None)
                return False
                
        except Exception as e:
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.error(f"Failed to add admin user_hash {user_hash}: {e}")
            return False
    
    async def remove_admin(self, user_id: int, removed_by: Optional[int] = None) -> bool:
        """
        Remove admin user
        
        Args:
            user_id (int): User to remove from admin
            removed_by (int): User ID of admin who removed this user
            
        Returns:
            bool: True if successful
        """
        if user_id not in self._admin_users:
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.warning(f"User_hash {user_hash} is not an admin")
            return False
        
        if user_id == self._bot_owner_id:
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.error(f"Cannot remove bot owner user_hash {user_hash} from admin")
            return False
        
        try:
            self._admin_users.discard(user_id)
            self._admin_levels.pop(user_id, None)
            
            success = await self._save_admin_data()
            
            if success:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.info(f"✅ Removed admin user_hash {user_hash}")
                await self._log_admin_action(
                    removed_by if removed_by is not None else user_id,
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
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.error(f"Failed to remove admin user_hash {user_hash}: {e}")
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
    
    async def set_maintenance_mode(self, enabled: bool, set_by: Optional[int] = None) -> bool:
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
    
    def _is_session_valid(self, user_id: int) -> bool:
        """
        Check if admin session is still valid
        
        Args:
            user_id (int): Admin user ID
            
        Returns:
            bool: True if session is valid
        """
        if user_id not in self._admin_sessions:
            return False
        
        session_start = self._admin_sessions[user_id]
        current_time = datetime.utcnow()
        
        if current_time - session_start > self._admin_session_duration:
            # Session expired
            self._admin_sessions.pop(user_id, None)
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.info(f"Admin session expired for user_hash {user_hash}")
            return False
        
        return True
    
    def refresh_admin_session(self, user_id: int) -> bool:
        """
        Refresh admin session if user is still valid admin
        
        Args:
            user_id (int): Admin user ID
            
        Returns:
            bool: True if session refreshed successfully
        """
        if user_id not in self._admin_users:
            return False
        
        self._admin_sessions[user_id] = datetime.utcnow()
        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
        logger.info(f"Admin session refreshed for user_hash {user_hash}")
        return True
    
    async def check_admin_rate_limit(self, user_id: int) -> tuple[bool, int]:
        """
        Check rate limit for admin users with enhanced security
        
        Args:
            user_id (int): Admin user ID
            
        Returns:
            tuple[bool, int]: (is_allowed, wait_time)
        """
        # Track access attempts
        current_time = datetime.utcnow()
        self._admin_access_history[user_id].append(current_time)
        
        # Clean old access history (keep last hour)
        hour_ago = current_time - timedelta(hours=1)
        self._admin_access_history[user_id] = [
            access for access in self._admin_access_history[user_id] 
            if access > hour_ago
        ]
        
        # Check for suspicious admin activity (too many commands in short time)
        recent_accesses = len(self._admin_access_history[user_id])
        if recent_accesses > 100:  # More than 100 admin commands per hour is suspicious
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
            logger.warning(f"SECURITY: Suspicious admin activity from user_hash {user_hash} - {recent_accesses} commands in last hour")
            return False, 300  # 5 minute timeout for suspicious activity
        
        # CRITICAL FIX: Check admin status without triggering rate limit recursion
        if user_id in self._admin_users:
            is_allowed, wait_time = admin_rate_limiter.is_allowed(user_id)
            return is_allowed, wait_time if wait_time is not None else 0
        else:
            # Track failed admin access attempts
            self._failed_admin_attempts[user_id] += 1
            if self._failed_admin_attempts[user_id] > 5:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.warning(f"SECURITY: Multiple failed admin access attempts from user_hash {user_hash}")
                return False, 600  # 10 minute timeout for repeated failures
            
            is_allowed, wait_time = rate_limiter.is_allowed(user_id)
            return is_allowed, wait_time if wait_time is not None else 0
    
    async def _log_admin_action(self, user_id: Optional[int], action: str, details: Optional[Dict[str, Any]] = None) -> None:
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
            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8] if user_id else 'system'
            logger.info(f"🔐 ADMIN ACTION: {action} by user_hash {user_hash} - {details}")
            
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
            if storage is not None and hasattr(storage, 'get_usage_stats'):
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
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive security statistics
        
        Returns:
            Dict[str, Any]: Security statistics
        """
        current_time = datetime.utcnow()
        active_sessions = sum(1 for user_id in self._admin_sessions 
                             if self._is_session_valid(user_id))
        
        return {
            'total_admins': len(self._admin_users),
            'active_admin_sessions': active_sessions,
            'expired_sessions': len(self._admin_sessions) - active_sessions,
            'privilege_escalation_attempts': dict(self._privilege_escalation_attempts),
            'failed_admin_attempts': dict(self._failed_admin_attempts),
            'suspicious_admin_activity': {
                user_id: len(accesses) for user_id, accesses in self._admin_access_history.items()
                if len(accesses) > 50  # Show users with high activity
            },
            'admin_levels_distribution': {
                level: sum(1 for l in self._admin_levels.values() if l == level)
                for level in self.ADMIN_LEVELS.keys()
            }
        }
    
    async def cleanup_security_data(self) -> None:
        """
        Clean up old security tracking data
        """
        current_time = datetime.utcnow()
        
        # Clean expired sessions
        expired_sessions = [user_id for user_id in self._admin_sessions
                           if not self._is_session_valid(user_id)]
        for user_id in expired_sessions:
            self._admin_sessions.pop(user_id, None)
        
        # Reset failed attempts for users who haven't tried recently
        day_ago = current_time - timedelta(days=1)
        users_to_reset = []
        for user_id in self._failed_admin_attempts:
            if user_id not in self._admin_access_history or \
               not self._admin_access_history[user_id] or \
               max(self._admin_access_history[user_id]) < day_ago:
                users_to_reset.append(user_id)
        
        for user_id in users_to_reset:
            self._failed_admin_attempts.pop(user_id, None)
    
    async def health_check(self) -> bool:
        """
        Perform health check on admin system with security validation
        
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
            
            # Security check: ensure owner still exists
            if self._bootstrap_completed and self._bot_owner_id not in self._admin_users:
                logger.critical("SECURITY: Bot owner not found in admin list - potential security breach")
                return False
            
            # Clean up old security data
            await self.cleanup_security_data()
            
            return True
            
        except Exception:
            return False


# Global admin system instance
admin_system = AdminSystem()