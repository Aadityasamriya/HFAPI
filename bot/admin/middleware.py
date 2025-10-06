"""
Admin middleware for security and access control
Provides decorators and utilities for admin-only functionality
"""

import asyncio
import logging
import hashlib
from functools import wraps
from typing import Callable, Any, Optional, Dict
from datetime import datetime

from telegram import Update
from telegram.ext import ContextTypes

from .system import admin_system
from bot.security_utils import escape_markdown, safe_markdown_format
from bot.config import Config

logger = logging.getLogger(__name__)

def admin_required(min_level: str = 'admin', allow_bootstrap: bool = False):
    """
    Decorator to require admin privileges for handler functions
    
    Args:
        min_level (str): Minimum admin level required ('owner', 'admin', 'moderator')
        allow_bootstrap (bool): Allow access if bootstrap not completed (for bootstrap command)
    
    Usage:
        @admin_required(min_level='owner')
        async def owner_only_command(update, context):
            pass
            
        @admin_required(allow_bootstrap=True)  # For bootstrap command
        async def bootstrap_command(update, context):
            pass
    """
    def decorator(handler_func: Callable) -> Callable:
        @wraps(handler_func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs) -> Any:
            user = update.effective_user
            if user is None:
                logger.error("No user found in update")
                return
            user_id = user.id
            username = user.username or "No username"
            
            try:
                # Ensure admin system is initialized
                if not admin_system._initialized:
                    await admin_system.initialize()
                
                # Check if bootstrap is needed and allowed
                if not admin_system.is_bootstrap_completed():
                    if allow_bootstrap:
                        # SECURITY ENFORCEMENT: Validate bootstrap authorization
                        # Only allow bootstrap if user is authorized via OWNER_ID or is first legitimate user
                        from bot.config import Config
                        
                        # Check if OWNER_ID is configured in environment
                        if Config.OWNER_ID and Config.OWNER_ID > 0:
                            # SECURITY: If OWNER_ID is set, ONLY that user can bootstrap
                            if user_id != Config.OWNER_ID:
                                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                                owner_hash = hashlib.sha256(f"{Config.OWNER_ID}".encode()).hexdigest()[:8]
                                logger.critical(f"🚨 SECURITY: Unauthorized bootstrap attempt by user_hash={user_hash} (expected owner_hash={owner_hash})")
                                
                                # Log security event for unauthorized bootstrap attempt
                                await AdminSecurityLogger.log_security_event(
                                    'unauthorized_bootstrap_attempt',
                                    {'user_id': user_id, 'username': username, 'expected_owner_id': Config.OWNER_ID}
                                )
                                
                                if update.message:
                                    await update.message.reply_text(
                                        "🚫 **Unauthorized Bootstrap Attempt**\n\n"
                                        "Bootstrap is restricted to the configured bot owner only.\n\n"
                                        "This incident has been logged.",
                                        parse_mode='Markdown'
                                    )
                                return
                        else:
                            # SECURITY WARNING: No OWNER_ID configured - allowing first-come-first-served
                            # This is a security risk in production environments
                            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                            logger.warning(f"⚠️ SECURITY: No OWNER_ID configured - allowing bootstrap by first user user_hash={user_hash}")
                            logger.warning(f"⚠️ SECURITY: Set OWNER_ID environment variable to restrict bootstrap to specific user")
                        
                        # Rate limiting for bootstrap attempts to prevent abuse
                        # Use admin rate limiter with stricter limits for bootstrap
                        from bot.security_utils import admin_rate_limiter
                        is_allowed, wait_time = await admin_system.check_admin_rate_limit(user_id)
                        if not is_allowed:
                            user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                            logger.warning(f"🚨 SECURITY: Bootstrap rate limit exceeded for user_hash={user_hash}")
                            
                            if update.message:
                                await update.message.reply_text(
                                    f"⚠️ **Rate Limit Exceeded**\n\n"
                                    f"Too many bootstrap attempts. Please wait {wait_time} seconds.",
                                    parse_mode='Markdown'
                                )
                            return
                        
                        # Log successful bootstrap access authorization
                        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                        logger.info(f"🔸 Bootstrap access granted for authorized user_hash={user_hash}")
                        
                        # Log security event for authorized bootstrap access
                        await AdminSecurityLogger.log_security_event(
                            'bootstrap_access_authorized',
                            {'user_id': user_id, 'username': username, 'owner_id_configured': bool(Config.OWNER_ID)}
                        )
                        
                        return await handler_func(update, context, *args, **kwargs)
                    else:
                        # In test mode, allow test admins to access commands even if bootstrap not completed
                        from bot.config import Config
                        if Config.is_test_mode() and user_id in getattr(admin_system, '_test_mode_admins', set()):
                            logger.info(f"🧪 Test mode admin access granted for user_hash={hashlib.sha256(f'{user_id}'.encode()).hexdigest()[:8]}")
                            return await handler_func(update, context, *args, **kwargs)
                        await _send_bootstrap_required_message(update)
                        return
                
                # CRITICAL FIX: Proper admin status validation with session management
                is_valid_admin = admin_system.is_admin(user_id)
                
                # If user is in admin list but session invalid, try to refresh
                if user_id in admin_system._admin_users and not is_valid_admin:
                    # Session may be expired - refresh it if user is still valid admin
                    refresh_success = admin_system.refresh_admin_session(user_id)
                    if refresh_success:
                        is_valid_admin = admin_system.is_admin(user_id)  # Re-check after refresh
                        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                        logger.info(f"🔄 Admin session renewed for user_hash={user_hash}")
                
                # Check final admin status after potential session renewal
                if not is_valid_admin:
                    user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                    logger.warning(f"🚫 Access denied: user_hash={user_hash} not an admin")
                    await _send_access_denied_message(update)
                    return
                
                # Check admin level
                user_level = admin_system.get_admin_level(user_id)
                if not _check_admin_level(user_level, min_level):
                    user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                    logger.warning(f"🚫 Insufficient privileges: user_hash={user_hash} level '{user_level}' < required '{min_level}'")
                    await _send_insufficient_privileges_message(update, min_level)
                    return
                
                # Refresh admin session on successful access
                admin_system.refresh_admin_session(user_id)
                
                # Check admin rate limit with enhanced security
                is_allowed, wait_time = await admin_system.check_admin_rate_limit(user_id)
                if not is_allowed:
                    if update.message:
                        await update.message.reply_text(
                            f"⚠️ **Admin Rate Limit Exceeded**\n\nSecurity protection activated. Please wait {wait_time} seconds before using admin commands.",
                            parse_mode='Markdown'
                        )
                    # Log suspicious activity
                    user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                    logger.warning(f"SECURITY: Admin rate limit triggered for user_hash={user_hash} on command {handler_func.__name__}")
                    return
                
                # Log admin access with privacy-safe structured logging
                handler_name = handler_func.__name__
                await log_admin_action(
                    user_id, 
                    'command_access', 
                    {
                        'command': handler_name,
                        'level_required': min_level,
                        'user_level': user_level
                    }
                )
                
                # SECURITY: Hash user_id for logging to prevent PII exposure
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.info(f"✅ Admin access granted: user_hash={user_hash} -> {handler_name}")
                
                # Execute the handler
                return await handler_func(update, context, *args, **kwargs)
                
            except Exception as e:
                user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                logger.error(f"Admin middleware error for user_hash {user_hash}: {e}")
                # Log potential security incident
                logger.warning(f"SECURITY: Admin middleware exception for user_hash={user_hash}: {e}")
                if update.message:
                    await update.message.reply_text(
                        "🚫 **Admin System Error**\n\nA security error occurred while checking admin privileges. This incident has been logged.",
                        parse_mode='Markdown'
                    )
                return
        
        return wrapper
    return decorator

def _check_admin_level(user_level: Optional[str], required_level: str) -> bool:
    """
    Check if user admin level meets requirement
    
    Args:
        user_level (Optional[str]): User's admin level
        required_level (str): Required minimum level
        
    Returns:
        bool: True if user meets requirement
    """
    if not user_level:
        return False
    
    # Admin level hierarchy (higher number = more privileges)
    level_hierarchy = {
        'moderator': 1,
        'admin': 2,
        'owner': 3
    }
    
    user_rank = level_hierarchy.get(user_level, 0)
    required_rank = level_hierarchy.get(required_level, 0)
    
    return user_rank >= required_rank

async def _send_access_denied_message(update: Update) -> None:
    """Send access denied message"""
    message = """
🚫 **Access Denied**

This command is restricted to bot administrators only.

If you believe you should have admin access, contact the bot owner.
    """
    
    if update.message:
        await update.message.reply_text(
            message.strip(),
            parse_mode='Markdown'
        )

async def _send_insufficient_privileges_message(update: Update, required_level: str) -> None:
    """Send insufficient privileges message"""
    safe_level = escape_markdown(required_level.title())
    message = f"""
🚫 **Insufficient Privileges**

This command requires **{safe_level}** level access\\.

Contact a higher\\-level administrator if you need elevated access\\.
    """
    
    if update.message:
        await update.message.reply_text(
            message.strip(),
            parse_mode='Markdown'
        )

async def _send_bootstrap_required_message(update: Update) -> None:
    """Send bootstrap required message"""
    message = """
🔸 **Bootstrap Required**

The bot admin system has not been initialized yet.

Use `/bootstrap` to set the first admin user (bot owner only).
    """
    
    if update.message:
        await update.message.reply_text(
            message.strip(),
            parse_mode='Markdown'
        )

async def check_admin_access(user_id: int, min_level: str = 'admin') -> tuple[bool, Optional[str]]:
    """
    Check if user has admin access without sending messages
    
    Args:
        user_id (int): User ID to check
        min_level (str): Minimum required level
        
    Returns:
        tuple[bool, Optional[str]]: (has_access, reason_if_denied)
    """
    try:
        # Ensure admin system is initialized
        if not admin_system._initialized:
            await admin_system.initialize()
        
        # Check if bootstrap completed
        if not admin_system.is_bootstrap_completed():
            return False, "Bootstrap not completed"
        
        # Check admin status
        if not admin_system.is_admin(user_id):
            return False, "Not an admin"
        
        # Check admin level
        user_level = admin_system.get_admin_level(user_id)
        if not _check_admin_level(user_level, min_level):
            return False, f"Level '{user_level}' insufficient (requires '{min_level}')"
        
        return True, None
        
    except Exception as e:
        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
        logger.error(f"Error checking admin access for user_hash {user_hash}: {e}")
        return False, "System error"

async def log_admin_action(user_id: int, action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log admin actions for audit trail
    
    Args:
        user_id (int): Admin user ID
        action (str): Action performed
        details (dict): Additional action details
    """
    try:
        await admin_system._log_admin_action(user_id, action, details)
    except Exception as e:
        logger.error(f"Failed to log admin action: {e}")

def maintenance_mode_check(allow_admins: bool = True):
    """
    Decorator to check maintenance mode before handler execution
    
    Args:
        allow_admins (bool): Allow admin users during maintenance mode
    """
    def decorator(handler_func: Callable) -> Callable:
        @wraps(handler_func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs) -> Any:
            user = update.effective_user
            if user is None:
                return await handler_func(update, context, *args, **kwargs)
            user_id = user.id
            
            try:
                # Ensure admin system is initialized
                if not admin_system._initialized:
                    await admin_system.initialize()
                
                # Check maintenance mode
                if admin_system.is_maintenance_mode():
                    # Allow admins to bypass maintenance mode if specified
                    if allow_admins and admin_system.is_admin(user_id):
                        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
                        logger.info(f"Admin user_hash {user_hash} bypassing maintenance mode")
                        return await handler_func(update, context, *args, **kwargs)
                    
                    # Send maintenance message to regular users
                    await _send_maintenance_mode_message(update)
                    return
                
                # Normal execution if not in maintenance mode
                return await handler_func(update, context, *args, **kwargs)
                
            except Exception as e:
                logger.error(f"Maintenance mode check error: {e}")
                return await handler_func(update, context, *args, **kwargs)
        
        return wrapper
    return decorator

async def _send_maintenance_mode_message(update: Update) -> None:
    """Send maintenance mode message"""
    message = """
🔧 **Maintenance Mode**

The bot is currently undergoing maintenance and is temporarily unavailable.

Please try again later. Thank you for your patience!
    """
    
    if update.message:
        await update.message.reply_text(
            message.strip(),
            parse_mode='Markdown'
        )

class AdminSecurityLogger:
    """Security logger for admin actions and access attempts"""
    
    @staticmethod
    async def log_access_attempt(user_id: int, username: str, command: str, success: bool, reason: Optional[str] = None):
        """Log admin access attempts"""
        status = "SUCCESS" if success else "DENIED"
        reason_text = f" - {reason}" if reason else ""
        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
        logger.info(f"🔐 ADMIN ACCESS {status}: user_hash {user_hash} (@{username}) -> {command}{reason_text}")
    
    @staticmethod 
    async def log_sensitive_action(user_id: int, action: str, target: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log sensitive admin actions"""
        target_text = f" on {target}" if target else ""
        details_text = f" - {details}" if details else ""
        user_hash = hashlib.sha256(f"{user_id}".encode()).hexdigest()[:8]
        logger.warning(f"🚨 SENSITIVE ACTION: {action}{target_text} by admin user_hash {user_hash}{details_text}")
    
    @staticmethod
    async def log_security_event(event_type: str, details: dict):
        """Log security-related events"""
        logger.warning(f"🛡️ SECURITY EVENT: {event_type} - {details}")