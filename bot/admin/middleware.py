"""
Admin middleware for security and access control
Provides decorators and utilities for admin-only functionality
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Any, Optional
from datetime import datetime

from telegram import Update
from telegram.ext import ContextTypes

from .system import admin_system
from bot.security_utils import escape_markdown, safe_markdown_format

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
            user_id = user.id
            username = user.username or "No username"
            
            try:
                # Ensure admin system is initialized
                if not admin_system._initialized:
                    await admin_system.initialize()
                
                # Check if bootstrap is needed and allowed
                if not admin_system.is_bootstrap_completed():
                    if allow_bootstrap:
                        logger.info(f"🔸 Bootstrap access granted for user {user_id} (@{username})")
                        return await handler_func(update, context, *args, **kwargs)
                    else:
                        await _send_bootstrap_required_message(update)
                        return
                
                # Check admin status
                if not admin_system.is_admin(user_id):
                    logger.warning(f"🚫 Access denied: User {user_id} (@{username}) not an admin")
                    await _send_access_denied_message(update)
                    return
                
                # Check admin level
                user_level = admin_system.get_admin_level(user_id)
                if not _check_admin_level(user_level, min_level):
                    logger.warning(f"🚫 Insufficient privileges: User {user_id} (@{username}) level '{user_level}' < required '{min_level}'")
                    await _send_insufficient_privileges_message(update, min_level)
                    return
                
                # Check admin rate limit
                is_allowed, wait_time = await admin_system.check_admin_rate_limit(user_id)
                if not is_allowed:
                    await update.message.reply_text(
                        f"⚠️ **Admin Rate Limit**\n\nPlease wait {wait_time} seconds before using admin commands.",
                        parse_mode='Markdown'
                    )
                    return
                
                # Log admin access
                handler_name = handler_func.__name__
                await log_admin_action(
                    user_id, 
                    'command_access', 
                    {
                        'command': handler_name,
                        'level_required': min_level,
                        'user_level': user_level,
                        'username': username
                    }
                )
                
                logger.info(f"✅ Admin access granted: {user_id} (@{username}) -> {handler_name}")
                
                # Execute the handler
                return await handler_func(update, context, *args, **kwargs)
                
            except Exception as e:
                logger.error(f"Admin middleware error for user {user_id}: {e}")
                await update.message.reply_text(
                    "🚫 **Admin System Error**\n\nAn error occurred while checking admin privileges. Please try again.",
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
        logger.error(f"Error checking admin access for {user_id}: {e}")
        return False, "System error"

async def log_admin_action(user_id: int, action: str, details: dict = None) -> None:
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
            user_id = update.effective_user.id
            
            try:
                # Ensure admin system is initialized
                if not admin_system._initialized:
                    await admin_system.initialize()
                
                # Check maintenance mode
                if admin_system.is_maintenance_mode():
                    # Allow admins to bypass maintenance mode if specified
                    if allow_admins and admin_system.is_admin(user_id):
                        logger.info(f"Admin {user_id} bypassing maintenance mode")
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
    
    await update.message.reply_text(
        message.strip(),
        parse_mode='Markdown'
    )

class AdminSecurityLogger:
    """Security logger for admin actions and access attempts"""
    
    @staticmethod
    async def log_access_attempt(user_id: int, username: str, command: str, success: bool, reason: str = None):
        """Log admin access attempts"""
        status = "SUCCESS" if success else "DENIED"
        reason_text = f" - {reason}" if reason else ""
        logger.info(f"🔐 ADMIN ACCESS {status}: User {user_id} (@{username}) -> {command}{reason_text}")
    
    @staticmethod 
    async def log_sensitive_action(user_id: int, action: str, target: str = None, details: dict = None):
        """Log sensitive admin actions"""
        target_text = f" on {target}" if target else ""
        details_text = f" - {details}" if details else ""
        logger.warning(f"🚨 SENSITIVE ACTION: {action}{target_text} by admin {user_id}{details_text}")
    
    @staticmethod
    async def log_security_event(event_type: str, details: dict):
        """Log security-related events"""
        logger.warning(f"🛡️ SECURITY EVENT: {event_type} - {details}")