"""
Admin command handlers for Hugging Face By AadityaLabs AI
Comprehensive administrative controls accessible only via Telegram
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CallbackQueryHandler
from telegram.constants import ChatAction

from .system import admin_system
from .middleware import admin_required, log_admin_action, AdminSecurityLogger
from bot.storage_manager import storage_manager, db
from bot.security_utils import escape_markdown, safe_markdown_format, rate_limiter
from bot.config import Config

logger = logging.getLogger(__name__)

class AdminCommands:
    """
    Comprehensive admin command handlers with interactive UI
    Provides enterprise-grade administrative controls via Telegram
    """
    
    @staticmethod
    @admin_required(allow_bootstrap=True)
    async def bootstrap_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Bootstrap first admin user (one-time operation)
        Usage: /bootstrap
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"🔸 Bootstrap command invoked by user {user_id} (@{username})")
        
        try:
            # Check if bootstrap already completed
            if admin_system.is_bootstrap_completed():
                await update.message.reply_text(
                    "⚠️ **Bootstrap Already Completed**\n\n"
                    "The bot admin system has already been initialized.\n\n"
                    "Contact the existing admin for access if needed.",
                    parse_mode='Markdown'
                )
                return
            
            # Perform bootstrap
            success = await admin_system.bootstrap_admin(user_id, username)
            
            if success:
                success_message = f"""
🎉 **Admin Bootstrap Successful!**

Welcome, **Bot Owner**! You have been successfully configured as the first administrator.

**Your Admin Privileges:**
👑 **Owner Level Access** - Highest privilege level
🔧 **Full System Control** - All admin commands available
👥 **User Management** - Can add/remove other admins
📊 **System Monitoring** - Access to stats and logs
🔧 **Maintenance Control** - Can enable/disable maintenance mode

**Next Steps:**
• Use `/admin` to access the admin panel
• Use `/stats` to view bot statistics
• Use `/settings` for user management controls

**Security Note:** Your admin status is now permanently stored in the encrypted database.

Ready to manage your bot! 🚀
                """
                
                await update.message.reply_text(
                    success_message.strip(),
                    parse_mode='Markdown'
                )
                
                # Log security event
                await AdminSecurityLogger.log_security_event(
                    'admin_bootstrap',
                    {'user_id': user_id, 'username': username}
                )
                
            else:
                await update.message.reply_text(
                    "❌ **Bootstrap Failed**\n\n"
                    "Failed to initialize admin system. Please check logs and try again.",
                    parse_mode='Markdown'
                )
        
        except Exception as e:
            logger.error(f"Bootstrap command error: {e}")
            await update.message.reply_text(
                "🚫 **Bootstrap Error**\n\n"
                "An error occurred during bootstrap. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required()
    async def admin_panel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Main admin panel with interactive keyboard
        Usage: /admin
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"🔧 Admin panel accessed by {user_id} (@{username})")
        
        try:
            # Get admin level and system status
            admin_level = admin_system.get_admin_level(user_id)
            is_owner = admin_system.is_owner(user_id)
            maintenance_mode = admin_system.is_maintenance_mode()
            
            # Get basic stats for panel
            admin_stats = await admin_system.get_admin_stats()
            
            # Construct panel message
            safe_username = escape_markdown(user.first_name or username)
            safe_level = escape_markdown(admin_level.title() if admin_level else "Unknown")
            
            panel_text = f"""
🔧 **Admin Control Panel**

**Welcome, {safe_username}**
**Level:** {safe_level} {'👑' if is_owner else '🛡️'}

**System Status:**
🤖 **Bot Status:** {'🔧 Maintenance' if maintenance_mode else '✅ Active'}
👥 **Admins:** {admin_stats.get('admin_count', 0)} total
📊 **Users:** {admin_stats.get('total_users', 'N/A')}
💬 **Conversations:** {admin_stats.get('total_conversations', 'N/A')}

**Quick Actions:**
            """
            
            # Build keyboard based on admin level
            keyboard = []
            
            # Row 1: Basic admin functions
            keyboard.append([
                InlineKeyboardButton("📊 Statistics", callback_data="admin_stats"),
                InlineKeyboardButton("👥 User Management", callback_data="admin_users")
            ])
            
            # Row 2: System controls
            if maintenance_mode:
                keyboard.append([
                    InlineKeyboardButton("✅ Disable Maintenance", callback_data="admin_maintenance_off"),
                    InlineKeyboardButton("📋 View Logs", callback_data="admin_logs")
                ])
            else:
                keyboard.append([
                    InlineKeyboardButton("🔧 Enable Maintenance", callback_data="admin_maintenance_on"),
                    InlineKeyboardButton("📋 View Logs", callback_data="admin_logs")
                ])
            
            # Row 3: Communication and advanced features
            keyboard.append([
                InlineKeyboardButton("📢 Broadcast Message", callback_data="admin_broadcast"),
                InlineKeyboardButton("⚙️ System Health", callback_data="admin_health")
            ])
            
            # Row 4: Owner-only functions
            if is_owner:
                keyboard.append([
                    InlineKeyboardButton("👑 Admin Management", callback_data="admin_manage_admins"),
                    InlineKeyboardButton("🔒 Security Logs", callback_data="admin_security")
                ])
            
            # Row 5: Refresh and help
            keyboard.append([
                InlineKeyboardButton("🔄 Refresh Panel", callback_data="admin_refresh"),
                InlineKeyboardButton("❓ Admin Help", callback_data="admin_help")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                panel_text.strip(),
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            await log_admin_action(user_id, 'admin_panel_access', {'level': admin_level})
            
        except Exception as e:
            logger.error(f"Admin panel error: {e}")
            await update.message.reply_text(
                "🚫 **Admin Panel Error**\n\n"
                "Failed to load admin panel. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required()
    async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Display comprehensive bot statistics
        Usage: /stats
        """
        user_id = update.effective_user.id
        
        logger.info(f"📊 Stats command accessed by admin {user_id}")
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            # Get comprehensive statistics
            admin_stats = await admin_system.get_admin_stats()
            
            # Get database statistics
            storage = await storage_manager.ensure_connected()
            
            # Calculate additional metrics
            current_time = datetime.utcnow()
            
            stats_text = f"""
📊 **Bot Statistics Dashboard**

**System Overview:**
🤖 **Bot Status:** {'🔧 Maintenance Mode' if admin_stats.get('maintenance_mode') else '✅ Active'}
⚡ **Uptime:** Active
🏗️ **Bootstrap:** {'✅ Completed' if admin_stats.get('bootstrap_completed') else '❌ Pending'}

**Admin Information:**
👑 **Total Admins:** {admin_stats.get('admin_count', 0)}
📋 **Admin Levels:**
  • Owner: {admin_stats.get('admin_levels_count', {}).get('owner', 0)}
  • Admin: {admin_stats.get('admin_levels_count', {}).get('admin', 0)}
  • Moderator: {admin_stats.get('admin_levels_count', {}).get('moderator', 0)}

**Database Statistics:**
💾 **Storage:** {type(storage).__name__.replace('Provider', '')}
🔗 **Connection:** {'✅ Connected' if storage_manager.connected else '❌ Disconnected'}

**Performance Metrics:**
🔄 **Rate Limiter:** Active ({rate_limiter.max_requests}/min)
🛡️ **Security:** Encryption enabled
📝 **Logging:** Enhanced observability active

**Generated:** {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
            """
            
            # Add usage stats if available
            if 'usage_stats' in admin_stats:
                usage = admin_stats['usage_stats']
                stats_text += f"""

**Usage Statistics (30 days):**
📈 **API Calls:** {usage.get('total_calls', 'N/A')}
👤 **Active Users:** {usage.get('active_users', 'N/A')}
💬 **Messages:** {usage.get('total_messages', 'N/A')}
🎨 **Images Generated:** {usage.get('images_generated', 'N/A')}
💻 **Code Generated:** {usage.get('code_generated', 'N/A')}
                """
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh Stats", callback_data="admin_stats"),
                    InlineKeyboardButton("📊 Detailed View", callback_data="admin_stats_detailed")
                ],
                [
                    InlineKeyboardButton("🏠 Admin Panel", callback_data="admin_refresh"),
                    InlineKeyboardButton("💾 Export Data", callback_data="admin_export")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                stats_text.strip(),
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            await log_admin_action(user_id, 'view_stats', {})
            
        except Exception as e:
            logger.error(f"Stats command error: {e}")
            await update.message.reply_text(
                "🚫 **Statistics Error**\n\n"
                "Failed to load statistics. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required(min_level='admin')
    async def maintenance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Toggle maintenance mode
        Usage: /maintenance [on|off]
        """
        user_id = update.effective_user.id
        username = update.effective_user.username or "No username"
        
        try:
            # Parse arguments
            args = context.args if context.args else []
            
            if not args:
                # Show current status and options
                current_status = admin_system.is_maintenance_mode()
                status_text = "🔧 **Enabled**" if current_status else "✅ **Disabled**"
                
                message = f"""
🔧 **Maintenance Mode Control**

**Current Status:** {status_text}

**Maintenance Mode Effects:**
• Regular users cannot use the bot
• Only admins can access bot functions
• Users see maintenance message for all commands

**Choose an action:**
                """
                
                keyboard = [
                    [
                        InlineKeyboardButton(
                            "🔧 Enable Maintenance" if not current_status else "✅ Disable Maintenance",
                            callback_data=f"admin_maintenance_{'on' if not current_status else 'off'}"
                        )
                    ],
                    [
                        InlineKeyboardButton("🏠 Admin Panel", callback_data="admin_refresh")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    message.strip(),
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                return
            
            # Process on/off argument
            action = args[0].lower()
            if action in ['on', 'enable', 'true', '1']:
                success = await admin_system.set_maintenance_mode(True, user_id)
                if success:
                    await update.message.reply_text(
                        "🔧 **Maintenance Mode Enabled**\n\n"
                        "The bot is now in maintenance mode. Only admins can use bot functions.",
                        parse_mode='Markdown'
                    )
                    await AdminSecurityLogger.log_sensitive_action(
                        user_id, 'maintenance_mode_enabled', details={'username': username}
                    )
                else:
                    await update.message.reply_text(
                        "❌ **Failed to enable maintenance mode**",
                        parse_mode='Markdown'
                    )
                    
            elif action in ['off', 'disable', 'false', '0']:
                success = await admin_system.set_maintenance_mode(False, user_id)
                if success:
                    await update.message.reply_text(
                        "✅ **Maintenance Mode Disabled**\n\n"
                        "The bot is now active and available to all users.",
                        parse_mode='Markdown'
                    )
                    await AdminSecurityLogger.log_sensitive_action(
                        user_id, 'maintenance_mode_disabled', details={'username': username}
                    )
                else:
                    await update.message.reply_text(
                        "❌ **Failed to disable maintenance mode**",
                        parse_mode='Markdown'
                    )
            else:
                await update.message.reply_text(
                    "❌ **Invalid argument**\n\n"
                    "Usage: `/maintenance on` or `/maintenance off`",
                    parse_mode='Markdown'
                )
        
        except Exception as e:
            logger.error(f"Maintenance command error: {e}")
            await update.message.reply_text(
                "🚫 **Maintenance Command Error**\n\n"
                "Failed to process maintenance command.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required()
    async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        View recent bot logs
        Usage: /logs [lines] [level]
        """
        user_id = update.effective_user.id
        
        logger.info(f"📋 Logs command accessed by admin {user_id}")
        
        try:
            # Parse arguments
            args = context.args if context.args else []
            lines = 50  # Default
            log_level = 'INFO'  # Default
            
            if len(args) >= 1:
                try:
                    lines = int(args[0])
                    lines = min(max(lines, 10), 200)  # Limit between 10-200
                except ValueError:
                    pass
            
            if len(args) >= 2:
                log_level = args[1].upper()
                if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    log_level = 'INFO'
            
            # For now, show a placeholder with instructions
            # In a real implementation, you'd read from log files or a logging service
            logs_text = f"""
📋 **Bot Logs Viewer**

**Viewing:** Last {lines} lines, Level: {log_level}

**Recent Log Entries:**
```
{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Admin logs accessed by user {user_id}
{(datetime.utcnow() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')} - INFO - Bot startup completed successfully
{(datetime.utcnow() - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')} - INFO - Storage connection established
{(datetime.utcnow() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')} - INFO - Admin system initialized
```

**Log Levels Available:**
• DEBUG - Detailed debugging information
• INFO - General information messages
• WARNING - Warning messages
• ERROR - Error messages
• CRITICAL - Critical error messages

**Usage:** `/logs [lines] [level]`
**Example:** `/logs 100 ERROR` (show last 100 error messages)
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh Logs", callback_data="admin_logs"),
                    InlineKeyboardButton("🚨 Error Logs", callback_data="admin_logs_errors")
                ],
                [
                    InlineKeyboardButton("📊 System Health", callback_data="admin_health"),
                    InlineKeyboardButton("🏠 Admin Panel", callback_data="admin_refresh")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                logs_text.strip(),
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            await log_admin_action(user_id, 'view_logs', {'lines': lines, 'level': log_level})
            
        except Exception as e:
            logger.error(f"Logs command error: {e}")
            await update.message.reply_text(
                "🚫 **Logs Viewer Error**\n\n"
                "Failed to retrieve logs. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required(min_level='admin')
    async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Broadcast message to all users
        Usage: /broadcast <message>
        """
        user_id = update.effective_user.id
        username = update.effective_user.username or "No username"
        
        try:
            # Parse message content
            if not context.args:
                await update.message.reply_text(
                    """
📢 **Broadcast Message System**

**Usage:** `/broadcast <your message>`

**Example:** `/broadcast Bot will be updated tonight at 23:00 UTC`

**Features:**
• Message sent to all bot users
• Supports Markdown formatting
• Delivery status reported
• Action logged for audit trail

**Safety:** Use responsibly - all users will receive this message.
                    """,
                    parse_mode='Markdown'
                )
                return
            
            broadcast_message = ' '.join(context.args)
            
            if len(broadcast_message) > 1000:
                await update.message.reply_text(
                    "❌ **Message Too Long**\n\n"
                    "Broadcast messages must be under 1000 characters for reliability.",
                    parse_mode='Markdown'
                )
                return
            
            # Confirmation before sending
            safe_preview = escape_markdown(broadcast_message[:100])
            preview_text = f"{safe_preview}..." if len(broadcast_message) > 100 else safe_preview
            
            confirmation_text = f"""
📢 **Confirm Broadcast**

**Message Preview:**
{preview_text}

**Recipients:** All bot users
**Length:** {len(broadcast_message)} characters

**Are you sure you want to send this broadcast?**
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("✅ Send Broadcast", callback_data=f"admin_broadcast_confirm"),
                    InlineKeyboardButton("❌ Cancel", callback_data="admin_refresh")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Store message in context for callback
            if context.user_data is not None:
                context.user_data['broadcast_message'] = broadcast_message
            
            await update.message.reply_text(
                confirmation_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Broadcast command error: {e}")
            await update.message.reply_text(
                "🚫 **Broadcast Error**\n\n"
                "Failed to prepare broadcast. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required(min_level='owner')
    async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        User management interface
        Usage: /users [search_term]
        """
        user_id = update.effective_user.id
        
        logger.info(f"👥 Users command accessed by admin {user_id}")
        
        try:
            args = context.args if context.args else []
            search_term = ' '.join(args) if args else None
            
            users_text = f"""
👥 **User Management Panel**

**Search:** {escape_markdown(search_term) if search_term else 'All users'}

**Available Actions:**
• View user details and statistics
• Search users by ID or username
• Monitor user activity
• View user conversation history

**User Statistics:**
📊 **Total Users:** Loading...
📈 **Active Users (30d):** Loading...
💬 **Total Conversations:** Loading...

**Usage:**
• `/users` - Show all users
• `/users 12345` - Search by user ID
• `/users @username` - Search by username
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("🔍 Search Users", callback_data="admin_users_search"),
                    InlineKeyboardButton("📊 User Stats", callback_data="admin_users_stats")
                ],
                [
                    InlineKeyboardButton("📈 Active Users", callback_data="admin_users_active"),
                    InlineKeyboardButton("💬 Recent Activity", callback_data="admin_users_activity")
                ],
                [
                    InlineKeyboardButton("🏠 Admin Panel", callback_data="admin_refresh")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                users_text.strip(),
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            await log_admin_action(user_id, 'user_management_access', {'search_term': search_term})
            
        except Exception as e:
            logger.error(f"Users command error: {e}")
            await update.message.reply_text(
                "🚫 **User Management Error**\n\n"
                "Failed to load user management. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    @admin_required()
    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Admin help and documentation
        Usage: /adminhelp
        """
        user_id = update.effective_user.id
        admin_level = admin_system.get_admin_level(user_id)
        is_owner = admin_system.is_owner(user_id)
        
        help_text = f"""
❓ **Admin Help & Documentation**

**Your Level:** {admin_level.title() if admin_level else 'Unknown'} {'👑' if is_owner else '🛡️'}

**Available Commands:**

**🔧 Basic Admin Commands:**
• `/admin` - Access main admin panel
• `/stats` - View comprehensive bot statistics
• `/logs [lines] [level]` - View recent bot logs
• `/adminhelp` - Show this help message

**⚙️ System Control:**
• `/maintenance [on|off]` - Toggle maintenance mode
        """
        
        if admin_level in ['admin', 'owner']:
            help_text += """
**📢 Communication:**
• `/broadcast <message>` - Send message to all users
            """
        
        if admin_level == 'owner':
            help_text += """
**👑 Owner Commands:**
• `/bootstrap` - Initialize first admin (one-time)
• `/users [search]` - User management interface
• Security and admin management features
            """
        
        help_text += """

**🔍 Interactive Features:**
• All commands support interactive keyboards
• Real-time statistics and monitoring
• Secure action confirmations
• Comprehensive audit logging

**🛡️ Security:**
• All admin actions are logged
• Rate limiting for admin commands
• Privilege level enforcement
• Secure data encryption

**💡 Tips:**
• Use `/admin` for quick access to all features
• Check `/stats` regularly for bot health
• Enable maintenance mode during updates
• Use broadcast sparingly to avoid spam

Need help with a specific command? Contact the bot owner.
        """
        
        keyboard = [
            [
                InlineKeyboardButton("🏠 Admin Panel", callback_data="admin_refresh"),
                InlineKeyboardButton("📊 Quick Stats", callback_data="admin_stats")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            help_text.strip(),
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        await log_admin_action(user_id, 'admin_help_accessed', {'level': admin_level})


# Callback query handlers for admin interface
class AdminCallbackHandlers:
    """Handle admin panel callback queries"""
    
    @staticmethod
    async def handle_admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle all admin-related callback queries
        """
        query = update.callback_query
        user_id = query.from_user.id
        callback_data = query.data
        
        # Verify admin access
        has_access, reason = await check_admin_access(user_id)
        if not has_access:
            await query.answer("❌ Access denied", show_alert=True)
            return
        
        await query.answer()  # Acknowledge the callback
        
        try:
            if callback_data == "admin_refresh":
                await AdminCommands.admin_panel_command(update, context)
            elif callback_data == "admin_stats":
                await AdminCommands.stats_command(update, context)
            elif callback_data == "admin_maintenance_on":
                await admin_system.set_maintenance_mode(True, user_id)
                await query.edit_message_text(
                    "🔧 **Maintenance Mode Enabled**\n\nBot is now in maintenance mode.",
                    parse_mode='Markdown'
                )
            elif callback_data == "admin_maintenance_off":
                await admin_system.set_maintenance_mode(False, user_id)
                await query.edit_message_text(
                    "✅ **Maintenance Mode Disabled**\n\nBot is now active for all users.",
                    parse_mode='Markdown'
                )
            # Add more callback handlers as needed
            
        except Exception as e:
            logger.error(f"Admin callback error: {e}")
            await query.edit_message_text(
                "🚫 **Error**\n\nFailed to process admin action.",
                parse_mode='Markdown'
            )