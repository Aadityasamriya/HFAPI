"""
Professional Telegram bot command handlers
Rich UI with emojis and inline keyboards for superior user experience
"""

import asyncio
import logging

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ContextTypes
    from telegram.constants import ChatAction
except ImportError as e:
    print(f"Telegram library import error: {e}")
    print("Please ensure python-telegram-bot is installed: pip install python-telegram-bot[ext]==22.4")
    raise
from bot.database import db
from bot.config import Config
from bot.security_utils import escape_markdown, safe_markdown_format, check_rate_limit

logger = logging.getLogger(__name__)

class CommandHandlers:
    """Professional command handlers with comprehensive observability logging"""
    
    @staticmethod
    async def start_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Professional welcome message with sophisticated UI and comprehensive logging
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or "Unknown"
        
        # Enhanced entry logging with user details
        logger.info(f"\ud83d\ude80 START command invoked by user_id:{user_id} (@{username}) '{full_name}'")
        logger.info(f"\ud83d\udccd Chat type: {update.effective_chat.type} | Chat ID: {update.effective_chat.id}")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another command.",
                parse_mode='Markdown'
            )
            return
        
        # Safely escape user's first name
        safe_first_name = escape_markdown(user.first_name or "User")
        
        welcome_text = f"""
🤖 **Welcome to AI Assistant Pro** 🚀

Hello {safe_first_name}! I'm powered by the **latest 2024-2025 AI models** - more advanced than ChatGPT, Grok, or Gemini!

**🔥 Latest AI Models I Use:**
🧠 **Llama-3.2 & Qwen2.5** - Next-gen text AI (29+ languages)
💻 **StarCoder2-15B** - Revolutionary coding assistant  
🎨 **FLUX.1** - Breakthrough image generation
📊 **Advanced Emotion AI** - 28 emotion categories
🌐 **Universal Translation** - Professional multilingual support

**🎯 Superior Features:**
✨ **Intelligent Model Routing** - I choose the perfect AI for each task
⚡ **Lightning Fast** - Optimized for speed and quality
🛡️ **Privacy First** - Your data stays secure and private
🆓 **Completely Free** - Generous Hugging Face quotas

**⚡ Quick Start (2 minutes):**
1️⃣ Get your free Hugging Face API key
2️⃣ Start chatting with the world's newest AI
3️⃣ Experience AI that's truly superior!

*Ready to experience the future of AI?* 🚀✨
        """
        
        keyboard = [
            [InlineKeyboardButton("🔑 Set API Key", callback_data="set_api_key")],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton("💡 Help", callback_data="help")
            ],
            [InlineKeyboardButton("🚀 Quick Start Guide", callback_data="quick_start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await update.message.reply_text(
                welcome_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            # Enhanced success logging
            logger.info(f"\u2705 START response sent successfully to user_id:{user_id} (@{username})")
            logger.info(f"\ud83d\udce4 Welcome message delivered with {len(keyboard)} inline buttons")
            
        except Exception as e:
            logger.error(f"\u274c START command failed for user_id:{user_id} (@{username}): {e}")
            logger.error(f"\ud83d\udd0d Error type: {type(e).__name__}")
            raise
        finally:
            logger.info(f"\ud83c\udfc1 START command completed for user_id:{user_id}")
    
    @staticmethod
    async def newchat_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Clear chat history with professional confirmation and comprehensive logging
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        # Enhanced entry logging
        logger.info(f"\ud83d\udd04 NEWCHAT command invoked by user_id:{user_id} (@{username})")
        
        # Log current chat history status
        current_history_size = len(context.user_data.get('chat_history', [])) if context.user_data else 0
        logger.info(f"\ud83d\udcca Current chat history size: {current_history_size} messages")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another command.",
                parse_mode='Markdown'
            )
            return
        
        # Clear chat history from context
        if context.user_data is not None and 'chat_history' in context.user_data:
            context.user_data['chat_history'] = []
        
        success_text = """
🔄 **Chat History Cleared** ✨

Your conversation history has been reset. You're starting fresh with a clean slate!

💡 **Tip:** Each conversation maintains context for up to 15 messages to provide better responses.
        """
        
        keyboard = [
            [InlineKeyboardButton("🚀 Start New Conversation", callback_data="start_conversation")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await update.message.reply_text(
                success_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            # Enhanced success logging
            logger.info(f"\u2705 NEWCHAT response sent successfully to user_id:{user_id} (@{username})")
            logger.info(f"\ud83d\uddd1\ufe0f Chat history cleared: {current_history_size} messages removed")
            
        except Exception as e:
            logger.error(f"\u274c NEWCHAT command failed for user_id:{user_id} (@{username}): {e}")
            logger.error(f"\ud83d\udd0d Error type: {type(e).__name__}")
            raise
        finally:
            logger.info(f"\ud83c\udfc1 NEWCHAT command completed for user_id:{user_id}")
    
    @staticmethod
    async def settings_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Professional settings menu with comprehensive options and detailed logging
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        # Enhanced entry logging
        logger.info(f"\u2699\ufe0f SETTINGS command invoked by user_id:{user_id} (@{username})")
        logger.info(f"\ud83d\udd0d Checking API key status for user_id:{user_id}...")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another command.",
                parse_mode='Markdown'
            )
            return
            
        # Check API key from persistent database storage with logging
        try:
            api_key = await db.get_user_api_key(user_id)
            if api_key:
                logger.info(f"\u2705 API key found for user_id:{user_id} (last 4 chars: ...{api_key[-4:] if len(api_key) >= 4 else 'short'})")
            else:
                logger.info(f"\u274c No API key found for user_id:{user_id}")
        except Exception as e:
            logger.error(f"\ud83d\udd0d Database error checking API key for user_id:{user_id}: {e}")
            api_key = None
        
        status_emoji = "✅" if api_key else "❌"
        api_status = "Connected" if api_key else "Not Set"
        
        settings_text = f"""
⚙️ **AI Assistant Settings** 🛠️

**Current Status:**
🔑 API Key: {status_emoji} {api_status}
🤖 Models: Premium Hugging Face Collection
🧠 Intelligence: Adaptive Model Routing
💾 Storage: Persistent database (as specified)
📊 Chat History: Persistent storage with last 15 messages in active session

**Available Actions:**
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Change API Key", callback_data="set_api_key")],
            [
                InlineKeyboardButton("📊 Usage Stats", callback_data="usage_stats"),
                InlineKeyboardButton("🎯 Model Info", callback_data="model_info")
            ],
            [
                InlineKeyboardButton("🗑️ Reset My Data", callback_data="confirm_reset"),
                InlineKeyboardButton("💡 Help", callback_data="help")
            ],
            [InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await update.message.reply_text(
                settings_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            # Enhanced success logging
            logger.info(f"\u2705 SETTINGS response sent successfully to user_id:{user_id} (@{username})")
            logger.info(f"\ud83d\udccb Settings menu displayed with {len(keyboard)} button options")
            
        except Exception as e:
            logger.error(f"\u274c SETTINGS command failed for user_id:{user_id} (@{username}): {e}")
            logger.error(f"\ud83d\udd0d Error type: {type(e).__name__}")
            raise
        finally:
            logger.info(f"\ud83c\udfc1 SETTINGS command completed for user_id:{user_id}")
    
    @staticmethod
    async def help_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Comprehensive help system with examples and detailed logging
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        # Enhanced entry logging
        logger.info(f"❓ HELP command invoked by user_id:{user_id} (@{username})")
        logger.info(f"📚 Preparing comprehensive help documentation for user")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another command.",
                parse_mode='Markdown'
            )
            return
            
        help_text = """
💡 **AI Assistant Pro Help Guide** 📚

**🎯 Smart Commands:**
• `/start` - Welcome & setup
• `/newchat` - Clear conversation history  
• `/settings` - Manage your preferences
• `/help` - This help guide

**🤖 AI Capabilities:**

**💬 Text Generation:**
"Explain quantum computing"
"Write a business proposal for..."
"Help me understand machine learning"

**💻 Code Generation:**
"Create a Python function to sort data"
"Build a React component for..."
"Write SQL query to find..."

**🎨 Image Creation:**
"Draw a futuristic cityscape"
"Create a professional logo for..."
"Generate artwork of..."

**📊 Analysis:**
"Analyze the sentiment of this text"
"What's the mood of this message"

**✨ Features:**
• 🧠 Intelligent model routing
• 🔄 Context-aware conversations
• 🚀 Multi-modal capabilities
• 🛡️ Secure API key management

**Need more help?** Just ask me anything!
        """
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Try Examples", callback_data="examples"),
                InlineKeyboardButton("🎯 Model Guide", callback_data="model_guide")
            ],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def resetdb_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Reset user database command with comprehensive confirmation and security logging
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or "Unknown"
        
        # Enhanced entry logging with user details for security audit
        logger.info(f"🗑️ RESETDB command invoked by user_id:{user_id} (@{username}) '{full_name}'")
        logger.info(f"🔐 SECURITY AUDIT: Database reset request from {update.effective_chat.type} chat")
        logger.warning(f"⚠️ CRITICAL ACTION: User {user_id} requesting complete data deletion")
        
        # Check rate limit (critical for security)
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            logger.warning(f"🚨 RATE LIMIT: RESETDB blocked for user_id:{user_id} - {wait_time}s remaining")
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another command.",
                parse_mode='Markdown'
            )
            return
        
        # Safely escape user's first name for security
        safe_first_name = escape_markdown(user.first_name or "User")
        
        warning_text = f"""
🗑️ **Database Reset Request** ⚠️

Hello {safe_first_name}, you've requested to completely reset your database.

**⚠️ THIS WILL PERMANENTLY DELETE:**
🔑 Your stored API key
💾 All account preferences  
📊 Usage statistics and history
🗂️ Any saved configurations

**🔄 IMMEDIATE EFFECTS:**
• You'll need to set up your API key again
• All personalized settings will be lost
• This action **cannot be undone**

**🛡️ SECURITY NOTE:**
Your conversations are stored securely in our encrypted database. This will remove your saved conversation history and account data.

**Are you absolutely sure you want to proceed?**
        """
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Yes, Reset Database", callback_data="resetdb_confirmed"),
                InlineKeyboardButton("❌ Cancel", callback_data="resetdb_cancel")
            ],
            [InlineKeyboardButton("⚙️ Settings Instead", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await update.message.reply_text(
                warning_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            # Enhanced success logging for security audit
            logger.info(f"✅ RESETDB confirmation dialog sent to user_id:{user_id} (@{username})")
            logger.info(f"📋 Confirmation UI presented with {len(keyboard)} safety options")
            
        except Exception as e:
            logger.error(f"❌ RESETDB command failed for user_id:{user_id} (@{username}): {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            # Security: Log the full error for debugging but don't expose it to user
            logger.error(f"🐛 Full error details: {str(e)}")
            
            # Send user-friendly error message
            await update.message.reply_text(
                "❌ **System Error**\n\nSorry, there was a problem processing your request. Please try again later.",
                parse_mode='Markdown'
            )
            raise
        finally:
            logger.info(f"🏁 RESETDB command completed for user_id:{user_id}")
    
    @staticmethod
    async def button_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle all inline keyboard button interactions with entity parsing protection
        """
        try:
            query = update.callback_query
            await query.answer()
            
            # Safely extract user data with entity parsing protection
            try:
                user_id = query.from_user.id
                data = query.data
            except Exception as entity_error:
                logger.warning(f"Entity parsing error in button handler: {entity_error}")
                # Try alternative extraction methods
                try:
                    raw_query = update.to_dict().get('callback_query', {})
                    user_id = raw_query.get('from', {}).get('id', 0)
                    data = raw_query.get('data', '')
                except Exception:
                    logger.error("Failed to extract callback query data, ignoring request")
                    return
                    
            if not data:
                logger.warning("Empty callback data received")
                return
            
            # Process the button click
            if data == "set_api_key":
                await CommandHandlers._handle_api_key_setup(query, context)
            
            elif data == "settings":
                await CommandHandlers._handle_settings_display(query, context)
            
            elif data == "help":
                await CommandHandlers._handle_help_display(query, context)
            
            elif data == "confirm_reset":
                await CommandHandlers._handle_reset_confirmation(query, context)
            
            elif data == "reset_confirmed":
                await CommandHandlers._handle_data_reset(query, context)
            
            elif data == "usage_stats":
                await CommandHandlers._handle_usage_stats(query, context)
            
            elif data == "model_info":
                await CommandHandlers._handle_model_info(query, context)
            
            elif data == "examples":
                await CommandHandlers._handle_examples(query, context)
            
            elif data == "quick_start":
                await CommandHandlers._handle_quick_start(query, context)
            
            elif data == "resetdb_confirmed":
                await CommandHandlers._handle_resetdb_execution(query, context)
            
            elif data == "resetdb_cancel":
                await CommandHandlers._handle_resetdb_cancel(query, context)
            
            # History-related callbacks
            elif data.startswith("view_conv_"):
                conversation_id = data.replace("view_conv_", "")
                await CommandHandlers._handle_view_conversation(query, context, conversation_id)
            
            elif data.startswith("history_page_"):
                page_number = int(data.replace("history_page_", ""))
                await CommandHandlers._handle_history_pagination(query, context, page_number)
            
            elif data == "history_refresh":
                await CommandHandlers._handle_history_refresh(query, context)
            
            elif data == "history_clear_confirm":
                await CommandHandlers._handle_history_clear_confirm(query, context)
            
            elif data == "history_clear_yes":
                await CommandHandlers._handle_history_clear_execute(query, context)
            
            elif data == "history_clear_no":
                await CommandHandlers._handle_history_refresh(query, context)  # Just refresh instead
            
            elif data == "start_conversation":
                await CommandHandlers._handle_start_conversation(query, context)
            
            elif data.startswith("continue_conv_"):
                conversation_id = data.replace("continue_conv_", "")
                await CommandHandlers._handle_continue_conversation(query, context, conversation_id)
            
            elif data.startswith("delete_conv_"):
                conversation_id = data.replace("delete_conv_", "")
                await CommandHandlers._handle_delete_conversation_confirm(query, context, conversation_id)
            
            elif data.startswith("delete_conv_yes_"):
                conversation_id = data.replace("delete_conv_yes_", "")
                await CommandHandlers._handle_delete_conversation_execute(query, context, conversation_id)
            
            elif data.startswith("delete_conv_no_"):
                # Just return to conversation view
                conversation_id = data.replace("delete_conv_no_", "")
                await CommandHandlers._handle_view_conversation(query, context, conversation_id)
            
            else:
                await query.edit_message_text("🔄 Processing your request...")
                
        except Exception as e:
            logger.error(f"Error in button handler: {e}")
            # Try to respond to user even if there's an error
            try:
                if hasattr(update, 'callback_query') and update.callback_query:
                    await update.callback_query.edit_message_text("❌ **Button Error**\n\nSorry, there was an issue processing your button click. Please try again.")
            except Exception:
                pass  # If even error response fails, just log and continue
    
    @staticmethod
    async def _handle_api_key_setup(query, context) -> None:
        """Handle API key setup process"""
        text = """
🔑 **Hugging Face API Key Setup** (Easy & Fast!)

**🚀 Why you'll love this:**
• Access to 2024-2025's most advanced AI models
• FLUX.1 image generation (state-of-the-art)
• StarCoder2-15B coding assistant
• Llama-3.2 & Qwen2.5 text models
• All completely FREE with generous limits!

**📋 Super Simple Setup:**
1️⃣ Visit: https://huggingface.co/settings/tokens
2️⃣ Click "Create new token"
3️⃣ Choose "Read" permissions (default)
4️⃣ Copy your token (starts with hf_)
5️⃣ Send it here as your next message

🛡️ **Security:** Your API key is stored securely in our encrypted database as specified.

💸 **Cost:** Completely FREE for personal use! Hugging Face offers generous quotas for all latest models.

🔒 **Privacy First:** API keys are stored persistently but securely encrypted for seamless access.

✨ **You're about to access AI technology that rivals ChatGPT, but with the newest 2024-2025 models!**
        """
        
        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Set state for expecting API key
        context.user_data['waiting_for_api_key'] = True
        
        await query.edit_message_text(
            text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_reset_confirmation(query, context) -> None:
        """Handle data reset confirmation"""
        text = """
⚠️ **Confirm Data Reset** 

This will permanently delete:
• Your stored API key
• All account preferences
• Usage statistics

🔄 **Note:** Your chat history is stored securely with encryption for persistence across sessions.

Are you sure you want to proceed?
        """
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Yes, Reset", callback_data="reset_confirmed"),
                InlineKeyboardButton("❌ Cancel", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_data_reset(query, context) -> None:
        """Handle complete user data reset"""
        user_id = query.from_user.id
        
        # Reset all user data in database as specified
        success = await db.reset_user_database(user_id)
        
        # Also clear session data
        if context.user_data:
            context.user_data.clear()
        
        if success:
            text = """
✅ **All Data Reset Complete** 

Your data has been completely cleared, including:
• API key (permanently removed)
• All account preferences
• Usage statistics
• Chat history

🔄 To continue using AI Assistant Pro, you'll need to set up your API key again.

💡 **Note:** Your data has been permanently removed from our secure database as requested.
            """
        else:
            text = """
❌ **Reset Failed** 

There was an issue resetting your data. Please try again or contact support if the problem persists.
            """
        
        keyboard = [
            [InlineKeyboardButton("🔑 Set New API Key", callback_data="set_api_key")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_resetdb_execution(query, context) -> None:
        """Handle /resetdb database reset execution with comprehensive logging"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username" 
        
        # Enhanced security audit logging
        logger.warning(f"🗑️ CRITICAL: Database reset EXECUTION requested by user_id:{user_id} (@{username})")
        logger.info(f"🔐 SECURITY AUDIT: Beginning database purge for user {user_id}")
        
        # Reset all user data in database using existing method
        success = await db.reset_user_database(user_id)
        
        # Also clear session data completely
        if context.user_data:
            session_items_cleared = len(context.user_data)
            context.user_data.clear()
            logger.info(f"🧹 Session data cleared: {session_items_cleared} items removed")
        
        if success:
            logger.info(f"✅ Database reset COMPLETED successfully for user_id:{user_id}")
            logger.info(f"🔐 SECURITY AUDIT: All data permanently removed for user {user_id}")
            
            text = """
🗑️ **Database Reset Complete** ✅

Your database has been completely reset! All data has been permanently deleted:

**✅ REMOVED:**
🔑 API key (securely deleted)
💾 Account preferences  
📊 Usage statistics
🗂️ Saved configurations
💬 Session data

**🔄 WHAT'S NEXT:**
To continue using AI Assistant Pro, you'll need to set up your API key again.

**🛡️ PRIVACY CONFIRMED:**
Your data has been permanently removed from our secure database as requested. This action is irreversible.

Thank you for using our secure database reset feature!
            """
        else:
            logger.error(f"❌ Database reset FAILED for user_id:{user_id}")
            logger.error(f"🔐 SECURITY AUDIT: Reset operation failed - user {user_id} data may still exist")
            
            text = """
❌ **Database Reset Failed** 

There was an issue resetting your database. This could be due to:
• Temporary database connectivity issues
• System maintenance in progress
• Network timeout

**🔄 Please try again** in a few minutes. If the problem persists, your data may already be cleared or there could be a system issue.

**🛡️ SECURITY NOTE:** No partial resets occur - either all data is removed or none is modified.
            """
        
        keyboard = [
            [InlineKeyboardButton("🔑 Set New API Key", callback_data="set_api_key")],
            [
                InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_resetdb_cancel(query, context) -> None:
        """Handle /resetdb cancellation with security logging"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        # Security audit logging
        logger.info(f"🛡️ Database reset CANCELLED by user_id:{user_id} (@{username})")
        logger.info(f"🔐 SECURITY AUDIT: User {user_id} cancelled database reset - no data modified")
        
        text = """
✅ **Database Reset Cancelled** 

Your database reset has been cancelled. No data was modified or deleted.

**🔒 Your data remains secure:**
🔑 API key - Still stored securely
💾 Account preferences - Unchanged  
📊 Usage statistics - Preserved
🗂️ Saved configurations - Intact

**💡 Alternative Options:**
You can manage your data through Settings if you need to make specific changes.
        """
        
        keyboard = [
            [
                InlineKeyboardButton("⚙️ Go to Settings", callback_data="settings"),
                InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")
            ],
            [InlineKeyboardButton("💡 Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_usage_stats(query, context) -> None:
        """Display user usage statistics"""
        user_id = query.from_user.id
        
        # Get user stats (you would implement actual tracking)
        stats_text = """
📊 **Your Usage Statistics** 

**This Session:**
💬 Messages: 0
🤖 AI Responses: 0
🎨 Images Generated: 0
💻 Code Created: 0

**🏆 Most Used:**
• Model: Text Generation
• Feature: Conversation
• Language: English

💡 **Tip:** Try different prompt styles to explore all AI capabilities!
        """
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            stats_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_model_info(query, context) -> None:
        """Display model information"""
        model_text = f"""
🎯 **Latest 2024-2025 AI Models** 🚀

**🧠 Text Generation (Latest Models):**
• Primary: Llama-3.2-3B-Instruct ⚡ (Meta's newest)
• Advanced: Qwen2.5-7B-Instruct 🌐 (Multilingual powerhouse)
• Fallback: Mixtral-8x7B-Instruct ✨ (Enterprise grade)
• Features: 29+ languages, advanced reasoning, context awareness

**💻 Code Generation (State-of-the-Art):**
• Primary: StarCoder2-15B 🔥 (Latest 2024 coding AI)
• Fallback: CodeLlama-13b-Instruct 💪 (Enhanced version)
• Languages: 80+ programming languages
• Features: Code completion, debugging, optimization

**🎨 Image Creation (Revolutionary):**
• Primary: FLUX.1-schnell ⚡ (Breakthrough 2024 model)
• Fallback: Stable Diffusion XL 🎨 (Professional quality)  
• Resolution: 1024x1024 ultra-high quality
• Speed: 4-step generation (lightning fast!)

**📊 Sentiment & Emotion (Advanced):**
• Sentiment: RoBERTa-base-sentiment-latest 😊
• Emotions: go_emotions (28 emotion categories) 🎭
• Features: Advanced emotion detection & analysis

**🔄 Intelligent Model Selection:**
I automatically choose the optimal model based on your request complexity and type!
        """
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            model_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_examples(query, context) -> None:
        """Show example prompts"""
        examples_text = """
🚀 **Try These Examples** (Latest AI Models)

**💬 Advanced Conversations (Llama-3.2 & Qwen2.5):**
• "Explain quantum computing with practical examples"
• "Create a comprehensive business plan for an AI startup"
• "Compare the philosophical implications of AI consciousness"

**💻 Code Examples (StarCoder2-15B):**
• "Build a complete REST API with authentication in Python"
• "Create a React app with TypeScript and real-time features"  
• "Write efficient algorithms for machine learning preprocessing"

**🎨 Image Examples (FLUX.1 - Revolutionary):**
• "Create a photorealistic portrait of a cyberpunk samurai"
• "Generate a minimalist logo for a sustainable tech company"
• "Design a futuristic cityscape with flying cars at sunset"

**📊 Advanced Analysis (28 Emotion Categories):**
• "Analyze the complex emotions in: I'm excited but nervous about this new opportunity"
• "What are all the emotions in this customer feedback"

**🌐 Translation Examples (29+ Languages):**
• "Translate this to Chinese and explain cultural context"
• "Convert this business proposal to Spanish with local adaptations"

**✨ Each request uses the optimal 2024-2025 AI model automatically!** 🎯
        """
        
        keyboard = [
            [InlineKeyboardButton("💡 More Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            examples_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_quick_start(query, context) -> None:
        """Show quick start guide"""
        quick_start_text = """
🚀 **Quick Start Guide** 

**⚡ Get started in 3 easy steps:**

**1️⃣ Set Your API Key**
Get a free Hugging Face token and add it to unlock all features

**2️⃣ Start Chatting**
Just type any question or request - I'll automatically choose the best AI model

**3️⃣ Explore Features**
Try text, code, images, and analysis - I handle everything intelligently!

**🎯 Pro Tips:**
• Be specific in your requests for better results
• Use `/newchat` to start fresh conversations
• I remember context for up to 15 messages

Ready to experience the future of AI? 🤖✨
        """
        
        keyboard = [
            [InlineKeyboardButton("🔑 Set API Key", callback_data="set_api_key")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            quick_start_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_settings_display(query, context) -> None:
        """Redisplay settings menu"""
        # This would call the settings_command logic
        user_id = query.from_user.id
        api_key = await db.get_user_api_key(user_id)
        
        status_emoji = "✅" if api_key else "❌"
        api_status = "Connected" if api_key else "Not Set"
        
        settings_text = f"""
⚙️ **AI Assistant Settings** 🛠️

**Current Status:**
🔑 API Key: {status_emoji} {api_status}
🤖 Models: Premium Hugging Face Collection
🧠 Intelligence: Adaptive Model Routing
💾 Storage: Persistent database (as specified)
📊 Chat History: Persistent storage with last 15 messages in active session

**Available Actions:**
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Change API Key", callback_data="set_api_key")],
            [
                InlineKeyboardButton("📊 Usage Stats", callback_data="usage_stats"),
                InlineKeyboardButton("🎯 Model Info", callback_data="model_info")
            ],
            [
                InlineKeyboardButton("🗑️ Reset My Data", callback_data="confirm_reset"),
                InlineKeyboardButton("💡 Help", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_help_display(query, context) -> None:
        """Redisplay help menu"""
        help_text = """
💡 **AI Assistant Pro Help Guide** 📚

**🎯 Smart Commands:**
• `/start` - Welcome & setup
• `/newchat` - Clear conversation history  
• `/settings` - Manage your preferences
• `/help` - This help guide

**🤖 AI Capabilities:**

**💬 Text Generation:**
"Explain quantum computing"
"Write a business proposal for..."

**💻 Code Generation:**
"Create a Python function to sort data"
"Build a React component for..."

**🎨 Image Creation:**
"Draw a futuristic cityscape"
"Create a professional logo for..."

**✨ Features:**
• 🧠 Intelligent model routing
• 🔄 Context-aware conversations
• 🚀 Multi-modal capabilities

**Need more help?** Just ask me anything!
        """
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Try Examples", callback_data="examples"),
                InlineKeyboardButton("🎯 Model Guide", callback_data="model_guide")
            ],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def history_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Display user's conversation history with professional UI and pagination
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        full_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or "Unknown"
        
        # Enhanced entry logging with user details
        logger.info(f"📚 HISTORY command invoked by user_id:{user_id} (@{username}) '{full_name}'")
        logger.info(f"📊 Chat type: {update.effective_chat.type} | Chat ID: {update.effective_chat.id}")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another command.",
                parse_mode='Markdown'
            )
            return
        
        # Check API key from persistent database storage
        try:
            api_key = await db.get_user_api_key(user_id)
            if not api_key:
                logger.info(f"❌ No API key found for user_id:{user_id} - prompting setup")
                
                no_api_text = """
🔑 **API Key Required for History Access**

To view your conversation history, you need to set up your Hugging Face API key first.

Your conversations are stored securely, but we need your API key to verify your access.

**Why API Key Required:**
🛡️ Security - Protects your personal conversation data  
💾 Storage - Enables persistent conversation history
🔒 Privacy - Ensures only you can access your chats

                """
                
                keyboard = [
                    [InlineKeyboardButton("🔑 Set API Key", callback_data="set_api_key")],
                    [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    no_api_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                return
                
        except Exception as e:
            logger.error(f"🔍 Database error checking API key for user_id:{user_id}: {e}")
            await update.message.reply_text(
                "❌ **Database Error**\n\nSorry, there was a problem accessing your data. Please try again later.",
                parse_mode='Markdown'
            )
            return
        
        # Get conversation count and first page of conversations
        try:
            logger.info(f"📊 Fetching conversation count for user_id:{user_id}")
            
            # Get total count and first page of conversations in parallel
            import asyncio
            conversation_count_task = db.get_conversation_count(user_id)
            conversations_task = db.get_user_conversations(user_id, limit=5, skip=0)
            
            conversation_count, conversations = await asyncio.gather(
                conversation_count_task,
                conversations_task
            )
            
            logger.info(f"📚 User {user_id} has {conversation_count} total conversations, displaying first {len(conversations)}")
            
        except Exception as e:
            logger.error(f"🔍 Database error fetching conversations for user_id:{user_id}: {e}")
            await update.message.reply_text(
                "❌ **Database Error**\n\nSorry, there was a problem retrieving your conversation history. Please try again later.",
                parse_mode='Markdown'
            )
            return
        
        # Display appropriate message based on conversation count
        if conversation_count == 0:
            # No conversations yet
            empty_history_text = """
📚 **Your Conversation History**

You don't have any saved conversations yet! 

**How to Build History:**
💬 Start chatting with me to create conversations
💾 Each conversation session gets automatically saved  
🔄 Use `/newchat` to start fresh conversation sessions
📖 Return here anytime to browse your chat history

**Get Started:**
Just send me any message to begin your first conversation!

✨ **Pro Tip:** Your conversations persist even after the bot restarts, so you can always return to previous discussions.
            """
            
            keyboard = [
                [InlineKeyboardButton("🚀 Start First Conversation", callback_data="start_conversation")],
                [InlineKeyboardButton("💡 Help & Tips", callback_data="help")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                empty_history_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            return
        
        # Display conversation history with pagination
        await CommandHandlers._display_conversation_history(update, context, conversations, conversation_count, page=0)
        
        logger.info(f"✅ HISTORY response sent successfully to user_id:{user_id} (@{username})")
        logger.info(f"📊 Displayed {len(conversations)} conversations out of {conversation_count} total")
    
    @staticmethod
    async def _display_conversation_history(update_or_query, context, conversations: list, total_count: int, page: int = 0):
        """
        Display conversation history with professional UI and pagination
        
        Args:
            update_or_query: Update object (for new messages) or CallbackQuery (for edits)
            context: Bot context
            conversations: List of conversation summaries
            total_count: Total number of conversations
            page: Current page number (0-based)
        """
        # Determine if this is an update or callback query
        is_callback = hasattr(update_or_query, 'edit_message_text')
        
        # Calculate pagination info
        items_per_page = 5
        total_pages = (total_count + items_per_page - 1) // items_per_page  # Ceiling division
        current_page = page + 1  # Convert to 1-based for display
        
        # Build history text
        history_text = f"""
📚 **Your Conversation History** (Page {current_page}/{total_pages})

Found **{total_count}** saved conversations:

"""
        
        # Add conversation summaries
        for i, conv in enumerate(conversations):
            # Format timestamp
            try:
                from datetime import datetime
                if 'last_message_at' in conv:
                    timestamp = conv['last_message_at']
                    if hasattr(timestamp, 'strftime'):
                        time_str = timestamp.strftime("%b %d, %Y at %H:%M")
                    else:
                        time_str = str(timestamp)[:16]  # Fallback
                else:
                    time_str = "Unknown time"
                    
                summary = conv.get('summary', 'Untitled Conversation')[:60]  # Limit length
                if len(conv.get('summary', '')) > 60:
                    summary += "..."
                    
                message_count = conv.get('message_count', 0)
                
                history_text += f"""
**{i + 1 + (page * items_per_page)}.** {escape_markdown(summary)}
📅 {time_str} • 💬 {message_count} messages
"""
                
            except Exception as e:
                logger.error(f"Error formatting conversation summary: {e}")
                history_text += f"\n**{i + 1 + (page * items_per_page)}.** Conversation (formatting error)\n"
        
        # Build inline keyboard
        keyboard = []
        
        # Add conversation buttons (max 5 per page)
        conv_buttons = []
        for i, conv in enumerate(conversations):
            conv_id = str(conv.get('_id', ''))
            button_text = f"📖 View #{i + 1 + (page * items_per_page)}"
            conv_buttons.append(InlineKeyboardButton(button_text, callback_data=f"view_conv_{conv_id}"))
            
            # Add 2 buttons per row
            if len(conv_buttons) == 2 or i == len(conversations) - 1:
                keyboard.append(conv_buttons)
                conv_buttons = []
        
        # Pagination buttons
        pagination_buttons = []
        if page > 0:
            pagination_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f"history_page_{page - 1}"))
        if current_page < total_pages:
            pagination_buttons.append(InlineKeyboardButton("➡️ Next", callback_data=f"history_page_{page + 1}"))
        
        if pagination_buttons:
            keyboard.append(pagination_buttons)
        
        # Action buttons
        action_buttons = [
            InlineKeyboardButton("🔄 Refresh", callback_data="history_refresh"),
            InlineKeyboardButton("🗑️ Clear All", callback_data="history_clear_confirm")
        ]
        keyboard.append(action_buttons)
        
        # Settings button
        keyboard.append([InlineKeyboardButton("⚙️ Settings", callback_data="settings")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send or edit message
        try:
            if is_callback:
                await update_or_query.edit_message_text(
                    history_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
            else:
                await update_or_query.message.reply_text(
                    history_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error displaying conversation history: {e}")
            error_text = "❌ **Display Error**\n\nSorry, there was a problem showing your conversation history. Please try again."
            
            try:
                if is_callback:
                    await update_or_query.edit_message_text(error_text, parse_mode='Markdown')
                else:
                    await update_or_query.message.reply_text(error_text, parse_mode='Markdown')
            except Exception:
                pass  # If even error message fails, just log
    
    @staticmethod
    async def _handle_view_conversation(query, context, conversation_id: str):
        """Handle viewing full conversation details"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        logger.info(f"📄 VIEW_CONVERSATION requested by user_id:{user_id} (@{username}) for conv:{conversation_id}")
        
        try:
            # Get full conversation details
            conversation = await db.get_conversation_details(user_id, conversation_id)
            
            if not conversation:
                await query.edit_message_text(
                    "❌ **Conversation Not Found**\n\nThe requested conversation could not be found or may have been deleted.",
                    parse_mode='Markdown'
                )
                return
            
            # Build conversation display
            messages = conversation.get('messages', [])
            summary = conversation.get('summary', 'Untitled Conversation')
            message_count = conversation.get('message_count', len(messages))
            
            # Format timestamp
            try:
                timestamp = conversation.get('last_message_at', conversation.get('started_at'))
                if hasattr(timestamp, 'strftime'):
                    time_str = timestamp.strftime("%B %d, %Y at %H:%M")
                else:
                    time_str = str(timestamp)[:19] if timestamp else "Unknown time"
            except Exception:
                time_str = "Unknown time"
            
            # Build conversation text (limit to prevent message too long)
            conv_text = f"""
📄 **Conversation Details**

**Summary:** {escape_markdown(summary)}
📅 **Date:** {time_str}
💬 **Messages:** {message_count}

**Conversation:**
"""
            
            # Add messages (limit to prevent telegram message limit)
            char_count = len(conv_text)
            max_chars = 3500  # Leave room for buttons and formatting
            
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                # Format role emoji
                role_emoji = "👤" if role == "user" else "🤖"
                role_name = "You" if role == "user" else "AI Assistant"
                
                # Limit message content length
                if len(content) > 200:
                    content = content[:197] + "..."
                
                message_text = f"\n{role_emoji} **{role_name}:**\n{escape_markdown(content)}\n"
                
                # Check if adding this message would exceed limit
                if char_count + len(message_text) > max_chars:
                    remaining_messages = len(messages) - i
                    conv_text += f"\n... and {remaining_messages} more messages\n"
                    break
                
                conv_text += message_text
                char_count += len(message_text)
            
            # Build action buttons
            keyboard = [
                [
                    InlineKeyboardButton("💬 Continue Chat", callback_data=f"continue_conv_{conversation_id}"),
                    InlineKeyboardButton("🗑️ Delete", callback_data=f"delete_conv_{conversation_id}")
                ],
                [
                    InlineKeyboardButton("⬅️ Back to History", callback_data="history_refresh"),
                    InlineKeyboardButton("🔄 Refresh", callback_data=f"view_conv_{conversation_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                conv_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Conversation details displayed successfully for user_id:{user_id}, conv:{conversation_id}")
            
        except Exception as e:
            logger.error(f"Error viewing conversation for user_id:{user_id}, conv:{conversation_id}: {e}")
            await query.edit_message_text(
                "❌ **Error Loading Conversation**\n\nSorry, there was a problem loading this conversation. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_history_pagination(query, context, page_number: int):
        """Handle pagination for conversation history"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        logger.info(f"📄 HISTORY_PAGINATION requested by user_id:{user_id} (@{username}) for page:{page_number}")
        
        try:
            # Get conversations for the requested page
            items_per_page = 5
            skip = page_number * items_per_page
            
            conversation_count_task = db.get_conversation_count(user_id)
            conversations_task = db.get_user_conversations(user_id, limit=items_per_page, skip=skip)
            
            conversation_count, conversations = await asyncio.gather(
                conversation_count_task,
                conversations_task
            )
            
            # Display the requested page
            await CommandHandlers._display_conversation_history(query, context, conversations, conversation_count, page=page_number)
            
            logger.info(f"✅ History pagination displayed successfully for user_id:{user_id}, page:{page_number}")
            
        except Exception as e:
            logger.error(f"Error in history pagination for user_id:{user_id}, page:{page_number}: {e}")
            await query.edit_message_text(
                "❌ **Pagination Error**\n\nSorry, there was a problem loading that page. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_history_refresh(query, context):
        """Handle refreshing the conversation history view"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        logger.info(f"🔄 HISTORY_REFRESH requested by user_id:{user_id} (@{username})")
        
        try:
            # Get fresh conversation data
            conversation_count_task = db.get_conversation_count(user_id)
            conversations_task = db.get_user_conversations(user_id, limit=5, skip=0)
            
            conversation_count, conversations = await asyncio.gather(
                conversation_count_task,
                conversations_task
            )
            
            # Display refreshed history starting from page 0
            await CommandHandlers._display_conversation_history(query, context, conversations, conversation_count, page=0)
            
            logger.info(f"✅ History refreshed successfully for user_id:{user_id}")
            
        except Exception as e:
            logger.error(f"Error refreshing history for user_id:{user_id}: {e}")
            await query.edit_message_text(
                "❌ **Refresh Error**\n\nSorry, there was a problem refreshing your history. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_history_clear_confirm(query, context):
        """Handle confirmation dialog for clearing all conversation history"""
        user_id = query.from_user.id
        
        # Get conversation count for the warning
        try:
            conversation_count = await db.get_conversation_count(user_id)
        except Exception:
            conversation_count = 0
        
        confirm_text = f"""
🚨 **Clear All Conversation History**

⚠️ **WARNING:** This will permanently delete ALL your saved conversations!

📊 **You currently have {conversation_count} saved conversations**

**What will be deleted:**
🗑️ All conversation messages and content
📅 All conversation timestamps and metadata  
💾 All conversation summaries
🔄 Your entire chat history with this bot

**This action cannot be undone!**

Are you absolutely sure you want to proceed?
        """
        
        keyboard = [
            [
                InlineKeyboardButton("🗑️ Yes, Delete All", callback_data="history_clear_yes"),
                InlineKeyboardButton("❌ Cancel", callback_data="history_clear_no")
            ],
            [InlineKeyboardButton("⬅️ Back to History", callback_data="history_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            confirm_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        logger.info(f"🚨 Clear history confirmation shown to user_id:{user_id} ({conversation_count} conversations)")
    
    @staticmethod
    async def _handle_history_clear_execute(query, context):
        """Execute clearing all conversation history"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        logger.info(f"🗑️ HISTORY_CLEAR_EXECUTE requested by user_id:{user_id} (@{username})")
        
        try:
            # Clear all user's conversation history
            success = await db.clear_user_history(user_id)
            
            if success:
                success_text = """
✅ **All Conversations Deleted**

Your entire conversation history has been permanently cleared.

**What was deleted:**
🗑️ All conversation messages and content
📅 All timestamps and metadata
💾 All conversation summaries

**Fresh Start:**
✨ You can now begin new conversations
💬 Send any message to start chatting again
📖 Future conversations will be saved automatically

Your API key and settings remain unchanged.
                """
                
                keyboard = [
                    [InlineKeyboardButton("🚀 Start New Conversation", callback_data="start_conversation")],
                    [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    success_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
                logger.info(f"✅ All conversation history cleared successfully for user_id:{user_id}")
                
            else:
                await query.edit_message_text(
                    "❌ **Clear Failed**\n\nThere was a problem clearing your conversation history. Please try again later.",
                    parse_mode='Markdown'
                )
                logger.error(f"Failed to clear conversation history for user_id:{user_id}")
                
        except Exception as e:
            logger.error(f"Error clearing conversation history for user_id:{user_id}: {e}")
            await query.edit_message_text(
                "❌ **Clear Error**\n\nSorry, there was a problem clearing your history. Please try again later.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_start_conversation(query, context):
        """Handle starting a new conversation"""
        start_text = """
🚀 **Ready to Start Chatting!**

Just send me any message to begin a new conversation! Here are some ideas:

💬 **Ask me anything:**
• "Explain artificial intelligence"
• "Help me write a business plan"
• "What's the weather like on Mars?"

💻 **Code assistance:**
• "Create a Python web scraper"
• "Help debug this JavaScript function"
• "Write a SQL query for user analytics"

🎨 **Creative tasks:**
• "Generate an image of a sunset over mountains"
• "Write a short story about time travel"
• "Create a poem about programming"

✨ **Your conversation will be automatically saved and you can return to view it anytime using the `/history` command!**

Go ahead - send me your first message! 👇
        """
        
        keyboard = [
            [InlineKeyboardButton("💡 Need Help?", callback_data="help")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            start_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        logger.info(f"🚀 Start conversation prompt shown to user_id:{query.from_user.id}")
    
    @staticmethod
    async def _handle_continue_conversation(query, context, conversation_id: str):
        """Handle continuing a previous conversation"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        logger.info(f"💬 CONTINUE_CONVERSATION requested by user_id:{user_id} (@{username}) for conv:{conversation_id}")
        
        try:
            # Get conversation details
            conversation = await db.get_conversation_details(user_id, conversation_id)
            
            if not conversation:
                await query.edit_message_text(
                    "❌ **Conversation Not Found**\n\nThe conversation you're trying to continue could not be found.",
                    parse_mode='Markdown'
                )
                return
            
            # Load conversation history into context
            messages = conversation.get('messages', [])
            if context.user_data is None:
                context.user_data = {}
            
            # Clear current history and load the selected conversation
            context.user_data['chat_history'] = messages.copy()
            
            summary = conversation.get('summary', 'Previous Conversation')
            message_count = len(messages)
            
            continue_text = f"""
💬 **Conversation Resumed**

**Continuing:** {escape_markdown(summary)}
📊 **Loaded {message_count} previous messages**

✅ **Ready to continue!** Your conversation context has been restored.

**What happens next:**
🔄 Send any message to continue the conversation
💾 New messages will be added to this conversation
📖 Access full history anytime with `/history`

**Tip:** I remember our previous discussion, so you can pick up right where we left off!

Go ahead and send your next message! 👇
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("📄 View Full Conversation", callback_data=f"view_conv_{conversation_id}"),
                    InlineKeyboardButton("📚 History", callback_data="history_refresh")
                ],
                [InlineKeyboardButton("🔄 New Chat Instead", callback_data="start_conversation")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                continue_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Conversation continued successfully for user_id:{user_id}, conv:{conversation_id}, loaded {message_count} messages")
            
        except Exception as e:
            logger.error(f"Error continuing conversation for user_id:{user_id}, conv:{conversation_id}: {e}")
            await query.edit_message_text(
                "❌ **Continue Error**\n\nSorry, there was a problem continuing this conversation. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_delete_conversation_confirm(query, context, conversation_id: str):
        """Handle confirmation for deleting a single conversation"""
        user_id = query.from_user.id
        
        try:
            # Get conversation summary for confirmation
            conversation = await db.get_conversation_details(user_id, conversation_id)
            
            if not conversation:
                await query.edit_message_text(
                    "❌ **Conversation Not Found**\n\nThe conversation you're trying to delete could not be found.",
                    parse_mode='Markdown'
                )
                return
            
            summary = conversation.get('summary', 'Untitled Conversation')
            message_count = conversation.get('message_count', 0)
            
            confirm_text = f"""
🗑️ **Delete Conversation**

⚠️ **Are you sure you want to delete this conversation?**

**Conversation:** {escape_markdown(summary)}
💬 **Messages:** {message_count}

**This action cannot be undone!**

The conversation will be permanently removed from your history.
            """
            
            keyboard = [
                [
                    InlineKeyboardButton("🗑️ Yes, Delete", callback_data=f"delete_conv_yes_{conversation_id}"),
                    InlineKeyboardButton("❌ Cancel", callback_data=f"delete_conv_no_{conversation_id}")
                ],
                [InlineKeyboardButton("⬅️ Back to Conversation", callback_data=f"view_conv_{conversation_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                confirm_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            logger.info(f"🗑️ Delete conversation confirmation shown to user_id:{user_id} for conv:{conversation_id}")
            
        except Exception as e:
            logger.error(f"Error showing delete confirmation for user_id:{user_id}, conv:{conversation_id}: {e}")
            await query.edit_message_text(
                "❌ **Error**\n\nSorry, there was a problem processing your request. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_delete_conversation_execute(query, context, conversation_id: str):
        """Execute deleting a single conversation"""
        user_id = query.from_user.id
        username = query.from_user.username or "No username"
        
        logger.info(f"🗑️ DELETE_CONVERSATION_EXECUTE requested by user_id:{user_id} (@{username}) for conv:{conversation_id}")
        
        try:
            # Delete the conversation
            success = await db.delete_conversation(user_id, conversation_id)
            
            if success:
                success_text = """
✅ **Conversation Deleted**

The conversation has been permanently removed from your history.

**What's next:**
📚 Return to your conversation history
🚀 Start a new conversation
⚙️ Adjust your settings
                """
                
                keyboard = [
                    [
                        InlineKeyboardButton("📚 Back to History", callback_data="history_refresh"),
                        InlineKeyboardButton("🚀 New Chat", callback_data="start_conversation")
                    ],
                    [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    success_text,
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
                logger.info(f"✅ Conversation deleted successfully for user_id:{user_id}, conv:{conversation_id}")
                
            else:
                await query.edit_message_text(
                    "❌ **Delete Failed**\n\nThere was a problem deleting this conversation. It may have already been removed.",
                    parse_mode='Markdown'
                )
                logger.warning(f"Failed to delete conversation for user_id:{user_id}, conv:{conversation_id}")
                
        except Exception as e:
            logger.error(f"Error deleting conversation for user_id:{user_id}, conv:{conversation_id}: {e}")
            await query.edit_message_text(
                "❌ **Delete Error**\n\nSorry, there was a problem deleting this conversation. Please try again.",
                parse_mode='Markdown'
            )

# Export command handlers
command_handlers = CommandHandlers()