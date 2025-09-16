"""
AI Assistant Pro - Advanced Telegram Bot
Sophisticated AI orchestrator with intelligent model routing
"""

import asyncio
import logging

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
except ImportError as e:
    print(f"Telegram library import error: {e}")
    print("Please ensure python-telegram-bot is installed: pip install python-telegram-bot[ext]==22.4")
    exit(1)

from bot.config import Config
from bot.database import db
from bot.handlers.command_handlers import command_handlers
from bot.handlers.message_handlers import message_handlers

# Configure logging with security
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Enhanced logging levels for better observability
# Set to INFO instead of WARNING to capture useful telemetry
logging.getLogger('httpx').setLevel(logging.INFO)
logging.getLogger('telegram').setLevel(logging.INFO)
logging.getLogger('telegram.ext').setLevel(logging.INFO)

# Still restrict overly verbose loggers  
logging.getLogger('telegram.vendor').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Enhanced observability logging (after logger is defined)
logger.info("📊 Enhanced logging levels configured for observability")
logger.info("🔍 Telegram/HTTP logs set to INFO for telemetry visibility")

class AIAssistantBot:
    """Main bot application class with comprehensive observability"""
    
    def __init__(self):
        self.application = None
        logger.info("🤖 AIAssistantBot instance created")
    
    async def _perform_startup_health_checks(self):
        """Perform comprehensive startup health checks"""
        logger.info("🩺 Starting comprehensive health checks...")
        
        try:
            # Health check 1: Bot token validation with getMe()
            logger.info("🔍 Validating bot token with getMe()...")
            bot_info = await self.application.bot.get_me()
            logger.info(f"✅ Bot authenticated: @{bot_info.username} (ID: {bot_info.id})")
            logger.info(f"📋 Bot details: {bot_info.first_name} | Can join groups: {bot_info.can_join_groups}")
            
        except Exception as e:
            logger.error(f"❌ Bot token validation failed: {e}")
            raise RuntimeError(f"Bot authentication failed: {e}")
        
        try:
            # Health check 2: Clear any existing webhook
            logger.info("🧹 Clearing any existing webhook for polling mode...")
            webhook_result = await self.application.bot.delete_webhook(drop_pending_updates=True)
            if webhook_result:
                logger.info("✅ Webhook cleared successfully")
            else:
                logger.warning("⚠️ No webhook was set or clearing failed")
                
        except Exception as e:
            logger.warning(f"⚠️ Webhook clearing failed (may be normal): {e}")
        
        try:
            # Health check 3: Set bot commands for better UX
            logger.info("📋 Setting bot commands menu...")
            from telegram import BotCommand
            
            commands = [
                BotCommand("start", "🚀 Welcome and setup"),
                BotCommand("newchat", "🔄 Clear chat history"), 
                BotCommand("settings", "⚙️ Bot settings"),
                BotCommand("help", "❓ Get help")
            ]
            
            await self.application.bot.set_my_commands(commands)
            logger.info(f"✅ Bot commands configured: {len(commands)} commands set")
            
        except Exception as e:
            logger.warning(f"⚠️ Bot commands setup failed (non-critical): {e}")
        
        logger.info("🩺 Health checks completed successfully")
    
    async def post_init(self, app: Application) -> None:
        """Post-initialization hook for database connection and health checks"""
        logger.info("🔄 Post-initialization started...")
        
        try:
            if not db.connected:
                logger.info("🔗 Connecting to database via post_init...")
                await db.connect()
                logger.info("✅ Database connected successfully via post_init")
            else:
                logger.info("✅ Database already connected - skipping duplicate connection")
                
            # Perform health checks after database is ready
            await self._perform_startup_health_checks()
            logger.info("🎯 Post-initialization completed successfully")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Post-initialization failed: {e}")
            logger.error("🚨 Database is required for API key storage and bot functionality")
            # This is critical - don't let the bot continue without database
            raise RuntimeError(f"Post-initialization failed: {e}. Bot cannot operate without database.")
    
    async def post_shutdown(self, app: Application) -> None:
        """Post-shutdown hook for comprehensive cleanup"""
        logger.info("🔄 Starting post-shutdown cleanup...")
        
        try:
            if db.connected:
                await db.disconnect()
                logger.info("✅ Database disconnected successfully")
            else:
                logger.info("ℹ️ Database already disconnected")
                
            logger.info("🎯 Post-shutdown cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during post-shutdown cleanup: {e}")
    
    async def initialize(self):
        """Initialize bot application and database"""
        try:
            # Validate configuration
            Config.validate_config()
            logger.info("Configuration validated successfully")
            
            # Connect to database
            await db.connect()
            logger.info("Database connected successfully")
            
            # Create application with enhanced logging
            if not Config.TELEGRAM_BOT_TOKEN:
                logger.error("❌ TELEGRAM_BOT_TOKEN environment variable is missing")
                raise ValueError("TELEGRAM_BOT_TOKEN is required")
                
            logger.info("🏗️ Building Telegram application...")
            self.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            logger.info("✅ Telegram application built successfully")
            
            # Register handlers
            self._register_handlers()
            logger.info("Bot handlers registered successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    def _register_handlers(self):
        """Register all bot handlers"""
        if self.application is None:
            raise RuntimeError("Application not initialized")
            
        # Command handlers
        self.application.add_handler(CommandHandler("start", command_handlers.start_command))
        self.application.add_handler(CommandHandler("newchat", command_handlers.newchat_command))
        self.application.add_handler(CommandHandler("settings", command_handlers.settings_command))
        self.application.add_handler(CommandHandler("help", command_handlers.help_command))
        
        # Callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(command_handlers.button_handler))
        
        # Message handler for text messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, message_handlers.text_message_handler)
        )
        
        # Error handler
        self.application.add_error_handler(message_handlers.error_handler)
        
        logger.info("All handlers registered successfully")
        logger.info(f"📊 Handler summary: {len(self.application.handlers)} handler groups configured")
        
        # Log detailed handler information for observability
        for group_id, handlers in self.application.handlers.items():
            logger.info(f"📋 Group {group_id}: {len(handlers)} handlers")
            for i, handler in enumerate(handlers):
                handler_type = type(handler).__name__
                logger.debug(f"  └── Handler {i}: {handler_type}")
    
    def start_polling(self):
        """Start the bot with polling"""
        try:
            logger.info("Starting AI Assistant Pro bot...")
            
            if self.application is None:
                raise RuntimeError("Application not initialized")
            
            # Start polling using run_polling which handles the event loop
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error during polling: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if db.connected:
                await db.disconnect()
                logger.info("Database disconnected")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point - synchronous version for Replit compatibility"""
    # Welcome message
    print("🤖 AI Assistant Pro - Starting Up...")
    print("🚀 Advanced Telegram Bot with Intelligent AI Routing")
    print("=" * 50)
    
    # Create bot instance
    bot = AIAssistantBot()
    
    # Initialize and run bot
    try:
        # Validate configuration first
        Config.validate_config()
        logger.info("Configuration validated successfully")
        
        # Build application with lifecycle hooks for proper database connection
        if not Config.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        bot.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).post_init(bot.post_init).post_shutdown(bot.post_shutdown).build()
        
        # Register handlers
        bot._register_handlers()
        logger.info("Bot handlers registered successfully")
        
        logger.info("🚀 Starting AI Assistant Pro bot...")
        logger.info("🎯 Initializing Telegram polling with enhanced observability...")
        logger.info("📡 Bot is now LIVE and listening for messages...")
        logger.info("✨ AI Models: FLUX.1, StarCoder2-15B, Llama-3.2, Qwen2.5")
        logger.info("🔍 Enhanced logging: User interactions, API calls, and errors tracked")
        logger.info("🛡️ Security: API keys redacted, sensitive data protected")
        logger.info("📊 Observability: All handlers and database operations logged")
        logger.info("⏹️ Press Ctrl+C to stop gracefully")
        
        # Start polling - this will handle its own event loop and token validation
        if bot.application is not None:
            bot.application.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
        else:
            raise RuntimeError("Bot application not initialized")
        
        # This line should never be reached if polling is working correctly
        logger.info("🔄 Polling ended normally (unexpected but graceful)")
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Bot crashed with error: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        import traceback
        logger.error(f"📋 Stack trace: {traceback.format_exc()}")
        raise
    finally:
        logger.info("🔄 Bot shutdown sequence complete")
        logger.info("👋 AI Assistant Pro has stopped")

if __name__ == "__main__":
    main()