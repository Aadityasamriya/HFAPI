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

# Set sensitive loggers to WARNING to prevent token exposure
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class AIAssistantBot:
    """Main bot application class"""
    
    def __init__(self):
        self.application = None
    
    async def post_init(self, app: Application) -> None:
        """Post-initialization hook for database connection"""
        try:
            if not db.connected:
                await db.connect()
                logger.info("Database connected via post_init")
            else:
                logger.info("Database already connected - skipping post_init connection")
        except Exception as e:
            logger.error(f"Error in post_init: {e}")
            # Don't fail the entire bot if database connection fails in post_init
            pass
    
    async def post_shutdown(self, app: Application) -> None:
        """Post-shutdown hook for cleanup"""
        await db.disconnect()
        logger.info("Database disconnected via post_shutdown")
    
    async def initialize(self):
        """Initialize bot application and database"""
        try:
            # Validate configuration
            Config.validate_config()
            logger.info("Configuration validated successfully")
            
            # Connect to database
            await db.connect()
            logger.info("Database connected successfully")
            
            # Create application
            if not Config.TELEGRAM_BOT_TOKEN:
                raise ValueError("TELEGRAM_BOT_TOKEN is required")
            self.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            
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
        
        # Build application without lifecycle hooks to avoid blocking issues
        if not Config.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        bot.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        
        # Register handlers
        bot._register_handlers()
        logger.info("Bot handlers registered successfully")
        
        logger.info("Starting AI Assistant Pro bot...")
        
        # Connect to database before starting polling - REQUIRED for bot functionality
        logger.info("🔗 Initializing database connection...")
        try:
            if not db.connected:
                # Database connection is critical - fail fast if it fails
                async def init_db():
                    await db.connect()
                    logger.info("✅ Database connected successfully")
                
                asyncio.run(init_db())
            else:
                logger.info("✅ Database already connected")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: Database connection failed: {e}")
            logger.error("Database is required for API key storage and bot functionality")
            logger.error("Please check your MONGO_URI and database configuration")
            raise RuntimeError(f"Failed to connect to database: {e}. Bot cannot operate without database.")
        
        logger.info("🎯 Initializing Telegram polling...")
        logger.info("🚀 AI Assistant Pro is now running! Bot will run indefinitely...")
        logger.info("✨ Using latest 2024-2025 AI models: FLUX.1, StarCoder2-15B, Llama-3.2, Qwen2.5")
        logger.info("📡 Listening for messages... Press Ctrl+C to stop")
        
        # Start polling - this will handle its own event loop and token validation
        if bot.application is not None:
            bot.application.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
        else:
            raise RuntimeError("Bot application not initialized")
        
        # This line should never be reached if polling is working correctly
        logger.info("🔄 Polling ended normally")
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    main()