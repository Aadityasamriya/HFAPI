"""
AI Assistant Pro - Advanced Telegram Bot
Sophisticated AI orchestrator with intelligent model routing
"""

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

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
        await db.connect()
        logger.info("Database connected via post_init")
    
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
        
        # Build application with lifecycle hooks
        if not Config.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        bot.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).post_init(bot.post_init).post_shutdown(bot.post_shutdown).build()
        
        # Register handlers
        bot._register_handlers()
        logger.info("Bot handlers registered successfully")
        
        logger.info("Starting AI Assistant Pro bot...")
        
        # Start polling - this will handle its own event loop
        bot.application.run_polling(
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    main()