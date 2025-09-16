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

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AIAssistantBot:
    """Main bot application class"""
    
    def __init__(self):
        self.application = None
    
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
    
    async def start_polling(self):
        """Start the bot with polling"""
        try:
            logger.info("Starting AI Assistant Pro bot...")
            
            # Start polling
            await self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
        except Exception as e:
            logger.error(f"Error during polling: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if db.connected:
                await db.disconnect()
                logger.info("Database disconnected")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point"""
    bot = AIAssistantBot()
    
    # Initialize bot
    initialized = await bot.initialize()
    if not initialized:
        logger.error("Failed to initialize bot. Exiting.")
        return
    
    # Start bot
    try:
        await bot.start_polling()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    # Welcome message
    print("🤖 AI Assistant Pro - Starting Up...")
    print("🚀 Advanced Telegram Bot with Intelligent AI Routing")
    print("=" * 50)
    
    # Run the bot
    asyncio.run(main())