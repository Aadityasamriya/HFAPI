#!/usr/bin/env python3
"""
Hugging Face By AadityaLabs AI - Main Entry Point
Superior Telegram Bot with Intelligent AI Routing that outperforms ChatGPT, Grok, and Gemini

PHASE 2: Advanced Features Implementation
- Intelligent Intent Classification System (≥90% accuracy)
- Secure File Processing Pipeline (Images, PDFs, ZIP)
- Streamlined User Onboarding with Inline Buttons
- Individual User Database System with Encryption
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Optional

# Core Telegram bot imports
try:
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
    from telegram.constants import ParseMode
    from telegram import Bot
except ImportError as e:
    print(f"❌ Critical Error: Telegram library not found: {e}")
    print("Please install: pip install python-telegram-bot[ext]==22.4")
    sys.exit(1)

# Bot configuration and components
from bot.config import Config
from bot.storage_manager import storage_manager
from bot.handlers.command_handlers import CommandHandlers
from bot.handlers.message_handlers import MessageHandlers
from bot.core.router import IntelligentRouter
from bot.core.intent_classifier import AdvancedIntentClassifier
from bot.file_processors import AdvancedFileProcessor
from bot.admin import AdminCommands, admin_system
from bot.dependency_validator import DependencyValidator
from health_server import health_server

# Enhanced AI Model Selection System - Superior to current AI systems
from bot.core.dynamic_model_selector import dynamic_model_selector
from bot.core.model_selection_explainer import model_selection_explainer
from bot.core.conversation_context_tracker import conversation_context_tracker
from bot.core.performance_predictor import performance_predictor
from bot.core.dynamic_fallback_strategy import dynamic_fallback_strategy

# Setup comprehensive logging with security
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', mode='a', encoding='utf-8')
    ]
)

# Disable overly verbose logs but keep important ones
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.INFO)
logging.getLogger('aiohttp').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class TelegramBotApplication:
    """
    Main application class for Hugging Face By AadityaLabs AI
    Handles initialization, lifecycle management, and graceful shutdown
    """
    
    def __init__(self):
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self.router: Optional[IntelligentRouter] = None
        self.intent_classifier: Optional[AdvancedIntentClassifier] = None
        self.file_processor: Optional[AdvancedFileProcessor] = None
        self.health_server_started = False
        self.shutdown_event = asyncio.Event()
        
        # Enhanced AI Model Selection System
        self.dynamic_model_selector = None
        self.enhanced_features_enabled = getattr(Config, 'ENHANCED_MODEL_SELECTION', True)
        
        # PHASE 2: Initialize core AI systems
        logger.info("🚀 Initializing Hugging Face By AadityaLabs AI - PHASE 2")
        
        # Initialize Enhanced AI Model Selection System
        if self.enhanced_features_enabled:
            logger.info("🎯 Initializing Enhanced AI Model Selection System - Superior to ChatGPT/Grok/Gemini")
            self.dynamic_model_selector = dynamic_model_selector
            logger.info("✅ Enhanced model selection system initialized")
        logger.info("🧠 Advanced Features: Intent Classification, File Processing, User Onboarding")
        
    async def validate_dependencies(self) -> bool:
        """
        Validate all critical dependencies before starting
        
        Returns:
            bool: True if all critical dependencies are valid
        """
        logger.info("🔍 Validating critical dependencies...")
        
        try:
            dependency_validator = DependencyValidator()
            
            # Validate critical dependencies
            if not dependency_validator.validate_critical_dependencies():
                logger.error("❌ Critical dependency validation failed")
                for error in dependency_validator.validation_results['errors']:
                    logger.error(f"   • {error}")
                return False
            
            # Validate environment variables
            if not dependency_validator.validate_environment_variables():
                logger.error("❌ Environment validation failed")
                for error in dependency_validator.validation_results['errors']:
                    logger.error(f"   • {error}")
                return False
            
            # Log warnings for optional dependencies
            if dependency_validator.validation_results['warnings']:
                logger.warning("⚠️ Optional dependency warnings:")
                for warning in dependency_validator.validation_results['warnings']:
                    logger.warning(f"   • {warning}")
            
            logger.info("✅ Dependency validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency validation failed with error: {e}")
            return False

    async def validate_production_security(self) -> bool:
        """
        Validate production security requirements
        
        Returns:
            bool: True if production security is valid
        """
        logger.info("🔒 Validating production security configuration...")
        
        # Check environment type
        env_type = Config.get_environment_type()
        is_production = env_type.lower() in ['production', 'prod']
        
        if is_production:
            logger.info("🏭 Production environment detected - enforcing strict security")
            
            # CRITICAL: Enforce HTTPS/TLS in production
            if not os.getenv('FORCE_HTTPS', '').lower() in ['true', '1', 'yes']:
                logger.warning("⚠️ FORCE_HTTPS not set - consider setting for production security")
            
            # CRITICAL: Require explicit encryption seed in production
            if not os.getenv('ENCRYPTION_SEED'):
                logger.error("❌ ENCRYPTION_SEED must be explicitly set in production")
                logger.error("   Auto-generated seeds are not allowed in production")
                return False
                
            # CRITICAL: Validate token security
            token = Config.TELEGRAM_BOT_TOKEN
            if token and (len(token) < 40 or ':' not in token):
                logger.error("❌ TELEGRAM_BOT_TOKEN appears invalid for production")
                return False
                
            # CRITICAL: Require secure database connections
            if Config.MONGODB_URI and not Config.MONGODB_URI.startswith(('mongodb+srv://', 'mongodb://')) and 'ssl=true' not in Config.MONGODB_URI.lower():
                logger.warning("⚠️ MongoDB connection should use SSL/TLS in production")
                
            # CRITICAL: Validate owner ID is set for admin functions
            if not Config.OWNER_ID:
                logger.warning("⚠️ OWNER_ID not set - admin functions may be insecure")
                
            logger.info("✅ Production security validation completed")
        else:
            logger.info(f"🛠️ Development environment ({env_type}) - relaxed security validation")
            
        return True

    async def validate_database_connectivity(self) -> bool:
        """
        Validate database connectivity and health
        
        Returns:
            bool: True if database is healthy
        """
        logger.info("💾 Validating database connectivity...")
        
        try:
            # Test database connectivity through storage manager
            if not storage_manager.initialized:
                await storage_manager.initialize()
                
            # Perform health check
            is_healthy = await storage_manager.health_check()
            if not is_healthy:
                logger.error("❌ Database health check failed")
                return False
                
            logger.info("✅ Database connectivity validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database validation failed: {e}")
            return False
    
    async def validate_configuration(self) -> bool:
        """
        Validate critical configuration before starting
        
        Returns:
            bool: True if configuration is valid
        """
        logger.info("🔧 Validating bot configuration...")
        
        # Check for required Telegram bot token
        if not Config.TELEGRAM_BOT_TOKEN:
            logger.error("❌ TELEGRAM_BOT_TOKEN is required but not set")
            logger.error("   Please set TELEGRAM_BOT_TOKEN environment variable")
            return False
        
        # Check database configuration
        if not Config.has_mongodb_config() and not Config.has_supabase_config():
            logger.error("❌ No database configuration found")
            logger.error("   Please set MONGODB_URI or SUPABASE_MGMT_URL")
            return False
        
        # Validate encryption seed for production security
        try:
            Config.ensure_encryption_seed()
            logger.info("🔐 Encryption configuration validated")
        except ValueError as e:
            logger.error(f"❌ Encryption validation failed: {e}")
            return False
        
        # Check AI functionality status
        ai_available, ai_status = Config.is_ai_functionality_available()
        if ai_available:
            logger.info(f"🤖 AI Functionality: {ai_status}")
        else:
            logger.warning(f"⚠️ AI Functionality Limited: {ai_status}")
            logger.warning("   Bot will work with reduced functionality")
        
        logger.info("✅ Configuration validation completed successfully")
        return True
    
    async def initialize_core_systems(self) -> bool:
        """
        Initialize core AI systems for PHASE 2 features
        
        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("🧠 Initializing PHASE 2: Advanced AI Systems...")
            
            # Initialize Intelligent Router (PHASE 2 Feature 1)
            logger.info("📡 Initializing Intelligent Router...")
            self.router = IntelligentRouter()
            logger.info("✅ Intelligent Router initialized with superior model selection")
            
            # Initialize Advanced Intent Classifier (PHASE 2 Feature 1)
            logger.info("🎯 Initializing Advanced Intent Classifier...")
            self.intent_classifier = AdvancedIntentClassifier()
            logger.info("✅ Intent Classifier initialized with ≥90% accuracy target")
            
            # Initialize Advanced File Processor (PHASE 2 Feature 2)
            logger.info("📁 Initializing Advanced File Processor...")
            self.file_processor = AdvancedFileProcessor()
            logger.info("✅ File Processor initialized with security features")
            
            # Initialize Storage Manager (PHASE 2 Feature 4)
            logger.info("💾 Initializing Individual User Database System...")
            await storage_manager.initialize()
            logger.info("✅ Database system initialized with encryption and per-user isolation")
            
            # Initialize Admin System
            logger.info("👑 Initializing Admin System...")
            await admin_system.initialize()
            logger.info("✅ Admin system initialized")
            
            logger.info("🎉 All PHASE 2 core systems initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Core system initialization failed: {e}")
            return False
    
    async def register_handlers(self):
        """
        Register all command and message handlers for PHASE 2 features
        CRITICAL FIX: Added comprehensive exception handling to prevent crashes
        """
        if not self.application:
            raise RuntimeError("Application not initialized")
        
        logger.info("📝 Registering PHASE 2 handlers...")
        
        # CRITICAL FIX: Track registration success/failure for graceful degradation
        registered_handlers = 0
        failed_handlers = []
        
        # Helper function for safe handler registration
        def safe_register_handler(handler_type, handler_name, handler_func, *args):
            nonlocal registered_handlers
            # Ensure application is available before attempting registration
            if not self.application:
                error_msg = "Application not initialized"
                failed_handlers.append({"type": handler_type, "name": handler_name, "error": error_msg})
                logger.error(f"❌ Failed to register {handler_type} handler '{handler_name}': {error_msg}")
                return False
                
            try:
                if handler_type == "command":
                    self.application.add_handler(CommandHandler(handler_name, handler_func))
                elif handler_type == "message":
                    # Fix: MessageHandler(filters, callback_func) - correct argument order
                    if args:
                        self.application.add_handler(MessageHandler(args[0], handler_func))
                    else:
                        raise ValueError("Message handler requires filter argument")
                elif handler_type == "callback":
                    self.application.add_handler(CallbackQueryHandler(handler_func))
                registered_handlers += 1
                logger.debug(f"✅ Registered {handler_type} handler: {handler_name}")
                return True
            except Exception as e:
                failed_handlers.append({"type": handler_type, "name": handler_name, "error": str(e)})
                logger.error(f"❌ Failed to register {handler_type} handler '{handler_name}': {e}")
                return False
        
        # PHASE 2 Feature 3: Streamlined User Onboarding Commands with exception handling
        safe_register_handler("command", "start", CommandHandlers.start_command)
        safe_register_handler("command", "settings", CommandHandlers.settings_command)
        safe_register_handler("command", "newchat", CommandHandlers.newchat_command)
        
        # Legacy commands for backward compatibility with exception handling
        safe_register_handler("command", "setup", CommandHandlers.setup_command)
        safe_register_handler("command", "status", CommandHandlers.status_command)
        safe_register_handler("command", "resetdb", CommandHandlers.resetdb_command)
        safe_register_handler("command", "help", CommandHandlers.help_command)
        
        # Admin commands with exception handling
        safe_register_handler("command", "admin", CommandHandlers.admin_command)
        safe_register_handler("command", "bootstrap", CommandHandlers.bootstrap_command)
        safe_register_handler("command", "broadcast", CommandHandlers.broadcast_command)
        safe_register_handler("command", "stats", CommandHandlers.admin_stats_command)
        safe_register_handler("command", "users", CommandHandlers.admin_users_command)
        
        # CRITICAL FIX: Register message handlers with proper exception handling
        safe_register_handler("callback", "button_handler", CommandHandlers.button_handler)
        safe_register_handler("message", "text_message_handler", MessageHandlers.text_message_handler, 
                            filters.TEXT & ~filters.COMMAND)
        safe_register_handler("message", "photo_handler", MessageHandlers.photo_handler, filters.PHOTO)
        safe_register_handler("message", "document_handler", MessageHandlers.document_handler, filters.Document.ALL)
        safe_register_handler("message", "voice_handler", MessageHandlers.voice_handler, filters.VOICE)
        safe_register_handler("message", "audio_handler", MessageHandlers.audio_handler, filters.AUDIO)
        
        # FIXED: Dynamic handler count calculation instead of hardcoded value
        # Count total handlers attempted (sum of all safe_register_handler calls)
        total_expected_handlers = registered_handlers + len(failed_handlers)
        success_rate = (registered_handlers / total_expected_handlers) * 100 if total_expected_handlers > 0 else 0
        
        if failed_handlers:
            logger.warning(f"⚠️ Handler Registration Summary: {registered_handlers}/{total_expected_handlers} succeeded ({success_rate:.1f}%)")
            logger.warning("❌ Failed handlers:")
            for handler in failed_handlers:
                logger.warning(f"   • {handler['type']} '{handler['name']}': {handler['error']}")
            
            # CRITICAL FIX: Continue with degraded functionality instead of crashing
            if registered_handlers == 0:
                raise RuntimeError("CRITICAL: No handlers could be registered - bot cannot function")
            elif len(failed_handlers) > 5:  # More than 35% failure rate
                logger.error(f"🚨 HIGH FAILURE RATE: {len(failed_handlers)} handlers failed - bot functionality severely limited")
        else:
            logger.info("✅ CRITICAL message handlers registered successfully")
        
        logger.info("🎯 Core systems (Intent Classification, File Processing, Database) are ready")
        logger.info(f"✅ Handler Registration Complete: {registered_handlers}/{total_expected_handlers} handlers active ({success_rate:.1f}% success rate)")
        
        # Log handler statistics
        handler_count = len(self.application.handlers[0])  # Default group handlers
        logger.info(f"📊 Total handlers registered: {handler_count}")
        logger.info("🎯 PHASE 2 Features Active:")
        logger.info("   • Intelligent Intent Classification (≥90% accuracy)")
        logger.info("   • Secure File Processing (Images, PDFs, ZIP)")
        logger.info("   • Streamlined User Onboarding (3 core commands)")
        logger.info("   • Individual User Database (Encrypted per-user isolation)")
    
    async def start_bot(self):
        """
        Start the Telegram bot with comprehensive error handling
        """
        try:
            # CRITICAL: Validate dependencies first (fail-fast behavior)
            if not await self.validate_dependencies():
                logger.error("❌ Dependency validation failed, cannot start bot")
                return False
            
            # CRITICAL: Validate production security requirements
            if not await self.validate_production_security():
                logger.error("❌ Production security validation failed, cannot start bot")
                return False
            
            # Validate configuration
            if not await self.validate_configuration():
                logger.error("❌ Configuration validation failed, cannot start bot")
                return False
            
            # CRITICAL: Validate database connectivity and health
            if not await self.validate_database_connectivity():
                logger.error("❌ Database validation failed, cannot start bot")
                return False
            
            # Initialize core systems
            if not await self.initialize_core_systems():
                logger.error("❌ Core system initialization failed, cannot start bot")
                return False
            
            # Start health check web server for Railway.com monitoring
            try:
                await health_server.start()
                self.health_server_started = True
                logger.info("✅ Health check web server started for Railway.com monitoring")
            except Exception as e:
                logger.error(f"❌ Failed to start health server: {e}")
                logger.error("   Health monitoring will not be available")
                # Don't fail startup for health server issues
            
            # Create Telegram application
            logger.info("🤖 Creating Telegram bot application...")
            if not Config.TELEGRAM_BOT_TOKEN:
                raise ValueError("TELEGRAM_BOT_TOKEN is required but not set")
            self.application = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            self.bot = self.application.bot
            
            # Register all handlers
            await self.register_handlers()
            
            # Setup graceful shutdown
            self.setup_signal_handlers()
            
            # Get bot info for logging
            try:
                me = await self.bot.get_me()
                logger.info(f"🎯 Bot Info: @{me.username} ({me.first_name})")
                logger.info(f"🔗 Bot ID: {me.id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch bot info: {e}")
            
            # Start the bot
            logger.info("🚀 Starting Hugging Face By AadityaLabs AI...")
            logger.info("🌟 SUPERIOR AI Assistant with Advanced Features Active!")
            logger.info("💡 Features: Intent Classification • File Processing • User Onboarding • Encrypted Database")
            
            # Start polling
            await self.application.initialize()
            await self.application.start()
            if self.application.updater:
                await self.application.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=["message", "callback_query", "inline_query"]
                )
            else:
                raise RuntimeError("Application updater not available")
            
            logger.info("✅ Bot started successfully and is now polling for updates")
            logger.info("🎉 Hugging Face By AadityaLabs AI is now LIVE!")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # CRITICAL FIX: Perform graceful shutdown when event is triggered
            await self.shutdown()
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            return False
        
        return True
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers with proper async handling"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            # CRITICAL FIX: Use event-based shutdown instead of creating tasks in signal handler
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """
        Graceful shutdown with cleanup
        """
        logger.info("🛑 Initiating graceful shutdown...")
        
        try:
            # Stop health server first
            if self.health_server_started:
                logger.info("🌐 Stopping health check web server...")
                try:
                    await health_server.stop()
                    logger.info("✅ Health check web server stopped")
                except Exception as e:
                    logger.error(f"❌ Error stopping health server: {e}")
            
            # Stop the application
            if self.application:
                logger.info("📱 Stopping Telegram application...")
                if self.application.updater:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                logger.info("✅ Telegram application stopped")
            
            # Close storage connections
            if storage_manager.initialized:
                logger.info("💾 Closing storage connections...")
                await storage_manager.disconnect()
                logger.info("✅ Storage connections closed")
            
            # Cleanup admin system
            if admin_system:
                logger.info("👑 Admin system is properly initialized")
                # Admin system doesn't require explicit cleanup
                logger.info("✅ Admin system handled")
            
            logger.info("✅ Graceful shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        
        finally:
            # Signal shutdown completion
            self.shutdown_event.set()

async def main():
    """
    Main entry point for Hugging Face By AadityaLabs AI
    """
    # Print startup banner
    print("\n" + "="*80)
    print("🤖 Hugging Face By AadityaLabs AI - PHASE 2")
    print("🧠 Superior Telegram Bot with Advanced AI Features")
    print("🚀 Outperforms ChatGPT, Grok, and Gemini")
    print("="*80)
    print(f"📦 Bot Name: {Config.BOT_NAME}")
    print(f"📦 Version: {Config.BOT_VERSION}")
    print(f"🔧 Environment: {Config.get_environment_type()}")
    print(f"🎯 HF Tier: {Config.HF_TIER}")
    print("="*80 + "\n")
    
    # Initialize and start the bot
    bot_app = TelegramBotApplication()
    
    try:
        await bot_app.start_bot()
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
        await bot_app.shutdown()
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}")
        await bot_app.shutdown()
    finally:
        logger.info("👋 Hugging Face By AadityaLabs AI has been shut down")

if __name__ == "__main__":
    """
    Entry point when running the script directly
    """
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)