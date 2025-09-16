"""
Professional Telegram bot command handlers
Rich UI with emojis and inline keyboards for superior user experience
"""

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

logger = logging.getLogger(__name__)

class CommandHandlers:
    """Professional command handlers with rich UI"""
    
    @staticmethod
    async def start_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Professional welcome message with sophisticated UI
        """
        user = update.effective_user
        user_id = user.id
        
        welcome_text = f"""
🤖 **Welcome to AI Assistant Pro** 🚀

Hello {user.first_name}! I'm your sophisticated AI companion powered by cutting-edge Hugging Face models. I can help you with:

🧠 **Text Generation** - Intelligent conversations and content creation
🎨 **Image Creation** - Professional artwork and visualizations  
💻 **Code Generation** - Multi-language programming assistance
📊 **Data Analysis** - Sentiment analysis and insights
🌐 **Translation** - Multi-language support
✍️ **Creative Writing** - Stories, poems, and creative content

**Getting Started:**
1️⃣ Set your Hugging Face API key
2️⃣ Start chatting with advanced AI models
3️⃣ Experience intelligent model routing

*I automatically select the best model for your needs!*
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
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        logger.info(f"User {user_id} started the bot")
    
    @staticmethod
    async def newchat_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Clear chat history with professional confirmation
        """
        user_id = update.effective_user.id
        
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
        
        await update.message.reply_text(
            success_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        logger.info(f"User {user_id} cleared chat history")
    
    @staticmethod
    async def settings_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Professional settings menu with comprehensive options
        """
        user_id = update.effective_user.id
        api_key = await db.get_user_api_key(user_id)
        
        status_emoji = "✅" if api_key else "❌"
        api_status = "Connected" if api_key else "Not Set"
        
        settings_text = f"""
⚙️ **AI Assistant Settings** 🛠️

**Current Status:**
🔑 API Key: {status_emoji} {api_status}
🤖 Models: Premium Hugging Face Collection
🧠 Intelligence: Adaptive Model Routing
📊 Chat History: Last 15 messages

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
        
        await update.message.reply_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def help_command(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Comprehensive help system with examples
        """
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
    async def button_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle all inline keyboard button interactions
        """
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        data = query.data
        
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
        
        else:
            await query.edit_message_text("🔄 Processing your request...")
    
    @staticmethod
    async def _handle_api_key_setup(query, context) -> None:
        """Handle API key setup process"""
        text = """
🔑 **Hugging Face API Key Setup** 

To use AI Assistant Pro, you need a Hugging Face API key:

**📋 Steps:**
1️⃣ Visit: https://huggingface.co/settings/tokens
2️⃣ Create a new token with 'Read' permissions
3️⃣ Copy your token
4️⃣ Send it here as your next message

🛡️ **Security:** Your API key is stored securely and encrypted. Only you can access your data.

💡 **Free Tier:** Hugging Face offers generous free usage for all models!
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

🔄 **Note:** Your chat history is never stored, so it's already private.

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
        """Handle actual data reset"""
        user_id = query.from_user.id
        success = await db.reset_user_database(user_id)
        
        if success:
            text = """
✅ **Data Reset Complete** 

Your data has been successfully removed from our system.

🔄 To continue using AI Assistant Pro, you'll need to set up your API key again.
            """
        else:
            text = """
❌ **Reset Failed** 

There was an issue resetting your data. Please try again or contact support.
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
🎯 **AI Models Information** 

**🧠 Text Generation:**
• Primary: {Config.DEFAULT_TEXT_MODEL}
• Fallback: {Config.FALLBACK_TEXT_MODEL}
• Specialty: Advanced reasoning & conversation

**💻 Code Generation:**
• Model: {Config.DEFAULT_CODE_MODEL}
• Languages: Python, JS, Java, C++, and more
• Features: Code explanation & debugging

**🎨 Image Creation:**
• Model: {Config.DEFAULT_IMAGE_MODEL}
• Resolution: 1024x1024
• Style: Professional quality artwork

**🔄 Smart Routing:**
I automatically select the best model based on your request type and complexity!
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
🚀 **Try These Examples** 

**💬 Conversations:**
• "Explain artificial intelligence in simple terms"
• "Help me plan a productive morning routine"
• "What are the latest trends in technology?"

**💻 Code Examples:**
• "Create a Python function to calculate fibonacci"
• "Build a React todo list component"
• "Write SQL to find top customers"

**🎨 Image Examples:**
• "Draw a minimalist mountain landscape"
• "Create a professional tech startup logo"
• "Generate a cyberpunk city at night"

**📊 Analysis Examples:**
• "Analyze sentiment: I love this new feature!"
• "What's the mood of this customer review"

Just copy and paste any example to try it out! 🎯
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
📊 Chat History: Last 15 messages

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

# Export command handlers
command_handlers = CommandHandlers()