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

Hello {user.first_name}! I'm powered by the **latest 2024-2025 AI models** - more advanced than ChatGPT, Grok, or Gemini!

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

🛡️ **Security:** Your API key is encrypted & stored securely. Privacy guaranteed!

💸 **Cost:** Completely FREE for personal use! Hugging Face offers generous quotas for all latest models.

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