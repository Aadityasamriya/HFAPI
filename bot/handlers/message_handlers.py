"""
Advanced message handlers for AI Assistant Pro
Handles intelligent routing, context management, and multi-modal responses
"""

import asyncio
import io
import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from bot.database import db
from bot.core.router import router, IntentType
from bot.core.model_caller import ModelCaller
from bot.config import Config

logger = logging.getLogger(__name__)

class MessageHandlers:
    """Advanced message processing with intelligent AI routing"""
    
    @staticmethod
    async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Main text message processor with intelligent AI routing
        """
        user = update.effective_user
        user_id = user.id
        message_text = update.message.text
        
        # Check if bot is waiting for API key
        if context.user_data.get('waiting_for_api_key', False):
            await MessageHandlers._handle_api_key_input(update, context, message_text)
            return
        
        # Get user's API key
        api_key = await db.get_user_api_key(user_id)
        if not api_key:
            await MessageHandlers._prompt_api_key_setup(update, context)
            return
        
        # Send typing action for better UX
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            # Get or initialize chat history
            chat_history = context.user_data.get('chat_history', [])
            
            # Analyze prompt and route to appropriate model
            intent, routing_info = router.route_prompt(message_text)
            
            logger.info(f"User {user_id} prompt routed to {intent.value} with confidence {routing_info['confidence']}")
            
            # Process based on intent
            if intent == IntentType.IMAGE_GENERATION:
                await MessageHandlers._handle_image_generation(update, context, message_text, api_key)
            
            elif intent == IntentType.CODE_GENERATION:
                await MessageHandlers._handle_code_generation(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.SENTIMENT_ANALYSIS:
                await MessageHandlers._handle_sentiment_analysis(update, context, message_text, api_key)
            
            else:  # Text generation and other intents
                await MessageHandlers._handle_text_generation(update, context, message_text, api_key, chat_history, routing_info)
            
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Processing Error**\n\nI encountered an issue processing your request. Please try again or contact support if the problem persists.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_api_key_input(update: Update, context: ContextTypes.DEFAULT_TYPE, api_key: str) -> None:
        """Handle API key input from user"""
        user_id = update.effective_user.id
        
        # Send processing message
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        # Validate API key format
        if not api_key.startswith('hf_') or len(api_key) < 30:
            await update.message.reply_text(
                "❌ **Invalid API Key Format**\n\nHugging Face API keys start with 'hf_' and are longer than 30 characters.\n\nPlease check your key and try again.",
                parse_mode='Markdown'
            )
            return
        
        # Test API key with a simple call
        try:
            async with ModelCaller() as model_caller:
                success, result = await model_caller._make_api_call(
                    "gpt2", 
                    {"inputs": "test"}, 
                    api_key
                )
            
            if success:
                # Save API key
                save_success = await db.save_user_api_key(user_id, api_key)
                
                if save_success:
                    context.user_data['waiting_for_api_key'] = False
                    
                    success_text = """
✅ **API Key Configured Successfully!** 🎉

Your Hugging Face API key has been securely stored and validated.

🚀 **You're ready to go!** Try asking me:
• "Explain machine learning"
• "Create a Python function to sort numbers" 
• "Draw a sunset over mountains"
• "What's the sentiment of: I love this!"

💡 **Smart Features Unlocked:**
• 🧠 Intelligent model routing
• 🎨 Multi-modal AI (text, images, code)
• 💬 Context-aware conversations
• 🔄 Automatic model selection

What would you like to explore first? 🤖✨
                    """
                    
                    await update.message.reply_text(
                        success_text,
                        parse_mode='Markdown'
                    )
                else:
                    await update.message.reply_text(
                        "❌ **Storage Error**\n\nYour API key is valid but couldn't be saved. Please try again.",
                        parse_mode='Markdown'
                    )
            else:
                await update.message.reply_text(
                    "❌ **API Key Invalid**\n\nThe provided API key couldn't be verified. Please check:\n\n• Key is copied correctly\n• Token has 'Read' permissions\n• Account is in good standing\n\nTry again with a valid key.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error validating API key for user {user_id}: {e}")
            await update.message.reply_text(
                "🔄 **Validation Error**\n\nCouldn't verify your API key due to a network issue. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _prompt_api_key_setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Prompt user to set up their API key"""
        setup_text = """
🔑 **API Key Required** 

To use AI Assistant Pro, you need to configure your Hugging Face API key.

**Quick Setup:**
1️⃣ Visit: https://huggingface.co/settings/tokens
2️⃣ Create a new token with 'Read' permissions  
3️⃣ Send the token here as your next message

🆓 **Free to use** - Hugging Face offers generous free tier access!

Use `/start` for detailed setup instructions.
        """
        
        context.user_data['waiting_for_api_key'] = True
        
        await update.message.reply_text(
            setup_text,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_text_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, chat_history: list, routing_info: dict) -> None:
        """Handle intelligent text generation"""
        user_id = update.effective_user.id
        
        # Update chat history with user message
        chat_history.append({'role': 'user', 'content': prompt})
        
        # Limit chat history size
        if len(chat_history) > Config.MAX_CHAT_HISTORY * 2:  # *2 for user + assistant pairs
            chat_history = chat_history[-Config.MAX_CHAT_HISTORY * 2:]
        
        try:
            async with ModelCaller() as model_caller:
                success, response = await model_caller.generate_text(
                    prompt, 
                    api_key, 
                    chat_history[:-1],  # Exclude the current message
                    routing_info.get('recommended_model')
                )
            
            if success and response:
                # Update chat history with assistant response
                chat_history.append({'role': 'assistant', 'content': response})
                context.user_data['chat_history'] = chat_history
                
                # Format response with smart features info
                formatted_response = f"🤖 **AI Response** ({routing_info['primary_intent'].value})\n\n{response}"
                
                # Add routing confidence if high
                if routing_info['confidence'] >= 2:
                    formatted_response += f"\n\n🎯 *Auto-selected {routing_info['primary_intent'].value} model*"
                
                await update.message.reply_text(
                    formatted_response,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Successfully generated text for user {user_id}")
                
            else:
                await update.message.reply_text(
                    f"❌ **Generation Failed**\n\n{response}\n\nPlease try again or use `/newchat` to start fresh.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in text generation for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Processing Error**\n\nFailed to generate response. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_code_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, routing_info: dict) -> None:
        """Handle specialized code generation"""
        user_id = update.effective_user.id
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            language = routing_info['analysis'].get('language_detected', 'python')
            
            async with ModelCaller() as model_caller:
                success, code_response = await model_caller.generate_code(prompt, api_key, language)
            
            if success and code_response:
                response_text = f"💻 **Code Generated** ({language})\n\n{code_response}\n\n🎯 *Optimized for {language} development*"
                
                await update.message.reply_text(
                    response_text,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Successfully generated code for user {user_id}")
                
            else:
                await update.message.reply_text(
                    f"❌ **Code Generation Failed**\n\n{code_response}\n\nTry rephrasing your request or specify the programming language.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in code generation for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Code Generation Error**\n\nFailed to generate code. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str) -> None:
        """Handle professional image generation"""
        user_id = update.effective_user.id
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎨 **Creating Your Image...**\n\nGenerating high-quality artwork with AI. This may take a moment...",
            parse_mode='Markdown'
        )
        
        try:
            async with ModelCaller() as model_caller:
                success, image_data = await model_caller.generate_image(prompt, api_key)
            
            if success and image_data:
                # Delete processing message
                await processing_msg.delete()
                
                # Send image with caption
                image_stream = io.BytesIO(image_data)
                image_stream.name = 'ai_generated_image.png'
                
                caption = f"🎨 **AI Generated Image**\n\n**Prompt:** {prompt}\n\n✨ *Created with professional AI image generation*"
                
                await update.message.reply_photo(
                    photo=image_stream,
                    caption=caption,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Successfully generated image for user {user_id}")
                
            else:
                await processing_msg.edit_text(
                    "❌ **Image Generation Failed**\n\nCouldn't create your image. This might be due to:\n\n• Content policy restrictions\n• Model availability\n• Network issues\n\nTry rephrasing your prompt or try again later.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in image generation for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **Image Generation Error**\n\nFailed to create image. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_sentiment_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str) -> None:
        """Handle sentiment analysis requests"""
        user_id = update.effective_user.id
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            # Extract text to analyze from prompt
            analysis_text = prompt
            if "analyze" in prompt.lower() or "sentiment" in prompt.lower():
                # Try to extract the actual text to analyze
                parts = prompt.split('"')
                if len(parts) >= 2:
                    analysis_text = parts[1]
                else:
                    # Look for text after colons or keywords
                    for keyword in ["analyze:", "sentiment:", "text:", "this:"]:
                        if keyword in prompt.lower():
                            analysis_text = prompt.lower().split(keyword)[1].strip()
                            break
            
            async with ModelCaller() as model_caller:
                success, sentiment_data = await model_caller.analyze_sentiment(analysis_text, api_key)
            
            if success and sentiment_data:
                # Format sentiment results
                if isinstance(sentiment_data, list) and sentiment_data:
                    sentiment_info = sentiment_data[0]
                    label = sentiment_info.get('label', 'Unknown')
                    score = sentiment_info.get('score', 0)
                    
                    # Emoji mapping
                    emoji_map = {
                        'POSITIVE': '😊',
                        'NEGATIVE': '😞', 
                        'NEUTRAL': '😐',
                        'MIXED': '🤔'
                    }
                    
                    sentiment_emoji = emoji_map.get(label.upper(), '🤖')
                    confidence_percent = round(score * 100, 1)
                    
                    response_text = f"""
📊 **Sentiment Analysis Results**

**Text:** "{analysis_text}"

**Result:** {sentiment_emoji} {label.title()}
**Confidence:** {confidence_percent}%

🎯 *Analyzed with advanced sentiment detection AI*
                    """
                else:
                    response_text = "📊 **Sentiment Analysis**\n\nReceived analysis results but couldn't parse the sentiment data."
                
                await update.message.reply_text(
                    response_text,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Successfully analyzed sentiment for user {user_id}")
                
            else:
                await update.message.reply_text(
                    "❌ **Sentiment Analysis Failed**\n\nCouldn't analyze the sentiment. Please try again with clear text.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Analysis Error**\n\nFailed to analyze sentiment. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler for the bot"""
        logger.error(f"Exception while handling an update: {context.error}")
        
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "🚫 **Unexpected Error**\n\nSomething went wrong. Our team has been notified. Please try again.",
                parse_mode='Markdown'
            )

# Export message handlers
message_handlers = MessageHandlers()