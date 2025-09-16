"""
Advanced message handlers for AI Assistant Pro
Handles intelligent routing, context management, and multi-modal responses
"""

import asyncio
import io
import logging
import os
import re
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from bot.database import db
from bot.core.router import router, IntentType
from bot.core.model_caller import ModelCaller, _redact_sensitive_data, _safe_log_error
from bot.config import Config
from bot.security_utils import escape_markdown, safe_markdown_format, check_rate_limit

logger = logging.getLogger(__name__)


class MessageHandlers:
    """Advanced message processing with intelligent AI routing"""
    
    @staticmethod
    async def text_message_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Main text message processor with intelligent AI routing
        """
        user = update.effective_user
        user_id = user.id
        
        # Safely extract message text, handling entity parsing errors
        message_text = None
        
        # Try multiple methods to extract text content
        try:
            # Primary method - direct text access
            message_text = update.message.text
        except Exception as entity_error:
            logger.warning(f"Direct text access failed for user {user_id}: {entity_error}")
            
            # Method 2: Try accessing raw update data
            try:
                raw_message = update.to_dict().get('message', {})
                message_text = raw_message.get('text', '')
                logger.info(f"Extracted text via raw update data for user {user_id}")
            except Exception:
                # Method 3: Try without entity processing
                try:
                    # Access the message object directly bypassing entity parsing
                    if hasattr(update.message, '_effective_message'):
                        message_text = update.message._effective_message.get('text', '')
                    elif hasattr(update, '_raw_data'):
                        message_text = update._raw_data.get('message', {}).get('text', '')
                    else:
                        # Last resort - reconstruct from available data
                        message_text = str(update.message).split('text=')[-1].split(',')[0].strip('"\'')
                        
                    logger.info(f"Extracted text via alternative method for user {user_id}")
                except Exception:
                    logger.warning(f"All text extraction methods failed for user {user_id}, using fallback")
                    message_text = "Hello"  # Safe fallback that won't cause issues
        
        # Ensure we have valid text to work with
        if not message_text or len(message_text.strip()) == 0:
            message_text = "Hello"  # Default safe message
            logger.info(f"Using default message text for user {user_id}")
        
        # Check rate limit for text messages
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before sending another message.",
                parse_mode='Markdown'
            )
            return
        
        # Check if bot is waiting for API key
        if context.user_data is not None and context.user_data.get('waiting_for_api_key', False):
            await MessageHandlers._handle_api_key_input(update, context, message_text)
            return
        
        # Get user's API key from persistent database storage as specified
        api_key = await db.get_user_api_key(user_id)
        if not api_key:
            await MessageHandlers._prompt_api_key_setup(update, context)
            return
        
        # Send typing action for better UX
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            # Get or initialize chat history
            chat_history = context.user_data.get('chat_history', []) if context.user_data is not None else []
            
            # Analyze prompt and route to appropriate model
            intent, routing_info = router.route_prompt(message_text)
            
            logger.info(f"User {user_id} prompt routed to {intent.value} with confidence {routing_info['confidence']}")
            
            # 2025: Enhanced processing with new intent types
            if intent == IntentType.IMAGE_GENERATION:
                await MessageHandlers._handle_image_generation(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.CODE_GENERATION:
                await MessageHandlers._handle_code_generation(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.SENTIMENT_ANALYSIS:
                await MessageHandlers._handle_sentiment_analysis(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.DATA_ANALYSIS:
                await MessageHandlers._handle_data_analysis(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.DOCUMENT_PROCESSING:
                await MessageHandlers._handle_document_processing(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.MULTI_MODAL:
                await MessageHandlers._handle_multi_modal(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.CONVERSATION:
                await MessageHandlers._handle_conversation(update, context, message_text, api_key, chat_history, routing_info)
            
            else:  # Text generation and other intents
                await MessageHandlers._handle_text_generation(update, context, message_text, api_key, chat_history, routing_info)
            
        except Exception as e:
            _safe_log_error(logger.error, f"Error processing message for user {user_id}: {e}")
            # Safely escape any error details that might be included
            safe_error_msg = safe_markdown_format(
                "🚫 **Processing Error**\n\nI encountered an issue processing your request. Please try again or contact support if the problem persists."
            )
            await update.message.reply_text(
                safe_error_msg,
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_api_key_input(update, context: ContextTypes.DEFAULT_TYPE, api_key: str) -> None:
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
                # Store API key persistently in database as specified
                success_stored = await db.save_user_api_key(user_id, api_key)
                if success_stored:
                    if context.user_data is not None:
                        context.user_data['waiting_for_api_key'] = False
                else:
                    await update.message.reply_text(
                        "❌ **Storage Error**\n\n"
                        "Failed to save your API key to the database. "
                        "Please try again or contact support if the problem persists.",
                        parse_mode='Markdown'
                    )
                    return
                
                success_text = """
✅ **API Key Configured Successfully!** 🎉

Your Hugging Face API key has been validated and stored for this session.

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

🔒 **Privacy Note:** Your API key is stored securely in our encrypted database as specified, allowing you to use the bot seamlessly across sessions.

What would you like to explore first? 🤖✨
                """
                
                await update.message.reply_text(
                    success_text,
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    "❌ **API Key Invalid**\n\nThe provided API key couldn't be verified. Please check:\n\n• Key is copied correctly\n• Token has 'Read' permissions\n• Account is in good standing\n\nTry again with a valid key.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"Error validating API key for user {user_id}: {e}")
            await update.message.reply_text(
                "🔄 **Validation Error**\n\nCouldn't verify your API key due to a network issue. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _prompt_api_key_setup(update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        
        if context.user_data is not None:
            context.user_data['waiting_for_api_key'] = True
        
        await update.message.reply_text(
            setup_text,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def _handle_text_generation(update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, chat_history: list, routing_info: dict) -> None:
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
                    routing_info.get('recommended_model') or Config.DEFAULT_TEXT_MODEL,
                    routing_info.get('special_parameters', {})  # Use advanced parameters
                )
            
            if success and response:
                # Update chat history with assistant response
                chat_history.append({'role': 'assistant', 'content': response})
                if context.user_data is not None:
                    context.user_data['chat_history'] = chat_history
                
                # Enhanced response formatting with model information - safely escaped
                model_used = escape_markdown(routing_info.get('recommended_model', 'Unknown').split('/')[-1])  # Get model name only
                safe_response = safe_markdown_format(response, preserve_code=True)
                safe_intent = escape_markdown(routing_info['primary_intent'].value)
                formatted_response = f"🤖 **AI Response** ({safe_intent})\n\n{safe_response}"
                
                # Add model info and routing confidence
                if routing_info['confidence'] >= 2:
                    formatted_response += f"\n\n🎯 *Powered by {model_used} \- Auto\-selected for optimal performance*"
                
                await update.message.reply_text(
                    formatted_response,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Successfully generated text for user {user_id}")
                
            else:
                safe_error_response = safe_markdown_format(str(response))
                await update.message.reply_text(
                    f"❌ **Generation Failed**\n\n{safe_error_response}\n\nPlease try again or use `/newchat` to start fresh\.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"Error in text generation for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Processing Error**\n\nFailed to generate response. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_code_generation(update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, routing_info: dict) -> None:
        """Handle specialized code generation"""
        user_id = update.effective_user.id
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            language = routing_info['analysis'].get('language_detected', 'python')
            
            async with ModelCaller() as model_caller:
                success, code_response = await model_caller.generate_code(
                    prompt, 
                    api_key, 
                    language,
                    routing_info.get('special_parameters', {})  # Use advanced parameters
                )
            
            if success and code_response:
                safe_model_used = escape_markdown(routing_info.get('recommended_model', 'Unknown').split('/')[-1])
                safe_language = escape_markdown(language)
                safe_code_response = safe_markdown_format(code_response, preserve_code=True)
                response_text = f"💻 **Code Generated** ({safe_language})\n\n{safe_code_response}\n\n🎯 *Generated by {safe_model_used} \- Latest 2024\-2025 coding AI*"
                
                await update.message.reply_text(
                    response_text,
                    parse_mode='Markdown'
                )
                
                logger.info(f"Successfully generated code for user {user_id}")
                
            else:
                safe_error_response = safe_markdown_format(str(code_response))
                await update.message.reply_text(
                    f"❌ **Code Generation Failed**\n\n{safe_error_response}\n\nTry rephrasing your request or specify the programming language\.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"Error in code generation for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Code Generation Error**\n\nFailed to generate code. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_image_generation(update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, routing_info: dict) -> None:
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
                success, image_data = await model_caller.generate_image(
                    prompt, 
                    api_key,
                    routing_info.get('special_parameters', {})  # Use FLUX.1 parameters
                )
            
            if success and image_data:
                # Delete processing message
                await processing_msg.delete()
                
                # Send image with caption - safely escaped
                image_stream = io.BytesIO(image_data)
                image_stream.name = 'ai_generated_image.png'
                
                safe_prompt = escape_markdown(prompt)
                caption = f"🎨 **AI Generated Image**\n\n**Prompt:** {safe_prompt}\n\n✨ *Created with FLUX\.1 \- State\-of\-the\-art 2024\-2025 image AI*"
                
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
            _safe_log_error(logger.error, f"Error in image generation for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **Image Generation Error**\n\nFailed to create image. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_sentiment_analysis(update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, routing_info: dict) -> None:
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
                # Check if advanced emotion detection is needed
                use_emotions = routing_info.get('special_parameters', {}).get('use_emotion_detection', False)
                success, sentiment_data = await model_caller.analyze_sentiment(analysis_text, api_key, use_emotions)
            
            response_text = "❌ **Analysis Error**\n\nUnable to format sentiment analysis results."
            
            if success and sentiment_data:
                # Enhanced sentiment formatting for both standard and emotion detection
                if sentiment_data.get('emotion_type') == 'advanced':
                    # Advanced emotion detection results
                    emotions = sentiment_data.get('emotions', [])
                    if emotions:
                        top_emotion = max(emotions, key=lambda x: x.get('score', 0))
                        emotion_name = top_emotion.get('label', 'Unknown')
                        confidence = round(top_emotion.get('score', 0) * 100, 1)
                        
                        safe_analysis_text = escape_markdown(analysis_text)
                        safe_emotion_name = escape_markdown(emotion_name.title())
                        safe_emotions_list = chr(10).join([f"• {escape_markdown(e.get('label', 'Unknown').title())}: {round(e.get('score', 0)*100, 1)}%" for e in emotions[:5]])
                        
                        response_text = f"""
📊 **Advanced Emotion Analysis**

**Text:** "{safe_analysis_text}"

**Primary Emotion:** {safe_emotion_name} 
**Confidence:** {confidence}%

**All Detected Emotions:**
{safe_emotions_list}

🎯 *Analyzed with go\_emotions \- 28 emotion categories*
                        """
                else:
                    # Standard sentiment analysis
                    result_data = sentiment_data.get('result', {})
                    if result_data:
                        label = result_data.get('label', 'Unknown')
                        score = result_data.get('score', 0)
                        
                        emoji_map = {
                            'POSITIVE': '😊', 'NEGATIVE': '😞', 'NEUTRAL': '😐',
                            'LABEL_0': '😞', 'LABEL_1': '😐', 'LABEL_2': '😊'  # RoBERTa labels
                        }
                        
                        sentiment_emoji = emoji_map.get(label.upper(), '🤖')
                        confidence_percent = round(score * 100, 1)
                        
                        response_text = f"""
📊 **Sentiment Analysis Results**

**Text:** "{analysis_text}"

**Result:** {sentiment_emoji} {label.title()}
**Confidence:** {confidence_percent}%

🎯 *Analyzed with RoBERTa - Latest sentiment AI*
                        """
                
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
            _safe_log_error(logger.error, f"Error in sentiment analysis for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Analysis Error**\n\nFailed to analyze sentiment. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def error_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler for the bot"""
        error_message = str(context.error)
        
        # Handle entity parsing errors - silently ignore them since we handle them in message processing
        if "Can't parse entities" in error_message or ("entity" in error_message.lower() and "offset" in error_message.lower()):
            logger.warning(f"Entity parsing error detected (ignoring): {error_message}")
            # Don't send any response to user - our message handler will process the message properly
            return
        
        # Log other errors
        _safe_log_error(logger.error, f"Exception while handling an update: {context.error}")
        
        if update and hasattr(update, 'effective_message') and update.effective_message:
            await update.effective_message.reply_text(
                "🚫 **Unexpected Error**\n\nSomething went wrong. Our team has been notified. Please try again.",
                parse_mode='Markdown'
            )

    # 2025: New advanced handler methods for enhanced AI capabilities
    @staticmethod
    async def _handle_data_analysis(update, context, prompt: str, api_key: str, routing_info: dict) -> None:
        """Handle data analysis requests with advanced AI models"""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        enhanced_prompt = f"""🔬 **Data Analysis Request:** {prompt}

Please provide detailed analysis including key insights, patterns, and actionable recommendations."""
        
        async with ModelCaller(provider="auto") as model_caller:
            success, result = await model_caller.generate_text(enhanced_prompt, api_key, special_params={'temperature': 0.3, 'max_new_tokens': 1500})
            
            if success:
                await update.message.reply_text(f"📊 **Data Analysis Results**\n\n{result}\n\n*🎯 Analyzed with advanced 2025 AI models*", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ **Analysis Failed** - Please try again with more specific data context.", parse_mode='Markdown')

    @staticmethod
    async def _handle_document_processing(update, context, prompt: str, api_key: str, routing_info: dict) -> None:
        """Handle document processing requests"""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        enhanced_prompt = f"📄 **Document Processing:** {prompt}\n\nProvide structured processing with key information extraction and summary."
        
        async with ModelCaller(provider="auto") as model_caller:
            success, result = await model_caller.generate_text(enhanced_prompt, api_key, special_params={'temperature': 0.2})
            
            if success:
                await update.message.reply_text(f"📋 **Document Processing Results**\n\n{result}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ **Processing Failed** - Please try again with clearer context.", parse_mode='Markdown')

    @staticmethod 
    async def _handle_multi_modal(update, context, prompt: str, api_key: str, routing_info: dict) -> None:
        """Handle multi-modal AI requests"""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        async with ModelCaller(provider="auto") as model_caller:
            success, result = await model_caller.generate_text(f"🔄 **Multi-Modal AI:** {prompt}", api_key, special_params={'temperature': 0.6})
            
            if success:
                await update.message.reply_text(f"🎭 **Multi-Modal Response**\n\n{result}\n\n*🌟 Generated with 2025 multi-modal AI*", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ **Multi-Modal Processing Failed**", parse_mode='Markdown')

    @staticmethod
    async def _handle_conversation(update, context, prompt: str, api_key: str, chat_history: list, routing_info: dict) -> None:
        """Handle natural conversation with enhanced context awareness"""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        async with ModelCaller(provider="auto") as model_caller:
            success, result = await model_caller.generate_text(prompt, api_key, chat_history, special_params={'temperature': 0.8})
            
            if success:
                # Update chat history for better context
                if context.user_data is not None:
                    if 'chat_history' not in context.user_data:
                        context.user_data['chat_history'] = []
                    context.user_data['chat_history'].append({'role': 'user', 'content': prompt})
                    context.user_data['chat_history'].append({'role': 'assistant', 'content': result})
                    
                    # Keep only recent history
                    if len(context.user_data['chat_history']) > Config.MAX_CHAT_HISTORY * 2:
                        context.user_data['chat_history'] = context.user_data['chat_history'][-Config.MAX_CHAT_HISTORY * 2:]
                
                await update.message.reply_text(result, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ **Conversation Error** - Let me try that again.", parse_mode='Markdown')

# Export message handlers
message_handlers = MessageHandlers()