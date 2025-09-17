"""
Advanced message handlers for Hugging Face By AadityaLabs AI
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
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MessageHandlers:
    """Advanced message processing for Hugging Face By AadityaLabs AI with intelligent AI routing and comprehensive observability"""
    
    @staticmethod
    async def _save_conversation_if_ready(user_id: int, chat_history: list, context=None) -> bool:
        """
        Save conversation to persistent storage if it meets criteria for saving
        
        Args:
            user_id (int): User ID for the conversation
            chat_history (list): Current chat history with user/assistant messages
            context: Bot context (optional)
        
        Returns:
            bool: True if conversation was saved, False otherwise
        """
        try:
            # Only save conversations with meaningful exchanges (at least 2 messages: user + assistant)
            if not chat_history or len(chat_history) < 2:
                return False
            
            # Check if conversation has at least one user-assistant exchange
            has_exchange = False
            user_messages = 0
            assistant_messages = 0
            
            for msg in chat_history:
                if msg.get('role') == 'user':
                    user_messages += 1
                elif msg.get('role') == 'assistant':
                    assistant_messages += 1
            
            has_exchange = user_messages >= 1 and assistant_messages >= 1
            
            if not has_exchange:
                logger.debug(f"Conversation not ready for saving - user:{user_messages}, assistant:{assistant_messages}")
                return False
            
            # Generate conversation summary from the first user message
            first_user_msg = None
            for msg in chat_history:
                if msg.get('role') == 'user':
                    first_user_msg = msg.get('content', '')
                    break
            
            if not first_user_msg:
                logger.warning(f"No user message found in chat_history for user {user_id}")
                return False
            
            # Create a meaningful summary (first 100 chars of first user message)
            summary = first_user_msg.strip()[:100]
            if len(first_user_msg) > 100:
                # Try to break at word boundary
                word_boundary = summary.rfind(' ')
                if word_boundary > 50:  # Don't make it too short
                    summary = summary[:word_boundary]
                summary += "..."
            
            # If summary is too short or generic, enhance it
            if len(summary.strip()) < 10 or summary.lower().strip() in ['hello', 'hi', 'hey']:
                # Try to get more context from the conversation
                context_parts = []
                for msg in chat_history[:4]:  # Look at first few messages
                    content = msg.get('content', '').strip()
                    if content and msg.get('role') == 'user' and len(content) > 10:
                        context_parts.append(content[:50])
                
                if context_parts:
                    summary = " | ".join(context_parts)[:100]
                else:
                    summary = "Conversation with AI Assistant"
            
            # Ensure summary is clean for display
            summary = summary.replace('\n', ' ').replace('\r', ' ').strip()
            if not summary:
                summary = "Conversation with AI Assistant"
            
            # Prepare conversation data with timestamps
            now = datetime.utcnow()
            
            # Add timestamps to messages that don't have them
            messages_with_timestamps = []
            for i, msg in enumerate(chat_history):
                msg_with_timestamp = msg.copy()
                if 'timestamp' not in msg_with_timestamp:
                    # Estimate timestamps for older messages (slightly earlier)
                    estimated_time = now - timedelta(minutes=len(chat_history) - i)
                    msg_with_timestamp['timestamp'] = estimated_time
                messages_with_timestamps.append(msg_with_timestamp)
            
            conversation_data = {
                'messages': messages_with_timestamps,
                'summary': summary,
                'started_at': messages_with_timestamps[0].get('timestamp', now),
                'last_message_at': messages_with_timestamps[-1].get('timestamp', now),
                'message_count': len(messages_with_timestamps)
            }
            
            # Save to database
            success = await db.save_conversation(user_id, conversation_data)
            
            if success:
                logger.info(f"💾 Successfully saved conversation for user {user_id}: {len(summary)} char summary ({len(messages_with_timestamps)} messages)")
                
                # Clear the session history since it's now saved persistently
                if context and context.user_data is not None:
                    context.user_data['chat_history'] = []
                    logger.info(f"🔄 Cleared session chat_history for user {user_id} - conversation saved persistently")
                
                return True
            else:
                logger.error(f"Failed to save conversation for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving conversation for user {user_id}: {e}")
            return False
    
    @staticmethod
    async def text_message_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Main text message processor with intelligent AI routing
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        # Enhanced entry logging for text message processing
        logger.info(f"📬 TEXT MESSAGE received from user_id:{user_id} (@{username})")
        
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
        
        # Get user's API key from persistent database storage with comprehensive logging
        try:
            logger.info(f"🔍 Database query: get_user_api_key for user_id:{user_id}")
            api_key = await db.get_user_api_key(user_id)
            if api_key:
                logger.info(f"✅ API key found for user_id:{user_id} (length: {len(api_key)} chars)")
            else:
                logger.info(f"❌ No API key found for user_id:{user_id} - prompting setup")
                await MessageHandlers._prompt_api_key_setup(update, context)
                return
        except Exception as e:
            logger.error(f"🔍 Database error retrieving API key for user_id:{user_id}: {e}")
            logger.error(f"🔍 Database error type: {type(e).__name__}")
            await MessageHandlers._prompt_api_key_setup(update, context)
            return
        
        # Send typing action for better UX
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            # Get or initialize chat history
            chat_history = context.user_data.get('chat_history', []) if context.user_data is not None else []
            
            # Analyze prompt and route to appropriate model
            intent, routing_info = router.route_prompt(message_text)
            
            logger.info(f"🚦 ROUTING: user_id:{user_id} prompt -> {intent.value} (confidence: {routing_info['confidence']})")
            logger.info(f"🤖 Selected model: {routing_info['recommended_model']}")
            
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
        
        # Update chat history with user message (include timestamp)
        chat_history.append({'role': 'user', 'content': prompt, 'timestamp': datetime.utcnow()})
        
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
                chat_history.append({'role': 'assistant', 'content': response, 'timestamp': datetime.utcnow()})
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
                
                # Save conversation to persistent storage after successful exchange
                try:
                    saved = await MessageHandlers._save_conversation_if_ready(user_id, chat_history, context)
                    if saved:
                        logger.info(f"💾 Conversation saved to persistent storage for user {user_id}")
                    else:
                        logger.debug(f"Conversation not yet ready for saving for user {user_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save conversation for user {user_id}: {save_error}")
                    # Don't fail the response if saving fails
                
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
                # Update chat history for better context (include timestamps)
                if context.user_data is not None:
                    if 'chat_history' not in context.user_data:
                        context.user_data['chat_history'] = []
                    
                    # Add timestamped messages
                    now = datetime.utcnow()
                    context.user_data['chat_history'].append({'role': 'user', 'content': prompt, 'timestamp': now})
                    context.user_data['chat_history'].append({'role': 'assistant', 'content': result, 'timestamp': now})
                    
                    # Keep only recent history
                    if len(context.user_data['chat_history']) > Config.MAX_CHAT_HISTORY * 2:
                        context.user_data['chat_history'] = context.user_data['chat_history'][-Config.MAX_CHAT_HISTORY * 2:]
                
                await update.message.reply_text(result, parse_mode='Markdown')
                
                # Save conversation to persistent storage after successful exchange
                user_id = update.effective_user.id
                try:
                    chat_history = context.user_data.get('chat_history', [])
                    saved = await MessageHandlers._save_conversation_if_ready(user_id, chat_history, context)
                    if saved:
                        logger.info(f"💾 Conversation saved to persistent storage for user {user_id}")
                    else:
                        logger.debug(f"Conversation not yet ready for saving for user {user_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save conversation for user {user_id}: {save_error}")
                    # Don't fail the response if saving fails
            else:
                await update.message.reply_text("❌ **Conversation Error** - Let me try that again.", parse_mode='Markdown')
    
    @staticmethod
    async def document_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle document uploads (PDF and ZIP files) with AI analysis
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"📎 DOCUMENT upload received from user_id:{user_id} (@{username})")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before uploading another file.",
                parse_mode='Markdown'
            )
            return
        
        try:
            # Get document info
            document = update.message.document
            if not document:
                await update.message.reply_text(
                    "❌ **No Document Found**\n\nPlease upload a valid PDF or ZIP file.",
                    parse_mode='Markdown'
                )
                return
            
            filename = document.file_name or "unknown_file"
            file_size = document.file_size or 0
            mime_type = document.mime_type or "application/octet-stream"
            
            logger.info(f"📄 Document details: {filename} ({file_size:,} bytes, {mime_type}) from user {user_id}")
            
            # Determine file type and validate
            is_pdf = mime_type in ['application/pdf'] or filename.lower().endswith('.pdf')
            is_zip = mime_type in ['application/zip', 'application/x-zip-compressed'] or filename.lower().endswith('.zip')
            
            if not (is_pdf or is_zip):
                await update.message.reply_text(
                    "❌ **Unsupported File Type**\n\n"
                    "I can only process:\n"
                    "• **PDF files** - For document analysis\n"
                    "• **ZIP archives** - For content analysis\n\n"
                    "Please upload a PDF or ZIP file.",
                    parse_mode='Markdown'
                )
                return
            
            # Check file size limits
            from bot.file_processors import FileProcessor
            max_size = FileProcessor.MAX_ZIP_SIZE if is_zip else FileProcessor.MAX_PDF_SIZE
            if file_size > max_size:
                max_size_mb = max_size / (1024 * 1024)
                await update.message.reply_text(
                    f"❌ **File Too Large**\n\n"
                    f"Maximum size for {'ZIP' if is_zip else 'PDF'} files: {max_size_mb:.1f}MB\n"
                    f"Your file: {file_size / (1024 * 1024):.1f}MB\n\n"
                    f"Please upload a smaller file.",
                    parse_mode='Markdown'
                )
                return
            
            # Download file data
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
            
            processing_msg = await update.message.reply_text(
                f"📥 **Downloading {filename}...**\n\n"
                f"Processing your {'ZIP archive' if is_zip else 'PDF document'} for analysis. This may take a moment...",
                parse_mode='Markdown'
            )
            
            try:
                # Download file
                file_obj = await document.get_file()
                file_data = await file_obj.download_as_bytearray()
                
                # Security validation
                expected_type = 'zip' if is_zip else 'pdf'
                is_valid, error_msg = FileProcessor.validate_file_security(bytes(file_data), filename, expected_type)
                
                if not is_valid:
                    await processing_msg.edit_text(
                        f"🚫 **Security Validation Failed**\n\n{error_msg}\n\n"
                        "Please upload a valid, safe file.",
                        parse_mode='Markdown'
                    )
                    return
                
                # Process based on file type
                if is_pdf:
                    await MessageHandlers._process_pdf_document(update, context, file_data, filename, processing_msg)
                elif is_zip:
                    await MessageHandlers._process_zip_archive(update, context, file_data, filename, processing_msg)
                    
            except Exception as download_error:
                logger.error(f"File download error for user {user_id}: {download_error}")
                await processing_msg.edit_text(
                    "❌ **Download Failed**\n\n"
                    "Failed to download your file. Please try uploading again.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"Error in document handler for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Document Processing Error**\n\n"
                "An error occurred while processing your document. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def photo_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle photo uploads for image analysis
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"🖼️ PHOTO upload received from user_id:{user_id} (@{username})")
        
        # Check rate limit
        is_allowed, wait_time = check_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"⚠️ **Rate Limit Exceeded**\n\nPlease wait {wait_time} seconds before uploading another image.",
                parse_mode='Markdown'
            )
            return
        
        try:
            # Get the largest photo size
            if not update.message.photo:
                await update.message.reply_text(
                    "❌ **No Photo Found**\n\nPlease upload a valid image file.",
                    parse_mode='Markdown'
                )
                return
            
            photo = update.message.photo[-1]  # Get highest resolution
            file_size = photo.file_size or 0
            
            logger.info(f"📸 Photo details: {photo.width}x{photo.height} ({file_size:,} bytes) from user {user_id}")
            
            # Check file size
            from bot.file_processors import FileProcessor
            if file_size > FileProcessor.MAX_IMAGE_SIZE:
                max_size_mb = FileProcessor.MAX_IMAGE_SIZE / (1024 * 1024)
                await update.message.reply_text(
                    f"❌ **Image Too Large**\n\n"
                    f"Maximum size for images: {max_size_mb:.1f}MB\n"
                    f"Your image: {file_size / (1024 * 1024):.1f}MB\n\n"
                    f"Please upload a smaller image.",
                    parse_mode='Markdown'
                )
                return
            
            # Download and process image
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            
            processing_msg = await update.message.reply_text(
                "🔍 **Analyzing Your Image...**\n\n"
                "Processing image content and extracting insights. This may take a moment...",
                parse_mode='Markdown'
            )
            
            try:
                # Download image
                file_obj = await photo.get_file()
                image_data = await file_obj.download_as_bytearray()
                
                # Security validation
                is_valid, error_msg = FileProcessor.validate_file_security(bytes(image_data), f"image_{photo.file_id}.jpg", 'image')
                
                if not is_valid:
                    await processing_msg.edit_text(
                        f"🚫 **Image Validation Failed**\n\n{error_msg}\n\n"
                        "Please upload a valid image file.",
                        parse_mode='Markdown'
                    )
                    return
                
                # Process image
                await MessageHandlers._process_image_analysis(update, context, image_data, f"image_{photo.file_id}.jpg", processing_msg)
                
            except Exception as download_error:
                logger.error(f"Image download error for user {user_id}: {download_error}")
                await processing_msg.edit_text(
                    "❌ **Download Failed**\n\n"
                    "Failed to download your image. Please try uploading again.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"Error in photo handler for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Image Processing Error**\n\n"
                "An error occurred while processing your image. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _process_pdf_document(update, context, file_data: bytes, filename: str, processing_msg) -> None:
        """Process PDF document with AI analysis"""
        user_id = update.effective_user.id
        
        try:
            # Get user's API key
            api_key = await db.get_user_api_key(user_id)
            if not api_key:
                await processing_msg.edit_text(
                    "🔑 **API Key Required**\n\n"
                    "Please provide your Hugging Face API key first using the /start command.",
                    parse_mode='Markdown'
                )
                return
            
            await processing_msg.edit_text(
                "📄 **Extracting PDF Content...**\n\n"
                "Extracting text, tables, and metadata from your PDF document...",
                parse_mode='Markdown'
            )
            
            # Extract PDF content
            from bot.file_processors import FileProcessor
            pdf_result = await FileProcessor.extract_pdf_content(file_data, filename)
            
            if not pdf_result.get('success'):
                await processing_msg.edit_text(
                    f"❌ **PDF Extraction Failed**\n\n{pdf_result.get('error', 'Unknown error')}\n\n"
                    "Please ensure you've uploaded a valid PDF file.",
                    parse_mode='Markdown'
                )
                return
            
            await processing_msg.edit_text(
                "🧠 **Analyzing PDF Content with AI...**\n\n"
                "Using advanced AI models to analyze your document content...",
                parse_mode='Markdown'
            )
            
            # AI analysis
            async with ModelCaller() as model_caller:
                success, analysis_result = await model_caller.analyze_pdf(
                    pdf_result['full_text'],
                    pdf_result['metadata'],
                    api_key,
                    analysis_type="comprehensive"
                )
            
            if success and analysis_result:
                # Format response
                metadata = pdf_result['metadata']
                stats = pdf_result.get('stats', {})
                
                safe_filename = escape_markdown(filename)
                safe_analysis = safe_markdown_format(analysis_result.get('analysis', 'Analysis unavailable'))
                
                response_text = f"""📄 **PDF Analysis Complete**

**Document:** {safe_filename}
**Pages:** {metadata.get('pages', 'Unknown')} | **Size:** {metadata.get('file_size', 0):,} bytes
{'**Title:** ' + escape_markdown(metadata.get('title', '')) if metadata.get('title') else ''}
{'**Author:** ' + escape_markdown(metadata.get('author', '')) if metadata.get('author') else ''}

📊 **Content Statistics:**
• Characters: {stats.get('total_characters', 0):,}
• Pages with text: {stats.get('pages_with_text', 0)}
• Tables detected: {stats.get('tables_found', 0)}

🤖 **AI Analysis:**

{safe_analysis}

🎯 *Analyzed by {escape_markdown(analysis_result.get('model_used', 'AI'))} \- Advanced 2025 document processing*"""
                
                # Delete processing message and send result
                await processing_msg.delete()
                await update.message.reply_text(response_text, parse_mode='Markdown')
                
                logger.info(f"✅ PDF analysis completed successfully for user {user_id}")
                
            else:
                error_msg = analysis_result.get('error', 'Analysis failed') if analysis_result else 'AI analysis unavailable'
                await processing_msg.edit_text(
                    f"❌ **AI Analysis Failed**\n\n{error_msg}\n\n"
                    "The PDF was extracted successfully, but AI analysis encountered an error.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"PDF processing error for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **PDF Processing Error**\n\n"
                "An error occurred while processing your PDF. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _process_zip_archive(update, context, file_data: bytes, filename: str, processing_msg) -> None:
        """Process ZIP archive with comprehensive analysis"""
        user_id = update.effective_user.id
        
        try:
            # Get user's API key
            api_key = await db.get_user_api_key(user_id)
            if not api_key:
                await processing_msg.edit_text(
                    "🔑 **API Key Required**\n\n"
                    "Please provide your Hugging Face API key first using the /start command.",
                    parse_mode='Markdown'
                )
                return
            
            await processing_msg.edit_text(
                "📦 **Extracting ZIP Contents...**\n\n"
                "Safely extracting and cataloging files in your ZIP archive...",
                parse_mode='Markdown'
            )
            
            # Extract ZIP contents
            from bot.file_processors import FileProcessor
            zip_result = await FileProcessor.analyze_zip_archive(file_data, filename)
            
            if not zip_result.get('success'):
                await processing_msg.edit_text(
                    f"❌ **ZIP Extraction Failed**\n\n{zip_result.get('error', 'Unknown error')}\n\n"
                    "Please ensure you've uploaded a valid ZIP file.",
                    parse_mode='Markdown'
                )
                return
            
            await processing_msg.edit_text(
                "🧠 **Analyzing ZIP Contents with AI...**\n\n"
                "Using advanced AI models to analyze your archive contents...",
                parse_mode='Markdown'
            )
            
            # Determine analysis depth based on content
            file_contents = zip_result.get('file_contents', [])
            has_code = any(f.get('extension', '') in ['.py', '.js', '.java', '.cpp', '.php', '.rb'] for f in file_contents)
            analysis_depth = "code_focus" if has_code else "overview"
            
            # AI analysis
            async with ModelCaller() as model_caller:
                success, analysis_result = await model_caller.analyze_zip_contents(
                    file_contents,
                    api_key,
                    analysis_depth=analysis_depth
                )
            
            if success and analysis_result:
                # Format response
                archive_info = zip_result.get('archive_info', {})
                stats = zip_result.get('stats', {})
                
                safe_filename = escape_markdown(filename)
                safe_analysis = safe_markdown_format(analysis_result.get('analysis', 'Analysis unavailable'))
                
                file_types_summary = ', '.join([f"{count} {ftype}" for ftype, count in 
                                               dict(list({f.get('type', 'unknown'): 1 for f in file_contents}.items())[:5]).items()])
                
                response_text = f"""📦 **ZIP Archive Analysis Complete**

**Archive:** {safe_filename}
**Files:** {archive_info.get('total_files', 0)} | **Size:** {archive_info.get('compressed_size', 0):,} bytes
**Uncompressed:** {archive_info.get('uncompressed_size', 0):,} bytes

📊 **Content Summary:**
• Text files analyzed: {stats.get('text_files', 0)}
• File types: {escape_markdown(file_types_summary)}
• Analysis depth: {escape_markdown(analysis_result.get('analysis_depth', 'standard'))}

🤖 **AI Analysis:**

{safe_analysis}

🎯 *Analyzed by {escape_markdown(analysis_result.get('model_used', 'AI'))} \- Advanced 2025 archive processing*"""
                
                # Delete processing message and send result
                await processing_msg.delete()
                await update.message.reply_text(response_text, parse_mode='Markdown')
                
                logger.info(f"✅ ZIP analysis completed successfully for user {user_id}")
                
            else:
                error_msg = analysis_result.get('error', 'Analysis failed') if analysis_result else 'AI analysis unavailable'
                await processing_msg.edit_text(
                    f"❌ **AI Analysis Failed**\n\n{error_msg}\n\n"
                    "The ZIP was extracted successfully, but AI analysis encountered an error.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            _safe_log_error(logger.error, f"ZIP processing error for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **ZIP Processing Error**\n\n"
                "An error occurred while processing your ZIP file. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _process_image_analysis(update, context, image_data: bytes, filename: str, processing_msg) -> None:
        """Process image with comprehensive AI analysis"""
        user_id = update.effective_user.id
        
        try:
            # Get user's API key
            api_key = await db.get_user_api_key(user_id)
            if not api_key:
                await processing_msg.edit_text(
                    "🔑 **API Key Required**\n\n"
                    "Please provide your Hugging Face API key first using the /start command.",
                    parse_mode='Markdown'
                )
                return
            
            await processing_msg.edit_text(
                "🖼️ **Processing Image...**\n\n"
                "Extracting image information and preparing for AI analysis...",
                parse_mode='Markdown'
            )
            
            # Process image
            from bot.file_processors import FileProcessor
            image_result = await FileProcessor.process_image_content(image_data, filename, "comprehensive")
            
            if not image_result.get('success'):
                await processing_msg.edit_text(
                    f"❌ **Image Processing Failed**\n\n{image_result.get('error', 'Unknown error')}\n\n"
                    "Please ensure you've uploaded a valid image file.",
                    parse_mode='Markdown'
                )
                return
            
            await processing_msg.edit_text(
                "🧠 **Analyzing Image with AI...**\n\n"
                "Using advanced computer vision and AI models to analyze your image...",
                parse_mode='Markdown'
            )
            
            # AI analysis
            async with ModelCaller() as model_caller:
                success, ai_analysis = await model_caller.analyze_image_content(
                    image_data,
                    "comprehensive",
                    api_key
                )
            
            # Prepare response
            image_info = image_result.get('image_info', {})
            ocr_result = image_result.get('ocr', {})
            
            safe_filename = escape_markdown(filename)
            
            # Build response text
            response_text = f"""🖼️ **Image Analysis Complete**

**Image:** {safe_filename}
**Dimensions:** {image_info.get('width', 0)}x{image_info.get('height', 0)} pixels
**Format:** {image_info.get('format', 'Unknown')} | **Size:** {image_info.get('file_size', 0):,} bytes"""
            
            # Add OCR results if text was found
            if ocr_result.get('has_text'):
                extracted_text = ocr_result.get('text', '')
                if len(extracted_text) > 300:
                    extracted_text = extracted_text[:300] + "..."
                safe_ocr_text = escape_markdown(extracted_text)
                response_text += f"\n\n📖 **Text Extracted (OCR):**\n{safe_ocr_text}"
            
            # Add AI analysis if available
            if success and ai_analysis:
                if 'detailed_analysis' in ai_analysis:
                    safe_analysis = safe_markdown_format(ai_analysis['detailed_analysis'])
                    response_text += f"\n\n🤖 **AI Visual Analysis:**\n{safe_analysis}"
                elif 'description' in ai_analysis:
                    safe_description = safe_markdown_format(ai_analysis['description'])
                    response_text += f"\n\n🤖 **AI Description:**\n{safe_description}"
                elif 'guidance' in ai_analysis:
                    safe_guidance = safe_markdown_format(ai_analysis['guidance'])
                    response_text += f"\n\n💡 **Analysis Guidance:**\n{safe_guidance}"
                
                model_used = ai_analysis.get('model_used', 'AI vision models')
                response_text += f"\n\n🎯 *Analyzed by {escape_markdown(model_used)} \- Advanced 2025 computer vision*"
            else:
                response_text += "\n\n⚠️ **Note:** AI visual analysis is currently limited, but OCR and basic image processing completed successfully."
            
            # Delete processing message and send result
            await processing_msg.delete()
            await update.message.reply_text(response_text, parse_mode='Markdown')
            
            logger.info(f"✅ Image analysis completed successfully for user {user_id}")
            
        except Exception as e:
            _safe_log_error(logger.error, f"Image processing error for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **Image Processing Error**\n\n"
                "An error occurred while processing your image. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def error_handler(update, context) -> None:
        """Enhanced error handler for all bot operations"""
        user_id = getattr(update.effective_user, 'id', 'Unknown') if update.effective_user else 'Unknown'
        
        # Log error safely without exposing sensitive information
        _safe_log_error(logger.error, f"Update {update} caused error {context.error} for user {user_id}")
        
        # Send user-friendly error message
        try:
            if update.message:
                await update.message.reply_text(
                    "🚫 **Something went wrong**\n\n"
                    "I encountered an unexpected error. Please try again, and if the problem persists, contact support.",
                    parse_mode='Markdown'
                )
        except Exception:
            # Fallback if even error message fails
            logger.error("Failed to send error message to user")

# Export message handlers
message_handlers = MessageHandlers()