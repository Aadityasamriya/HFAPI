"""
Advanced message handlers for Hugging Face By AadityaLabs AI
Superior multi-modal processing that outperforms ChatGPT, Grok, and Gemini
Handles intelligent routing, context management, and enhanced file processing
"""

import asyncio
import io
import logging
import os
import re
import uuid
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from bot.storage_manager import db
from bot.core.router import router, IntentType
from bot.core.model_caller import ModelCaller, model_caller
# Enhanced AI Model Selection Integration
from bot.core.enhanced_integration import enhanced_integration
from bot.security_utils import redact_sensitive_data, get_secure_logger
from bot.core.response_processor import response_processor, ResponseQuality
from bot.core.smart_cache import smart_cache
from bot.core.code_generator import code_generator
from bot.core.response_formatter import response_formatter, ResponseType
from bot.file_processors import AdvancedFileProcessor
from bot.config import Config
from bot.security_utils import escape_markdown, safe_markdown_format, check_rate_limit
from datetime import datetime, timedelta
import time
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)
secure_logger = get_secure_logger(logger)


class MessageHandlers:
    """Advanced message processing for Hugging Face By AadityaLabs AI with intelligent AI routing and comprehensive observability"""
    
    @staticmethod
    async def _send_progress_updates(update, context, interval: int = 12) -> asyncio.Task:
        """
        Send periodic progress updates to user during long AI operations
        
        Args:
            update: Telegram update object
            context: Bot context
            interval: Seconds between progress messages (default 12s)
            
        Returns:
            asyncio.Task that can be cancelled when operation completes
        """
        progress_messages = [
            "⏳ Still working on your request...",
            "🔄 Processing your query...",
            "⚙️ AI is analyzing your request...",
            "🤖 Almost there, generating response...",
            "💭 Thinking through your request...",
        ]
        
        async def send_updates():
            try:
                message_index = 0
                while True:
                    await asyncio.sleep(interval)
                    await context.bot.send_chat_action(
                        chat_id=update.effective_chat.id, 
                        action=ChatAction.TYPING
                    )
                    await update.message.reply_text(
                        progress_messages[message_index % len(progress_messages)],
                        parse_mode='Markdown'
                    )
                    message_index += 1
            except asyncio.CancelledError:
                # Task was cancelled, which is expected when response is ready
                pass
            except Exception as e:
                logger.debug(f"Progress update error (non-critical): {e}")
        
        return asyncio.create_task(send_updates())
    
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
        
        # FIXED: Robust text extraction with proper error handling
        message_text = None
        
        # Primary method - direct text access (most reliable)
        try:
            message_text = update.message.text
            if message_text is not None:
                logger.debug(f"Successfully extracted text directly for user {user_id}")
        except AttributeError as e:
            logger.warning(f"Direct text access failed for user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in direct text access for user {user_id}: {e}")
        
        # Fallback method - raw update data
        if not message_text:
            try:
                raw_message = update.to_dict().get('message', {})
                message_text = raw_message.get('text', '')
                if message_text:
                    logger.info(f"Extracted text via raw update data for user {user_id}")
            except Exception as e:
                logger.warning(f"Raw update data extraction failed for user {user_id}: {e}")
        
        # Final validation and safe fallback
        if not message_text or not isinstance(message_text, str) or len(message_text.strip()) == 0:
            message_text = "Hello"  # Safe default message
            logger.info(f"Using safe default message text for user {user_id}")
        else:
            # Sanitize message text to prevent potential issues
            message_text = message_text.strip()[:4000]  # Limit length to prevent memory issues
            logger.debug(f"Text extraction successful for user {user_id}: {len(message_text)} chars")
        
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
            
            # SUPERIOR AI ROUTING: Advanced analysis with context and complexity
            start_time = time.time()
            
            # Build user context for intelligent routing
            user_context = await MessageHandlers._build_user_context(user_id, chat_history, context)
            
            # First, check smart cache for cached response
            cache_entry = await smart_cache.get(
                message_text, 
                "general",  # Will be updated with actual intent
                user_context
            )
            
            if cache_entry:
                logger.info(f"🎯 SUPERIOR CACHE HIT: Serving cached response (quality: {cache_entry.quality_level.value})")
                await update.message.reply_text(
                    cache_entry.content,
                    parse_mode='Markdown' if '**' in cache_entry.content or '_' in cache_entry.content else None
                )
                
                # Update user satisfaction tracking
                await smart_cache.update_user_satisfaction(cache_entry.key, 0.9)  # Assume high satisfaction for cache hit
                return
            
            # Enhanced dynamic model selection with superior intelligence
            request_id = str(uuid.uuid4())
            conversation_id = f"conv_{user_id}_{int(time.time())}"
            
            # First get intent classification 
            intent, routing_info = await router.route_prompt(message_text, user_id, user_context)
            
            # Use enhanced model selection if enabled
            if hasattr(enhanced_integration, 'enhanced_enabled') and enhanced_integration.enhanced_enabled:
                try:
                    selected_model, enhanced_metadata = await enhanced_integration.select_model_for_request(
                        prompt=message_text,
                        intent_type=intent.value,
                        user_id=str(user_id),
                        conversation_id=conversation_id,
                        request_id=request_id,
                        enable_fallback=True
                    )
                    
                    # Update routing info with enhanced selection
                    routing_info['selected_model'] = selected_model
                    routing_info['enhanced_metadata'] = enhanced_metadata
                    routing_info['request_id'] = request_id
                    routing_info['conversation_id'] = conversation_id
                    
                    logger.info(f"🎯 ENHANCED DYNAMIC ROUTING: user_id:{user_id} -> {intent.value}")
                    logger.info(f"   📊 Enhanced Confidence: {enhanced_metadata.get('confidence', 'N/A'):.2f}")
                    logger.info(f"   🤖 Selected Model: {selected_model.split('/')[-1]}")
                    logger.info(f"   🚀 Strategy: {enhanced_metadata.get('selection_strategy', 'N/A')}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Enhanced selection failed, using traditional: {redact_sensitive_data(str(e))}")
                    # Traditional routing already set
                    
            else:
                logger.info(f"🚀 TRADITIONAL ROUTING: user_id:{user_id} -> {intent.value}")
                logger.info(f"   📊 Confidence: {routing_info['confidence']:.2f}")
                logger.info(f"   🤖 Model: {routing_info['selected_model']}")
                logger.info(f"   ⚡ Quality Score: {routing_info.get('routing_quality_score', 'N/A')}")
            
            # Store routing info for processing
            routing_time = time.time() - start_time
            
            # 2025: Enhanced processing with new intent types
            if intent == IntentType.IMAGE_GENERATION:
                await MessageHandlers._handle_image_generation(update, context, message_text, api_key, routing_info)
            
            elif intent == IntentType.CODE_GENERATION:
                await MessageHandlers._handle_enhanced_code_generation(update, context, message_text, api_key, routing_info)
            
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
            safe_exception_msg = redact_sensitive_data(str(e))
            secure_logger.error(f"Error processing message for user {user_id}: {safe_exception_msg}")
            
            # UX7 fix: Provide specific error messages based on error type
            error_lower = str(e).lower()
            
            if 'rate limit' in error_lower or 'too many requests' in error_lower:
                error_msg = "⏰ **Rate Limit Reached**\n\nYou've sent too many requests. Please wait a moment and try again."
            elif 'timeout' in error_lower or 'timed out' in error_lower:
                error_msg = "⏱️ **Request Timeout**\n\nThe AI took too long to respond. Please try again with a simpler request."
            elif 'network' in error_lower or 'connection' in error_lower:
                error_msg = "🌐 **Connection Issue**\n\nI'm having trouble connecting to the AI service. Please check your internet and try again."
            elif 'invalid' in error_lower or 'format' in error_lower:
                error_msg = "📝 **Invalid Format**\n\nThere's an issue with your request format. Please try rephrasing or simplifying your message."
            elif 'unauthorized' in error_lower or 'forbidden' in error_lower or 'api key' in error_lower:
                error_msg = "🔑 **Authentication Issue**\n\nThere's a problem with your API key. Try setting it up again with `/start`"
            elif 'file' in error_lower and 'size' in error_lower:
                error_msg = "📁 **File Too Large**\n\nThe file you sent is too large. Please try a smaller file."
            else:
                error_msg = "🚫 **Processing Error**\n\nI encountered an issue processing your request. Please try:\n\n• Rephrasing your message\n• Simplifying your request\n• Using `/newchat` to start fresh\n\nIf the problem persists, contact support."
            
            safe_error_msg = safe_markdown_format(error_msg)
            await update.message.reply_text(
                safe_error_msg,
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _build_user_context(user_id: int, chat_history: list, context) -> Dict[str, Any]:
        """Build comprehensive user context for superior AI routing"""
        user_context = {
            'user_id': user_id,
            'domain_context': 'general',
            'technical_level': 'intermediate',
            'language': 'en',
            'complexity_level': 'medium'
        }
        
        # Analyze chat history for context
        if chat_history and len(chat_history) > 0:
            recent_messages = [msg.get('content', '') for msg in chat_history[-5:] if msg.get('role') == 'user']
            combined_text = ' '.join(recent_messages).lower()
            
            # Detect technical level
            tech_indicators = ['python', 'javascript', 'api', 'database', 'algorithm', 'code', 'programming']
            if sum(1 for word in tech_indicators if word in combined_text) >= 2:
                user_context['technical_level'] = 'advanced'
                user_context['domain_context'] = 'programming'
            
            # Detect creative context
            creative_indicators = ['story', 'creative', 'write', 'design', 'art', 'image', 'generate']
            if sum(1 for word in creative_indicators if word in combined_text) >= 2:
                user_context['domain_context'] = 'creative'
            
            # Detect complexity level based on message length and technical terms
            avg_length = sum(len(msg) for msg in recent_messages) / max(len(recent_messages), 1)
            if avg_length > 200:
                user_context['complexity_level'] = 'high'
            elif avg_length < 50:
                user_context['complexity_level'] = 'low'
        
        return user_context
    
    @staticmethod
    def _get_language_identifier(filename: str, content_type: str) -> str:
        """Get proper language identifier for Telegram syntax highlighting"""
        # Extract extension from filename
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Language mapping for Telegram syntax highlighting
        language_map = {
            'py': 'python',
            'js': 'javascript', 
            'jsx': 'javascript',
            'ts': 'typescript',
            'tsx': 'typescript',
            'java': 'java',
            'cs': 'csharp',
            'cpp': 'cpp',
            'cxx': 'cpp',
            'cc': 'cpp',
            'c': 'c',
            'h': 'c',
            'html': 'html',
            'htm': 'html',
            'css': 'css',
            'scss': 'css',
            'sass': 'css',
            'less': 'css',
            'sql': 'sql',
            'sh': 'bash',
            'bash': 'bash',
            'zsh': 'bash',
            'fish': 'bash',
            'json': 'json',
            'xml': 'xml',
            'yaml': 'yaml',
            'yml': 'yaml',
            'md': 'markdown',
            'markdown': 'markdown',
            'go': 'go',
            'rs': 'rust',
            'php': 'php',
            'rb': 'ruby',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala',
            'r': 'r',
            'm': 'matlab',
            'dockerfile': 'dockerfile'
        }
        
        # First try extension-based detection
        if extension in language_map:
            return language_map[extension]
        
        # Fallback to content-type analysis
        if 'python' in content_type:
            return 'python'
        elif 'javascript' in content_type:
            return 'javascript'
        elif 'html' in content_type:
            return 'html'
        elif 'css' in content_type:
            return 'css'
        elif 'json' in content_type:
            return 'json'
        elif 'xml' in content_type:
            return 'xml'
        
        # Default to text for unknown types
        return 'text'
    
    @staticmethod
    def _get_enhanced_mime_type(filename: str) -> str:
        """Get enhanced MIME type with proper file association"""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Comprehensive MIME type mapping
        mime_types = {
            'py': 'text/x-python',
            'js': 'text/javascript',
            'jsx': 'text/javascript',
            'ts': 'application/typescript',
            'tsx': 'application/typescript',
            'java': 'text/x-java-source',
            'cs': 'text/x-csharp',
            'cpp': 'text/x-c++src',
            'cxx': 'text/x-c++src',
            'cc': 'text/x-c++src',
            'c': 'text/x-csrc',
            'h': 'text/x-chdr',
            'html': 'text/html',
            'htm': 'text/html',
            'css': 'text/css',
            'scss': 'text/x-scss',
            'sass': 'text/x-sass',
            'less': 'text/x-less',
            'sql': 'application/sql',
            'sh': 'application/x-shellscript',
            'bash': 'application/x-shellscript',
            'zsh': 'application/x-shellscript',
            'fish': 'application/x-shellscript',
            'json': 'application/json',
            'xml': 'application/xml',
            'yaml': 'application/x-yaml',
            'yml': 'application/x-yaml',
            'md': 'text/markdown',
            'markdown': 'text/markdown',
            'go': 'text/x-go',
            'rs': 'text/x-rust',
            'php': 'text/x-php',
            'rb': 'text/x-ruby',
            'swift': 'text/x-swift',
            'kt': 'text/x-kotlin',
            'scala': 'text/x-scala',
            'r': 'text/x-r',
            'm': 'text/x-matlab',
            'dockerfile': 'text/x-dockerfile',
            'txt': 'text/plain',
            'log': 'text/plain',
            'conf': 'text/plain',
            'config': 'text/plain'
        }
        
        return mime_types.get(extension, 'text/plain')
    
    @staticmethod
    async def _handle_enhanced_code_generation(update, context, message_text: str, api_key: str, routing_info: Dict) -> None:
        """
        Enhanced code generation that surpasses ChatGPT, Grok, and Gemini
        Features: Multi-file support, syntax validation, professional formatting, actual file downloads
        """
        user_id = update.effective_user.id
        
        try:
            # Send enhanced typing action
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            logger.info(f"🚀 ENHANCED CODE GENERATION started for user_id:{user_id}")
            logger.info(f"📝 Prompt: {message_text[:100]}...")
            
            # Start progress updates for long operations (UX4 fix)
            progress_task = await MessageHandlers._send_progress_updates(update, context, interval=12)
            
            try:
                # Use the superior code generator with context
                user_context = await MessageHandlers._build_user_context(user_id, [], context)
                generation_result = await code_generator.generate_code_files(message_text, user_context)
            finally:
                # Cancel progress updates when response is ready
                progress_task.cancel()
            
            # Format response using advanced formatter
            formatted_response = await response_formatter.format_code_generation_response(
                generation_result, 
                {'user_id': user_id, 'routing_info': routing_info}
            )
            
            # Send the main formatted response with enhanced copy instructions
            await update.message.reply_text(
                formatted_response.main_text,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            # Enhanced file handling: Send both copy-friendly text AND downloadable files
            if formatted_response.attachments and len(formatted_response.attachments) <= 10:
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
                
                # Send instruction message for both options
                instructions_message = f"""📂 **File Options Available:**

🔹 **Option 1: Copy from Messages** - Individual file contents below for easy copying
🔹 **Option 2: Download Files** - Downloadable files with proper extensions

📋 *For copying: Tap any code block below*
⬇️ *For downloads: Use the file attachments*"""
                
                await update.message.reply_text(
                    instructions_message,
                    parse_mode='Markdown'
                )
                
                # Send both copy-friendly messages AND downloadable files
                for attachment in formatted_response.attachments:
                    # Get proper language identifier for syntax highlighting
                    language_id = MessageHandlers._get_language_identifier(attachment.filename, attachment.content_type)
                    
                    # Create enhanced copy-friendly message with better formatting
                    file_content_message = f"""📄 **{attachment.filename}**
*{attachment.description}*
📊 {attachment.file_size:,} bytes • {len(attachment.content.splitlines())} lines

```{language_id}
{attachment.content}
```

💡 **Tap the code block above to copy {attachment.filename}**"""
                    
                    # Send copy-friendly message
                    await update.message.reply_text(
                        file_content_message,
                        parse_mode='Markdown',
                        disable_web_page_preview=True
                    )
                    
                    # Send actual downloadable file with proper MIME type
                    try:
                        import io
                        
                        # Create file buffer with proper encoding
                        file_buffer = io.BytesIO(attachment.content.encode('utf-8') if isinstance(attachment.content, str) else attachment.content)
                        file_buffer.name = attachment.filename
                        
                        # Send as downloadable document with proper MIME type
                        await update.message.reply_document(
                            document=file_buffer,
                            filename=attachment.filename,
                            caption=f"⬇️ **{attachment.filename}**\n📋 {attachment.description}\n💾 Ready to download and use",
                            parse_mode='Markdown'
                        )
                        
                        logger.info(f"✅ Sent downloadable file: {attachment.filename} for user {user_id}")
                        
                    except Exception as file_error:
                        logger.error(f"❌ Failed to send file {attachment.filename} for user {user_id}: {file_error}")
                        # Continue with other files even if one fails
                        await update.message.reply_text(
                            f"⚠️ **Download Error for {attachment.filename}**\nFile content is available in the message above for copying.",
                            parse_mode='Markdown'
                        )
                
                # Send summary message
                summary_message = f"""✅ **Code Generation Complete!**

📁 **Generated:** {len(formatted_response.attachments)} files
📋 **Copy Option:** Tap any code block above
⬇️ **Download Option:** Use file attachments above
⭐ **Quality Score:** {formatted_response.metadata.get('quality_score', 8.5):.1f}/10

🎯 *All files are ready for both copying and downloading!*"""
                
                await update.message.reply_text(
                    summary_message,
                    parse_mode='Markdown'
                )
            
            # Cache the successful response  
            generation_time = 1.0  # Default generation time
            user_context = {'user_id': user_id, 'files_generated': len(formatted_response.attachments)}
            
            await smart_cache.store(
                prompt=message_text,
                intent_type="code_generation",
                content=formatted_response.main_text,
                user_context=user_context,
                model_used=routing_info.get('selected_model', 'unknown'),
                response_time=generation_time,
                quality_score=generation_result.quality_score if hasattr(generation_result, 'quality_score') else 7.5
            )
            
            logger.info(f"✅ ENHANCED CODE GENERATION completed: {generation_result.total_files} files, quality {generation_result.quality_score:.1f}/10")
            
        except Exception as e:
            logger.error(f"❌ Enhanced code generation failed for user {user_id}: {e}")
            error_response = await response_formatter.format_error_response(
                f"Code generation failed: {str(e)}",
                "Please try with a more specific code request or simpler requirements."
            )
            await update.message.reply_text(
                error_response.main_text,
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def document_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Enhanced document handler with superior analysis capabilities
        Supports PDF and ZIP files with intelligent processing
        
        SECURITY: DoS Prevention - Rate limiting, concurrency limits, and timeouts
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"📄 DOCUMENT UPLOAD from user_id:{user_id} (@{username})")
        
        # CRITICAL SECURITY: Check file upload rate limit (5 files per 5 minutes)
        from bot.security_utils import check_file_upload_rate_limit
        is_allowed, wait_time = check_file_upload_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"🚫 **File Upload Limit Exceeded**\n\n"
                f"You've uploaded too many files recently.\n"
                f"Please wait {wait_time} seconds before uploading another file.\n\n"
                f"📋 Limit: 5 files per 5 minutes",
                parse_mode='Markdown'
            )
            return
        
        # Get user's API key
        try:
            api_key = await db.get_user_api_key(user_id)
            if not api_key:
                await update.message.reply_text(
                    "🔐 **API Key Required**\n\nPlease set up your Hugging Face API key first using /start",
                    parse_mode='Markdown'
                )
                return
        except Exception as e:
            logger.error(f"Error getting API key for user {user_id}: {e}")
            return
        
        document = update.message.document
        if not document:
            await update.message.reply_text("❌ **No Document Detected**\n\nPlease send a valid document file.")
            return
        
        filename = document.file_name or "unknown_document"
        file_size = document.file_size
        
        # SECURITY FIX: Validate file size BEFORE downloading to prevent resource exhaustion
        if file_size and file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            max_size_mb = AdvancedFileProcessor.MAX_FILE_SIZE / (1024 * 1024)
            actual_size_mb = file_size / (1024 * 1024)
            await update.message.reply_text(
                f"🚫 **File Too Large**\n\n"
                f"Maximum allowed size: {max_size_mb:.1f}MB\n"
                f"Your file size: {actual_size_mb:.1f}MB\n\n"
                f"Please upload a smaller file.",
                parse_mode='Markdown'
            )
            logger.warning(f"🚫 Pre-download validation rejected {filename}: {file_size:,} bytes exceeds limit of {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes")
            return
        
        logger.info(f"📁 Processing document: {filename} ({file_size:,} bytes) - pre-download validation passed")
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            f"🔄 **Processing Document...**\n\n📄 **{filename}**\n📊 Size: {file_size:,} bytes\n\n⚡ Using advanced AI analysis...",
            parse_mode='Markdown'
        )
        
        # CRITICAL SECURITY: Acquire semaphore for concurrent file processing control
        from bot.file_processors import file_processing_semaphore, with_file_processing_timeout, FileProcessingTimeoutError, FileConcurrencyLimitError
        from bot.config import Config
        
        # Check if we can process this file (concurrency limits)
        can_process, limit_error = await file_processing_semaphore.acquire(user_id)
        if not can_process:
            await update.message.reply_text(
                f"🚫 **Processing Limit Reached**\n\n{limit_error}",
                parse_mode='Markdown'
            )
            return
        
        try:
            # Download file (only after size validation)
            file = await context.bot.get_file(document.file_id)
            file_data = await file.download_as_bytearray()
            
            # Determine file type and validate security
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            
            if file_ext == 'pdf':
                is_valid, error_msg = AdvancedFileProcessor.validate_file_security(
                    bytes(file_data), filename, 'pdf'
                )
                
                if not is_valid:
                    await processing_msg.edit_text(f"🚫 **Security Check Failed**\n\n{error_msg}")
                    return
                
                # SECURITY: Enhanced PDF analysis with 30s timeout
                doc_structure = await with_file_processing_timeout(
                    AdvancedFileProcessor.enhanced_pdf_analysis(bytes(file_data), filename),
                    timeout_seconds=Config.FILE_PROCESSING_TIMEOUT,
                    operation_name=f"PDF analysis for {filename}"
                )
                formatted_response = await response_formatter.format_document_analysis_response(doc_structure, filename)
                
            elif file_ext == 'zip':
                is_valid, error_msg = AdvancedFileProcessor.validate_file_security(
                    bytes(file_data), filename, 'zip'
                )
                
                if not is_valid:
                    await processing_msg.edit_text(f"🚫 **Security Check Failed**\n\n{error_msg}")
                    return
                
                # SECURITY: Intelligent ZIP analysis with 30s timeout
                zip_analysis = await with_file_processing_timeout(
                    AdvancedFileProcessor.intelligent_zip_analysis(bytes(file_data), filename),
                    timeout_seconds=Config.FILE_PROCESSING_TIMEOUT,
                    operation_name=f"ZIP analysis for {filename}"
                )
                formatted_response = await response_formatter.format_zip_analysis_response(zip_analysis, filename)
                
            else:
                await processing_msg.edit_text(
                    f"❌ **Unsupported Format**\n\nCurrently supported: PDF, ZIP\n\nReceived: {file_ext.upper() if file_ext else 'Unknown'}"
                )
                return
            
            # Send the formatted analysis result
            await processing_msg.edit_text(
                formatted_response.main_text,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            logger.info(f"✅ Document analysis completed for {filename}")
            
        except FileProcessingTimeoutError as e:
            logger.error(f"⏱️ Document processing timeout for user {user_id}: {e}")
            await processing_msg.edit_text(
                f"⏱️ **Processing Timeout**\n\n"
                f"Your file took too long to process (>{Config.FILE_PROCESSING_TIMEOUT}s).\n"
                f"This usually happens with very large or complex files.\n\n"
                f"💡 Try uploading a smaller file or splitting it into parts.",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Document processing failed for user {user_id}: {e}")
            await processing_msg.edit_text(
                f"❌ **Processing Failed**\n\nSorry, I couldn't process your document. Please try again or contact support."
            )
        finally:
            # CRITICAL: Always release semaphore, even if processing fails
            file_processing_semaphore.release(user_id)
    
    @staticmethod
    async def photo_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Enhanced photo handler with superior image analysis
        Features: Advanced OCR, object detection, intelligent content description
        
        SECURITY: DoS Prevention - Rate limiting, concurrency limits, and timeouts
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"🖼️ PHOTO UPLOAD from user_id:{user_id} (@{username})")
        
        # CRITICAL SECURITY: Check file upload rate limit (5 files per 5 minutes)
        from bot.security_utils import check_file_upload_rate_limit
        is_allowed, wait_time = check_file_upload_rate_limit(user_id)
        if not is_allowed:
            await update.message.reply_text(
                f"🚫 **File Upload Limit Exceeded**\n\n"
                f"You've uploaded too many files recently.\n"
                f"Please wait {wait_time} seconds before uploading another file.\n\n"
                f"📋 Limit: 5 files per 5 minutes",
                parse_mode='Markdown'
            )
            return
        
        # Get the largest photo size
        photo = update.message.photo[-1]  # Largest size
        file_size = photo.file_size
        
        # SECURITY FIX: Validate file size BEFORE downloading to prevent resource exhaustion
        if file_size and file_size > AdvancedFileProcessor.MAX_IMAGE_SIZE:
            max_size_mb = AdvancedFileProcessor.MAX_IMAGE_SIZE / (1024 * 1024)
            actual_size_mb = file_size / (1024 * 1024)
            await update.message.reply_text(
                f"🚫 **Image Too Large**\n\n"
                f"Maximum allowed size: {max_size_mb:.1f}MB\n"
                f"Your image size: {actual_size_mb:.1f}MB\n\n"
                f"Please upload a smaller image.",
                parse_mode='Markdown'
            )
            logger.warning(f"🚫 Pre-download validation rejected image: {file_size:,} bytes exceeds limit of {AdvancedFileProcessor.MAX_IMAGE_SIZE:,} bytes")
            return
        
        logger.info(f"📸 Processing photo: {photo.width}×{photo.height} ({file_size:,} bytes) - pre-download validation passed")
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            f"🔄 **Analyzing Image...**\n\n📊 Size: {photo.width}×{photo.height} pixels\n📦 File: {file_size:,} bytes\n\n🧠 Running advanced AI analysis...",
            parse_mode='Markdown'
        )
        
        # CRITICAL SECURITY: Acquire semaphore for concurrent file processing control
        from bot.file_processors import file_processing_semaphore, with_file_processing_timeout, FileProcessingTimeoutError, FileConcurrencyLimitError
        from bot.config import Config
        
        # Check if we can process this file (concurrency limits)
        can_process, limit_error = await file_processing_semaphore.acquire(user_id)
        if not can_process:
            await update.message.reply_text(
                f"🚫 **Processing Limit Reached**\n\n{limit_error}",
                parse_mode='Markdown'
            )
            return
        
        try:
            # Download image (only after size validation)
            file = await context.bot.get_file(photo.file_id)
            image_data = await file.download_as_bytearray()
            
            # Validate security
            is_valid, error_msg = AdvancedFileProcessor.validate_file_security(
                bytes(image_data), "uploaded_image.jpg", 'image'
            )
            
            if not is_valid:
                await processing_msg.edit_text(f"🚫 **Security Check Failed**\n\n{error_msg}")
                return
            
            # SECURITY: Advanced image analysis with 30s timeout
            image_analysis = await with_file_processing_timeout(
                AdvancedFileProcessor.advanced_image_analysis(bytes(image_data), "uploaded_image.jpg"),
                timeout_seconds=Config.FILE_PROCESSING_TIMEOUT,
                operation_name="Image analysis"
            )
            formatted_response = await response_formatter.format_image_analysis_response(image_analysis, "uploaded_image.jpg")
            
            # Send the formatted analysis result
            await processing_msg.edit_text(
                formatted_response.main_text,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            logger.info(f"✅ Image analysis completed - OCR: {len(image_analysis.ocr_text)} chars, Objects: {len(image_analysis.detected_objects)}")
            
        except FileProcessingTimeoutError as e:
            logger.error(f"⏱️ Image processing timeout for user {user_id}: {e}")
            await processing_msg.edit_text(
                f"⏱️ **Processing Timeout**\n\n"
                f"Your image took too long to process (>{Config.FILE_PROCESSING_TIMEOUT}s).\n"
                f"This usually happens with very large or complex images.\n\n"
                f"💡 Try uploading a smaller or simpler image.",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Image processing failed for user {user_id}: {e}")
            await processing_msg.edit_text(
                f"❌ **Analysis Failed**\n\nSorry, I couldn't analyze your image. Please try again or contact support."
            )
        finally:
            # CRITICAL: Always release semaphore, even if processing fails
            file_processing_semaphore.release(user_id)
    
    @staticmethod
    async def error_handler(update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Enhanced error handler with intelligent error analysis and recovery"""
        
        try:
            user_id = update.effective_user.id if update.effective_user else 0
            
            # Log the error with enhanced context
            logger.error(f"🚨 TELEGRAM ERROR for user_id:{user_id}")
            logger.error(f"🔍 Error: {context.error}")
            logger.error(f"📋 Update type: {type(update).__name__}")
            
            # Format error response
            error_response = await response_formatter.format_error_response(
                "An unexpected error occurred while processing your request.",
                "Please try again in a moment. If the problem persists, contact support."
            )
            
            # Try to send error response to user
            if update.effective_chat:
                try:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=error_response.main_text,
                        parse_mode='Markdown'
                    )
                except Exception:
                    # If we can't send the formatted response, try a simple one
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text="⚠️ An error occurred. Please try again."
                    )
            
        except Exception as error_in_handler:
            logger.error(f"🚨 ERROR IN ERROR HANDLER: {error_in_handler}")
    
    @staticmethod
    async def _handle_multi_modal(update, context, message_text: str, api_key: str, routing_info: Dict) -> None:
        """
        Handle multi-modal requests that combine text, code, image, and document processing
        Superior unified intelligence that outperforms ChatGPT, Grok, and Gemini
        """
        user_id = update.effective_user.id
        
        try:
            logger.info(f"🌟 MULTI-MODAL PROCESSING started for user_id:{user_id}")
            
            # Send enhanced processing message
            await update.message.reply_text(
                "🌟 **Multi-Modal AI Processing**\n\n🧠 Analyzing your request with unified intelligence...\n⚡ Superior to ChatGPT, Grok, and Gemini",
                parse_mode='Markdown'
            )
            
            # Determine what type of multi-modal processing is needed
            request_lower = message_text.lower()
            
            # Check if it's a code generation request with file output
            if any(keyword in request_lower for keyword in ['create', 'generate', 'build', 'make', 'code', 'app', 'website']):
                await MessageHandlers._handle_enhanced_code_generation(update, context, message_text, api_key, routing_info)
                return
            
            # Otherwise, handle as advanced conversation with context awareness
            await MessageHandlers._handle_conversation(update, context, message_text, api_key, [], routing_info)
            
        except Exception as e:
            logger.error(f"❌ Multi-modal processing failed for user {user_id}: {e}")
            error_response = await response_formatter.format_error_response(
                f"Multi-modal processing failed: {str(e)}",
                "Please try breaking down your request into smaller parts."
            )
            await update.message.reply_text(
                error_response.main_text,
                parse_mode='Markdown'
            )
    
    
    @staticmethod
    async def _handle_text_generation(update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, chat_history: list, routing_info: dict) -> None:
        """SUPERIOR text generation with advanced processing and quality assurance"""
        user_id = update.effective_user.id
        start_time = time.time()
        
        # Update chat history with user message (include timestamp)
        chat_history.append({'role': 'user', 'content': prompt, 'timestamp': datetime.utcnow()})
        
        # Limit chat history size
        if len(chat_history) > Config.MAX_CHAT_HISTORY * 2:  # *2 for user + assistant pairs
            chat_history = chat_history[-Config.MAX_CHAT_HISTORY * 2:]
        
        try:
            # Use superior model caller with monitoring
            selected_model = routing_info.get('selected_model') or routing_info.get('recommended_model') or Config.DEFAULT_TEXT_MODEL
            intent_type = routing_info.get('primary_intent', 'text_generation').value if hasattr(routing_info.get('primary_intent'), 'value') else 'text_generation'
            
            logger.info(f"🤖 SUPERIOR GENERATION: Using {selected_model} for {intent_type}")
            
            # Start progress updates for long operations (UX4 fix)
            progress_task = await MessageHandlers._send_progress_updates(update, context, interval=12)
            
            try:
                # Call with performance monitoring and timeout protection
                try:
                    success, response, perf_metrics = await asyncio.wait_for(
                        model_caller.call_with_monitoring(
                            'generate_text',
                            selected_model,
                            intent_type,
                            prompt, 
                            api_key, 
                            chat_history[:-1],  # Exclude the current message
                            selected_model,
                            routing_info.get('special_parameters', {})
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ AI model call timeout (30s) for user {user_id} with model {selected_model}")
                    success = False
                    response = "AI request timed out after 30 seconds"
                    perf_metrics = {'response_time': 30.0}
            finally:
                # Cancel progress updates when response is ready
                progress_task.cancel()
            
            if success and response:
                # SUPERIOR RESPONSE PROCESSING with quality assessment
                complexity_data = routing_info.get('complexity_analysis')
                
                # Process response with quality validation and enhancement
                enhanced_response, quality_metrics = response_processor.process_response(
                    response,
                    prompt,
                    intent_type,
                    selected_model,
                    complexity_data.__dict__ if complexity_data else None
                )
                
                logger.info(f"🔍 RESPONSE QUALITY:")
                logger.info(f"   📊 Overall Score: {quality_metrics.overall_score:.1f}/10 ({quality_metrics.quality_level.value})")
                logger.info(f"   🎯 Relevance: {quality_metrics.relevance_score:.1f}/10")
                logger.info(f"   ✅ Completeness: {quality_metrics.completeness_score:.1f}/10")
                logger.info(f"   🔧 Technical: {quality_metrics.technical_score:.1f}/10")
                logger.info(f"   💎 Clarity: {quality_metrics.clarity_score:.1f}/10")
                
                # Update chat history with enhanced response
                chat_history.append({'role': 'assistant', 'content': enhanced_response, 'timestamp': datetime.utcnow()})
                if context.user_data is not None:
                    context.user_data['chat_history'] = chat_history
                
                # Format for user display
                safe_response = safe_markdown_format(enhanced_response, preserve_code=True)
                
                # Add quality indicator for excellent responses
                if quality_metrics.quality_level == ResponseQuality.EXCELLENT:
                    quality_badge = "✨ **Premium AI Response**\n\n"
                elif quality_metrics.quality_level == ResponseQuality.GOOD:
                    quality_badge = "🎯 **High-Quality Response**\n\n"
                else:
                    quality_badge = ""
                
                formatted_response = quality_badge + safe_response
                
                # Send response to user
                await update.message.reply_text(
                    formatted_response,
                    parse_mode='Markdown'
                )
                
                # SUPERIOR CACHING: Store high-quality responses
                if quality_metrics.overall_score >= 6.0:  # Only cache decent responses
                    user_context = await MessageHandlers._build_user_context(user_id, chat_history, context)
                    
                    await smart_cache.store(
                        prompt=prompt,
                        intent_type=intent_type,
                        content=enhanced_response,
                        user_context=user_context,
                        model_used=selected_model,
                        response_time=perf_metrics['response_time'],
                        quality_score=quality_metrics.overall_score,
                        complexity_score=complexity_data.complexity_score if complexity_data else 5.0
                    )
                    
                    logger.info(f"💾 CACHED: Response stored (quality: {quality_metrics.overall_score:.1f})")
                
                # Performance monitoring update
                total_time = time.time() - start_time
                logger.info(f"⚡ PERFORMANCE: Total time: {total_time:.2f}s, Model time: {perf_metrics['response_time']:.2f}s")
                
                # Save conversation to persistent storage
                try:
                    saved = await MessageHandlers._save_conversation_if_ready(user_id, chat_history, context)
                    if saved:
                        logger.info(f"💾 Conversation saved to persistent storage for user {user_id}")
                    else:
                        logger.debug(f"Conversation not yet ready for saving for user {user_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save conversation for user {user_id}: {save_error}")
                
                logger.info(f"🏆 SUPERIOR GENERATION COMPLETE: user_id={user_id}, quality={quality_metrics.overall_score:.1f}/10")
                
            else:
                # Handle generation failure with fallback
                logger.warning(f"❌ Generation failed: {response}")
                
                # Try fallback model if available
                fallback_model = Config.FALLBACK_TEXT_MODEL
                if fallback_model and fallback_model != selected_model:
                    logger.info(f"🔄 Trying fallback model: {fallback_model}")
                    
                    try:
                        success_fallback, response_fallback, _ = await asyncio.wait_for(
                            model_caller.call_with_monitoring(
                                'generate_text',
                                fallback_model,
                                intent_type,
                                prompt,
                                api_key,
                                chat_history[:-1],
                                fallback_model,
                                {}  # Basic parameters for fallback
                            ),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"⏱️ Fallback model call timeout (30s) for user {user_id} with model {fallback_model}")
                        success_fallback = False
                        response_fallback = "Fallback AI request also timed out"
                    
                    if success_fallback and response_fallback:
                        safe_response = safe_markdown_format(response_fallback, preserve_code=True)
                        await update.message.reply_text(
                            f"⚠️ **Using backup system**\n\n{safe_response}",
                            parse_mode='Markdown'
                        )
                        
                        # Update chat history with fallback response
                        chat_history.append({'role': 'assistant', 'content': response_fallback, 'timestamp': datetime.utcnow()})
                        if context.user_data is not None:
                            context.user_data['chat_history'] = chat_history
                        return
                
                safe_error_response = safe_markdown_format(str(response))
                await update.message.reply_text(
                    f"❌ **Generation Failed**\n\n{safe_error_response}\n\nPlease try again or use `/newchat` to start fresh\.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            secure_logger.error(f"Error in text generation for user {user_id}: {e}")
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
                try:
                    success, code_response = await asyncio.wait_for(
                        model_caller.generate_code(
                            prompt, 
                            api_key, 
                            language,
                            routing_info.get('special_parameters', {})  # Use advanced parameters
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Code generation timeout (30s) for user {user_id}")
                    success = False
                    code_response = "Code generation timed out after 30 seconds. Please try a simpler request."
            
            if success and code_response:
                safe_language = escape_markdown(language)
                safe_code_response = safe_markdown_format(code_response, preserve_code=True)
                response_text = f"💻 **Here's your {safe_language} code:**\n\n{safe_code_response}"
                
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
            secure_logger.error(f"Error in code generation for user {user_id}: {e}")
            await update.message.reply_text(
                "🚫 **Code Generation Error**\n\nFailed to generate code. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_image_generation(update, context: ContextTypes.DEFAULT_TYPE, prompt: str, api_key: str, routing_info: dict) -> None:
        """Handle enhanced image generation with free-tier optimization"""
        user_id = update.effective_user.id
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎨 **Creating Enhanced Artistic Guidance...**\n\n✨ Generating detailed descriptions and professional guidance for your image request...",
            parse_mode='Markdown'
        )
        
        try:
            async with ModelCaller() as model_caller:
                try:
                    success, generation_result = await asyncio.wait_for(
                        model_caller.generate_image(
                            prompt, 
                            api_key,
                            routing_info.get('special_parameters', {})  # Enhanced parameters
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Image generation timeout (30s) for user {user_id}")
                    success = False
                    generation_result = None
                    await processing_msg.edit_text(
                        "❌ **Request Timeout**\n\nImage generation took too long (>30s). Please try a simpler prompt.",
                        parse_mode='Markdown'
                    )
                    return
            
            if success and generation_result:
                # Delete processing message
                await processing_msg.delete()
                
                # Check if we got actual image data or enhanced description
                if generation_result.get('type') == 'image':
                    # Actual image was generated (rare case when models are available)
                    image_data = generation_result.get('content')
                    image_stream = io.BytesIO(image_data or b"")
                    image_stream.name = 'ai_generated_image.png'
                    
                    safe_prompt = escape_markdown(prompt)
                    caption = f"🎨 **AI Generated Image**\n\n**Prompt:** {safe_prompt}\n\n✨ *Created with advanced AI models*"
                    
                    await update.message.reply_photo(
                        photo=image_stream,
                        caption=caption,
                        parse_mode='Markdown'
                    )
                    
                    logger.info(f"Successfully generated actual image for user {user_id}")
                    
                elif generation_result.get('type') == 'enhanced_description':
                    # Enhanced description was generated (primary path for free tier)
                    formatted_response = await response_formatter.format_image_generation_response(
                        generation_result, prompt
                    )
                    
                    # Send the main formatted response
                    await update.message.reply_text(
                        formatted_response.main_text,
                        parse_mode='Markdown',
                        disable_web_page_preview=False  # Allow link previews for better UX
                    )
                    
                    # Send attachments if any (copy-ready prompts)
                    for attachment in formatted_response.attachments:
                        if attachment.content and len(attachment.content) < 4000:  # Telegram message limit
                            attachment_message = f"📋 **{attachment.description}**\n\n```\n{attachment.content}\n```\n\n💡 *Tap the text above to copy*"
                            await update.message.reply_text(
                                attachment_message,
                                parse_mode='Markdown'
                            )
                    
                    logger.info(f"Successfully provided enhanced image guidance for user {user_id}")
                
            else:
                await processing_msg.edit_text(
                    "❌ **Image Processing Failed**\n\nCouldn't process your image request. This might be due to:\n\n• Temporary service issues\n• Invalid prompt format\n• Network connectivity\n\nPlease try again with a different prompt.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            secure_logger.error(f"Error in enhanced image generation for user {user_id}: {e}")
            
            # Provide fallback response with manual suggestions
            fallback_message = f"""🎨 **Image Creation Guidance**

**Your Request:** {escape_markdown(prompt)}

🚀 **Try these free AI generators:**
• [Bing Image Creator](https://www.bing.com/images/create) - Free DALL-E 3
• [Leonardo.ai](https://leonardo.ai) - Professional AI art
• [Playground AI](https://playground.com) - Easy to use

💡 **Optimized Prompt:**
```
Create a high-quality image of: {prompt}. Style: photorealistic, highly detailed, professional quality.
```

📋 *Tap the prompt above to copy and paste into any AI generator*"""
            
            await processing_msg.edit_text(
                fallback_message,
                parse_mode='Markdown',
                disable_web_page_preview=False
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
                try:
                    success, sentiment_data = await asyncio.wait_for(
                        model_caller.analyze_sentiment(analysis_text, api_key, use_emotions),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Sentiment analysis timeout (30s) for user {user_id}")
                    success = False
                    sentiment_data = None
            
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
            secure_logger.error(f"Error in sentiment analysis for user {user_id}: {e}")
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
            try:
                success, result = await asyncio.wait_for(
                    model_caller.generate_text(enhanced_prompt, api_key, special_params={'temperature': 0.3, 'max_new_tokens': 1500}),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Data analysis timeout (30s) for user {update.effective_user.id}")
                success = False
                result = "Analysis timed out after 30 seconds"
            
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
            try:
                success, result = await asyncio.wait_for(
                    model_caller.generate_text(enhanced_prompt, api_key, special_params={'temperature': 0.2}),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Document processing timeout (30s) for user {update.effective_user.id}")
                success = False
                result = "Processing timed out after 30 seconds"
            
            if success:
                await update.message.reply_text(f"📋 **Document Processing Results**\n\n{result}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ **Processing Failed** - Please try again with clearer context.", parse_mode='Markdown')


    @staticmethod
    async def _handle_conversation(update, context, prompt: str, api_key: str, chat_history: list, routing_info: dict) -> None:
        """Handle natural conversation with enhanced context awareness"""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        async with ModelCaller(provider="auto") as model_caller:
            try:
                success, result = await asyncio.wait_for(
                    model_caller.generate_text(prompt, api_key, chat_history, special_params={'temperature': 0.8}),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Conversation timeout (30s) for user {update.effective_user.id}")
                success = False
                result = "Response timed out after 30 seconds"
            
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
    async def _process_photo_handler(update, context, file_data: bytes, filename: str, processing_msg) -> None:
        """Process photo/image with AI analysis"""
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
            
            # SECURITY FIX C5: Validate BEFORE download
            # Pre-download validation: Check file type and basic properties
            file_obj = await photo.get_file()
            
            # Validate file extension before download
            file_path = file_obj.file_path or ""
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            file_extension = os.path.splitext(file_path.lower())[1] if file_path else '.jpg'
            
            if file_extension not in allowed_extensions:
                await update.message.reply_text(
                    f"🚫 **Invalid File Type**\n\n"
                    f"File type '{file_extension}' is not allowed.\n"
                    f"Supported formats: JPG, PNG, GIF, WEBP",
                    parse_mode='Markdown'
                )
                return
            
            logger.info(f"🔒 Pre-download validation passed for user {user_id}: {file_extension}, {file_size} bytes")
            
            # Download and process image
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            
            processing_msg = await update.message.reply_text(
                "🔍 **Analyzing Your Image...**\n\n"
                "Processing image content and extracting insights. This may take a moment...",
                parse_mode='Markdown'
            )
            
            try:
                # Download image (after pre-validation)
                image_data = await file_obj.download_as_bytearray()
                
                # Post-download security validation (content-based)
                is_valid, error_msg = FileProcessor.validate_file_security(bytes(image_data), f"image_{photo.file_id}{file_extension}", 'image')
                
                if not is_valid:
                    await processing_msg.edit_text(
                        f"🚫 **Image Validation Failed**\n\n{error_msg}\n\n"
                        "Please upload a valid image file.",
                        parse_mode='Markdown'
                    )
                    return
                
                # Process image
                await MessageHandlers._handle_image_processing(update, context, image_data, f"image_{photo.file_id}{file_extension}", processing_msg)
                
            except Exception as download_error:
                logger.error(f"Image download error for user {user_id}: {download_error}")
                await processing_msg.edit_text(
                    "❌ **Download Failed**\n\n"
                    "Failed to download your image. Please try uploading again.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            secure_logger.error(f"Error in photo handler for user {user_id}: {e}")
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
                try:
                    success, analysis_result = await asyncio.wait_for(
                        model_caller.analyze_pdf(
                            pdf_result['full_text'],
                            pdf_result['metadata'],
                            api_key,
                            analysis_type="comprehensive"
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ PDF analysis timeout (30s) for user {user_id}")
                    success = False
                    analysis_result = {'error': 'PDF analysis timed out after 30 seconds. Please try a smaller document.'}
            
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
            secure_logger.error(f"PDF processing error for user {user_id}: {e}")
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
                try:
                    success, analysis_result = await asyncio.wait_for(
                        model_caller.analyze_zip_contents(
                            file_contents,
                            api_key,
                            analysis_depth=analysis_depth
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ ZIP analysis timeout (30s) for user {user_id}")
                    success = False
                    analysis_result = {'error': 'ZIP analysis timed out after 30 seconds. Please try a smaller archive.'}
            
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
            secure_logger.error(f"ZIP processing error for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **ZIP Processing Error**\n\n"
                "An error occurred while processing your ZIP file. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_image_processing(update, context, image_data: bytes, filename: str, processing_msg) -> None:
        """Process image with comprehensive AI analysis - CRITICAL FIX: Renamed from _process_image_analysis"""
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
                try:
                    success, ai_analysis = await asyncio.wait_for(
                        model_caller.analyze_image_content(
                            image_data,
                            "comprehensive",
                            api_key
                        ),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Image analysis timeout (30s) for user {user_id}")
                    success = False
                    ai_analysis = None
            
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
            secure_logger.error(f"Image processing error for user {user_id}: {e}")
            await processing_msg.edit_text(
                "🚫 **Image Processing Error**\n\n"
                "An error occurred while processing your image. Please try again.",
                parse_mode='Markdown'
            )
    
    @staticmethod
    async def _handle_api_key_input(update, context, message_text: str) -> None:
        """
        Handle API key input with verification via actual test API calls
        
        Args:
            update: Telegram update object
            context: Bot context
            message_text: The text message from user (potential API key)
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"🔑 API key input received from user_id:{user_id} (@{username})")
        
        try:
            # Clear the waiting state first
            if context.user_data is not None:
                context.user_data['waiting_for_api_key'] = False
                
            # Check which step we're in for proper progress display
            current_step = context.user_data.get('setup_step', 2) if context.user_data else 2
            
            # Validate API key format (Hugging Face tokens start with 'hf_')
            if not message_text or len(message_text.strip()) < 10:
                await update.message.reply_text(
                    "❌ **Invalid API Key**\n\n"
                    "The API key seems too short. Hugging Face tokens usually start with `hf_` and are much longer.\n\n"
                    "📍 **Please send me a valid Hugging Face token to continue.**\n\n"
                    "💡 *Need help? Get your token at: https://huggingface.co/settings/tokens*",
                    parse_mode='Markdown'
                )
                logger.warning(f"Invalid API key format from user_id:{user_id} - too short")
                # Re-enable waiting state for another attempt
                if context.user_data is not None:
                    context.user_data['waiting_for_api_key'] = True
                return
            
            api_key = message_text.strip()
            
            # Update progress to step 2 - receiving token
            if context.user_data is not None:
                context.user_data['setup_step'] = 2
            
            # Show verification progress
            verification_msg = await update.message.reply_text(
                "🔄 **Verifying Your API Key...**\n\n"
                "🧪 Testing your token with Hugging Face to ensure it works correctly.\n"
                "⚡ This may take a few seconds...\n\n"
                "⏳ *Please wait...*",
                parse_mode='Markdown'
            )
            
            logger.info(f"🧪 Starting API key verification for user_id:{user_id}")
            
            # Perform actual API verification
            is_valid, error_message = await MessageHandlers._verify_api_key(api_key, user_id)
            
            # Delete the verification message
            await verification_msg.delete()
            
            if is_valid:
                # Save the verified API key to database
                success = await db.save_user_api_key(user_id, api_key)
                
                if success:
                    # Clear setup step tracking - setup complete!
                    if context.user_data is not None:
                        context.user_data.pop('setup_step', None)
                    
                    await update.message.reply_text(
                        "🎉 **Setup Complete! Congratulations!** 🎊\n\n"
                        "✅ **Verification Successful!**\n\n"
                        "🔗 **Successfully connected to Hugging Face AI**\n"
                        "🔒 **Your token is securely stored and encrypted**\n\n"
                        "🚀 **AI Features Now Active:**\n"
                        "💬 Advanced AI conversations\n"
                        "🎨 Image descriptions & analysis\n"
                        "💻 Code generation & programming help\n"
                        "📊 Smart text analysis\n"
                        "📝 Creative writing assistance\n"
                        "🔍 Intelligent Q&A\n"
                        "🧮 Mathematical reasoning\n\n"
                        "🌟 **You're all set!** Just send me any message to start using your AI assistant! 🚀\n\n"
                        "💡 *Try asking me something or upload an image!*",
                        parse_mode='Markdown'
                    )
                    secure_logger.info(f"✅ API key verified and saved successfully for user_id:{user_id}")
                else:
                    await update.message.reply_text(
                        "✅ **API Key Verified Successfully!**\n\n"
                        "⚠️ However, there was an issue storing your key securely.\n\n"
                        "🔗 Your token works correctly with Hugging Face, but please try the setup again.\n\n"
                        "📍 **Please send your token again to complete setup.**",
                        parse_mode='Markdown'
                    )
                    logger.error(f"API key verified but failed to save for user_id:{user_id}")
                    # Re-enable waiting state for another attempt, reset to step 1
                    if context.user_data is not None:
                        context.user_data['waiting_for_api_key'] = True
                        context.user_data['setup_step'] = 1
            else:
                # API key verification failed - UX2 fix: user-friendly error messages
                await update.message.reply_text(
                    "❌ **I couldn't connect to Hugging Face**\n\n"
                    "Your token seems invalid. Please double-check:\n\n"
                    "**Common issues:**\n"
                    "🔹 Make sure you copied the **entire token** (starts with `hf_`)\n"
                    "🔹 Verify your token has at least '**Read**' permissions\n"
                    "🔹 Check that your token is **not expired**\n"
                    "🔹 Ensure you're **connected to the internet**\n\n"
                    "📍 **Try again:** Send me your valid Hugging Face token 👇\n\n"
                    "🔗 Need a new token? Visit: https://huggingface.co/settings/tokens",
                    parse_mode='Markdown'
                )
                logger.warning(f"API key verification failed for user_id:{user_id}: {error_message}")
                # Re-enable waiting state for another attempt
                if context.user_data is not None:
                    context.user_data['waiting_for_api_key'] = True
                
        except Exception as e:
            secure_logger.error(f"Error handling API key input for user_id:{user_id}: {e}")
            await update.message.reply_text(
                "🚫 **API Key Processing Error**\n\n"
                "An error occurred while processing your API key. Please try again.",
                parse_mode='Markdown'
            )
            
            # Re-enable waiting state for another attempt, reset to step 1
            if context.user_data is not None:
                context.user_data['waiting_for_api_key'] = True
                context.user_data['setup_step'] = 1
    
    @staticmethod
    async def _verify_api_key(api_key: str, user_id: int) -> Tuple[bool, str]:
        """
        Verify API key by making actual test API calls to Hugging Face
        
        Args:
            api_key (str): The API key to verify
            user_id (int): User ID for logging purposes
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Import ModelCaller for verification
            from bot.core.model_caller import ModelCaller
            
            logger.info(f"🔍 Verifying API key for user_id:{user_id}")
            
            # Create a ModelCaller instance for testing
            model_caller_instance = ModelCaller()
            
            # Use a simple, reliable model for testing - use the default text model
            from bot.config import Config
            test_model = Config.DEFAULT_TEXT_MODEL  # This should be Qwen/Qwen2.5-0.5B-Instruct or similar
            
            # Simple test prompt
            test_prompt = "Hello"
            
            # Test parameters (minimal for quick verification)
            test_params = {
                "max_new_tokens": 10,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
            
            # Create test payload for the API call
            payload = {
                "inputs": test_prompt,
                "parameters": test_params
            }
            
            logger.info(f"🧪 Making test API call to {test_model} for user_id:{user_id}")
            
            # Make the test API call with the provided API key with timeout
            try:
                success, result = await asyncio.wait_for(
                    model_caller_instance._make_api_call(
                        model_name=test_model,
                        payload=payload,
                        api_key=api_key,
                        retries=0  # No retries for verification to keep it fast
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error(f"⏱️ API key verification timeout (30s) for user {user_id}")
                return False, "API key verification timed out. The Hugging Face API might be slow. Please try again."
            
            if success:
                logger.info(f"✅ API key verification successful for user_id:{user_id}")
                return True, "API key is valid and working"
            else:
                # Parse the error message to provide specific feedback
                error_msg = str(result).lower()
                
                if "unauthorized" in error_msg or "invalid api key" in error_msg or "401" in error_msg:
                    logger.warning(f"❌ API key verification failed - unauthorized for user_id:{user_id}")
                    return False, "Invalid or unauthorized API key"
                elif "forbidden" in error_msg or "403" in error_msg:
                    logger.warning(f"❌ API key verification failed - forbidden for user_id:{user_id}")
                    return False, "API key doesn't have required permissions"
                elif "rate limit" in error_msg or "429" in error_msg:
                    logger.warning(f"❌ API key verification failed - rate limited for user_id:{user_id}")
                    return False, "Rate limit exceeded, please try again later"
                else:
                    logger.warning(f"❌ API key verification failed - other error for user_id:{user_id}: {result}")
                    return False, "API key test failed - please check your token"
                    
        except Exception as e:
            safe_error = redact_sensitive_data(str(e))
            logger.error(f"❌ API key verification error for user_id:{user_id}: {safe_error}")
            return False, "Verification error - please try again"
    
    @staticmethod
    async def _prompt_api_key_setup(update, context) -> None:
        """
        Prompt the user to set up their API key with instructions
        
        Args:
            update: Telegram update object  
            context: Bot context
        """
        user = update.effective_user
        user_id = user.id
        username = user.username or "No username"
        
        logger.info(f"🔧 Prompting API key setup for user_id:{user_id} (@{username})")
        
        try:
            # Set waiting state for API key
            if context.user_data is None:
                context.user_data = {}
            context.user_data['waiting_for_api_key'] = True
            
            setup_message = """
🤖 **AI Setup Required**

To unlock AI features, you need a free Hugging Face token:

**🔧 Quick Setup Steps:**
1. Visit: https://huggingface.co/settings/tokens
2. Click "**New token**" 
3. Choose "**Read**" access (free)
4. Copy the token (starts with `hf_`)
5. Send it to me in your next message

**✅ Once set up, you'll have access to:**
• Advanced AI conversations
• Image description generation  
• Code generation and help
• Text analysis and sentiment
• Creative writing assistance
• Intelligent Q&A

🔒 **Your token is stored securely and encrypted.**

**Ready?** Just send me your Hugging Face token in the next message!
            """
            
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [InlineKeyboardButton("🌐 Get HF Token", url="https://huggingface.co/settings/tokens")],
                [InlineKeyboardButton("❓ Need Help?", callback_data="setup_help")],
                [InlineKeyboardButton("⏭️ Skip for now", callback_data="skip_setup")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                setup_message.strip(),
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            secure_logger.info(f"✅ API key setup prompt sent to user_id:{user_id}")
            
        except Exception as e:
            secure_logger.error(f"Error prompting API key setup for user_id:{user_id}: {e}")
            
            # Fallback message without formatting
            await update.message.reply_text(
                "AI Setup Required: Please visit https://huggingface.co/settings/tokens to get your free token, then send it to me."
            )
            
            # Clear waiting state on error
            if context.user_data is not None:
                context.user_data['waiting_for_api_key'] = False


# Export message handlers
message_handlers = MessageHandlers()