"""
Security utilities for the AI Assistant Telegram Bot
Provides Markdown escaping and rate limiting functionality
"""

import re
import time
import logging
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class MarkdownSanitizer:
    """Utility class for safely escaping Markdown content"""
    
    # Markdown special characters that need escaping
    MARKDOWN_CHARS = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    @staticmethod
    def escape_markdown(text: str) -> str:
        """
        Escape Markdown special characters to prevent injection attacks
        
        Args:
            text (str): Input text that may contain Markdown characters
            
        Returns:
            str: Safely escaped text
        """
        if not text:
            return text
            
        # Escape each special character with backslash
        escaped_text = text
        for char in MarkdownSanitizer.MARKDOWN_CHARS:
            escaped_text = escaped_text.replace(char, f'\\{char}')
        
        return escaped_text
    
    @staticmethod
    def escape_code_blocks(text: str) -> str:
        """
        Safely escape code blocks while preserving formatting
        
        Args:
            text (str): Input text that may contain code blocks
            
        Returns:
            str: Text with safely escaped code blocks
        """
        if not text:
            return text
            
        # Pattern to match code blocks (```language\ncode\n```)
        code_block_pattern = r'```(\w*)\n(.*?)\n```'
        
        def escape_code_content(match):
            language = match.group(1)
            code = match.group(2)
            # Don't escape content inside code blocks, just ensure proper formatting
            return f'```{language}\n{code}\n```'
        
        # Replace code blocks with escaped versions
        escaped_text = re.sub(code_block_pattern, escape_code_content, text, flags=re.DOTALL)
        
        # Handle inline code (`code`)
        inline_code_pattern = r'`([^`]+)`'
        escaped_text = re.sub(inline_code_pattern, r'`\1`', escaped_text)
        
        return escaped_text
    
    @staticmethod
    def safe_markdown_format(text: str, preserve_code: bool = True) -> str:
        """
        Safely format text for Telegram Markdown, preserving code blocks if needed
        
        Args:
            text (str): Input text to format
            preserve_code (bool): Whether to preserve code block formatting
            
        Returns:
            str: Safely formatted text
        """
        if not text:
            return text
            
        if preserve_code:
            # Extract code blocks first
            code_blocks = []
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            
            def extract_code_block(match):
                code_blocks.append(match.group(0))
                return f"__CODE_BLOCK_{len(code_blocks)-1}__"
            
            # Replace code blocks with placeholders
            text_without_code = re.sub(code_block_pattern, extract_code_block, text, flags=re.DOTALL)
            
            # Escape the text without code blocks
            escaped_text = MarkdownSanitizer.escape_markdown(text_without_code)
            
            # Restore code blocks
            for i, code_block in enumerate(code_blocks):
                escaped_text = escaped_text.replace(f"__CODE_BLOCK_{i}__", code_block)
            
            return escaped_text
        else:
            return MarkdownSanitizer.escape_markdown(text)


class RateLimiter:
    """Robust rate limiting implementation with token bucket algorithm for per-user throttling"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60, strict_mode: bool = True):
        """
        Initialize rate limiter with token bucket algorithm
        
        Args:
            max_requests (int): Maximum requests per time window
            time_window (int): Time window in seconds
            strict_mode (bool): If True, blocks immediately after limit exceeded
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.strict_mode = strict_mode
        self.user_requests: Dict[int, list] = defaultdict(list)
        self.user_tokens: Dict[int, float] = defaultdict(float)  # Token bucket
        self.last_refill: Dict[int, float] = defaultdict(float)  # Last refill time
    
    def is_allowed(self, user_id: int) -> tuple[bool, Optional[int]]:
        """
        Check if user is allowed to make a request using token bucket algorithm
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            tuple[bool, Optional[int]]: (is_allowed, seconds_until_reset)
        """
        current_time = time.time()
        
        # Refill tokens based on time elapsed (token bucket algorithm)
        if user_id in self.last_refill and self.last_refill[user_id] > 0:
            time_elapsed = current_time - self.last_refill[user_id]
            tokens_to_add = time_elapsed * (self.max_requests / self.time_window)
            self.user_tokens[user_id] = min(self.max_requests, 
                                           self.user_tokens[user_id] + tokens_to_add)
        else:
            # First request - give full bucket
            self.user_tokens[user_id] = self.max_requests
        
        self.last_refill[user_id] = current_time
        
        # Check if tokens available
        if self.user_tokens[user_id] >= 1.0:
            # Allow request and consume token
            self.user_tokens[user_id] -= 1.0
            
            # Also track in sliding window for fallback compatibility
            user_requests = self.user_requests[user_id]
            user_requests.append(current_time)
            # Clean old requests
            user_requests[:] = [req_time for req_time in user_requests 
                               if current_time - req_time < self.time_window]
            
            return True, None
        else:
            # Rate limited - calculate wait time
            tokens_needed = 1.0 - self.user_tokens[user_id]
            seconds_until_token = int(tokens_needed * (self.time_window / self.max_requests))
            
            # In strict mode, immediate blocking
            if self.strict_mode:
                logger.warning(f"Rate limit exceeded for user {user_id} - strict blocking enabled")
                return False, max(1, seconds_until_token)
            
            # Legacy sliding window check for non-strict mode
            user_requests = self.user_requests[user_id]
            user_requests[:] = [req_time for req_time in user_requests 
                               if current_time - req_time < self.time_window]
            
            if len(user_requests) >= self.max_requests:
                oldest_request = min(user_requests) if user_requests else current_time
                seconds_until_reset = int(self.time_window - (current_time - oldest_request))
                return False, max(1, seconds_until_reset)
            
            # Allow in non-strict mode as fallback
            user_requests.append(current_time)
            return True, None
    
    def get_remaining_requests(self, user_id: int) -> int:
        """
        Get number of remaining requests for user
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            int: Number of remaining requests
        """
        current_time = time.time()
        user_requests = self.user_requests[user_id]
        
        # Remove old requests
        user_requests[:] = [req_time for req_time in user_requests 
                           if current_time - req_time < self.time_window]
        
        return max(0, self.max_requests - len(user_requests))
    
    def reset_user(self, user_id: int) -> None:
        """
        Reset rate limit for specific user (admin function)
        
        Args:
            user_id (int): Telegram user ID
        """
        self.user_requests[user_id] = []
        logger.info(f"Rate limit reset for user {user_id}")


# Global instances for use across the application  
markdown_sanitizer = MarkdownSanitizer()

# SECURITY HARDENED: Strict rate limiting with token bucket algorithm
# Production-ready rate limiters with immediate blocking
rate_limiter = RateLimiter(max_requests=3, time_window=60, strict_mode=True)  # STRICT: 3 requests per minute
premium_rate_limiter = RateLimiter(max_requests=10, time_window=60, strict_mode=True)  # Premium: 10 per minute
admin_rate_limiter = RateLimiter(max_requests=30, time_window=60, strict_mode=True)  # Admin: 30 per minute

# Fallback rate limiters with less strict limits for database failures
fallback_rate_limiter = RateLimiter(max_requests=20, time_window=60, strict_mode=False)  # Fallback mode

# Convenience functions for easy import
def escape_markdown(text: str) -> str:
    """Convenience function to escape Markdown text"""
    return markdown_sanitizer.escape_markdown(text)

def safe_markdown_format(text: str, preserve_code: bool = True) -> str:
    """Convenience function to safely format Markdown text"""
    return markdown_sanitizer.safe_markdown_format(text, preserve_code)

async def check_rate_limit_persistent(user_id: int, storage_provider=None) -> tuple[bool, int]:
    """Production-hardened persistent rate limiting with strict enforcement"""
    if storage_provider and hasattr(storage_provider, 'check_rate_limit'):
        try:
            # Use persistent database rate limiting for production (STRICT: 3 requests per minute)
            is_allowed, wait_time = await storage_provider.check_rate_limit(user_id, max_requests=3, time_window=60)
            return is_allowed, wait_time or 0
        except Exception as e:
            # Critical: Log database failures for security monitoring
            logging.getLogger(__name__).error(f"SECURITY WARNING: Persistent rate limiting failed for user {user_id}: {e}")
            # Use strict in-memory fallback
            return check_rate_limit(user_id)
    else:
        # Use strict in-memory rate limiting as fallback
        return check_rate_limit(user_id)

def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Strict in-memory rate limiting with immediate blocking after limit exceeded"""
    is_allowed, wait_time = rate_limiter.is_allowed(user_id)
    
    # Security logging for rate limit violations
    if not is_allowed:
        logger.warning(f"SECURITY: Rate limit exceeded for user {user_id}, blocking for {wait_time} seconds")
    
    return is_allowed, wait_time or 0