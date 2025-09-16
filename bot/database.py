"""
MongoDB database integration for AI Assistant Telegram Bot
Basic database connectivity for future use
"""

import asyncio
import os
from urllib.parse import urlparse, parse_qs
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError
from bot.config import Config
import logging

logger = logging.getLogger(__name__)

class Database:
    """MongoDB database manager for user data"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
    
    def _check_tls_enabled(self, mongo_uri: str) -> bool:
        """
        Safely check if TLS is enabled in MongoDB URI without mutating the original URI
        
        Args:
            mongo_uri (str): The original MongoDB connection string
            
        Returns:
            bool: True if TLS is enabled, False otherwise
        """
        try:
            # mongodb+srv:// always uses TLS by design
            if mongo_uri.startswith('mongodb+srv://'):
                return True
            
            # Parse the URI to check query parameters safely
            parsed_uri = urlparse(mongo_uri)
            query_params = parse_qs(parsed_uri.query)
            
            # Check for TLS/SSL parameters (case-insensitive)
            for key, values in query_params.items():
                key_lower = key.lower()
                if key_lower in ['tls', 'ssl'] and values:
                    # Check if any value indicates TLS is enabled
                    return any(value.lower() in ['true', '1', 'yes'] for value in values)
            
            return False
            
        except Exception as e:
            logger.warning(f"Error parsing MongoDB URI for TLS detection: {e}")
            # Default to False for safety in case of parsing errors
            return False
    
    async def connect(self):
        """Establish connection to MongoDB with TLS security validation"""
        try:
            if not Config.MONGO_URI:
                raise ValueError("MONGO_URI is not configured")
            
            # Security validation: Ensure TLS is enabled for production
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            # Safe TLS detection without corrupting the original URI
            has_tls = self._check_tls_enabled(Config.MONGO_URI)
            
            if is_production and not has_tls:
                raise ValueError(
                    "CRITICAL SECURITY ERROR: Production MongoDB connections must use TLS. "
                    "Use mongodb+srv:// connection string or add tls=true parameter."
                )
            
            if not has_tls:
                logger.warning(
                    "Database connection is not using TLS encryption. "
                    "This is acceptable for development but NOT for production."
                )
                
            self.client = AsyncIOMotorClient(Config.MONGO_URI)
            self.db = self.client.ai_assistant_bot
            
            # Test connection with retry logic
            max_ping_retries = 3
            for attempt in range(max_ping_retries):
                try:
                    await self.client.admin.command('ping')
                    break
                except Exception as e:
                    if attempt == max_ping_retries - 1:
                        raise e
                    await asyncio.sleep(1)
                    logger.warning(f"Ping attempt {attempt + 1} failed, retrying...")
            
            self.connected = True
            logger.info("✅ Successfully connected to MongoDB with security validation and performance optimizations")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            # Clean up partial state
            self.client = None
            self.db = None
            raise
    
    async def disconnect(self):
        """Close MongoDB connection gracefully"""
        if self.client:
            try:
                # Close connection gracefully
                self.client.close()
                self.connected = False
                self.client = None
                self.db = None
                logger.info("✅ Gracefully disconnected from MongoDB")
            except Exception as e:
                logger.error(f"Error during MongoDB disconnection: {e}")
                # Force cleanup
                self.connected = False
                self.client = None
                self.db = None

# Global database instance
db = Database()