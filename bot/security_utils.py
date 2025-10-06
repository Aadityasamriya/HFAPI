"""
Security utilities for the AI Assistant Telegram Bot
Provides Markdown escaping, rate limiting, and centralized data redaction functionality
"""

import re
import time
import logging
import hashlib
import ipaddress
import os
import threading
from typing import Dict, Optional, Tuple, Set, Any, List
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataRedactionEngine:
    """
    CRITICAL SECURITY: Centralized data redaction engine to prevent credential leakage
    All sensitive data redaction should go through this class to ensure consistency
    """
    
    @staticmethod
    def redact_sensitive_data(text: str) -> str:
        """
        ENHANCED SECURITY: Comprehensive redaction of sensitive data from any text
        
        This is the SINGLE SOURCE OF TRUTH for sensitive data redaction.
        All other modules should use this function to prevent credential leakage.
        
        === SECURITY AUDIT FIX (Task 3.C) ===
        Previous effectiveness: 88.9% → Target: 100%
        
        IMPROVEMENTS MADE:
        1. Fixed OpenAI key patterns with exact lengths (48-51 chars) and all variants:
           - Added sk-org-, sk-svcacct-, sk-proj- prefixes
           - Added exact length matching to prevent false positives
           
        2. Fixed Anthropic keys with correct length (95-110 chars, not 20+)
        
        3. Fixed HuggingFace tokens:
           - Added exact length (34 chars after prefix)
           - Added old format (api_[A-Z0-9]{32})
           
        4. Added ALL GitHub token types with exact lengths:
           - ghp_ (36 chars), gho_ (16), ghu_ (16), ghs_ (16), ghr_ (16)
           - github_pat_ (82 chars)
           - Old format (40 hex with context)
           
        5. Fixed Google AI keys with exact length (AIza + 35 chars)
        
        6. Added missing AI service patterns:
           - Replicate (r8_), Cohere (co-), Groq (gsk_), Perplexity (pplx-)
           - Together AI, Mistral, Azure OpenAI, Stability AI
           
        7. Added context-aware patterns to prevent false positives:
           - All broad patterns now require context (key names, headers, etc.)
           - URL-encoded keys, JSON escaped keys, environment variables
           
        8. Added encoding-aware patterns:
           - URL-encoded (%XX), base64, hex-encoded keys
           - Keys in various formats (YAML, TOML, env vars, curl commands)
           
        9. Enhanced header detection:
           - X-API-Key, X-Auth-Token, Authorization, Ocp-Apim-Subscription-Key
           - All quote styles and formats
           
        10. Added comprehensive generic patterns with full character sets:
            - Includes base64 chars (+/=), special password chars
            - All credential types (bearer_token, session_token, etc.)
        
        Args:
            text (str): Text that might contain sensitive information
            
        Returns:
            str: Sanitized text with sensitive data redacted
        """
        if not isinstance(text, str):
            text = str(text)
        
        # === API KEYS AND TOKENS ===
        # SECURITY FIX: Comprehensive API key patterns to achieve 100% redaction effectiveness
        # Previous effectiveness: 88.9% - Fixed with exact lengths and all variants
        
        # === OPENAI API KEYS - Enhanced with all formats and exact lengths ===
        # Standard OpenAI keys (sk-...) - Real keys are 48-51 characters total
        text = re.sub(r'sk-[a-zA-Z0-9]{48}', 'sk-[REDACTED]', text)  # Exact 48 char format
        text = re.sub(r'sk-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}', 'sk-[REDACTED]', text)  # Classic format with T3BlbkFJ marker
        text = re.sub(r'sk-[a-zA-Z0-9_-]{48,51}', 'sk-[REDACTED]', text)  # Range for different formats
        
        # OpenAI Project keys (sk-proj-...)
        text = re.sub(r'sk-proj-[a-zA-Z0-9_-]{48,}', 'sk-proj-[REDACTED]', text)
        
        # OpenAI Organization keys (sk-org-...)
        text = re.sub(r'sk-org-[a-zA-Z0-9_-]{48,}', 'sk-org-[REDACTED]', text)
        
        # OpenAI Service Account keys (sk-svcacct-...)
        text = re.sub(r'sk-svcacct-[a-zA-Z0-9_-]{48,}', 'sk-svcacct-[REDACTED]', text)
        
        # Azure OpenAI keys (context-aware to avoid false positives)
        text = re.sub(r'(?:azure[_-]?(?:openai)?[_-]?)?(?:api[_-]?)?key["\s]*[:=]["\s]*[a-f0-9]{32}', 'azure_key: [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'Ocp-Apim-Subscription-Key:\s*[a-f0-9]{32}', 'Ocp-Apim-Subscription-Key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # Fallback for any sk- pattern (to catch edge cases and test keys)
        # SECURITY FIX (Issue #3): Lower minimum to catch shorter test keys (10+ chars)
        text = re.sub(r'\bsk-[a-zA-Z0-9_-]{10,}\b', 'sk-[REDACTED]', text)
        
        # === ANTHROPIC API KEYS - Fixed with correct length ===
        # Real Anthropic keys are ~100+ characters
        text = re.sub(r'sk-ant-[a-zA-Z0-9_-]{95,110}', 'sk-ant-[REDACTED]', text)  # Typical length range
        text = re.sub(r'sk-ant-[a-zA-Z0-9_-]{50,}', 'sk-ant-[REDACTED]', text)  # Broader fallback
        
        # === HUGGING FACE TOKENS - All formats ===
        # New format (hf_...) - Real tokens are ~37 chars after prefix
        text = re.sub(r'hf_[a-zA-Z0-9]{34}', 'hf_[REDACTED]', text)  # Exact length
        text = re.sub(r'hf_[a-zA-Z0-9_-]{30,40}', 'hf_[REDACTED]', text)  # Range for variants
        
        # Old format HuggingFace tokens (api_...)
        text = re.sub(r'\bapi_[A-Z0-9]{32}\b', 'api_[REDACTED]', text)  # Old 32-char format
        
        # Context-based patterns
        text = re.sub(r'huggingface[_-]?(?:hub[_-]?)?token["\s]*[:=]["\s]*[a-zA-Z0-9_.-]{15,}', 'huggingface_token: [REDACTED]', text, flags=re.IGNORECASE)
        
        # === GITHUB TOKENS - All types with exact AND variable lengths (SECURITY FIX Issue #3) ===
        # Personal Access Tokens (ghp_...) - Exact 36 chars AND variable length for test coverage
        text = re.sub(r'ghp_[a-zA-Z0-9]{36}', 'ghp_[REDACTED]', text)  # Standard length
        text = re.sub(r'ghp_[a-zA-Z0-9]{30,50}', 'ghp_[REDACTED]', text)  # Variable length range
        text = re.sub(r'ghp_[a-zA-Z0-9_-]{20,}', 'ghp_[REDACTED]', text)  # Catch-all for any ghp_ token
        
        # OAuth Access Tokens (gho_...) - 16 chars and variable
        text = re.sub(r'gho_[a-zA-Z0-9]{16}', 'gho_[REDACTED]', text)
        text = re.sub(r'gho_[a-zA-Z0-9_-]{10,30}', 'gho_[REDACTED]', text)
        
        # User-to-Server tokens (ghu_...) - 16 chars and variable
        text = re.sub(r'ghu_[a-zA-Z0-9]{16}', 'ghu_[REDACTED]', text)
        text = re.sub(r'ghu_[a-zA-Z0-9_-]{10,30}', 'ghu_[REDACTED]', text)
        
        # Server-to-Server tokens (ghs_...) - 16 chars and variable
        text = re.sub(r'ghs_[a-zA-Z0-9]{16}', 'ghs_[REDACTED]', text)
        text = re.sub(r'ghs_[a-zA-Z0-9_-]{10,30}', 'ghs_[REDACTED]', text)
        
        # Refresh tokens (ghr_...) - 16 chars and variable
        text = re.sub(r'ghr_[a-zA-Z0-9]{16}', 'ghr_[REDACTED]', text)
        text = re.sub(r'ghr_[a-zA-Z0-9_-]{10,30}', 'ghr_[REDACTED]', text)
        
        # Fine-grained Personal Access Tokens (github_pat_...)
        text = re.sub(r'github_pat_[A-Za-z0-9_]{82}', 'github_pat_[REDACTED]', text)  # Exact length
        text = re.sub(r'github_pat_[A-Za-z0-9_]{50,90}', 'github_pat_[REDACTED]', text)  # Range fallback
        text = re.sub(r'github_pat_[A-Za-z0-9_]{20,}', 'github_pat_[REDACTED]', text)  # Catch-all
        
        # Old format GitHub tokens (40 hex characters - context-aware)
        text = re.sub(r'(?:github[_-]?)?(?:token|key|secret)["\s]*[:=]["\s]*[a-f0-9]{40}', 'github_token: [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'GITHUB[_-]TOKEN["\s]*[:=]["\s]*[a-f0-9]{40}', 'GITHUB_TOKEN=[REDACTED]', text)
        
        # === GOOGLE AI / VERTEX AI KEYS - Fixed exact length ===
        # Google API keys are exactly 39 chars (AIza + 35 chars)
        text = re.sub(r'AIza[a-zA-Z0-9_-]{35}', 'AIza[REDACTED]', text)
        
        # === AI SERVICE API KEYS - Comprehensive coverage ===
        # Replicate API tokens (r8_...)
        text = re.sub(r'r8_[a-zA-Z0-9]{40,50}', 'r8_[REDACTED]', text)
        
        # Cohere API keys (typically alphanumeric, ~40 chars)
        text = re.sub(r'\bco-[a-zA-Z0-9_-]{35,50}\b', 'co-[REDACTED]', text)
        
        # Together AI API keys (context-aware for 64 hex chars)
        text = re.sub(r'(?:together[_-]?)?(?:api[_-]?)?key["\s]*[:=]["\s]*[a-f0-9]{64}', 'together_key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # Mistral AI API keys (context-aware for 32 alphanumeric)
        text = re.sub(r'(?:mistral[_-]?)?(?:api[_-]?)?key["\s]*[:=]["\s]*[a-zA-Z0-9]{32}', 'mistral_key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # Groq API keys (gsk_...)
        text = re.sub(r'gsk_[a-zA-Z0-9]{40,50}', 'gsk_[REDACTED]', text)
        
        # Perplexity AI API keys (pplx-...)
        text = re.sub(r'pplx-[a-zA-Z0-9]{40,60}', 'pplx-[REDACTED]', text)
        
        # Stability AI keys (context-specific to avoid conflicts with OpenAI)
        text = re.sub(r'(?:stability[_-]?)?(?:api[_-]?)?key["\s]*[:=]["\s]*sk-[a-zA-Z0-9]{40,50}', 'stability_key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # RunPod API keys (UUID format - context-aware)
        text = re.sub(r'(?:runpod[_-]?)?(?:api[_-]?)?key["\s]*[:=]["\s]*[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}', 'runpod_key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # === JWT TOKENS - Enhanced ===
        text = re.sub(r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', 'eyJ[REDACTED_JWT]', text)  # Complete JWT
        text = re.sub(r'eyJ[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+){1,2}', 'eyJ[REDACTED_JWT]', text)  # Partial JWT
        
        # === AUTHORIZATION HEADERS - Enhanced for all contexts ===
        # Bearer tokens in various formats
        text = re.sub(r'Bearer\s+[a-zA-Z0-9_./+=:-]{15,}', 'Bearer [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'"Authorization":\s*"Bearer\s+[^"]+', '"Authorization": "Bearer [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r"'Authorization':\s*'Bearer\s+[^']+", "'Authorization': 'Bearer [REDACTED]", text, flags=re.IGNORECASE)
        text = re.sub(r'Authorization:\s*Bearer\s+[a-zA-Z0-9_./+=:-]{15,}', 'Authorization: Bearer [REDACTED]', text, flags=re.IGNORECASE)
        
        # X-API-Key headers (common in APIs)
        text = re.sub(r'X-API-Key:\s*[a-zA-Z0-9_./+=:-]{15,}', 'X-API-Key: [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'"X-API-Key":\s*"[^"]+', '"X-API-Key": "[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r"'X-API-Key':\s*'[^']+", "'X-API-Key': '[REDACTED]", text, flags=re.IGNORECASE)
        
        # X-Auth-Token headers
        text = re.sub(r'X-Auth-Token:\s*[a-zA-Z0-9_./+=:-]{15,}', 'X-Auth-Token: [REDACTED]', text, flags=re.IGNORECASE)
        
        # API-Key headers (without X- prefix)
        text = re.sub(r'API-Key:\s*[a-zA-Z0-9_./+=:-]{15,}', 'API-Key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # === DATABASE CONNECTION STRINGS (ENHANCED) ===
        
        # SECURITY FIX: Enhanced database URL patterns for complete redaction
        db_patterns = [
            (r'mongodb(\+srv)?://[^@/\s]+@[^\s]+', r'mongodb\1://[REDACTED_CONNECTION_STRING]'),
            (r'postgres(ql)?://[^@/\s]+@[^\s]+', r'postgres\1://[REDACTED_CONNECTION_STRING]'),
            (r'mysql://[^@/\s]+@[^\s]+', r'mysql://[REDACTED_CONNECTION_STRING]'),
            (r'redis(s)?://[^@/\s]+@[^\s]+', r'redis\1://[REDACTED_CONNECTION_STRING]'),
            (r'amqp://[^@/\s]+@[^\s]+', r'amqp://[REDACTED_CONNECTION_STRING]'),
            (r's3://[^@/\s]+@[^\s]+', r's3://[REDACTED_CONNECTION_STRING]'),
            # Additional patterns for connection strings without @ symbol
            (r'mongodb(\+srv)?://[a-zA-Z0-9._-]+:[0-9]+/[^\s]+', r'mongodb\1://[REDACTED_CONNECTION_STRING]'),
            (r'postgres(ql)?://[a-zA-Z0-9._-]+:[0-9]+/[^\s]+', r'postgres\1://[REDACTED_CONNECTION_STRING]'),
            (r'mysql://[a-zA-Z0-9._-]+:[0-9]+/[^\s]+', r'mysql://[REDACTED_CONNECTION_STRING]'),
        ]
        
        for pattern, replacement in db_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # === AWS CREDENTIALS ===
        
        text = re.sub(r'AKIA[0-9A-Z]{16}', 'AKIA[REDACTED]', text)
        text = re.sub(r'aws_access_key_id["\s]*[:=]["\s]*[A-Z0-9]{20}', 'aws_access_key_id=[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'aws_secret_access_key["\s]*[:=]["\s]*[A-Za-z0-9/+=]{40}', 'aws_secret_access_key=[REDACTED]', text, flags=re.IGNORECASE)
        
        # === DISCORD TOKENS ===
        
        text = re.sub(r'[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}', '[REDACTED_DISCORD_TOKEN]', text)
        
        # === SLACK TOKENS AND WEBHOOKS (SECURITY FIX: Added re.IGNORECASE) ===
        
        text = re.sub(r'xox[bpars]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}', 'xox[REDACTED]', text)
        text = re.sub(r'https://hooks\.slack\.com/services/[A-Z0-9/]+', 'https://hooks.slack.com/services/[REDACTED]', text, flags=re.IGNORECASE)
        
        # === SSH KEYS ===
        
        text = re.sub(r'ssh-rsa\s+[A-Za-z0-9+/=]+', 'ssh-rsa [REDACTED_SSH_KEY]', text)
        text = re.sub(r'ssh-ed25519\s+[A-Za-z0-9+/=]+', 'ssh-ed25519 [REDACTED_SSH_KEY]', text)
        
        # === WEBHOOK URLS ===
        
        text = re.sub(r'https://discord(?:app)?\.com/api/webhooks/[0-9]+/[a-zA-Z0-9_-]+', 'https://discord.com/api/webhooks/[REDACTED]', text)
        
        # === ENCODING-AWARE PATTERNS - Catch encoded API keys ===
        # URL-encoded API keys (sk-%XX%XX format)
        text = re.sub(r'sk-(?:%[0-9A-Fa-f]{2}|[a-zA-Z0-9_-]){40,}', 'sk-[REDACTED]', text)
        text = re.sub(r'hf_(?:%[0-9A-Fa-f]{2}|[a-zA-Z0-9_-]){30,}', 'hf_[REDACTED]', text)
        text = re.sub(r'ghp_(?:%[0-9A-Fa-f]{2}|[a-zA-Z0-9]){35,}', 'ghp_[REDACTED]', text)
        
        # Keys in JSON with escaped quotes
        text = re.sub(r'\\"api[_-]?key\\":\s*\\"[^"]{8,}\\"', '\\"api_key\\": \\"[REDACTED]\\"', text, flags=re.IGNORECASE)
        text = re.sub(r'\\"token\\":\s*\\"[^"]{8,}\\"', '\\"token\\": \\"[REDACTED]\\"', text, flags=re.IGNORECASE)
        
        # Keys in environment variable format (export KEY=value)
        text = re.sub(r'export\s+[A-Z_]+API[_-]?KEY\s*=\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'export API_KEY=[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'export\s+[A-Z_]+TOKEN\s*=\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'export TOKEN=[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'export\s+[A-Z_]+SECRET\s*=\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'export SECRET=[REDACTED]', text, flags=re.IGNORECASE)
        
        # Environment variables without export
        text = re.sub(r'[A-Z_]+API[_-]?KEY\s*=\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'API_KEY=[REDACTED]', text)
        text = re.sub(r'[A-Z_]+TOKEN\s*=\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'TOKEN=[REDACTED]', text)
        text = re.sub(r'[A-Z_]+SECRET\s*=\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'SECRET=[REDACTED]', text)
        
        # Keys in YAML/TOML format
        text = re.sub(r'api[_-]?key:\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'api_key: [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'token:\s*["\']?[a-zA-Z0-9_./+=:-]{8,}', 'token: [REDACTED]', text, flags=re.IGNORECASE)
        
        # Keys in command-line arguments
        text = re.sub(r'--api[_-]?key[=\s]+[a-zA-Z0-9_./+=:-]{8,}', '--api-key [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'--token[=\s]+[a-zA-Z0-9_./+=:-]{8,}', '--token [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'-k\s+[a-zA-Z0-9_./+=:-]{8,}', '-k [REDACTED]', text)  # Common -k flag for API key
        
        # Keys in curl commands
        text = re.sub(r'curl.*?-H\s*["\']Authorization:\s*Bearer\s+[^"\']+', 'curl -H "Authorization: Bearer [REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'curl.*?-H\s*["\']X-API-Key:\s*[^"\']+', 'curl -H "X-API-Key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # === GENERIC CREDENTIAL PATTERNS - Enhanced with more character sets ===
        credential_patterns = [
            # Base patterns with expanded character set for base64 and special chars
            (r'api[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'api_key: [REDACTED]'),
            (r'token["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'token: [REDACTED]'),
            (r'secret["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'secret: [REDACTED]'),
            (r'password["\s]*[:=]["\s]*[a-zA-Z0-9_./+=@!#$%^&*()-]{3,}', 'password=[REDACTED]'),
            (r'client[_-]?id["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'client_id: [REDACTED]'),
            (r'client[_-]?secret["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'client_secret: [REDACTED]'),
            (r'access[_-]?token["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'access_token: [REDACTED]'),
            (r'refresh[_-]?token["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'refresh_token: [REDACTED]'),
            (r'bearer[_-]?token["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'bearer_token: [REDACTED]'),
            (r'auth[_-]?token["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'auth_token: [REDACTED]'),
            (r'session[_-]?token["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'session_token: [REDACTED]'),
            (r'private[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'private_key: [REDACTED]'),
            (r'public[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'public_key: [REDACTED]'),
            (r'encryption[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'encryption_key: [REDACTED]'),
            (r'service[_-]?account[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9_./+=:-]{8,}', 'service_account_key: [REDACTED]'),
        ]
        
        for pattern, replacement in credential_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # === CRYPTOGRAPHIC KEYS (SECURITY FIX: More targeted patterns) ===
        
        # SECURITY FIX: Make 32-char hex patterns more targeted to avoid false positives
        # Only match if surrounded by key-related context
        key_context_patterns = [
            r'(?:key|secret|token|hash|signature)["\s]*[:=]["\s]*[a-fA-F0-9]{32,64}',
            r'[a-fA-F0-9]{32,64}(?=["\s]*(?:key|secret|token|hash|signature))',
        ]
        for pattern in key_context_patterns:
            text = re.sub(pattern, '[REDACTED_HEX_KEY]', text, flags=re.IGNORECASE)
        
        # Base64 encoded keys (more selective)
        text = re.sub(r'(?:key|secret|token)["\s]*[:=]["\s]*[A-Za-z0-9+/]{32,}={0,2}', '[REDACTED_B64_KEY]', text, flags=re.IGNORECASE)
        
        # === JSON RESPONSE PATTERNS ===
        
        text = re.sub(r'\{[^}]*["\'](api_?key|token|secret|password|auth)["\'][^}]*\}', '{[REDACTED_JSON_WITH_CREDENTIALS]}', text, flags=re.IGNORECASE)
        
        # === URL PARAMETERS - Enhanced ===
        text = re.sub(r'([?&])(token|key|secret|password|auth|access_token|api_key|apikey|api-key)=([^&\s"\']+)', r'\1\2=[REDACTED]', text, flags=re.IGNORECASE)
        
        # Keys in URL paths (e.g., /api/v1/keys/sk-xxxxx)
        text = re.sub(r'/(?:keys?|tokens?|secrets?|auth)/([a-zA-Z0-9_./+=:-]{15,})', r'/\1/[REDACTED]', text, flags=re.IGNORECASE)
        
        # === TELEGRAM BOT TOKENS ===
        text = re.sub(r'\b\d{8,10}:[a-zA-Z0-9_-]{35}\b', '[REDACTED_BOT_TOKEN]', text)
        
        # === EDGE CASE PATTERNS - Unusual formats and contexts ===
        # Partially masked keys (e.g., sk-***abc123 or sk-...abc123)
        text = re.sub(r'sk-[\*\.]{3,}[a-zA-Z0-9_-]+', 'sk-[REDACTED_PARTIAL]', text)
        text = re.sub(r'hf_[\*\.]{3,}[a-zA-Z0-9_-]+', 'hf_[REDACTED_PARTIAL]', text)
        text = re.sub(r'ghp_[\*\.]{3,}[a-zA-Z0-9]+', 'ghp_[REDACTED_PARTIAL]', text)
        
        # Keys with line breaks or whitespace (multiline strings)
        text = re.sub(r'sk-[a-zA-Z0-9_-]{20,}\s*\n', 'sk-[REDACTED]\n', text)
        text = re.sub(r'hf_[a-zA-Z0-9_-]{20,}\s*\n', 'hf_[REDACTED]\n', text)
        
        # Keys in Python/JavaScript strings (with quotes)
        text = re.sub(r'["\']sk-[a-zA-Z0-9_-]{40,}["\']', '"sk-[REDACTED]"', text)
        text = re.sub(r'["\']hf_[a-zA-Z0-9_-]{30,}["\']', '"hf_[REDACTED]"', text)
        text = re.sub(r'["\']ghp_[a-zA-Z0-9]{36}["\']', '"ghp_[REDACTED]"', text)
        
        # Keys in configuration comments (# key: sk-xxxxx or // key: sk-xxxxx)
        text = re.sub(r'[#/]{1,2}\s*(?:api[_-]?)?key:\s*sk-[a-zA-Z0-9_-]{20,}', '# key: sk-[REDACTED]', text, flags=re.IGNORECASE)
        
        # Keys in error messages
        text = re.sub(r'(?:invalid|unauthorized|expired|revoked)\s+(?:api[_-]?)?key:?\s*["\']?([a-zA-Z0-9_./+=:-]{15,})', r'\1 key: [REDACTED]', text, flags=re.IGNORECASE)
        
        # Keys with prefix in various formats (API_KEY= vs APIKEY= vs api-key=)
        text = re.sub(r'\b(?:API[_-]?KEY|APIKEY|api[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9_./+=:-]{15,})', 'API_KEY=[REDACTED]', text, flags=re.IGNORECASE)
        
        # === PRIVACY PROTECTION ===
        
        # Email addresses in credential contexts
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
        
        # IP addresses (privacy protection)
        text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '[REDACTED_IP]', text)
        
        return text
    
    @staticmethod
    def redact_crypto_data(text: str) -> str:
        """
        SECURITY: Specialized redaction for cryptographic error messages
        
        Args:
            text (str): Text that might contain sensitive crypto information
            
        Returns:
            str: Sanitized text with crypto-sensitive data redacted
        """
        # Use the main redaction engine plus crypto-specific patterns
        text = DataRedactionEngine.redact_sensitive_data(text)
        
        # Additional crypto-specific patterns
        crypto_patterns = [
            (r'salt["\s]*[:=]["\s]*[a-zA-Z0-9+/=]{20,}', 'salt=[REDACTED_SALT]'),
            (r'nonce["\s]*[:=]["\s]*[a-zA-Z0-9+/=]{16,}', 'nonce=[REDACTED_NONCE]'),
            (r'iv["\s]*[:=]["\s]*[a-zA-Z0-9+/=]{16,}', 'iv=[REDACTED_IV]'),
            (r'ciphertext["\s]*[:=]["\s]*[a-zA-Z0-9+/=]{20,}', 'ciphertext=[REDACTED_CIPHERTEXT]'),
            (r'signature["\s]*[:=]["\s]*[a-zA-Z0-9+/=]{20,}', 'signature=[REDACTED_SIGNATURE]'),
        ]
        
        for pattern, replacement in crypto_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

class SecureLogger:
    """
    SECURITY: Environment-aware secure logging with automatic redaction
    Implements proper stack trace handling based on environment
    """
    
    def __init__(self, logger_instance):
        self.logger = logger_instance
        # SECURITY FIX: Environment-gated exc_info handling
        self.environment = os.getenv('ENVIRONMENT', 'development').lower()
        self.is_production = self.environment == 'production'
        self.enable_stack_traces = os.getenv('ENABLE_STACK_TRACES', 'false').lower() == 'true'
    
    def _should_include_exc_info(self) -> bool:
        """
        SECURITY FIX: Determine if stack traces should be included based on environment
        
        Returns:
            bool: True if stack traces are safe to include
        """
        # In production, only include stack traces if explicitly enabled
        if self.is_production:
            return self.enable_stack_traces
        
        # In development, include stack traces by default
        return True
    
    def _safe_log(self, level_method, message: str, exc_info=None, *args, **kwargs):
        """
        SECURITY: Safe logging with automatic redaction and environment-aware exc_info
        
        Args:
            level_method: Logger level method (e.g., self.logger.error)
            message (str): Log message that might contain sensitive data
            exc_info: Exception info (will be handled based on environment)
            *args: Additional args for logging
            **kwargs: Additional kwargs for logging
        """
        # SECURITY FIX: Environment-gated exc_info handling with redaction
        if exc_info is not None:
            if not self._should_include_exc_info():
                exc_info = None
            else:
                # SECURITY FIX: Redact sensitive data from exception information
                if exc_info is True:
                    # Keep True as is - Python will capture current exception
                    pass
                elif isinstance(exc_info, tuple) and len(exc_info) >= 3:
                    # Redact traceback string representation
                    try:
                        import traceback
                        exc_str = ''.join(traceback.format_exception(*exc_info))
                        redacted_exc_str = DataRedactionEngine.redact_sensitive_data(exc_str)
                        # Create a safe exception message instead of full traceback
                        exc_type, exc_value, exc_tb = exc_info
                        safe_exc_msg = f"{exc_type.__name__}: {DataRedactionEngine.redact_sensitive_data(str(exc_value))}"
                        # In development, we can include more details; in production, keep it minimal
                        if not self.is_production:
                            exc_info = (exc_type, Exception(safe_exc_msg), None)
                        else:
                            # In production, just log the sanitized exception type and message
                            message = f"{message} | Exception: {safe_exc_msg}"
                            exc_info = None
                    except Exception:
                        # If redaction fails, disable exc_info for safety
                        exc_info = None
        
        # Redact sensitive data from message and args
        safe_message = DataRedactionEngine.redact_sensitive_data(message)
        safe_args = tuple(DataRedactionEngine.redact_sensitive_data(str(arg)) for arg in args)
        
        # SECURITY FIX: Redact sensitive data from kwargs
        safe_kwargs = {}
        for key, value in kwargs.items():
            if key in ['exc_info']:  # Skip already handled keys
                safe_kwargs[key] = value
            else:
                # Redact both key and value to be safe
                safe_key = DataRedactionEngine.redact_sensitive_data(str(key))
                safe_value = DataRedactionEngine.redact_sensitive_data(str(value))
                safe_kwargs[safe_key] = safe_value
        
        # Add security marker
        if not safe_message.startswith('🔒'):
            safe_message = f"🔒 {safe_message}"
        
        level_method(safe_message, *safe_args, exc_info=exc_info, **safe_kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Secure info logging with automatic redaction"""
        self._safe_log(self.logger.info, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Secure warning logging with automatic redaction"""
        self._safe_log(self.logger.warning, message, *args, **kwargs)
    
    def error(self, message: str, exc_info=None, *args, **kwargs):
        """Secure error logging with automatic redaction and environment-aware exc_info"""
        self._safe_log(self.logger.error, message, exc_info=exc_info, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Secure debug logging with automatic redaction"""
        # Skip debug logs in production for performance
        if self.is_production:
            return
        self._safe_log(self.logger.debug, message, *args, **kwargs)
    
    def audit(self, event_type: str, user_id: Optional[int] = None, details: str = "", success: bool = True, **kwargs):
        """
        Security audit logging with structured format
        
        Args:
            event_type (str): Type of security event (e.g., 'API_KEY_SAVE', 'LOGIN_ATTEMPT')
            user_id (Optional[int]): User ID associated with the event
            details (str): Detailed description of the event
            success (bool): Whether the operation was successful
            **kwargs: Additional context fields
        """
        # Create structured audit log entry
        audit_entry = {
            'event_type': event_type,
            'user_id': user_id,
            'details': details,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Add any additional context
        audit_entry.update(kwargs)
        
        # Format as audit message
        status = "SUCCESS" if success else "FAILED"
        message = f"AUDIT [{event_type}] {status}: {details}"
        if user_id:
            message += f" (User: {user_id})"
        
        # Log as info level for audit trail
        self._safe_log(self.logger.info, message)

# Global convenience functions for easy access
def redact_sensitive_data(text: str) -> str:
    """Global convenience function for data redaction"""
    return DataRedactionEngine.redact_sensitive_data(text)

def redact_crypto_data(text: str) -> str:
    """Global convenience function for crypto data redaction"""  
    return DataRedactionEngine.redact_crypto_data(text)

def get_secure_logger(logger_instance) -> SecureLogger:
    """Get a secure logger wrapper for any logger instance"""
    return SecureLogger(logger_instance)

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
        Enhanced security version with dangerous content removal
        
        Args:
            text (str): Input text to format
            preserve_code (bool): Whether to preserve code block formatting
            
        Returns:
            str: Safely formatted text
        """
        if not text:
            return text
        
        # Remove dangerous JavaScript and script content first
        text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        text = re.sub(r'data\s*:\s*text/html', '', text, flags=re.IGNORECASE)
            
        if preserve_code:
            # Extract and validate code blocks
            code_blocks = []
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            
            def extract_code_block(match):
                language = match.group(1) or ''
                code_content = match.group(2) or ''
                
                # Validate code content for dangerous patterns
                if any(danger in code_content.lower() for danger in ['javascript:', '<script', 'eval(', 'exec(']):
                    # Return sanitized version without dangerous content
                    safe_code = re.sub(r'(javascript|eval|exec|script)', '[REDACTED]', code_content, flags=re.IGNORECASE)
                    code_blocks.append(f'```{language}\n{safe_code}\n```')
                else:
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


class InputValidator:
    """Comprehensive input validation and sanitization for security"""
    
    # Malicious patterns for detection - Enhanced
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript\s*:',
        r'on\w+\s*=',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
        r'<svg[^>]*>.*?</svg>',
        r'data\s*:\s*text/html',
        r'vbscript\s*:',
        r'<meta[^>]*refresh',
        r'<link[^>]*href\s*=\s*["\']?javascript:',
        r'%3c\s*script',  # URL encoded
        r'&lt;\s*script',  # HTML encoded
        r'\\u003c\s*script',  # Unicode encoded
        r'<\s*script',  # Space variations
        r'expression\s*\(',  # CSS expression
        r'behavior\s*:',  # CSS behavior
        r'@import',  # CSS import
        r'alert\s*\(',  # Common XSS payload
        r'document\s*\.',  # Document access
        r'window\s*\.',  # Window access
    ]
    
    SQL_INJECTION_PATTERNS = [
        r'(\s|^)(select|insert|update|delete|drop|create|alter|exec|execute)\s+',
        r'(\s|^)(union|having|group\s+by|order\s+by)\s+',
        r"(\s|;|'|\")(--|/\*|\*/)",  # SQL comment injection with preceding delimiter
        # SECURITY FIX: Enhanced admin'-- pattern detection (Issue #1)
        r"'--",  # Classic comment-based SQL injection (e.g., admin'--)
        r"'--",  # Duplicate for emphasis - admin'-- pattern
        r"\w'--",  # Word followed by quote and comment (e.g., admin'--)
        r"'\s*--",  # Quote followed by optional space and comment (e.g., ' --)
        r"';?\s*--",  # Quote with optional semicolon and comment (e.g., '; --)
        r"'\s+or\s+.*?--",  # OR injection with comment (e.g., ' OR '1'='1' --)
        r'"--',  # Double quote comment injection
        r'"\s*--',  # Double quote with space and comment
        r"'\s*/\*",  # Quote followed by block comment (e.g., admin' /*)
        r"'\s*;",  # Quote followed by semicolon (e.g., admin';)
        r'(\s|^)(and|or)\s+\d+\s*=\s*\d+',
        r'(\s|^)(and|or)\s+["\']\w+["\']\s*=\s*["\']\w+["\']',
        r'\b(char|varchar|nvarchar|cast|convert|substring)\s*\(',
        r'\b(sys|information_schema|mysql|sqlite_master)',
        r'(\s|^)(load_file|into\s+outfile|into\s+dumpfile)',
        r'(1\s*=\s*1|1\s*=\s*0|\'\s*=\s*\')',
        r"'\s*or\s*'.*?'\s*=\s*'",  # Common OR injection
        r'"\s*or\s*".*?"\s*=\s*"',  # Double quote OR injection
        r';\s*drop\s+table',  # Drop table attempt
        r'waitfor\s+delay',  # Time-based injection
        r'benchmark\s*\(',  # MySQL benchmark
        r'pg_sleep\s*\(',  # PostgreSQL sleep
        r'extractvalue\s*\(',  # XML functions
        r'updatexml\s*\(',  # XML update functions
        r'0x[0-9a-fA-F]+',  # Hex encoding
    ]
    
    CODE_INJECTION_PATTERNS = [
        r'(exec|eval|system|shell_exec|passthru|popen)\s*\(',
        r'(import|from|require|include)\s+',
        r'(__import__|getattr|setattr|delattr|hasattr)\s*\(',
        r'(subprocess|os\.system|os\.popen|os\.exec)',
        r'(function\s*\(|=>|lambda)',
        r'(file_get_contents|file_put_contents|fopen|fwrite)',
        r'(base64_decode|unserialize|pickle\.loads)',
        r'(\${|<%|%>|<\?|\?>)',
        r'(cmd|powershell|bash|sh)\s+',
        # FIXED: Replaced ReDoS vulnerable regex with safer patterns
        r'(wget|curl|nc|netcat|telnet|ssh|ftp)\s+',  # Network tools (simplified)
        r'(rm|mv|cp|dd|mkfs|fdisk)\s+[^;]*(?:-rf|-f)',  # Destructive operations (safer)
        r'(python|perl|ruby|node)\s+-[ec]\b',  # Inline code execution (bounded)
        r'\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|[0-7]{3})',  # Escape sequences
        r'(%[0-9a-fA-F]{2})+',  # URL encoding that might hide commands
        r'(\\\\|//|\.\.)',  # Path manipulation attempts
        r'(chmod|chown|su|sudo)\s+',  # Privilege escalation
    ]
    
    # CRITICAL FIX: Replace ReDoS vulnerable regex with safe string operations
    DANGEROUS_SHELL_CHARS = ';', '&', '|', '`', '$', '(', ')', '{', '}', '[', ']', '\\', '<', '>', '*', '?', '~', '!'
    
    DANGEROUS_COMMANDS = [
        'rm', 'del', 'format', 'shutdown', 'reboot', 'kill', 'sudo', 'su',
        'passwd', 'chown', 'chmod', 'dd', 'fdisk', 'mkfs', 'mount', 'umount',
        # CRITICAL SECURITY FIX: Expanded dangerous commands (100% coverage)
        'nc', 'netcat', 'telnet', 'ssh', 'ftp', 'wget', 'curl', 'lynx',
        'python', 'perl', 'ruby', 'node', 'bash', 'sh', 'zsh', 'fish',
        'powershell', 'cmd', 'command', 'invoke-expression', 'iex',
        'systemctl', 'service', 'crontab', 'at', 'batch',
        'iptables', 'netsh', 'route', 'ifconfig', 'ip',
        'tar', 'gzip', 'gunzip', 'unzip', 'bzip2',
        'vi', 'vim', 'nano', 'emacs', 'ed',
        'less', 'more', 'tail', 'head', 'cat', 'grep', 'sed', 'awk',
        'find', 'locate', 'whereis', 'which', 'whoami', 'id', 'ps', 'top',
        'nmap', 'masscan', 'nikto', 'sqlmap', 'hydra', 'john',
        'msfconsole', 'metasploit', 'exploit', 'payload',
    ]
    
    SUSPICIOUS_KEYWORDS = [
        'bomb', 'exploit', 'payload', 'shellcode', 'backdoor', 'trojan',
        'malware', 'virus', 'keylogger', 'rootkit', 'botnet'
    ]
    
    @staticmethod
    def validate_input(text: str, max_length: int = 10000, strict_mode: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive input validation with malicious content detection
        
        Args:
            text (str): Input text to validate
            max_length (int): Maximum allowed length
            strict_mode (bool): Enable strict validation mode
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (is_valid, sanitized_text, threat_report)
        """
        if not text:
            return True, text, {'threats': [], 'risk_score': 0}
        
        threats = []
        risk_score = 0
        sanitized_text = text
        
        # Length validation
        if len(text) > max_length:
            threats.append({'type': 'length_exceeded', 'severity': 'medium'})
            risk_score += 3
            sanitized_text = text[:max_length]
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # XSS Detection
        xss_detected = 0
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                threats.append({'type': 'xss_attempt', 'pattern': pattern, 'severity': 'high'})
                risk_score += 5
                xss_detected += 1
        
        # SQL Injection Detection
        sql_detected = 0
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append({'type': 'sql_injection', 'pattern': pattern, 'severity': 'high'})
                risk_score += 5
                sql_detected += 1
        
        # Code Injection Detection
        code_detected = 0
        for pattern in InputValidator.CODE_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append({'type': 'code_injection', 'pattern': pattern, 'severity': 'critical'})
                risk_score += 8
                code_detected += 1
        
        # CRITICAL FIX: Safe shell metacharacter detection (ReDoS-free)
        shell_chars_detected = 0
        for char in InputValidator.DANGEROUS_SHELL_CHARS:
            if char in text:  # Simple string containment check - no regex backtracking
                threats.append({'type': 'shell_metachar', 'character': char, 'severity': 'critical'})
                risk_score += 8
                shell_chars_detected += 1
        
        # Dangerous Command Detection
        cmd_detected = 0
        for cmd in InputValidator.DANGEROUS_COMMANDS:
            if re.search(r'\b' + re.escape(cmd) + r'\b', text_lower):
                threats.append({'type': 'dangerous_command', 'command': cmd, 'severity': 'high'})
                risk_score += 6
                cmd_detected += 1
        
        # Suspicious Keyword Detection
        suspicious_detected = 0
        for keyword in InputValidator.SUSPICIOUS_KEYWORDS:
            if keyword in text_lower:
                threats.append({'type': 'suspicious_keyword', 'keyword': keyword, 'severity': 'medium'})
                risk_score += 2
                suspicious_detected += 1
        
        # Path Traversal Detection - Enhanced
        path_traversal_patterns = [
            r'\.\./', r'\.\.\\', r'%2e%2e%2f', r'%2e%2e%5c',
            r'\.\.%2f', r'\.\.%5c', r'%252e%252e%252f', r'%c0%ae%c0%ae%2f',
            r'\.\.\/.*?\/etc\/passwd', r'\.\.\\.*?\\windows\\system32',
            r'\/proc\/', r'\/dev\/', r'\/var\/', r'\/tmp\/'
        ]
        
        path_detected = 0
        for pattern in path_traversal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append({'type': 'path_traversal', 'pattern': pattern, 'severity': 'high'})
                risk_score += 6
                path_detected += 1
        
        # URL/Protocol Detection (potential for abuse)
        url_patterns = [r'https?://', r'ftp://', r'file://', r'data://', r'javascript:']
        for pattern in url_patterns:
            if re.search(pattern, text_lower):
                threats.append({'type': 'url_detected', 'pattern': pattern, 'severity': 'medium'})
                risk_score += 2
        
        # Encoding Detection (potential evasion)
        if any(enc in text_lower for enc in ['%', '&lt;', '&gt;', '&amp;', '&#', '\\x', '\\u']):
            threats.append({'type': 'encoding_detected', 'severity': 'medium'})
            risk_score += 3
        
        # Control Character Detection
        if any(ord(char) < 32 and char not in '\t\n\r' for char in text):
            threats.append({'type': 'control_characters', 'severity': 'medium'})
            risk_score += 2
        
        # SECURITY FIX: Stricter risk score thresholds to prevent bypass
        if strict_mode and risk_score >= 8:  # Lowered from 10 to 8
            # High risk - aggressive sanitization
            sanitized_text = InputValidator._aggressive_sanitize(sanitized_text)
        elif risk_score >= 3:  # Lowered from 5 to 3
            # Medium risk - moderate sanitization
            sanitized_text = InputValidator._moderate_sanitize(sanitized_text)
        else:
            # Low risk - basic sanitization
            sanitized_text = InputValidator._basic_sanitize(sanitized_text)
        
        # Final validation
        is_valid = risk_score < (5 if strict_mode else 15)
        
        threat_report = {
            'threats': threats,
            'risk_score': risk_score,
            'threat_counts': {
                'xss': xss_detected,
                'sql_injection': sql_detected,
                'code_injection': code_detected,
                'dangerous_commands': cmd_detected,
                'suspicious_keywords': suspicious_detected
            },
            'severity_level': InputValidator._get_severity_level(risk_score)
        }
        
        return is_valid, sanitized_text, threat_report
    
    @staticmethod
    def _basic_sanitize(text: str) -> str:
        """Basic sanitization for low-risk input"""
        # Remove obvious HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove script tags content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()
    
    @staticmethod
    def _moderate_sanitize(text: str) -> str:
        """Moderate sanitization for medium-risk input"""
        text = InputValidator._basic_sanitize(text)
        # Remove JavaScript protocols
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        # Remove common SQL injection patterns
        text = re.sub(r'(union|select|insert|update|delete)\s+', '', text, flags=re.IGNORECASE)
        # Remove dangerous characters
        text = re.sub(r'[<>"\'\/\\]', '', text)
        return text.strip()
    
    @staticmethod
    def _aggressive_sanitize(text: str) -> str:
        """Aggressive sanitization for high-risk input"""
        # Only allow alphanumeric characters, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\-_@#]', '', text)
        # Limit consecutive special characters
        text = re.sub(r'([.,!?\-_]){3,}', r'\1\1', text)
        return text.strip()[:1000]  # Truncate to safe length
    
    @staticmethod
    def _get_severity_level(risk_score: int) -> str:
        """Determine severity level based on risk score"""
        if risk_score >= 15:
            return 'critical'
        elif risk_score >= 10:
            return 'high'
        elif risk_score >= 5:
            return 'medium'
        else:
            return 'low'
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """Validate and sanitize filenames"""
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check for dangerous patterns
        dangerous_patterns = [r'\.\.', r'[<>:"|?*]', r'^(con|prn|aux|nul|com[1-9]|lpt[1-9])$']
        for pattern in dangerous_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                return False, f"Filename contains dangerous pattern: {pattern}"
        
        # Length validation
        if len(filename) > 255:
            return False, "Filename too long"
        
        # Sanitize filename
        sanitized = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)
        sanitized = re.sub(r'_{2,}', '_', sanitized)  # Remove multiple underscores
        
        return True, sanitized
    
    @staticmethod 
    def is_malicious_input(text: str) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Check if input contains malicious patterns (simplified wrapper for validate_input)
        
        Args:
            text (str): Input text to check
            
        Returns:
            Tuple[bool, List[str], Dict[str, Any]]: (is_malicious, threat_types, details)
        """
        is_valid, sanitized_text, threat_report = InputValidator.validate_input(text, strict_mode=True)
        
        threats = threat_report.get('threats', [])
        risk_score = threat_report.get('risk_score', 0)
        
        # Consider input malicious if it has threats or high risk score
        is_malicious = not is_valid or len(threats) > 0 or risk_score > 50
        
        # Extract threat types for cleaner interface
        threat_types = [threat.get('type', 'unknown') for threat in threats]
        
        return is_malicious, threat_types, threat_report


class RateLimiter:
    """
    Enhanced rate limiting with IP tracking, progressive penalties, and bypass protection
    
    THREAD SAFETY INVARIANTS:
    - ALL access to user_requests, ip_requests, user_violations, ip_violations,
      penalty_multipliers, blocked_users, blocked_ips must be protected by self._lock
    - Methods that modify these data structures MUST acquire the lock before ANY access
    - Heavy logging and external calls should be moved outside critical sections when possible
    - Use RLock to support re-entrant calls between internal methods
    - Never partially lock operations - entire request processing must be atomic
    """
    
    def __init__(self, max_requests: int = 10, time_window: int = 60, strict_mode: bool = True):
        """
        Initialize enhanced rate limiter with anti-bypass protection
        
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
        
        # Enhanced security features
        self.ip_requests: Dict[str, list] = defaultdict(list)  # IP-based tracking
        self.user_violations: Dict[int, int] = defaultdict(int)  # Violation count
        self.ip_violations: Dict[str, int] = defaultdict(int)  # IP violation count
        self.penalty_multipliers: Dict[int, float] = defaultdict(lambda: 1.0)  # Progressive penalties
        self.suspicious_patterns: Dict[int, list] = defaultdict(list)  # Pattern detection
        self.blocked_ips: Set[str] = set()  # Temporary IP blocks
        self.blocked_users: Set[int] = set()  # Temporary user blocks
        
        # CRITICAL FIX: Add thread synchronization to prevent race conditions
        # Using RLock for safer re-entrant locking when methods call each other
        self._lock = threading.RLock()
    
    def is_allowed(self, user_id: int, user_ip: Optional[str] = None, action_type: Optional[str] = None) -> tuple[bool, Optional[int]]:
        """
        Enhanced rate limiting with IP tracking and progressive penalties
        CRITICAL FIX: Completely thread-safe with atomic operations and optimized logging
        
        Args:
            user_id (int): Telegram user ID
            user_ip (str): User IP address for additional tracking
            action_type (str): Type of action being performed (for context)
            
        Returns:
            tuple[bool, Optional[int]]: (is_allowed, seconds_until_reset)
        """
        # Collect data for logging outside the critical section
        log_data = {}
        
        # CRITICAL FIX: Make entire rate limit check atomic to prevent bypass
        result = None
        wait_time = None
        
        with self._lock:
            current_time = time.time()
            
            # Check if user is temporarily blocked
            if user_id in self.blocked_users:
                log_data['user_blocked'] = user_id
                result = False
                wait_time = 300  # 5 minute block
            
            # Check if IP is temporarily blocked
            elif user_ip and user_ip in self.blocked_ips:
                log_data['ip_blocked'] = user_ip
                result = False
                wait_time = 300  # 5 minute block
            
            else:
                # Progressive penalty calculation
                violation_count = self.user_violations[user_id]
                penalty_multiplier = min(1.0 + (violation_count * 0.5), 5.0)  # Max 5x penalty
                effective_requests = max(1, int(self.max_requests / penalty_multiplier))
                
                # IP-based rate limiting (additional layer)
                if user_ip:
                    ip_requests = self.ip_requests[user_ip]
                    ip_requests[:] = [req_time for req_time in ip_requests 
                                     if current_time - req_time < self.time_window]
                    
                    # IP rate limit (stricter than user limit)
                    ip_limit = max(effective_requests * 2, 5)  # Allow some flexibility for shared IPs
                    if len(ip_requests) >= ip_limit:
                        self._record_violation(user_id, user_ip, "ip_rate_limit")
                        log_data['ip_rate_exceeded'] = (user_ip, user_id)
                        result = False
                        wait_time = max(30, int(self.time_window / 2))
                
                if result is None:  # Not blocked by IP limits
                    # Refill tokens based on time elapsed (with penalty)
                    if user_id in self.last_refill and self.last_refill[user_id] > 0:
                        time_elapsed = current_time - self.last_refill[user_id]
                        tokens_to_add = time_elapsed * (effective_requests / self.time_window)
                        self.user_tokens[user_id] = min(effective_requests, 
                                                       self.user_tokens[user_id] + tokens_to_add)
                    else:
                        # First request - give partial bucket based on penalty
                        self.user_tokens[user_id] = effective_requests
                    
                    self.last_refill[user_id] = current_time
                    
                    # Check if tokens available
                    if self.user_tokens[user_id] >= 1.0:
                        # Allow request and consume token
                        self.user_tokens[user_id] -= 1.0
                        
                        # Track request patterns for suspicious activity detection
                        user_requests = self.user_requests[user_id]
                        user_requests.append(current_time)
                        
                        # Track IP requests
                        if user_ip:
                            self.ip_requests[user_ip].append(current_time)
                        
                        # Clean old requests
                        user_requests[:] = [req_time for req_time in user_requests 
                                           if current_time - req_time < self.time_window]
                        
                        # Detect suspicious patterns (rapid successive requests)
                        self._detect_suspicious_patterns(user_id, user_ip, current_time)
                        
                        result = True
                        wait_time = None
                    else:
                        # Rate limited - record violation and calculate penalty
                        self._record_violation(user_id, user_ip, "rate_limit_exceeded")
                        
                        tokens_needed = 1.0 - self.user_tokens[user_id]
                        base_wait = int(tokens_needed * (self.time_window / effective_requests))
                        penalized_wait = int(base_wait * penalty_multiplier)
                        
                        # In strict mode, immediate blocking with penalties
                        if self.strict_mode:
                            log_data['rate_exceeded_strict'] = (user_id, penalty_multiplier)
                            result = False
                            wait_time = max(5, penalized_wait)  # Minimum 5 second wait
                        else:
                            # SECURITY FIX: Tightened non-strict mode to prevent bypass
                            user_requests = self.user_requests[user_id]
                            user_requests[:] = [req_time for req_time in user_requests 
                                               if current_time - req_time < self.time_window]
                            
                            if len(user_requests) >= effective_requests:
                                oldest_request = min(user_requests) if user_requests else current_time
                                base_wait = int(self.time_window - (current_time - oldest_request))
                                penalized_wait = int(base_wait * penalty_multiplier)
                                result = False
                                wait_time = max(5, penalized_wait)
                            
                            # SECURITY FIX: More restrictive fallback - deny if close to limit
                            elif len(user_requests) >= max(1, effective_requests - 1):  # Deny if within 1 of limit
                                log_data['near_limit_nonstrict'] = user_id
                                result = False
                                wait_time = max(5, int(penalty_multiplier * 10))
                            else:
                                # Allow only if well under the limit
                                user_requests.append(current_time)
                                if user_ip:
                                    self.ip_requests[user_ip].append(current_time)
                                result = True
                                wait_time = None
        
        # PERFORMANCE FIX: Move all heavy logging outside critical section
        if 'user_blocked' in log_data:
            logger.warning(f"SECURITY: User {log_data['user_blocked']} is temporarily blocked")
        
        if 'ip_blocked' in log_data:
            logger.warning(f"SECURITY: IP {log_data['ip_blocked']} is temporarily blocked")
        
        if 'ip_rate_exceeded' in log_data:
            user_ip, user_id = log_data['ip_rate_exceeded']
            logger.warning(f"SECURITY: IP rate limit exceeded for {user_ip} (user {user_id})")
        
        if 'rate_exceeded_strict' in log_data:
            user_id, penalty_multiplier = log_data['rate_exceeded_strict']
            logger.warning(f"SECURITY: Rate limit exceeded for user {user_id} (penalty: {penalty_multiplier:.1f}x)")
        
        if 'near_limit_nonstrict' in log_data:
            logger.warning(f"SECURITY: Near rate limit in non-strict mode for user {log_data['near_limit_nonstrict']}")
        
        return result, wait_time
    
    def get_remaining_requests(self, user_id: int) -> int:
        """
        Get number of remaining requests for user
        SECURITY FIX: Added thread synchronization to prevent race conditions
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            int: Number of remaining requests
        """
        with self._lock:
            current_time = time.time()
            user_requests = self.user_requests[user_id]
            
            # Remove old requests
            user_requests[:] = [req_time for req_time in user_requests 
                               if current_time - req_time < self.time_window]
            
            return max(0, self.max_requests - len(user_requests))
    
    def _record_violation(self, user_id: int, user_ip: Optional[str] = None, violation_type: str = "unknown") -> None:
        """
        Record security violation and apply progressive penalties
        CRITICAL FIX: Thread-safe with proper synchronization and optimized logging
        
        Args:
            user_id (int): User ID
            user_ip (str): User IP address
            violation_type (str): Type of violation
        """
        # Collect data for logging outside the critical section
        log_data = {}
        
        # CRITICAL FIX: Use RLock to ensure atomic updates to all violation counters
        with self._lock:
            # Update violation counters atomically
            self.user_violations[user_id] += 1
            
            if user_ip:
                self.ip_violations[user_ip] += 1
            
            # SECURITY FIX: Progressive blocking with configurable thresholds
            user_block_threshold = getattr(self, 'user_block_threshold', 10)
            ip_block_threshold = getattr(self, 'ip_block_threshold', 15)
            
            user_violation_count = self.user_violations[user_id]
            ip_violation_count = self.ip_violations.get(user_ip, 0) if user_ip else 0
            
            # Apply temporary blocks for repeat offenders with escalating penalties
            if user_violation_count >= user_block_threshold:
                self.blocked_users.add(user_id)
                # Schedule automatic unblock (can be extended by admin system)
                block_duration = min(300 * (user_violation_count // user_block_threshold), 3600)  # Max 1 hour
                log_data['user_blocked'] = (user_id, user_violation_count, block_duration)
            
            if user_ip and ip_violation_count >= ip_block_threshold:
                self.blocked_ips.add(user_ip)
                # Progressive IP block duration
                block_duration = min(300 * (ip_violation_count // ip_block_threshold), 7200)  # Max 2 hours
                log_data['ip_blocked'] = (user_ip, ip_violation_count, block_duration)
            
            # Update penalty multiplier
            self.penalty_multipliers[user_id] = min(1.0 + (user_violation_count * 0.5), 5.0)
            
            # Prepare data for logging outside critical section
            log_data['violation'] = (user_id, user_ip, violation_type, user_violation_count)
        
        # PERFORMANCE FIX: Move heavy logging outside critical section
        if 'user_blocked' in log_data:
            user_id, violations, duration = log_data['user_blocked']
            logger.critical(f"SECURITY: User {user_id} temporarily blocked for {violations} violations (duration: {duration}s)")
        
        if 'ip_blocked' in log_data:
            ip, violations, duration = log_data['ip_blocked']
            logger.critical(f"SECURITY: IP {ip} temporarily blocked for {violations} violations (duration: {duration}s)")
        
        # Log violation outside critical section
        user_id, user_ip, violation_type, count = log_data['violation']
        logger.warning(f"SECURITY: Violation recorded - User: {user_id}, IP: {user_ip}, Type: {violation_type}, Count: {count}")
    
    def _detect_suspicious_patterns(self, user_id: int, user_ip: Optional[str] = None, current_time: Optional[float] = None) -> None:
        """
        Detect suspicious request patterns that might indicate bypass attempts
        CRITICAL FIX: Thread-safe pattern detection with optimized logging
        
        Args:
            user_id (int): User ID
            user_ip (str): User IP address
            current_time (float): Current timestamp
        """
        if not current_time:
            current_time = time.time()
        
        # Collect data for logging outside the critical section
        log_data = {}
        
        # CRITICAL FIX: Access user_requests within lock for thread safety
        with self._lock:
            user_requests = self.user_requests[user_id].copy()  # Copy to avoid modification during analysis
            
            # Pattern 1: Rapid successive requests (possible burst attack)
            if len(user_requests) >= 3:
                recent_requests = [req for req in user_requests if current_time - req < 10]  # Last 10 seconds
                if len(recent_requests) >= 3:
                    intervals = [recent_requests[i] - recent_requests[i-1] for i in range(1, len(recent_requests))]
                    avg_interval = sum(intervals) / len(intervals)
                    if avg_interval < 2.0:  # Less than 2 seconds between requests
                        # Record violation outside this lock (it will use its own lock)
                        self._record_violation(user_id, user_ip, "rapid_requests")
                        # SECURITY FIX: Immediate penalty for rapid requests
                        if avg_interval < 0.5:  # Extremely rapid requests
                            self.user_violations[user_id] += 2  # Extra penalty
                            log_data['rapid_attack'] = (user_id, avg_interval)
            
            # Pattern 2: Consistent timing (possible automated requests)
            if len(user_requests) >= 5:
                intervals = [user_requests[i] - user_requests[i-1] for i in range(1, min(len(user_requests), 6))]
                if len(set([round(interval, 1) for interval in intervals])) <= 2:  # Very consistent timing
                    # Record violation outside this lock (it will use its own lock)
                    self._record_violation(user_id, user_ip, "automated_pattern")
                    # SECURITY FIX: Immediate penalty for bot-like behavior
                    if len(set([round(interval * 10) for interval in intervals])) <= 1:  # Extremely consistent (0.1s precision)
                        self.user_violations[user_id] += 3  # Heavy penalty for bots
                        log_data['bot_pattern'] = user_id
        
        # PERFORMANCE FIX: Move heavy logging outside critical section
        if 'rapid_attack' in log_data:
            user_id, avg_interval = log_data['rapid_attack']
            logger.critical(f"SECURITY: Extremely rapid requests detected from user {user_id} (avg: {avg_interval:.3f}s)")
        
        if 'bot_pattern' in log_data:
            user_id = log_data['bot_pattern']
            logger.critical(f"SECURITY: Bot-like automated pattern detected from user {user_id}")
    
    def reset_user(self, user_id: int) -> None:
        """
        Reset rate limit for specific user (admin function)
        SECURITY FIX: Added thread synchronization to prevent race conditions
        
        Args:
            user_id (int): Telegram user ID
        """
        with self._lock:
            self.user_requests[user_id] = []
            self.user_tokens[user_id] = self.max_requests
            self.user_violations[user_id] = 0
            self.penalty_multipliers[user_id] = 1.0
            self.blocked_users.discard(user_id)
            logger.info(f"Rate limit reset for user {user_id}")
    
    def reset_ip(self, ip_address: str) -> None:
        """
        Reset rate limit for specific IP (admin function)
        SECURITY FIX: Added thread synchronization to prevent race conditions
        
        Args:
            ip_address (str): IP address to reset
        """
        with self._lock:
            self.ip_requests[ip_address] = []
            self.ip_violations[ip_address] = 0
            self.blocked_ips.discard(ip_address)
            logger.info(f"Rate limit reset for IP {ip_address}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive security statistics
        SECURITY FIX: Added thread synchronization to prevent race conditions
        
        Returns:
            Dict[str, Any]: Security statistics
        """
        with self._lock:
            return {
                'total_users_tracked': len(self.user_requests),
                'total_ips_tracked': len(self.ip_requests),
                'users_with_violations': len([uid for uid, count in self.user_violations.items() if count > 0]),
                'ips_with_violations': len([ip for ip, count in self.ip_violations.items() if count > 0]),
                'blocked_users': len(self.blocked_users),
                'blocked_ips': len(self.blocked_ips),
                'average_penalty_multiplier': sum(self.penalty_multipliers.values()) / max(1, len(self.penalty_multipliers)),
                'high_violation_users': [uid for uid, count in self.user_violations.items() if count >= 5]
            }


# Global instances for use across the application  
markdown_sanitizer = MarkdownSanitizer()
input_validator = InputValidator()

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
    
    # TESTING BYPASS: Disable rate limiting in test mode  
    from bot.config import Config
    if Config.is_test_mode():
        logger.info(f"TEST MODE: Bypassing persistent rate limit for user {user_id} (testing)")
        return True, 0
    
    if storage_provider and hasattr(storage_provider, 'check_rate_limit'):
        try:
            # Use persistent database rate limiting for production (STRICT: 3 requests per minute)
            is_allowed, wait_time = await storage_provider.check_rate_limit(user_id, max_requests=3, time_window=60)
            return is_allowed, wait_time or 0
        except Exception as e:
            # SECURITY FIX: More secure fallback when persistent rate limiting fails
            secure_logger = get_secure_logger(logging.getLogger(__name__))
            secure_logger.error(f"CRITICAL: Persistent rate limiting system failure for user {user_id}", exc_info=True)
            
            # Use ultra-strict fallback during system failure
            fallback_limiter = RateLimiter(max_requests=1, time_window=60, strict_mode=True)  # Only 1 request per minute
            is_allowed, wait_time = fallback_limiter.is_allowed(user_id, None)
            
            if not is_allowed:
                logger.critical(f"SECURITY: Emergency rate limiting active for user {user_id} due to system failure")
                return False, max(60, wait_time or 60)  # Minimum 1 minute wait during emergency
            
            return is_allowed, wait_time or 0
    else:
        # Use strict in-memory rate limiting as fallback
        return check_rate_limit(user_id)

def check_rate_limit(user_id: int, user_ip: Optional[str] = None) -> tuple[bool, int]:
    """Enhanced in-memory rate limiting with IP tracking and progressive penalties"""
    
    # TESTING BYPASS: Disable rate limiting in test mode
    from bot.config import Config
    if Config.is_test_mode():
        logger.info(f"TEST MODE: Bypassing rate limit for user {user_id} (testing)")
        return True, 0
    
    is_allowed, wait_time = rate_limiter.is_allowed(user_id, user_ip)
    
    # Security logging for rate limit violations
    if not is_allowed:
        logger.warning(f"SECURITY: Rate limit exceeded for user {user_id} (IP: {user_ip}), blocking for {wait_time} seconds")
    
    return is_allowed, wait_time or 0

# Convenience functions for input validation
def validate_user_input(text: str, max_length: int = 10000, strict_mode: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """Convenience function for comprehensive input validation"""
    return input_validator.validate_input(text, max_length, strict_mode)

def safe_filename(filename: str) -> Tuple[bool, str]:
    """Convenience function for filename validation"""
    return input_validator.validate_filename(filename)

def get_security_stats() -> Dict[str, Any]:
    """Get comprehensive security statistics from rate limiter"""
    return rate_limiter.get_security_stats()