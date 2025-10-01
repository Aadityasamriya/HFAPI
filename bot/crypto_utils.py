"""
Secure Cryptographic Utilities for Hugging Face By AadityaLabs AI
Implements P0 CRITICAL SECURITY encryption hardening with authenticated encryption

SECURITY ARCHITECTURE:
- Versioned envelope format: v1 || salt || nonce || ciphertext || auth_tag
- HKDF(SHA256, salt) for key derivation from ENCRYPTION_SEED
- AES-256-GCM authenticated encryption with integrity verification
- Strict error propagation - NO silent fallbacks
- Key rotation utility for seamless migration
- Backward compatibility for existing encrypted data

This module replaces the vulnerable encryption implementations with
production-grade security to prevent data corruption and credential exposure.
"""

import secrets
import logging
import base64
from typing import Optional, Tuple, Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from bot.security_utils import redact_crypto_data, get_secure_logger

logger = logging.getLogger(__name__)

# Removed duplicate redaction function - now using centralized redaction from security_utils

# Security constants
ENVELOPE_VERSION = b'v1'  # Version identifier for envelope format
SALT_SIZE = 32  # 256-bit salt for HKDF
NONCE_SIZE = 12  # 96-bit nonce for AES-GCM (recommended)
KEY_SIZE = 32   # 256-bit encryption key
VERSION_SIZE = 2  # 2 bytes for version

class CryptoError(Exception):
    """Base exception for all cryptographic operations"""
    pass

class EncryptionError(CryptoError):
    """Raised when encryption operations fail"""
    pass

class DecryptionError(CryptoError):
    """Raised when decryption operations fail"""
    pass

class TamperDetectionError(DecryptionError):
    """Raised when data tampering is detected"""
    pass

class KeyDerivationError(CryptoError):
    """Raised when key derivation fails"""
    pass

# ELIMINATED: Legacy SecurityLogger class removed - using centralized security_utils.get_secure_logger instead

# Initialize secure logger using centralized security utilities
secure_logger = get_secure_logger(logger)

class SecureCrypto:
    """
    Production-grade cryptographic operations with authenticated encryption
    
    Implements versioned envelope format for forward compatibility:
    Envelope = version(2) || salt(32) || nonce(12) || ciphertext || auth_tag(16)
    
    Total overhead: 62 bytes (version + salt + nonce + auth_tag)
    """
    
    def __init__(self, encryption_seed: str):
        """
        Initialize secure crypto with encryption seed
        
        Args:
            encryption_seed (str): Master encryption seed from ENCRYPTION_SEED env var
            
        Raises:
            KeyDerivationError: If seed is invalid or key derivation fails
        """
        if not encryption_seed or not isinstance(encryption_seed, str):
            raise KeyDerivationError("Encryption seed must be a non-empty string")
        
        if len(encryption_seed) < 32:
            raise KeyDerivationError("Encryption seed must be at least 32 characters for security")
        
        self._seed = encryption_seed.encode('utf-8')
        secure_logger.debug("🔐 SecureCrypto initialized with validated encryption seed")
    
    def _derive_key(self, salt: bytes, context: Optional[bytes] = None) -> bytes:
        """
        Derive encryption key using HKDF(SHA256, salt)
        
        Args:
            salt (bytes): Random salt for key derivation
            context (bytes, optional): Additional context for key derivation
            
        Returns:
            bytes: 32-byte derived encryption key
            
        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            if len(salt) != SALT_SIZE:
                raise KeyDerivationError(f"Salt must be exactly {SALT_SIZE} bytes")
            
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=KEY_SIZE,
                salt=salt,
                info=context,
                backend=default_backend()
            )
            
            derived_key = hkdf.derive(self._seed)
            secure_logger.debug("🔑 Encryption key derived successfully using HKDF-SHA256")
            return derived_key
            
        except Exception as e:
            # SECURITY FIX: Apply crypto-specific redaction to ALL error logging
            safe_exception_msg = redact_crypto_data(str(e))
            secure_logger.error(f"CRITICAL: Key derivation failed: {safe_exception_msg}")
            raise KeyDerivationError(f"Failed to derive encryption key: {safe_exception_msg}")
    
    def _derive_user_key(self, user_id: int, salt: bytes) -> bytes:
        """
        Derive per-user encryption key for data isolation
        
        Args:
            user_id (int): User identifier for key derivation context
            salt (bytes): Random salt for key derivation
            
        Returns:
            bytes: 32-byte user-specific encryption key
            
        Raises:
            KeyDerivationError: If user key derivation fails
        """
        try:
            if not isinstance(user_id, int):
                raise KeyDerivationError("User ID must be an integer")
            if user_id <= 0:
                raise KeyDerivationError("User ID must be a positive integer")
            
            # Use user_id as additional context for key derivation
            context = f"user_{user_id}".encode('utf-8')
            user_key = self._derive_key(salt, context)
            
            secure_logger.debug(f"🔑 User-specific encryption key derived for user {user_id}")
            return user_key
            
        except Exception as e:
            # SECURITY FIX: Apply centralized crypto redaction to ALL error messages  
            safe_exception_msg = redact_crypto_data(str(e))
            secure_logger.error(f"CRITICAL: User key derivation failed for user {user_id}: {safe_exception_msg}")
            raise KeyDerivationError(f"Failed to derive user key: {safe_exception_msg}")
    
    def encrypt(self, plaintext: str, user_id: Optional[int] = None) -> str:
        """
        Encrypt plaintext with authenticated encryption using versioned envelope
        
        Envelope format: version(2) || salt(32) || nonce(12) || ciphertext || auth_tag(16)
        
        Args:
            plaintext (str): Data to encrypt
            user_id (int, optional): User ID for per-user encryption
            
        Returns:
            str: Base64-encoded encrypted envelope
            
        Raises:
            EncryptionError: If encryption fails
        """
        try:
            if not plaintext or not isinstance(plaintext, str):
                raise EncryptionError("Plaintext must be a non-empty string")
            
            # Generate cryptographically secure random values
            salt = secrets.token_bytes(SALT_SIZE)
            nonce = secrets.token_bytes(NONCE_SIZE)
            
            # Derive encryption key
            if user_id is not None:
                encryption_key = self._derive_user_key(user_id, salt)
                secure_logger.debug(f"🔒 Encrypting data for user {user_id}")
            else:
                encryption_key = self._derive_key(salt)
                secure_logger.debug("🔒 Encrypting data with global key")
            
            # Encrypt with AES-256-GCM (authenticated encryption)
            aesgcm = AESGCM(encryption_key)
            ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
            
            # Construct envelope: salt || nonce || ciphertext+auth_tag (no version in binary)
            envelope = salt + nonce + ciphertext
            
            # Encode to base64 and add string version prefix
            encoded_envelope = base64.b64encode(envelope).decode('ascii')
            versioned_result = f"v1{encoded_envelope}"
            
            secure_logger.debug("✅ Data encrypted successfully with authenticated encryption")
            return versioned_result
            
        except KeyDerivationError:
            # Re-raise key derivation errors as-is
            raise
        except Exception as e:
            # SECURITY FIX: Apply crypto-specific redaction to ALL error logging
            safe_exception_msg = redact_crypto_data(str(e))
            secure_logger.error(f"CRITICAL: Encryption failed: {safe_exception_msg}")
            raise EncryptionError(f"Failed to encrypt data: {safe_exception_msg}")
    
    def decrypt(self, encrypted_data: str, user_id: Optional[int] = None) -> str:
        """
        Decrypt authenticated encrypted data with strict integrity verification
        
        Args:
            encrypted_data (str): Base64-encoded encrypted envelope
            user_id (int, optional): User ID for per-user decryption
            
        Returns:
            str: Decrypted plaintext
            
        Raises:
            DecryptionError: If decryption fails
            TamperDetectionError: If data tampering is detected
        """
        try:
            if not encrypted_data or not isinstance(encrypted_data, str):
                raise DecryptionError("Encrypted data must be a non-empty string")
            
            # Handle versioned format: v1<base64_data>
            if not encrypted_data.startswith('v1'):
                raise DecryptionError("Missing version prefix - expected v1 format")
            
            # Extract base64 part after version prefix
            base64_part = encrypted_data[2:]  # Remove 'v1' prefix
            
            # Decode from base64
            try:
                envelope = base64.b64decode(base64_part.encode('ascii'))
            except Exception as e:
                raise DecryptionError(f"Invalid base64 encoding: {e}")
            
            # Validate minimum envelope size (no version in binary part)
            min_size = SALT_SIZE + NONCE_SIZE + 16  # +16 for GCM auth tag
            if len(envelope) < min_size:
                raise TamperDetectionError(f"Envelope too small: {len(envelope)} < {min_size} bytes")
            
            # Extract components from envelope (no version in binary)
            salt = envelope[:SALT_SIZE]
            nonce = envelope[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
            ciphertext = envelope[SALT_SIZE + NONCE_SIZE:]
            
            # Derive decryption key (must match encryption key)
            if user_id is not None:
                decryption_key = self._derive_user_key(user_id, salt)
                secure_logger.debug(f"🔓 Decrypting data for user {user_id}")
            else:
                decryption_key = self._derive_key(salt)
                secure_logger.debug("🔓 Decrypting data with global key")
            
            # Decrypt with AES-256-GCM (includes authentication verification)
            aesgcm = AESGCM(decryption_key)
            try:
                plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            except Exception as e:
                # AES-GCM auth tag verification failed - data was tampered with
                secure_logger.error("🚨 SECURITY ALERT: Data tampering detected during decryption")
                raise TamperDetectionError(f"Data integrity verification failed: {e}")
            
            plaintext = plaintext_bytes.decode('utf-8')
            secure_logger.debug("✅ Data decrypted successfully with integrity verification")
            return plaintext
            
        except (KeyDerivationError, TamperDetectionError):
            # Re-raise specific crypto errors as-is
            raise
        except DecryptionError:
            # Re-raise decryption errors as-is
            raise
        except Exception as e:
            # SECURITY FIX: Apply centralized crypto redaction to ALL error messages
            safe_exception_msg = redact_crypto_data(str(e))
            secure_logger.error(f"CRITICAL: Unexpected decryption error: {safe_exception_msg}")
            raise DecryptionError(f"Failed to decrypt data: {safe_exception_msg}")
    
    def is_encrypted(self, data: str) -> bool:
        """
        Check if data appears to be encrypted using our envelope format
        
        Implements robust validation to prevent false positives:
        - Validates v1 prefix AND base64 decodability
        - Checks minimum envelope size (salt+nonce+tag)  
        - Prevents false positives from plaintext starting with "v1"
        
        Args:
            data (str): Data to check
            
        Returns:
            bool: True if data appears encrypted, False otherwise
        """
        try:
            # Basic validation
            if not data or not isinstance(data, str):
                return False
            
            # Check for version prefix
            if not data.startswith('v1'):
                return False
            
            # Extract base64 part after version prefix
            base64_part = data[2:]  # Remove 'v1' prefix
            
            # Validate minimum length for base64 encoded envelope
            # Minimum: base64(salt(32) + nonce(12) + min_ciphertext+auth_tag(17)) ≈ 82 chars
            if len(base64_part) < 82:
                return False
            
            # Validate base64 format - must contain only valid base64 characters
            import string
            valid_b64_chars = string.ascii_letters + string.digits + '+/='
            if not all(c in valid_b64_chars for c in base64_part):
                return False
            
            # Try to decode as base64 with strict validation
            try:
                envelope = base64.b64decode(base64_part.encode('ascii'), validate=True)
            except Exception:
                # Not valid base64 - likely plaintext starting with "v1"
                return False
            
            # Check exact minimum size for our envelope format
            min_size = SALT_SIZE + NONCE_SIZE + 17  # salt(32) + nonce(12) + min_ciphertext_with_auth_tag(17)
            if len(envelope) < min_size:
                return False
            
            # Additional entropy check: encrypted data should have high entropy
            # More robust entropy validation for the first 32 bytes (salt portion)
            entropy_sample = envelope[:min(32, len(envelope))]
            unique_bytes = len(set(entropy_sample))
            
            # Encrypted data should have high entropy - require at least 75% unique bytes
            min_unique = max(8, int(len(entropy_sample) * 0.75))
            if unique_bytes < min_unique:
                return False
            
            # Additional check: reject obvious patterns
            # Check for repeating byte patterns (like 0x00, 0x01, 0x02... or repeated sequences)
            if len(entropy_sample) >= 16:
                # Check for ascending/descending sequences
                is_sequence = all(entropy_sample[i] == (entropy_sample[0] + i) % 256 for i in range(min(16, len(entropy_sample))))
                if is_sequence:
                    return False
                
                # Check for repeated small patterns
                for pattern_size in [1, 2, 4]:
                    if len(entropy_sample) >= pattern_size * 4:
                        pattern = entropy_sample[:pattern_size]
                        repeated = pattern * (len(entropy_sample) // pattern_size)
                        if entropy_sample.startswith(repeated[:len(entropy_sample)]):
                            return False
            
            return True
            
        except Exception:
            return False

class KeyRotationManager:
    """
    Secure key rotation utility for migrating between encryption keys
    
    Provides safe migration from old encryption seed to new encryption seed
    while maintaining data accessibility during transition period.
    """
    
    def __init__(self, old_seed: str, new_seed: str):
        """
        Initialize key rotation manager
        
        Args:
            old_seed (str): Current encryption seed
            new_seed (str): New encryption seed for migration
            
        Raises:
            KeyDerivationError: If seeds are invalid
        """
        if not old_seed or not new_seed:
            raise KeyDerivationError("Both old and new seeds must be provided")
        
        if old_seed == new_seed:
            raise KeyDerivationError("New seed must be different from old seed")
        
        self.old_crypto = SecureCrypto(old_seed)
        self.new_crypto = SecureCrypto(new_seed)
        
        secure_logger.debug("🔄 Key rotation manager initialized")
    
    def rotate_encrypted_data(self, encrypted_data: str, user_id: Optional[int] = None) -> str:
        """
        Rotate encrypted data from old key to new key
        
        Args:
            encrypted_data (str): Data encrypted with old key
            user_id (int, optional): User ID for per-user encryption
            
        Returns:
            str: Data re-encrypted with new key
            
        Raises:
            DecryptionError: If old data cannot be decrypted
            EncryptionError: If new encryption fails
        """
        try:
            # Decrypt with old key
            plaintext = self.old_crypto.decrypt(encrypted_data, user_id)
            
            # Re-encrypt with new key
            new_encrypted = self.new_crypto.encrypt(plaintext, user_id)
            
            secure_logger.debug(f"✅ Data rotated successfully for user {user_id if user_id else 'global'}")
            return new_encrypted
            
        except Exception as e:
            # SECURITY FIX: Apply crypto-specific redaction to ALL error logging
            safe_exception_msg = redact_crypto_data(str(e))
            secure_logger.error(f"CRITICAL: Key rotation failed: {safe_exception_msg}")
            raise
    
    def rotate_key(self, encrypted_data: str, user_id: Optional[int] = None) -> str:
        """
        Rotate encryption key for encrypted data (alias for rotate_encrypted_data)
        
        Args:
            encrypted_data (str): Data encrypted with old key
            user_id (int, optional): User ID for per-user encryption
            
        Returns:
            str: Data re-encrypted with new key
        """
        return self.rotate_encrypted_data(encrypted_data, user_id)

# Global crypto instance - will be initialized when Config is validated
_global_crypto: Optional[SecureCrypto] = None

def initialize_crypto(encryption_seed: str) -> None:
    """
    Initialize global crypto instance with encryption seed
    
    Args:
        encryption_seed (str): Master encryption seed
        
    Raises:
        KeyDerivationError: If initialization fails
    """
    global _global_crypto
    try:
        _global_crypto = SecureCrypto(encryption_seed)
        secure_logger.debug("🌐 Global crypto instance initialized successfully")
    except Exception as e:
        # SECURITY FIX: Apply crypto-specific redaction to ALL error logging
        safe_exception_msg = redact_crypto_data(str(e))
        secure_logger.error(f"CRITICAL: Failed to initialize global crypto: {safe_exception_msg}")
        raise

def get_crypto() -> SecureCrypto:
    """
    Get the global crypto instance
    
    Returns:
        SecureCrypto: Global crypto instance
        
    Raises:
        RuntimeError: If crypto is not initialized
    """
    if _global_crypto is None:
        raise RuntimeError("Crypto not initialized. Call initialize_crypto() first.")
    return _global_crypto

def encrypt_api_key(api_key: str, user_id: int) -> str:
    """
    Convenience function to encrypt API key with user-specific encryption
    
    Args:
        api_key (str): API key to encrypt
        user_id (int): User ID for per-user encryption
        
    Returns:
        str: Encrypted API key
        
    Raises:
        EncryptionError: If encryption fails
    """
    crypto = get_crypto()
    return crypto.encrypt(api_key, user_id)

def decrypt_api_key(encrypted_api_key: str, user_id: int) -> str:
    """
    Convenience function to decrypt API key with user-specific decryption
    
    Args:
        encrypted_api_key (str): Encrypted API key
        user_id (int): User ID for per-user decryption
        
    Returns:
        str: Decrypted API key
        
    Raises:
        DecryptionError: If decryption fails
        TamperDetectionError: If tampering is detected
    """
    crypto = get_crypto()
    return crypto.decrypt(encrypted_api_key, user_id)

def is_encrypted_data(data: str) -> bool:
    """
    Check if data is encrypted using our secure format
    
    Args:
        data (str): Data to check
        
    Returns:
        bool: True if encrypted, False otherwise
    """
    try:
        crypto = get_crypto()
        return crypto.is_encrypted(data)
    except:
        return False

# Export public API
__all__ = [
    'SecureCrypto',
    'KeyRotationManager', 
    'CryptoError',
    'EncryptionError', 
    'DecryptionError',
    'TamperDetectionError',
    'KeyDerivationError',
    'initialize_crypto',
    'get_crypto',
    'encrypt_api_key',
    'decrypt_api_key', 
    'is_encrypted_data'
]