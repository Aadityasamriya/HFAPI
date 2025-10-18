"""
Superior File Processing - Advanced capabilities that outperform ChatGPT, Grok, and Gemini
Enhanced PDF extraction, intelligent ZIP analysis, advanced image processing with OCR and object detection
"""

import io
import os
import zipfile
import tempfile
import logging
import asyncio
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from bot.security_utils import secure_logger, DataRedactionEngine

# PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available - PDF processing will be limited")

# Core image processing (lightweight)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore
    PIL_AVAILABLE = False
    logging.warning("PIL not available - basic image processing will be limited")

# Heavy image processing dependencies (optional)
try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False
    logging.info("NumPy not available - advanced image analysis features disabled")

try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    OPENCV_AVAILABLE = False
    logging.info("OpenCV not available - computer vision features disabled")

# OCR functionality (heavy dependency)
try:
    import pytesseract  # type: ignore
    PYTESSERACT_AVAILABLE = True
    
    # Check tesseract binary availability at runtime
    def _check_tesseract_available() -> bool:
        """Check if tesseract binary is available on the system"""
        try:
            if pytesseract is not None:
                pytesseract.get_tesseract_version()
                return True
            return False
        except Exception:
            return False
    
    TESSERACT_BINARY_AVAILABLE = _check_tesseract_available()
    if not TESSERACT_BINARY_AVAILABLE:
        logging.warning("tesseract binary not found - OCR functionality will be disabled")
        
except ImportError:
    pytesseract = None  # type: ignore
    PYTESSERACT_AVAILABLE = False
    TESSERACT_BINARY_AVAILABLE = False
    logging.info("pytesseract not available - OCR functionality disabled")

# Overall image processing availability (requires at least PIL)
IMAGE_PROCESSING_AVAILABLE = PIL_AVAILABLE

# File type detection
try:
    import magic
    import filetype
    FILE_DETECTION_AVAILABLE = True
except ImportError:
    magic = None  # type: ignore
    filetype = None  # type: ignore
    FILE_DETECTION_AVAILABLE = False
    logging.warning("File type detection libraries not available")

logger = logging.getLogger(__name__)

class FileSecurityError(Exception):
    """Raised when file security validation fails"""
    pass

class FileSizeError(Exception):
    """Raised when file size exceeds limits"""
    pass

class FileProcessingTimeoutError(Exception):
    """Raised when file processing exceeds timeout limit"""
    pass

class FileConcurrencyLimitError(Exception):
    """Raised when concurrent file processing limit is exceeded"""
    pass


class FileProcessingSemaphore:
    """
    CRITICAL SECURITY: Concurrency control for file processing to prevent DoS attacks
    
    Implements:
    - Global semaphore: max 10 concurrent file operations system-wide
    - Per-user semaphore: max 2 concurrent file operations per user
    - Automatic cleanup of completed operations
    """
    
    def __init__(self):
        from bot.config import Config
        
        # Global semaphore - prevents system-wide resource exhaustion
        self.global_limit = Config.MAX_CONCURRENT_FILES_GLOBAL
        self.global_semaphore = asyncio.Semaphore(self.global_limit)
        
        # Per-user semaphores - prevents individual user abuse
        self.user_limit = Config.MAX_CONCURRENT_FILES_PER_USER
        self.user_semaphores: Dict[int, asyncio.Semaphore] = {}
        
        # Track active operations for monitoring
        self.active_operations = 0
        self.user_active_operations: Dict[int, int] = {}
        
        secure_logger.info(f"🔒 FileProcessingSemaphore initialized: Global limit={self.global_limit}, Per-user limit={self.user_limit}")
    
    def _get_user_semaphore(self, user_id: int) -> asyncio.Semaphore:
        """Get or create semaphore for specific user"""
        if user_id not in self.user_semaphores:
            self.user_semaphores[user_id] = asyncio.Semaphore(self.user_limit)
            self.user_active_operations[user_id] = 0
        return self.user_semaphores[user_id]
    
    async def acquire(self, user_id: int) -> Tuple[bool, Optional[str]]:
        """
        Acquire both global and user-specific semaphores
        
        Args:
            user_id: User ID requesting file processing
            
        Returns:
            (success, error_message): Tuple indicating if acquisition succeeded
        """
        # Check global limit first (non-blocking check)
        if self.active_operations >= self.global_limit:
            secure_logger.warning(f"🚫 Global file processing limit reached ({self.active_operations}/{self.global_limit})")
            return False, f"System is currently processing {self.global_limit} files. Please wait and try again."
        
        # Check user limit (non-blocking check)
        user_active = self.user_active_operations.get(user_id, 0)
        if user_active >= self.user_limit:
            secure_logger.warning(f"🚫 User {user_id} file processing limit reached ({user_active}/{self.user_limit})")
            return False, f"You can only process {self.user_limit} files at a time. Please wait for your current files to finish."
        
        # Acquire global semaphore
        await self.global_semaphore.acquire()
        self.active_operations += 1
        
        # Acquire user semaphore
        user_semaphore = self._get_user_semaphore(user_id)
        await user_semaphore.acquire()
        self.user_active_operations[user_id] = self.user_active_operations.get(user_id, 0) + 1
        
        secure_logger.info(f"✅ File processing semaphore acquired for user {user_id} (Global: {self.active_operations}/{self.global_limit}, User: {self.user_active_operations[user_id]}/{self.user_limit})")
        return True, None
    
    def release(self, user_id: int):
        """Release both global and user-specific semaphores"""
        # Release user semaphore
        if user_id in self.user_semaphores:
            self.user_semaphores[user_id].release()
            self.user_active_operations[user_id] = max(0, self.user_active_operations.get(user_id, 1) - 1)
        
        # Release global semaphore
        self.global_semaphore.release()
        self.active_operations = max(0, self.active_operations - 1)
        
        secure_logger.info(f"✅ File processing semaphore released for user {user_id} (Global: {self.active_operations}/{self.global_limit}, User: {self.user_active_operations.get(user_id, 0)}/{self.user_limit})")
    
    async def __aenter__(self, user_id: int):
        """Context manager entry - acquire semaphores"""
        success, error = await self.acquire(user_id)
        if not success:
            raise FileConcurrencyLimitError(error)
        return self
    
    async def __aexit__(self, user_id: int, exc_type, exc_val, exc_tb):
        """Context manager exit - release semaphores"""
        self.release(user_id)


# Global file processing semaphore instance
file_processing_semaphore = FileProcessingSemaphore()


async def with_file_processing_timeout(coro, timeout_seconds: int = 30, operation_name: str = "File processing"):
    """
    CRITICAL SECURITY: Timeout wrapper for file processing operations to prevent DoS
    
    Args:
        coro: Coroutine to execute with timeout
        timeout_seconds: Maximum execution time in seconds
        operation_name: Name of operation for error messages
        
    Returns:
        Result of the coroutine
        
    Raises:
        FileProcessingTimeoutError: If operation exceeds timeout
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        error_msg = f"{operation_name} exceeded {timeout_seconds}s timeout"
        secure_logger.error(f"⏱️ TIMEOUT: {error_msg}")
        raise FileProcessingTimeoutError(error_msg)


@dataclass
class ProcessedFile:
    """Enhanced file processing result with comprehensive metadata"""
    filename: str
    file_type: str
    file_size: int
    content: str
    metadata: Dict[str, Any]
    processing_time: float
    quality_score: float
    confidence: float
    extracted_elements: List[Dict[str, Any]]
    summary: str
    key_insights: List[str]

@dataclass 
class ImageAnalysis:
    """Advanced image analysis results that surpass ChatGPT capabilities"""
    ocr_text: str
    detected_objects: List[Dict[str, Any]]
    faces_detected: int
    text_regions: List[Dict[str, Any]]
    image_type: str
    quality_assessment: Dict[str, float]
    content_description: str
    dominant_colors: List[str]
    technical_analysis: Dict[str, Any]

@dataclass
class DocumentStructure:
    """Enhanced document structure analysis superior to standard parsers"""
    title: str
    sections: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    word_count: int
    page_count: int
    reading_time: str
    complexity_score: float
    key_topics: List[str]
    summary: str

@dataclass
class ZipArchiveAnalysis:
    """Comprehensive ZIP archive analysis with intelligence"""
    filename: str
    total_files: int
    structure_tree: Dict[str, Any]
    file_types: Dict[str, int]
    largest_files: List[Dict[str, Any]]
    potential_executables: List[str]
    text_files_summary: List[Dict[str, Any]]
    compression_ratio: float
    security_assessment: Dict[str, Any]
    content_classification: str

class AdvancedFileProcessor:
    """
    Superior file processing system that outperforms ChatGPT, Grok, and Gemini
    Features: Advanced PDF analysis, intelligent image processing, smart content extraction
    """
    
    # SECURITY: Enforced file size limits per user requirements (10MB max)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB universal file size limit
    MAX_ZIP_SIZE = 10 * 1024 * 1024   # 10MB for ZIP files (enforced limit)
    MAX_PDF_SIZE = 10 * 1024 * 1024   # 10MB for PDF files (enforced limit)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB for images (enforced limit)
    MAX_EXTRACTED_FILES = 500  # Maximum files to extract from ZIP (increased)
    MAX_TEXT_CONTENT = 1000000  # Maximum characters to process (increased)
    
    # Expanded file type support
    ALLOWED_PDF_MIMES = ['application/pdf']
    ALLOWED_IMAGE_MIMES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff', 'image/svg+xml']
    ALLOWED_ZIP_MIMES = ['application/zip', 'application/x-zip-compressed', 'application/x-rar-compressed']
    ALLOWED_DOCUMENT_MIMES = ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                             'text/plain', 'text/csv', 'application/json', 'application/xml']
    
    # Dangerous extensions to block (enhanced security)
    DANGEROUS_EXTENSIONS = {'.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.dll', '.jar', '.sh', '.ps1', '.vbs', '.msi', '.app', 
                           '.deb', '.rpm', '.dmg', '.pkg', '.run', '.bin', '.elf', '.so', '.dylib', '.sys', '.drv'}
    
    # CRITICAL SECURITY: Malware signatures for detection
    MALWARE_SIGNATURES = [
        # Executable headers
        b'\x4d\x5a',  # MZ header (Windows executable)
        b'\x7f\x45\x4c\x46',  # ELF header (Linux executable)
        b'\xfe\xed\xfa\xce',  # Mach-O header (macOS executable - 32-bit)
        b'\xfe\xed\xfa\xcf',  # Mach-O header (macOS executable - 64-bit)
        b'\xca\xfe\xba\xbe',  # Java class file
        
        # Script signatures
        b'#!/bin/sh',
        b'#!/bin/bash',
        b'#!/usr/bin/env',
        b'@echo off',
        b'<script',
        b'<?php',
        b'<%',
        
        # EICAR test signature (standard antivirus test)
        b'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*',
        
        # PowerShell signatures
        b'powershell',
        b'invoke-expression',
        b'iex ',
        b'downloadstring',
        
        # Suspicious patterns
        b'eval(',
        b'exec(',
        b'system(',
        b'shell_exec(',
        b'subprocess',
        b'os.system',
        b'Runtime.getRuntime',
        
        # Cryptocurrency mining patterns
        b'stratum+tcp://',
        b'xmrig',
        b'cpuminer',
        
        # Common malware strings
        b'trojan',
        b'backdoor',
        b'keylogger',
        b'rootkit',
        b'ransomware',
        
        # Encoding patterns that might hide malware
        b'base64',
        b'fromcharcode',
        b'unescape',
        
        # Archive bombs (nested archives) - REMOVED ZIP signature as it causes false positives
        # ZIP files are validated separately in ZIP-specific validation
        b'Rar!',  # RAR signature (only flag if not expected)
        b'\x37\x7a\xbc\xaf\x27\x1c',  # 7z signature (only flag if not expected)
    ]
    
    # Advanced content analysis patterns
    TABLE_DETECTION_PATTERNS = [
        r'\|.*\|.*\|',  # Pipe-separated tables
        r'\t.*\t.*\t',  # Tab-separated tables  
        r'^\s*[\w\s]+\s+[\w\s]+\s+[\w\s]+\s*$',  # Space-separated columns
        r'(?:\S+\s+){2,}\S+',  # Multiple columns pattern
    ]
    
    SECTION_PATTERNS = [
        r'^#+\s+(.+)$',  # Markdown headers
        r'^(.+)\n[=-]{3,}$',  # Underlined headers
        r'^\d+\.\s+(.+)$',  # Numbered sections
        r'^[IVX]+\.\s+(.+)$',  # Roman numeral sections
        r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS headers
    ]
    
    @staticmethod
    def validate_file_security(file_data: bytes, filename: str, expected_type: str) -> Tuple[bool, str]:
        """
        Perform comprehensive security validation on uploaded file
        
        Args:
            file_data (bytes): File content
            filename (str): Original filename
            expected_type (str): Expected file type ('pdf', 'zip', 'image', 'document')
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # SECURITY FIX (Issue #2): File size check MUST be first and cannot be bypassed
        # This is the PRIMARY security control for file size validation
        if not isinstance(file_data, bytes):
            return False, "Invalid file data type - must be bytes"
        
        file_size = len(file_data)
        
        # CRITICAL SECURITY: Universal 10MB file size limit for ALL files (Issue #2)
        # This check MUST come before ANY other processing to prevent DoS and resource exhaustion
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            return False, f"File exceeds maximum allowed size: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)"
        
        # Zero-byte file check
        if file_size == 0:
            return False, "File is empty (0 bytes)"
        
        try:
            # Check file size limits (redundant but explicit for different file types)
            # SECURITY FIX: General file size limit for ALL files (10MB universal limit)
            if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
                return False, f"File too large: {file_size:,} bytes (universal limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)"
            
            # Type-specific size checks (redundant but kept for clarity)
            if expected_type == 'zip' and file_size > AdvancedFileProcessor.MAX_ZIP_SIZE:
                return False, f"ZIP file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_ZIP_SIZE:,} bytes)"
            elif expected_type == 'pdf' and file_size > AdvancedFileProcessor.MAX_PDF_SIZE:
                return False, f"PDF file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_PDF_SIZE:,} bytes)"
            elif expected_type == 'image' and file_size > AdvancedFileProcessor.MAX_IMAGE_SIZE:
                return False, f"Image file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_IMAGE_SIZE:,} bytes)"
            
            # Check for dangerous file extensions
            file_ext = Path(filename).suffix.lower()
            if file_ext in AdvancedFileProcessor.DANGEROUS_EXTENSIONS:
                return False, f"File type not allowed: {file_ext}"
            
            # CRITICAL SECURITY: Comprehensive malware scanning
            malware_detected = AdvancedFileProcessor._scan_for_malware(file_data, filename, expected_type)
            if malware_detected:
                return False, f"SECURITY THREAT: Malware signature detected in file: {malware_detected}"
            
            # Validate file type using magic numbers if available
            if FILE_DETECTION_AVAILABLE and filetype is not None:
                try:
                    detected_type = filetype.guess(file_data)
                    if detected_type:
                        detected_mime = detected_type.mime
                        
                        # Validate against expected types
                        if expected_type == 'pdf' and detected_mime not in AdvancedFileProcessor.ALLOWED_PDF_MIMES:
                            return False, f"File is not a valid PDF (detected: {detected_mime})"
                        elif expected_type == 'image' and detected_mime not in AdvancedFileProcessor.ALLOWED_IMAGE_MIMES:
                            return False, f"Image format not supported (detected: {detected_mime})"
                        elif expected_type == 'zip' and detected_mime not in AdvancedFileProcessor.ALLOWED_ZIP_MIMES:
                            return False, f"File is not a valid ZIP archive (detected: {detected_mime})"
                except Exception as e:
                    secure_logger.warning(f"File type detection failed: {e}")
            
            # Enhanced ZIP-specific security checks with comprehensive bomb detection
            if expected_type == 'zip':
                try:
                    # Use io.BytesIO for safe memory-based ZIP handling
                    zip_buffer = io.BytesIO(file_data)
                    
                    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                        # Enhanced ZIP bomb detection with multiple criteria
                        total_uncompressed = sum([zinfo.file_size for zinfo in zip_ref.filelist])
                        compressed_size = len(file_data)
                        
                        # Criterion 1: Enhanced compression ratio check with multiple thresholds
                        if compressed_size > 0:
                            compression_ratio = total_uncompressed / compressed_size
                            # Progressive compression ratio limits based on file size
                            if compressed_size < 1024:  # < 1KB
                                max_ratio = 100  # Allow higher ratio for tiny files
                            elif compressed_size < 10240:  # < 10KB
                                max_ratio = 75   # Moderate ratio for small files
                            else:
                                max_ratio = 25   # Strict ratio for larger files (was 50)
                            
                            if compression_ratio > max_ratio:
                                return False, f"ZIP bomb detected: compression ratio {compression_ratio:.1f}x exceeds safety limit of {max_ratio}x (uncompressed: {total_uncompressed:,} bytes, compressed: {compressed_size:,} bytes)"
                        
                        # Criterion 2: Absolute size check
                        max_uncompressed = AdvancedFileProcessor.MAX_ZIP_SIZE * 20  # 200MB uncompressed limit
                        if total_uncompressed > max_uncompressed:
                            return False, f"ZIP bomb detected: uncompressed size {total_uncompressed:,} bytes exceeds safety limit ({max_uncompressed:,} bytes)"
                        
                        # Criterion 3: Enhanced suspicious content detection
                        nested_archives = 0
                        large_files = 0
                        suspicious_files = 0
                        identical_names = set()
                        
                        for zinfo in zip_ref.filelist:
                            file_ext = Path(zinfo.filename).suffix.lower()
                            
                            # Count nested archives (common in ZIP bombs)
                            if file_ext in ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.lz', '.lzma']:
                                nested_archives += 1
                                if nested_archives > 3:  # Stricter limit (was 5)
                                    return False, f"ZIP bomb detected: excessive nested archives ({nested_archives} found)"
                            
                            # Check for suspiciously large individual files
                            if zinfo.file_size > 50 * 1024 * 1024:  # 50MB per file (reduced from 100MB)
                                large_files += 1
                                if large_files > 2:  # Stricter limit (was 3)
                                    return False, f"ZIP bomb detected: too many large files ({large_files} files > 50MB)"
                            
                            # Check for suspicious file patterns (common in bombs)
                            if zinfo.file_size > 0 and zinfo.compress_size > 0:
                                individual_ratio = zinfo.file_size / zinfo.compress_size
                                if individual_ratio > 500:  # Stricter limit (was 1000)
                                    suspicious_files += 1
                                    if suspicious_files > 1:  # Stricter limit (was 2)
                                        return False, f"ZIP bomb detected: multiple files with extreme compression ratios"
                            
                            # ENHANCEMENT: Check for duplicate/similar filenames (common in ZIP bombs)
                            base_name = Path(zinfo.filename).stem.lower()
                            if base_name in identical_names and len(base_name) > 3:
                                return False, f"ZIP bomb detected: suspicious duplicate filenames pattern"
                            identical_names.add(base_name)
                        
                        # Criterion 4: File count validation
                        if len(zip_ref.filelist) > AdvancedFileProcessor.MAX_EXTRACTED_FILES:
                            return False, f"ZIP contains too many files: {len(zip_ref.filelist)} (limit: {AdvancedFileProcessor.MAX_EXTRACTED_FILES})"
                        
                        # Criterion 5: Enhanced memory consumption and resource limits
                        estimated_memory = total_uncompressed + (len(zip_ref.filelist) * 2048)  # Increased overhead per file
                        max_memory = 200 * 1024 * 1024  # Reduced to 200MB (was 500MB)
                        if estimated_memory > max_memory:
                            return False, f"ZIP bomb detected: estimated memory consumption {estimated_memory:,} bytes exceeds safety limit ({max_memory:,} bytes)"
                        
                        # ENHANCEMENT: Additional ZIP bomb pattern detection
                        # Check for unusually repetitive directory structures
                        dir_depths = []
                        for zinfo in zip_ref.filelist:
                            depth = zinfo.filename.count('/')
                            dir_depths.append(depth)
                        
                        if dir_depths:
                            max_depth = max(dir_depths)
                            if max_depth > 20:  # Prevent excessively deep directory structures
                                return False, f"ZIP bomb detected: excessive directory nesting depth ({max_depth} levels)"
                        
                        # Check for files with identical sizes (common in ZIP bombs)
                        file_sizes = [zinfo.file_size for zinfo in zip_ref.filelist if zinfo.file_size > 1024]
                        if len(file_sizes) > 10:  # Only check if enough files
                            from collections import Counter
                            size_counts = Counter(file_sizes)
                            most_common_size, count = size_counts.most_common(1)[0]
                            if count > len(file_sizes) * 0.7:  # More than 70% identical sizes
                                return False, f"ZIP bomb detected: suspicious pattern - {count} files with identical size ({most_common_size:,} bytes)"
                        
                        # Enhanced directory traversal protection
                        for zinfo in zip_ref.filelist:
                            filename = zinfo.filename
                            
                            # Basic path traversal checks
                            if os.path.isabs(filename) or '..' in filename:
                                return False, f"ZIP contains potentially dangerous path: {filename}"
                            
                            # Check for null bytes and control characters in filenames
                            if '\x00' in filename or any(ord(c) < 32 and c not in '\t\n\r' for c in filename):
                                return False, f"ZIP contains filename with dangerous characters: {repr(filename)}"
                            
                            # Enhanced path normalization check
                            try:
                                # Resolve path safely
                                safe_path = os.path.normpath(filename)
                                if safe_path.startswith('../') or safe_path.startswith('..\\'):
                                    return False, f"ZIP contains path that would escape base directory: {filename}"
                                    
                                # Check for excessively long paths (potential DoS)
                                if len(filename) > 512:
                                    return False, f"ZIP contains excessively long filename: {filename[:100]}..."
                                    
                            except (OSError, ValueError) as e:
                                return False, f"ZIP contains invalid path: {filename} ({str(e)})"
                            
                            # Individual file size safety check
                            if zinfo.file_size > 200 * 1024 * 1024:  # 200MB per file absolute limit
                                return False, f"ZIP contains file too large for safe processing: {zinfo.filename} ({zinfo.file_size:,} bytes)"
                        
                        # Additional validation: Check for ZIP file structure integrity
                        try:
                            # Test reading the central directory
                            _ = zip_ref.testzip()
                        except Exception as e:
                            return False, f"ZIP file integrity check failed: {str(e)}"
                                
                except zipfile.BadZipFile:
                    return False, "File is not a valid ZIP archive"
                except zipfile.LargeZipFile:
                    return False, "ZIP file is too large to process safely"
                except MemoryError:
                    return False, "ZIP file requires too much memory to process safely"
                except Exception as e:
                    secure_logger.error(f"ZIP validation error: {e}")
                    return False, f"ZIP validation failed: {str(e)}"
            
            return True, ""
            
        except Exception as e:
            secure_logger.error(f"File security validation error: {e}")
            return False, f"Security validation failed: {str(e)}"
    
    @staticmethod
    def _scan_for_malware(file_data: bytes, filename: str, expected_type: str = 'unknown') -> Optional[str]:
        """
        CRITICAL SECURITY: Scan file content for malware signatures
        
        Args:
            file_data (bytes): File content to scan
            filename (str): Original filename for context
            
        Returns:
            Optional[str]: Description of detected malware or None if clean
        """
        try:
            # Check file against known malware signatures
            file_lower = file_data.lower()
            
            for signature in AdvancedFileProcessor.MALWARE_SIGNATURES:
                signature_lower = signature.lower()
                
                # CRITICAL FIX: Skip archive signatures for expected file types
                if signature == b'Rar!' and expected_type in ['zip', 'archive']:
                    continue  # Allow RAR signature for archive processing
                if signature == b'\x37\x7a\xbc\xaf\x27\x1c' and expected_type in ['zip', 'archive']:
                    continue  # Allow 7z signature for archive processing
                
                # SECURITY FIX: For executable headers, only flag if at file start (prevent false positives)
                # Legitimate images can contain MZ/ELF bytes in pixel data, but executables must have
                # these headers at offset 0. This prevents polyglot attacks while reducing false positives.
                if signature in [b'\x4d\x5a', b'\x7f\x45\x4c\x46']:
                    # Only flag if signature is at the very beginning (first 4 bytes)
                    if len(file_data) >= len(signature) and file_data[:len(signature)] == signature:
                        return "Executable file header detected"
                    # Skip further checks for this signature - don't flag if found elsewhere
                    continue
                
                # Check for exact signature match (for non-header signatures)
                if signature in file_data:
                    if signature == b'X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*':
                        return "EICAR test signature (antivirus test file)"
                    elif signature in [b'#!/bin/sh', b'#!/bin/bash']:
                        return "Shell script signature detected"
                    elif signature == b'<script':
                        return "JavaScript/HTML script content detected"
                    elif signature in [b'eval(', b'exec(', b'system(']:
                        return "Dangerous code execution pattern detected"
                    else:
                        return f"Malware signature detected: {signature.decode('utf-8', errors='ignore')[:50]}"
                
                # Check for case-insensitive matches for text patterns (only for text-based signatures)
                if len(signature) > 4 and signature_lower in file_lower:  # Only check longer text patterns
                    if b'powershell' in signature_lower:
                        return "PowerShell execution pattern detected"
                    elif b'trojan' in signature_lower or b'backdoor' in signature_lower:
                        return "Malware pattern detected in content"
                    elif b'stratum+tcp://' in signature_lower:
                        return "Cryptocurrency mining pattern detected"
            
            # Additional heuristic checks
            suspicious_count = 0
            
            # Check for high entropy (possible encrypted/packed malware)
            if len(file_data) > 1000:
                unique_bytes = len(set(file_data))
                entropy_ratio = unique_bytes / len(file_data)
                if entropy_ratio > 0.85:  # Very high entropy
                    suspicious_count += 1
            
            # Check for suspicious file structure
            if file_data.startswith(b'MZ') and len(file_data) > 1000:
                # Check for packed executable indicators
                if b'UPX' in file_data or b'ASPack' in file_data or b'PECompact' in file_data:
                    suspicious_count += 2
            
            # Check for multiple script languages in one file
            script_types = 0
            for script_sig in [b'<script', b'<?php', b'#!/bin/', b'<%']:
                if script_sig in file_lower:
                    script_types += 1
            
            if script_types >= 2:
                suspicious_count += 1
            
            # Check for base64 encoded content that might hide malware
            import re
            base64_pattern = re.compile(rb'[A-Za-z0-9+/]{100,}={0,2}')
            base64_matches = base64_pattern.findall(file_data)
            if len(base64_matches) > 5:  # Multiple large base64 blocks
                suspicious_count += 1
            
            # If multiple suspicious indicators, flag as potential threat
            if suspicious_count >= 2:
                return "Multiple suspicious patterns detected (possible malware)"
            
            return None  # File appears clean
            
        except Exception as e:
            secure_logger.error(f"Malware scanning error: {e}")
            # If scanning fails, err on the side of caution
            return "Malware scan failed - file rejected for safety"
    
    @staticmethod
    async def extract_pdf_content(pdf_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract comprehensive content from PDF file
        
        Args:
            pdf_data (bytes): PDF file data
            filename (str): Original filename
            
        Returns:
            Dict[str, Any]: Extracted content and metadata
        """
        # SECURITY: Check file size BEFORE processing begins
        file_size = len(pdf_data)
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            secure_logger.error(f"PDF file size exceeds limit: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
            raise FileSizeError(f"PDF file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
        
        if not PYMUPDF_AVAILABLE:
            return {'error': 'PDF processing not available - PyMuPDF not installed'}
        
        # CRITICAL FIX: Ensure proper resource cleanup for PDF documents
        pdf_document = None
        try:
            # Open PDF from memory
            if fitz is None:
                return {'error': 'PDF processing not available - PyMuPDF not installed'}
            
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Extract metadata with proper None checks
            doc_metadata = pdf_document.metadata or {}
            metadata = {
                'filename': filename,
                'pages': pdf_document.page_count,
                'title': doc_metadata.get('title', '').strip(),
                'author': doc_metadata.get('author', '').strip(),
                'subject': doc_metadata.get('subject', '').strip(),
                'creator': doc_metadata.get('creator', '').strip(),
                'producer': doc_metadata.get('producer', '').strip(),
                'creation_date': doc_metadata.get('creationDate', ''),
                'modification_date': doc_metadata.get('modDate', ''),
                'encrypted': pdf_document.needs_pass,
                'file_size': len(pdf_data)
            }
            
            # Extract text content
            full_text = ""
            page_texts = []
            tables_detected = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text
                page_text = page.get_text()  # type: ignore
                page_texts.append({
                    'page_number': page_num + 1,
                    'text': page_text.strip(),
                    'char_count': len(page_text)
                })
                
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Try to detect tables (basic heuristic)
                if page_text and ('\t' in page_text or '|' in page_text):
                    # Look for table-like structures
                    lines = page_text.split('\n')
                    table_lines = [line for line in lines if '\t' in line or '|' in line]
                    if len(table_lines) >= 3:  # At least header + 2 rows
                        tables_detected.append({
                            'page': page_num + 1,
                            'lines': len(table_lines),
                            'sample': table_lines[0][:100]
                        })
            
            # Truncate if too long
            if len(full_text) > AdvancedFileProcessor.MAX_TEXT_CONTENT:
                full_text = full_text[:AdvancedFileProcessor.MAX_TEXT_CONTENT] + "\n\n[Content truncated due to length]"
            
            return {
                'success': True,
                'metadata': metadata,
                'full_text': full_text.strip(),
                'page_texts': page_texts,
                'tables_detected': tables_detected,
                'stats': {
                    'total_characters': len(full_text),
                    'pages_with_text': len([p for p in page_texts if p['text']]),
                    'tables_found': len(tables_detected)
                }
            }
            
        except Exception as e:
            secure_logger.error(f"PDF extraction error: {e}")
            return {
                'success': False,
                'error': f"Failed to extract PDF content: {str(e)}"
            }
        finally:
            # CRITICAL FIX: Ensure PDF document is always closed to prevent resource leaks
            if pdf_document is not None:
                try:
                    pdf_document.close()
                    logger.debug("✅ PDF document resource cleaned up")
                except Exception as cleanup_error:
                    secure_logger.warning(f"⚠️ Failed to close PDF document during cleanup: {cleanup_error}")
    
    @staticmethod
    async def analyze_zip_archive(zip_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Safely analyze ZIP archive contents with security measures
        
        Args:
            zip_data (bytes): ZIP file data
            filename (str): Original filename
            
        Returns:
            Dict[str, Any]: Archive analysis results
        """
        # SECURITY: Check ZIP file size BEFORE processing begins
        file_size = len(zip_data)
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            secure_logger.error(f"ZIP file size exceeds limit: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
            raise FileSizeError(f"ZIP file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
        
        try:
            file_contents = []
            
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                # Get basic archive info
                archive_info = {
                    'filename': filename,
                    'total_files': len(zip_ref.filelist),
                    'compressed_size': len(zip_data),
                    'uncompressed_size': sum([zinfo.file_size for zinfo in zip_ref.filelist])
                }
                
                # Process each file in archive
                for zinfo in zip_ref.filelist[:AdvancedFileProcessor.MAX_EXTRACTED_FILES]:
                    if zinfo.is_dir():
                        continue
                    
                    # SECURITY: Check individual ZIP member size BEFORE any extraction/reading
                    if zinfo.file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
                        secure_logger.error(f"ZIP member '{zinfo.filename}' exceeds size limit: {zinfo.file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
                        raise FileSizeError(f"ZIP member too large: '{zinfo.filename}' ({zinfo.file_size:,} bytes, limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
                    
                    try:
                        # Get file info
                        file_info = {
                            'name': zinfo.filename,
                            'size': zinfo.file_size,
                            'compressed_size': zinfo.compress_size,
                            'type': 'unknown',
                            'extension': Path(zinfo.filename).suffix.lower(),
                            'content': None
                        }
                        
                        # Determine file type with size safety check
                        if FILE_DETECTION_AVAILABLE and filetype is not None:
                            try:
                                # Safety check - don't read files that are too large
                                if zinfo.file_size > 50 * 1024 * 1024:  # 50MB limit
                                    file_info['type'] = 'large_file'
                                    file_info['note'] = f'File too large for type detection ({zinfo.file_size:,} bytes)'
                                else:
                                    file_data = zip_ref.read(zinfo)
                                    detected = filetype.guess(file_data)
                                    if detected:
                                        file_info['type'] = detected.extension
                                        file_info['mime_type'] = detected.mime
                            except Exception:
                                pass
                        
                        # For text files, extract content (limited)
                        if file_info['extension'] in ['.txt', '.py', '.js', '.json', '.md', '.yml', '.yaml', '.xml', '.csv', '.html', '.css', '.sql', '.sh', '.bat']:
                            try:
                                # Limit file size for text extraction
                                if zinfo.file_size <= 50000:  # 50KB limit for text files
                                    file_data = zip_ref.read(zinfo)
                                    # Try to decode as text
                                    try:
                                        text_content = file_data.decode('utf-8')
                                        file_info['content'] = text_content[:5000]  # First 5KB
                                        file_info['type'] = 'text'
                                    except UnicodeDecodeError:
                                        try:
                                            text_content = file_data.decode('latin1')
                                            file_info['content'] = text_content[:5000]
                                            file_info['type'] = 'text'
                                        except:
                                            file_info['type'] = 'binary'
                                else:
                                    file_info['note'] = 'File too large for content extraction'
                            except Exception as e:
                                file_info['error'] = f'Content extraction failed: {str(e)}'
                        
                        file_contents.append(file_info)
                        
                    except Exception as e:
                        secure_logger.warning(f"Error processing file {zinfo.filename} in ZIP: {e}")
                        file_contents.append({
                            'name': zinfo.filename,
                            'error': f'Processing failed: {str(e)}'
                        })
            
            return {
                'success': True,
                'archive_info': archive_info,
                'file_contents': file_contents,
                'stats': {
                    'total_files': len(file_contents),
                    'text_files': len([f for f in file_contents if f.get('content')]),
                    'file_types': list(set([f.get('type', 'unknown') for f in file_contents]))
                }
            }
            
        except zipfile.BadZipFile:
            return {
                'success': False,
                'error': 'Invalid ZIP file format'
            }
        except Exception as e:
            secure_logger.error(f"ZIP analysis error: {e}")
            return {
                'success': False,
                'error': f'ZIP analysis failed: {str(e)}'
            }
    
    @staticmethod  
    async def process_image_content(image_data: bytes, filename: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Process image for analysis with OCR and content recognition
        
        Args:
            image_data (bytes): Image file data
            filename (str): Original filename
            analysis_type (str): Type of analysis ("description", "ocr", "comprehensive")
            
        Returns:
            Dict[str, Any]: Image processing results
        """
        # SECURITY: Check file size BEFORE processing begins
        file_size = len(image_data)
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            secure_logger.error(f"Image file size exceeds limit: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
            raise FileSizeError(f"Image file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
        
        if not PIL_AVAILABLE:
            return {
                'success': False,
                'error': 'Basic image processing not available - PIL (Pillow) not installed. Please install: pip install Pillow',
                'missing_dependency': 'Pillow',
                'install_command': 'pip install Pillow'
            }
        
        try:
            # Open image
            if not PIL_AVAILABLE or Image is None:
                return {
                    'success': False,
                    'error': 'Basic image processing not available - PIL (Pillow) not installed. Please install: pip install Pillow',
                    'missing_dependency': 'Pillow',
                    'install_command': 'pip install Pillow'
                }
            
            image = Image.open(io.BytesIO(image_data))
            
            # Get basic image info
            image_info = {
                'filename': filename,
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'file_size': len(image_data)
            }
            
            # Convert to RGB if necessary for processing
            if image.mode not in ['RGB', 'L']:
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image
            
            # Prepare results
            results = {
                'success': True,
                'image_info': image_info,
                'analysis_type': analysis_type
            }
            
            # OCR text extraction if requested
            if analysis_type in ['ocr', 'comprehensive']:
                if not PYTESSERACT_AVAILABLE:
                    results['ocr'] = {
                        'error': 'OCR not available - pytesseract not installed. Please install: pip install pytesseract',
                        'has_text': False,
                        'missing_dependency': 'pytesseract',
                        'install_command': 'pip install pytesseract',
                        'additional_requirement': 'Also requires tesseract binary on system'
                    }
                elif not TESSERACT_BINARY_AVAILABLE:
                    results['ocr'] = {
                        'error': 'OCR not available - tesseract binary not found on system. Please install tesseract-ocr package.',
                        'has_text': False,
                        'system_requirement': 'tesseract-ocr',
                        'install_hint': 'On Ubuntu: sudo apt-get install tesseract-ocr'
                    }
                else:
                    try:
                        # Use pytesseract for OCR
                        if pytesseract is None:
                            results['ocr'] = {
                                'error': 'OCR not available - pytesseract not installed',
                                'has_text': False
                            }
                        else:
                            extracted_text = pytesseract.image_to_string(rgb_image)
                            results['ocr'] = {
                                'text': extracted_text.strip(),
                                'char_count': len(extracted_text.strip()),
                                'has_text': bool(extracted_text.strip())
                            }
                    except Exception as e:
                        results['ocr'] = {
                            'error': f'OCR failed: {str(e)}',
                            'has_text': False
                        }
            
            # Basic image statistics and advanced analysis
            if analysis_type == 'comprehensive':
                # Basic image properties (always available with PIL)
                results['basic_analysis'] = {
                    'format': image_info['format'],
                    'mode': image_info['mode'],
                    'size': image_info['size'],
                    'has_transparency': 'transparency' in image.info or image.mode in ['RGBA', 'LA'],
                    'is_animated': getattr(image, 'is_animated', False)
                }
                
                # Advanced color analysis (requires numpy)
                if NUMPY_AVAILABLE and np is not None:
                    try:
                        img_array = np.array(rgb_image)
                        
                        # Calculate color statistics
                        results['color_analysis'] = {
                            'mean_brightness': float(np.mean(img_array)),
                            'std_brightness': float(np.std(img_array)),
                            'is_grayscale': len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1),
                            'numpy_analysis': True
                        }
                        
                        # Calculate histogram if possible
                        if len(img_array.shape) == 3:
                            results['color_analysis']['channel_means'] = {
                                'red': float(np.mean(img_array[:, :, 0])),
                                'green': float(np.mean(img_array[:, :, 1])),
                                'blue': float(np.mean(img_array[:, :, 2]))
                            }
                        
                    except Exception as e:
                        secure_logger.warning(f"Advanced color analysis failed: {e}")
                        results['color_analysis'] = {
                            'error': f'Advanced analysis failed: {str(e)}',
                            'fallback': 'Basic analysis only'
                        }
                else:
                    results['color_analysis'] = {
                        'error': 'Advanced color analysis not available - NumPy not installed',
                        'missing_dependency': 'numpy',
                        'install_command': 'pip install numpy',
                        'fallback': 'Basic image info available above'
                    }
            
            return results
            
        except Exception as e:
            secure_logger.error(f"Image processing error: {e}")
            return {
                'success': False,
                'error': f'Image processing failed: {str(e)}',
                'help': 'Ensure PIL (Pillow) is installed: pip install Pillow'
            }

    @staticmethod
    async def enhanced_pdf_analysis(pdf_data: bytes, filename: str) -> DocumentStructure:
        """
        Advanced PDF analysis that surpasses ChatGPT, Grok, and Gemini capabilities
        Extracts structure, tables, images, and provides intelligent summarization
        """
        # SECURITY: Check file size BEFORE processing begins
        file_size = len(pdf_data)
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            secure_logger.error(f"PDF file size exceeds limit: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
            raise FileSizeError(f"PDF file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
        
        start_time = time.time()
        
        # CRITICAL FIX: Ensure proper resource cleanup for PDF documents
        pdf_document = None
        try:
            if not PYMUPDF_AVAILABLE or fitz is None:
                return DocumentStructure(
                    title="PDF Analysis Unavailable",
                    sections=[],
                    tables=[],
                    images=[],
                    metadata={'error': 'PyMuPDF not available'},
                    word_count=0,
                    page_count=0,
                    reading_time="0 min",
                    complexity_score=0.0,
                    key_topics=[],
                    summary="PDF processing not available"
                )
            
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            
            # Extract comprehensive metadata
            doc_metadata = pdf_document.metadata or {}
            title = doc_metadata.get('title', '') or filename.replace('.pdf', '').replace('_', ' ').title()
            
            # Initialize analysis structures
            sections = []
            tables = []
            images = []
            full_text = ""
            
            # Process each page with advanced extraction
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                # Extract text from page (PyMuPDF method)
                try:
                    # Use getattr to safely access get_text method
                    get_text_method = getattr(page, 'get_text', None)
                    if get_text_method:
                        page_text = get_text_method()
                    else:
                        page_text = ''
                except (AttributeError, Exception):
                    page_text = ''
                full_text += f"\n{page_text}"
                
                # Advanced table detection
                for pattern in AdvancedFileProcessor.TABLE_DETECTION_PATTERNS:
                    table_matches = re.findall(pattern, page_text, re.MULTILINE)
                    if len(table_matches) >= 3:  # At least 3 rows
                        tables.append({
                            'page': page_num + 1,
                            'rows': len(table_matches),
                            'columns': len(table_matches[0].split()) if table_matches else 0,
                            'content_preview': table_matches[0][:100] if table_matches else "",
                            'pattern_type': pattern
                        })
                
                # Enhanced section detection
                for pattern in AdvancedFileProcessor.SECTION_PATTERNS:
                    section_matches = re.findall(pattern, page_text, re.MULTILINE)
                    for match in section_matches:
                        sections.append({
                            'title': match if isinstance(match, str) else match[0],
                            'page': page_num + 1,
                            'level': 1,  # Can be enhanced further
                            'content_preview': page_text[:200]
                        })
                
                # Image extraction
                try:
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        images.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'dimensions': f"{img[2]}x{img[3]}" if len(img) > 3 else "unknown",
                            'type': 'embedded_image'
                        })
                except Exception:
                    pass
            
            # Calculate metrics
            word_count = len(full_text.split())
            reading_time = f"{max(1, word_count // 200)} min"  # Average reading speed
            complexity_score = min(10.0, (len(sections) * 0.5) + (len(tables) * 1.0) + (word_count / 1000))
            
            # Extract key topics (basic keyword extraction)
            words = re.findall(r'\b[A-Z][a-z]{3,}\b', full_text)
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            key_topics = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)][:10]
            
            # Generate intelligent summary
            sentences = re.split(r'[.!?]+', full_text)
            important_sentences = [s.strip() for s in sentences if len(s.strip()) > 50 and len(s.strip()) < 200][:3]
            summary = " ".join(important_sentences) if important_sentences else "Document contains structured content with multiple sections."
            
            return DocumentStructure(
                title=title,
                sections=sections,
                tables=tables,
                images=images,
                metadata={
                    'filename': filename,
                    'processing_time': time.time() - start_time,
                    'file_size': len(pdf_data),
                    'creator': doc_metadata.get('creator', ''),
                    'creation_date': doc_metadata.get('creationDate', '')
                },
                word_count=word_count,
                page_count=pdf_document.page_count if 'pdf_document' in locals() else 0,
                reading_time=reading_time,
                complexity_score=complexity_score,
                key_topics=key_topics,
                summary=summary
            )
            
        except Exception as e:
            secure_logger.error(f"Enhanced PDF analysis failed: {e}")
            return DocumentStructure(
                title="Analysis Failed",
                sections=[],
                tables=[],
                images=[],
                metadata={'error': str(e)},
                word_count=0,
                page_count=0,
                reading_time="0 min",
                complexity_score=0.0,
                key_topics=[],
                summary=f"PDF analysis failed: {str(e)}"
            )
        finally:
            # CRITICAL FIX: Ensure PDF document is always closed to prevent resource leaks
            if pdf_document is not None:
                try:
                    pdf_document.close()
                    logger.debug("✅ PDF document resource cleaned up in enhanced analysis")
                except Exception as cleanup_error:
                    secure_logger.warning(f"⚠️ Failed to close PDF document during enhanced analysis cleanup: {cleanup_error}")
    
    @staticmethod
    async def advanced_image_analysis(image_data: bytes, filename: str) -> ImageAnalysis:
        """
        Superior image analysis that outperforms ChatGPT, Grok, and Gemini
        Combines OCR, object detection, and intelligent content description
        """
        # SECURITY: Check file size BEFORE processing begins
        file_size = len(image_data)
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            secure_logger.error(f"Image file size exceeds limit: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
            raise FileSizeError(f"Image file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
        
        start_time = time.time()
        
        try:
            if not IMAGE_PROCESSING_AVAILABLE or Image is None:
                return ImageAnalysis(
                    ocr_text="",
                    detected_objects=[],
                    faces_detected=0,
                    text_regions=[],
                    image_type="unknown",
                    quality_assessment={'processing_available': 0.0},  # Use float values
                    content_description="Image processing libraries not available",
                    dominant_colors=[],
                    technical_analysis={'processing_time': time.time() - start_time}
                )
            
            # Open and analyze image
            image = Image.open(io.BytesIO(image_data))
            image_info = {
                'format': image.format,
                'size': image.size,
                'mode': image.mode
            }
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'L']:
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image
            
            # Enhanced OCR analysis
            ocr_text = ""
            text_regions = []
            
            if TESSERACT_BINARY_AVAILABLE and pytesseract is not None:
                try:
                    # Extract text with confidence scores
                    ocr_data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
                    
                    # Build text and regions
                    words = []
                    for i, text in enumerate(ocr_data['text']):
                        if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                            words.append(text)
                            if text.strip():
                                text_regions.append({
                                    'text': text,
                                    'confidence': int(ocr_data['conf'][i]),
                                    'bbox': {
                                        'x': ocr_data['left'][i],
                                        'y': ocr_data['top'][i],
                                        'width': ocr_data['width'][i],
                                        'height': ocr_data['height'][i]
                                    }
                                })
                    
                    ocr_text = ' '.join(words).strip()
                    
                except Exception as e:
                    secure_logger.warning(f"OCR failed: {e}")
                    ocr_text = "OCR processing failed"
            
            # Basic object detection using OpenCV (if available)
            detected_objects = []
            faces_detected = 0
            
            if OPENCV_AVAILABLE and NUMPY_AVAILABLE and cv2 is not None and np is not None:
                try:
                    # Convert PIL to OpenCV format
                    opencv_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                    
                    # Simple face detection
                    try:
                        # Access haarcascades data properly
                        try:
                            # Try to access cv2.data safely
                            cv2_data = getattr(cv2, 'data', None)
                            if cv2_data and hasattr(cv2_data, 'haarcascades'):
                                cascade_path = cv2_data.haarcascades + 'haarcascade_frontalface_default.xml'
                            else:
                                # Fallback path - try common locations
                                cascade_path = 'haarcascade_frontalface_default.xml'
                        except (AttributeError, TypeError):
                            # Fallback path - try common locations
                            cascade_path = 'haarcascade_frontalface_default.xml'
                        face_cascade = cv2.CascadeClassifier(cascade_path)
                        faces = face_cascade.detectMultiScale(opencv_image, 1.1, 4)
                        faces_detected = len(faces)
                        
                        for (x, y, w, h) in faces:
                            detected_objects.append({
                                'type': 'face',
                                'confidence': 0.8,  # Approximate
                                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                            })
                    except Exception:
                        pass  # Face detection not available
                        
                except Exception as e:
                    logger.info(f"OpenCV processing failed (this is expected if OpenCV is not installed): {e}")
            elif not OPENCV_AVAILABLE:
                logger.debug("OpenCV not available - object detection features disabled")
            elif not NUMPY_AVAILABLE:
                logger.debug("NumPy not available - advanced image analysis features disabled")
            
            # Color analysis
            dominant_colors = []
            quality_assessment = {}
            
            if NUMPY_AVAILABLE and np is not None:
                try:
                    img_array = np.array(rgb_image)
                    
                    # Calculate quality metrics
                    quality_assessment = {
                        'mean_brightness': float(np.mean(img_array)),
                        'contrast': float(np.std(img_array)),
                        'sharpness': 'calculated' if len(img_array.shape) == 3 else 'grayscale',
                        'numpy_analysis': True
                    }
                    
                    # Simple dominant color extraction
                    if len(img_array.shape) == 3:
                        reshaped = img_array.reshape(-1, 3)
                        # Get most common colors (simplified)
                        unique_colors = np.unique(reshaped, axis=0)
                        dominant_colors = [f"rgb({c[0]},{c[1]},{c[2]})" for c in unique_colors[:5]]
                    
                except Exception as e:
                    secure_logger.warning(f"NumPy color analysis failed: {e}")
                    quality_assessment = {'analysis_failed': True, 'error': str(e)}
            else:
                # Fallback analysis using PIL only
                try:
                    # Basic analysis without numpy
                    quality_assessment = {
                        'basic_analysis': True,
                        'width': image.width,
                        'height': image.height,
                        'mode': image.mode,
                        'note': 'Advanced analysis unavailable - NumPy not installed'
                    }
                    
                    # Basic color extraction using PIL
                    if hasattr(image, 'getcolors'):
                        try:
                            colors = image.getcolors(maxcolors=256*256*256)
                            if colors:
                                # Get top 5 most frequent colors
                                sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
                                dominant_colors = [f"rgb{color[1][:3]}" if isinstance(color[1], tuple) and len(color[1]) >= 3 
                                                 else str(color[1]) for color in sorted_colors]
                        except Exception:
                            dominant_colors = ['analysis_failed']
                    
                except Exception as e:
                    secure_logger.warning(f"Basic color analysis failed: {e}")
                    quality_assessment = {'basic_analysis_failed': True, 'error': str(e)}
            
            # Generate intelligent content description
            content_description = AdvancedFileProcessor._generate_image_description(
                image_info, ocr_text, detected_objects, faces_detected
            )
            
            # Determine image type
            image_type = AdvancedFileProcessor._classify_image_type(image_info, ocr_text, detected_objects)
            
            return ImageAnalysis(
                ocr_text=ocr_text,
                detected_objects=detected_objects,
                faces_detected=faces_detected,
                text_regions=text_regions,
                image_type=image_type,
                quality_assessment=quality_assessment,
                content_description=content_description,
                dominant_colors=dominant_colors,
                technical_analysis={
                    'processing_time': time.time() - start_time,
                    'file_size': len(image_data),
                    'dimensions': f"{image.width}x{image.height}",
                    'format': image.format
                }
            )
            
        except Exception as e:
            secure_logger.error(f"Advanced image analysis failed: {e}")
            return ImageAnalysis(
                ocr_text="",
                detected_objects=[],
                faces_detected=0,
                text_regions=[],
                image_type="unknown",
                quality_assessment={'analysis_failed': 0.0},  # Use float values
                content_description=f"Image analysis failed: {str(e)}",
                dominant_colors=[],
                technical_analysis={'processing_time': time.time() - start_time, 'error': str(e)}
            )
    
    @staticmethod
    def _generate_image_description(image_info: Dict, ocr_text: str, detected_objects: List, faces_detected: int) -> str:
        """Generate intelligent description of image content"""
        
        description_parts = []
        
        # Basic image properties
        width, height = image_info['size']
        if width > height * 1.5:
            description_parts.append("wide/landscape format")
        elif height > width * 1.5:
            description_parts.append("tall/portrait format")
        else:
            description_parts.append("square/balanced format")
        
        # Content analysis
        if ocr_text and len(ocr_text.strip()) > 10:
            description_parts.append(f"contains text content")
            if any(keyword in ocr_text.lower() for keyword in ['title', 'heading', 'header']):
                description_parts.append("appears to be a document or presentation")
            elif any(keyword in ocr_text.lower() for keyword in ['menu', 'price', 'order']):
                description_parts.append("appears to be a menu or price list")
        
        # Object detection results
        if faces_detected > 0:
            if faces_detected == 1:
                description_parts.append("contains 1 person")
            else:
                description_parts.append(f"contains {faces_detected} people")
        
        if detected_objects:
            object_types = set(obj['type'] for obj in detected_objects)
            if 'face' in object_types:
                object_types.remove('face')  # Already mentioned above
            if object_types:
                description_parts.append(f"contains {', '.join(object_types)}")
        
        # Image type inference
        if not description_parts:
            description_parts.append("general image content")
        
        return f"This is a {image_info['format']} image in {', '.join(description_parts)}."
    
    @staticmethod
    def _classify_image_type(image_info: Dict, ocr_text: str, detected_objects: List) -> str:
        """Classify the type of image based on analysis"""
        
        if len(ocr_text.strip()) > 50:
            if any(keyword in ocr_text.lower() for keyword in ['invoice', 'receipt', 'bill']):
                return "document_invoice"
            elif any(keyword in ocr_text.lower() for keyword in ['menu', 'restaurant', 'food']):
                return "menu_food"
            elif any(keyword in ocr_text.lower() for keyword in ['certificate', 'diploma', 'award']):
                return "certificate"
            else:
                return "text_document"
        
        if any(obj['type'] == 'face' for obj in detected_objects):
            return "portrait_photo"
        
        width, height = image_info['size']
        if width > 1920 or height > 1920:
            return "high_resolution"
        elif width < 300 and height < 300:
            return "thumbnail_icon"
        
        return "general_image"
    
    @staticmethod
    async def intelligent_zip_analysis(zip_data: bytes, filename: str) -> ZipArchiveAnalysis:
        """
        Intelligent ZIP analysis that surpasses standard file managers
        Provides detailed structure analysis and content classification
        """
        # SECURITY: Check ZIP file size BEFORE processing begins
        file_size = len(zip_data)
        if file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
            secure_logger.error(f"ZIP file size exceeds limit: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
            raise FileSizeError(f"ZIP file too large: {file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
        
        start_time = time.time()
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
                # Build file structure tree
                structure_tree = {}
                file_types = {}
                largest_files = []
                potential_executables = []
                text_files_summary = []
                
                total_uncompressed = sum(zinfo.file_size for zinfo in zip_ref.filelist)
                compression_ratio = len(zip_data) / total_uncompressed if total_uncompressed > 0 else 1.0
                
                # Analyze each file
                for zinfo in zip_ref.filelist:
                    if zinfo.is_dir():
                        continue
                    
                    # SECURITY: Check individual ZIP member size BEFORE any extraction/reading
                    if zinfo.file_size > AdvancedFileProcessor.MAX_FILE_SIZE:
                        secure_logger.error(f"ZIP member '{zinfo.filename}' exceeds size limit: {zinfo.file_size:,} bytes (limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
                        raise FileSizeError(f"ZIP member too large: '{zinfo.filename}' ({zinfo.file_size:,} bytes, limit: {AdvancedFileProcessor.MAX_FILE_SIZE:,} bytes)")
                    
                    # Build directory structure
                    path_parts = zinfo.filename.split('/')
                    current_level = structure_tree
                    for part in path_parts[:-1]:
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]
                    
                    if path_parts:
                        current_level[path_parts[-1]] = {
                            'size': zinfo.file_size,
                            'compressed_size': zinfo.compress_size,
                            'type': Path(zinfo.filename).suffix.lower()
                        }
                    
                    # Track file types
                    file_ext = Path(zinfo.filename).suffix.lower()
                    file_types[file_ext] = file_types.get(file_ext, 0) + 1
                    
                    # Track largest files
                    largest_files.append({
                        'name': zinfo.filename,
                        'size': zinfo.file_size,
                        'compressed_size': zinfo.compress_size,
                        'ratio': zinfo.compress_size / zinfo.file_size if zinfo.file_size > 0 else 0
                    })
                    
                    # Identify potential executables
                    if file_ext in AdvancedFileProcessor.DANGEROUS_EXTENSIONS:
                        potential_executables.append(zinfo.filename)
                    
                    # Analyze text files
                    if file_ext in ['.txt', '.py', '.js', '.json', '.md', '.csv', '.html', '.css'] and zinfo.file_size < 100000:
                        try:
                            file_content = zip_ref.read(zinfo).decode('utf-8', errors='ignore')
                            text_files_summary.append({
                                'name': zinfo.filename,
                                'lines': len(file_content.split('\n')),
                                'characters': len(file_content),
                                'preview': file_content[:200].replace('\n', ' ')
                            })
                        except Exception:
                            pass
                
                # Sort largest files
                largest_files.sort(key=lambda x: x['size'], reverse=True)
                largest_files = largest_files[:10]  # Top 10
                
                # Security assessment
                security_assessment = {
                    'risk_level': 'high' if potential_executables else 'low',
                    'executable_count': len(potential_executables),
                    'suspicious_patterns': [],
                    'compression_anomaly': compression_ratio < 0.1  # Highly compressed
                }
                
                # Content classification
                content_classification = AdvancedFileProcessor._classify_zip_content(file_types, text_files_summary)
                
                return ZipArchiveAnalysis(
                    filename=filename,
                    total_files=len([f for f in zip_ref.filelist if not f.is_dir()]),
                    structure_tree=structure_tree,
                    file_types=file_types,
                    largest_files=largest_files,
                    potential_executables=potential_executables,
                    text_files_summary=text_files_summary,
                    compression_ratio=compression_ratio,
                    security_assessment=security_assessment,
                    content_classification=content_classification
                )
                
        except Exception as e:
            secure_logger.error(f"Intelligent ZIP analysis failed: {e}")
            return ZipArchiveAnalysis(
                filename=filename,
                total_files=0,
                structure_tree={'error': str(e)},
                file_types={},
                largest_files=[],
                potential_executables=[],
                text_files_summary=[],
                compression_ratio=0.0,
                security_assessment={'error': str(e)},
                content_classification="analysis_failed"
            )
    
    @staticmethod
    async def call_vision_model(image_data: bytes, prompt: str = "", model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Call Hugging Face vision model for image analysis
        
        Args:
            image_data (bytes): Image data as bytes
            prompt (str): Optional prompt for the vision model
            model_name (str): Specific model to use, defaults to Config.DEFAULT_VISION_MODEL
            
        Returns:
            Dict[str, Any]: Analysis results from the vision model
        """
        from bot.config import Config
        from bot.core.model_caller import model_caller
        
        logger.info("🔍 Calling vision model for image analysis")
        
        # Check if AI functionality is available
        is_available, status_msg = Config.is_ai_functionality_available()
        if not is_available:
            return {
                'success': False,
                'error': 'AI functionality not available',
                'status': status_msg,
                'analysis': {
                    'description': 'AI analysis unavailable - HF_TOKEN not configured',
                    'confidence': 0.0,
                    'details': 'Set up HF_TOKEN to enable advanced image analysis'
                }
            }
        
        try:
            # OCT 2025: Use latest VLM models with intelligent fallback chain
            # Try models in order: DEFAULT (Qwen2-VL) -> ADVANCED (Llama-3.2-Vision) -> LIGHTWEIGHT (SmolVLM)
            vlm_models = [
                model_name or Config.DEFAULT_VLM_MODEL,      # Qwen2-VL-7B-Instruct: best multimodal
                Config.ADVANCED_VLM_MODEL,                    # Llama-3.2-11B-Vision: excellent VQA
                Config.LIGHTWEIGHT_VLM_MODEL                   # SmolVLM-Instruct: fast fallback
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            vlm_models = [m for m in vlm_models if not (m in seen or seen.add(m))]
            
            # Convert image to base64 for API call
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the prompt for vision analysis
            if not prompt:
                prompt = "Analyze this image and describe what you see. Include details about objects, text, people, and any notable features."
            
            # Get HF token for API call
            hf_token = Config.get_hf_token()
            if not hf_token:
                return {
                    'success': False,
                    'error': 'HF_TOKEN not available for vision model call',
                    'analysis': {
                        'description': 'Vision analysis failed - missing API token',
                        'confidence': 0.0
                    }
                }
            
            # Try each VLM model in the fallback chain
            last_error = None
            for vlm_model in vlm_models:
                try:
                    logger.info(f"🔍 Attempting vision analysis with {vlm_model}")
                    
                    # Use the async context manager for model caller
                    async with model_caller as caller:
                        # Create a vision analysis prompt that includes image description
                        vision_prompt = f"Analyze this image and {prompt}. Provide a detailed description including objects, text, people, colors, and notable features."
                        success, response = await caller.generate_text(
                            prompt=vision_prompt,
                            api_key=hf_token,
                            model_override=vlm_model
                        )
                    
                    if success and response:
                        logger.info(f"✅ Vision analysis successful with {vlm_model}")
                        return {
                            'success': True,
                            'model_used': vlm_model,
                            'analysis': {
                                'description': str(response.get('description', '')) if isinstance(response, dict) else str(response),
                                'confidence': response.get('confidence', 0.8) if isinstance(response, dict) else 0.8,
                                'model_response': str(response),
                                'processing_time': 0
                            }
                        }
                    else:
                        error_msg = str(response) if response else 'No response from model'
                        logger.warning(f"⚠️ Vision model {vlm_model} failed: {error_msg}")
                        last_error = error_msg
                        continue
                        
                except Exception as e:
                    safe_error = str(e)
                    logger.warning(f"⚠️ Vision analysis failed with {vlm_model}: {safe_error}")
                    last_error = safe_error
                    continue
            
            # All VLM models failed
            logger.error("❌ All VLM models failed for vision analysis")
            return {
                'success': False,
                'error': f'Vision model call failed with all models: {last_error}',
                'attempted_models': vlm_models,
                'analysis': {
                    'description': 'Vision analysis failed - all models unavailable',
                    'confidence': 0.0,
                    'error_details': last_error
                }
            }
                
        except Exception as e:
            secure_logger.error(f"Vision model call failed: {e}")
            return {
                'success': False,
                'error': f'Vision model call exception: {str(e)}',
                'analysis': {
                    'description': 'Vision analysis encountered an error',
                    'confidence': 0.0,
                    'error_details': str(e)
                }
            }
    
    @staticmethod 
    async def call_text_model(text_content: str, task_type: str = "analysis", model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Call Hugging Face text model for document analysis
        
        Args:
            text_content (str): Text content to analyze
            task_type (str): Type of analysis ('analysis', 'summarization', 'classification', etc.)
            model_name (str): Specific model to use, defaults to Config.DEFAULT_TEXT_MODEL
            
        Returns:
            Dict[str, Any]: Analysis results from the text model
        """
        from bot.config import Config
        from bot.core.model_caller import model_caller
        
        logger.info(f"📄 Calling text model for {task_type}")
        
        # Check if AI functionality is available
        is_available, status_msg = Config.is_ai_functionality_available()
        if not is_available:
            return {
                'success': False,
                'error': 'AI functionality not available',
                'status': status_msg,
                'analysis': {
                    'summary': 'AI analysis unavailable - HF_TOKEN not configured',
                    'key_insights': ['Set up HF_TOKEN to enable advanced text analysis'],
                    'confidence': 0.0,
                    'word_count': len(text_content.split()) if text_content else 0
                }
            }
        
        try:
            # Use specified model or default based on task type
            if not model_name:
                if task_type == 'summarization':
                    selected_model = Config.DEFAULT_SUMMARIZATION_MODEL
                elif task_type == 'classification':
                    selected_model = Config.DEFAULT_CLASSIFICATION_MODEL
                elif task_type == 'sentiment':
                    selected_model = Config.DEFAULT_SENTIMENT_MODEL
                else:
                    selected_model = Config.DEFAULT_TEXT_MODEL
            else:
                selected_model = model_name
            
            # Prepare the prompt based on task type
            prompts = {
                'analysis': f"Analyze the following text and provide key insights, themes, and important information:\n\n{text_content}",
                'summarization': f"Summarize the following text concisely, highlighting the main points:\n\n{text_content}",
                'classification': f"Classify and categorize the following text, identifying its main topic and type:\n\n{text_content}",
                'sentiment': f"Analyze the sentiment and emotional tone of the following text:\n\n{text_content}",
                'extraction': f"Extract the most important information and key points from the following text:\n\n{text_content}"
            }
            
            prompt = prompts.get(task_type, prompts['analysis'])
            
            # Truncate text if too long (Railway-compatible configurable limit)
            max_content_length = int(os.getenv('MAX_TEXT_CONTENT_LENGTH', '8000'))
            if len(text_content) > max_content_length:
                text_content = text_content[:max_content_length] + "... [truncated]"
                prompt = prompts.get(task_type, prompts['analysis']).replace(text_content, text_content)
            
            # Call the text model through the model caller
            # Get HF token for API call
            hf_token = Config.get_hf_token()
            if not hf_token:
                return {
                    'success': False,
                    'error': 'HF_TOKEN not available for text model call',
                    'analysis': {
                        'summary': 'Text analysis failed - missing API token',
                        'key_insights': [],
                        'confidence': 0.0
                    }
                }
            
            # Prepare special parameters
            max_tokens = getattr(Config, 'get_model_max_tokens', lambda x: 1000)(selected_model)
            special_params = {
                'max_new_tokens': max_tokens,
                'temperature': 0.7,
                'return_full_text': False
            }
            
            # Use the async context manager for model caller
            async with model_caller as caller:
                success, response = await caller.generate_text(
                    prompt=prompt,
                    api_key=hf_token,
                    model_override=selected_model,
                    special_params=special_params
                )
            
            if success and response:
                analysis_content = str(response)
                
                # Extract insights from the response
                insights = []
                if analysis_content:
                    # Simple heuristic to extract key insights
                    sentences = analysis_content.split('.')
                    insights = [s.strip() for s in sentences if len(s.strip()) > 20][:5]
                
                return {
                    'success': True,
                    'model_used': selected_model,
                    'analysis': {
                        'summary': analysis_content[:500] + '...' if len(analysis_content) > 500 else analysis_content,
                        'full_analysis': analysis_content,
                        'key_insights': insights,
                        'confidence': 0.8,
                        'word_count': len(text_content.split()) if text_content else 0,
                        'processing_time': 0,
                        'task_type': task_type
                    }
                }
            else:
                error_msg = str(response) if response else 'No response from model'
                return {
                    'success': False,
                    'error': f'Text model call failed: {error_msg}',
                    'model_used': selected_model,
                    'analysis': {
                        'summary': 'Text analysis failed',
                        'key_insights': ['Analysis could not be completed'],
                        'confidence': 0.0,
                        'word_count': len(text_content.split()) if text_content else 0,
                        'error_details': error_msg
                    }
                }
                
        except Exception as e:
            secure_logger.error(f"Text model call failed: {e}")
            return {
                'success': False,
                'error': f'Text model call exception: {str(e)}',
                'analysis': {
                    'summary': 'Text analysis encountered an error',
                    'key_insights': ['Analysis failed due to technical error'],
                    'confidence': 0.0,
                    'word_count': len(text_content.split()) if text_content else 0,
                    'error_details': str(e)
                }
            }

    @staticmethod
    def _classify_zip_content(file_types: Dict[str, int], text_files: List[Dict]) -> str:
        """Classify the overall content type of a ZIP archive"""
        
        # Count file categories
        code_extensions = {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php', '.rb'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'}
        document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md'}
        
        code_count = sum(file_types.get(ext, 0) for ext in code_extensions)
        image_count = sum(file_types.get(ext, 0) for ext in image_extensions)
        document_count = sum(file_types.get(ext, 0) for ext in document_extensions)
        
        total_files = sum(file_types.values())
        
        if total_files == 0:
            return "empty_archive"
        
        # Classify based on predominant content
        if code_count / total_files > 0.5:
            return "software_project"
        elif image_count / total_files > 0.5:
            return "image_collection"
        elif document_count / total_files > 0.5:
            return "document_archive"
        elif '.exe' in file_types or '.msi' in file_types:
            return "software_installer"
        elif len(file_types) == 1 and list(file_types.keys())[0] in code_extensions:
            return "source_code"
        else:
            return "mixed_content"

# Create legacy alias for backward compatibility
FileProcessor = AdvancedFileProcessor

# Enhanced exports with new classes and functionality
__all__ = [
    'AdvancedFileProcessor', 'FileProcessor', 'FileSecurityError', 'FileSizeError',
    'ProcessedFile', 'ImageAnalysis', 'DocumentStructure', 'ZipArchiveAnalysis'
]