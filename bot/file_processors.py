"""
Advanced file processing utilities for Hugging Face By AadityaLabs AI
Handles PDF extraction, ZIP analysis, and image processing with security measures
"""

import io
import os
import zipfile
import tempfile
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available - PDF processing will be limited")

# Image processing
try:
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np
    IMAGE_PROCESSING_AVAILABLE = True
    
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
    Image = None  # type: ignore
    pytesseract = None  # type: ignore
    cv2 = None  # type: ignore
    np = None  # type: ignore
    IMAGE_PROCESSING_AVAILABLE = False
    TESSERACT_BINARY_AVAILABLE = False
    logging.warning("Image processing libraries not available - some features will be limited")

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

class FileProcessor:
    """Advanced file processing with security measures and AI integration"""
    
    # Security limits
    MAX_ZIP_SIZE = 10 * 1024 * 1024  # 10MB for ZIP files
    MAX_PDF_SIZE = 25 * 1024 * 1024  # 25MB for PDF files  
    MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB for images
    MAX_EXTRACTED_FILES = 100  # Maximum files to extract from ZIP
    MAX_TEXT_CONTENT = 200000  # Maximum characters to process
    
    # Allowed file types
    ALLOWED_PDF_MIMES = ['application/pdf']
    ALLOWED_IMAGE_MIMES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp']
    ALLOWED_ZIP_MIMES = ['application/zip', 'application/x-zip-compressed']
    
    # Dangerous extensions to block
    DANGEROUS_EXTENSIONS = {'.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.dll', '.jar', '.sh', '.ps1', '.vbs', '.js'}
    
    @staticmethod
    def validate_file_security(file_data: bytes, filename: str, expected_type: str) -> Tuple[bool, str]:
        """
        Perform comprehensive security validation on uploaded file
        
        Args:
            file_data (bytes): File content
            filename (str): Original filename
            expected_type (str): Expected file type ('pdf', 'zip', 'image')
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Check file size limits
            file_size = len(file_data)
            
            if expected_type == 'zip' and file_size > FileProcessor.MAX_ZIP_SIZE:
                return False, f"ZIP file too large: {file_size:,} bytes (limit: {FileProcessor.MAX_ZIP_SIZE:,} bytes)"
            elif expected_type == 'pdf' and file_size > FileProcessor.MAX_PDF_SIZE:
                return False, f"PDF file too large: {file_size:,} bytes (limit: {FileProcessor.MAX_PDF_SIZE:,} bytes)"
            elif expected_type == 'image' and file_size > FileProcessor.MAX_IMAGE_SIZE:
                return False, f"Image file too large: {file_size:,} bytes (limit: {FileProcessor.MAX_IMAGE_SIZE:,} bytes)"
            
            # Check for dangerous file extensions
            file_ext = Path(filename).suffix.lower()
            if file_ext in FileProcessor.DANGEROUS_EXTENSIONS:
                return False, f"File type not allowed: {file_ext}"
            
            # Validate file type using magic numbers if available
            if FILE_DETECTION_AVAILABLE and filetype is not None:
                try:
                    detected_type = filetype.guess(file_data)
                    if detected_type:
                        detected_mime = detected_type.mime
                        
                        # Validate against expected types
                        if expected_type == 'pdf' and detected_mime not in FileProcessor.ALLOWED_PDF_MIMES:
                            return False, f"File is not a valid PDF (detected: {detected_mime})"
                        elif expected_type == 'image' and detected_mime not in FileProcessor.ALLOWED_IMAGE_MIMES:
                            return False, f"Image format not supported (detected: {detected_mime})"
                        elif expected_type == 'zip' and detected_mime not in FileProcessor.ALLOWED_ZIP_MIMES:
                            return False, f"File is not a valid ZIP archive (detected: {detected_mime})"
                except Exception as e:
                    logger.warning(f"File type detection failed: {e}")
            
            # Additional ZIP-specific security checks
            if expected_type == 'zip':
                try:
                    with zipfile.ZipFile(io.BytesIO(file_data), 'r') as zip_ref:
                        # Check for zip bomb (excessive compression ratio)
                        total_size = sum([zinfo.file_size for zinfo in zip_ref.filelist])
                        if total_size > FileProcessor.MAX_ZIP_SIZE * 10:  # 10x compression ratio limit
                            return False, f"ZIP file has excessive compression ratio (uncompressed: {total_size:,} bytes)"
                        
                        # Check number of files
                        if len(zip_ref.filelist) > FileProcessor.MAX_EXTRACTED_FILES:
                            return False, f"ZIP contains too many files: {len(zip_ref.filelist)} (limit: {FileProcessor.MAX_EXTRACTED_FILES})"
                        
                        # Check for directory traversal attacks with enhanced path validation
                        for zinfo in zip_ref.filelist:
                            # Basic checks for dangerous paths
                            if os.path.isabs(zinfo.filename) or '..' in zinfo.filename:
                                return False, f"ZIP contains potentially dangerous path: {zinfo.filename}"
                            
                            # Enhanced path normalization check
                            try:
                                # Normalize the path and ensure it stays within bounds
                                normalized_path = Path(zinfo.filename).resolve()
                                safe_base = Path().resolve()
                                
                                # Check if normalized path would escape the base directory
                                if not str(normalized_path).startswith(str(safe_base)):
                                    return False, f"ZIP contains path that would escape base directory: {zinfo.filename}"
                                    
                            except (OSError, ValueError) as e:
                                return False, f"ZIP contains invalid path: {zinfo.filename} ({str(e)})"
                            
                            # Check individual file size limits for extraction safety
                            if zinfo.file_size > 50 * 1024 * 1024:  # 50MB per file limit
                                return False, f"ZIP contains file too large for safe processing: {zinfo.filename} ({zinfo.file_size:,} bytes)"
                                
                except zipfile.BadZipFile:
                    return False, "File is not a valid ZIP archive"
                except Exception as e:
                    return False, f"ZIP validation failed: {str(e)}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"File security validation error: {e}")
            return False, f"Security validation failed: {str(e)}"
    
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
        if not PYMUPDF_AVAILABLE:
            return {'error': 'PDF processing not available - PyMuPDF not installed'}
        
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
            
            # Close document
            pdf_document.close()
            
            # Truncate if too long
            if len(full_text) > FileProcessor.MAX_TEXT_CONTENT:
                full_text = full_text[:FileProcessor.MAX_TEXT_CONTENT] + "\n\n[Content truncated due to length]"
            
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
            logger.error(f"PDF extraction error: {e}")
            return {
                'success': False,
                'error': f"Failed to extract PDF content: {str(e)}"
            }
    
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
                for zinfo in zip_ref.filelist[:FileProcessor.MAX_EXTRACTED_FILES]:
                    if zinfo.is_dir():
                        continue
                    
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
                        logger.warning(f"Error processing file {zinfo.filename} in ZIP: {e}")
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
            logger.error(f"ZIP analysis error: {e}")
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
        if not IMAGE_PROCESSING_AVAILABLE:
            return {
                'success': False,
                'error': 'Image processing not available - required libraries not installed'
            }
        
        try:
            # Open image
            if Image is None:
                return {
                    'success': False,
                    'error': 'Image processing not available - PIL not installed'
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
                if not TESSERACT_BINARY_AVAILABLE:
                    results['ocr'] = {
                        'error': 'OCR not available - tesseract binary not found on system. Please install tesseract-ocr package.',
                        'has_text': False,
                        'system_requirement': 'tesseract-ocr'
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
            
            # Basic image statistics
            if analysis_type == 'comprehensive':
                try:
                    # Convert to numpy array for analysis
                    if np is None:
                        results['color_analysis'] = {
                            'error': 'Color analysis not available - numpy not installed'
                        }
                    else:
                        img_array = np.array(rgb_image)
                        
                        # Calculate color statistics
                        results['color_analysis'] = {
                            'mean_brightness': float(np.mean(img_array)),
                            'dominant_colors': 'analysis_available',
                            'is_grayscale': len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1)
                        }
                    
                except Exception as e:
                    logger.warning(f"Color analysis failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {
                'success': False,
                'error': f'Image processing failed: {str(e)}'
            }

# Export main class
__all__ = ['FileProcessor', 'FileSecurityError', 'FileSizeError']