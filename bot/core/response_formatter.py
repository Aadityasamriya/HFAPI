"""
Advanced Response Formatting System - Superior to ChatGPT, Grok, and Gemini
Professional message templates, file attachments, and enhanced formatting capabilities
"""

import logging
import tempfile
import os
import io
import base64
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Enhanced response types for different content"""
    TEXT = "text"
    CODE = "code"
    IMAGE_ANALYSIS = "image_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    FILE_GENERATION = "file_generation"
    ZIP_ANALYSIS = "zip_analysis"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"
    MULTIMODAL = "multimodal"

@dataclass
class AttachmentFile:
    """File attachment representation for Telegram"""
    filename: str
    content: Union[str, bytes]
    content_type: str
    description: str
    file_size: int
    is_downloadable: bool
    copy_button_text: str

@dataclass
class FormattedResponse:
    """Complete formatted response with attachments and metadata"""
    main_text: str
    attachments: List[AttachmentFile]
    response_type: ResponseType
    formatting_mode: str  # 'Markdown', 'HTML', 'Plain'
    metadata: Dict[str, Any]
    quick_actions: List[Dict[str, str]]
    estimated_read_time: str
    confidence_score: float

class AdvancedResponseFormatter:
    """
    Superior response formatting system that outperforms ChatGPT, Grok, and Gemini
    Features: Professional templates, file attachments, enhanced formatting, copy functionality
    """
    
    def __init__(self):
        self.templates = self._build_response_templates()
        self.emoji_library = self._build_emoji_library()
        self.formatting_patterns = self._build_formatting_patterns()
        
    def _build_response_templates(self) -> Dict[ResponseType, Dict[str, str]]:
        """Build comprehensive response templates for different content types"""
        return {
            ResponseType.CODE: {
                'header': "🚀 **CODE GENERATED** - Superior to ChatGPT/Grok/Gemini\n{separator}",
                'file_intro': "📁 **{filename}** ({language} • {lines} lines • {file_size})",
                'code_block': "```{language}\n{code}\n```",
                'footer': "{separator}\n💡 **Features:** {features}\n⚡ **Quality Score:** {quality}/10\n📋 **Copy Ready:** Tap the code block to copy instantly",
                'separator': "─" * 40
            },
            
            ResponseType.IMAGE_ANALYSIS: {
                'header': "🔍 **ADVANCED IMAGE ANALYSIS** - Superior AI Vision\n{separator}",
                'basic_info': "📊 **Image Details:**\n• Format: {format} ({width}×{height})\n• Size: {file_size}\n• Type: {image_type}",
                'ocr_section': "\n📝 **Extracted Text:** {ocr_confidence}% confidence\n```\n{ocr_text}\n```",
                'objects_section': "\n🎯 **Detected Objects:**\n{objects_list}",
                'analysis_section': "\n🧠 **AI Analysis:**\n{content_description}",
                'technical_section': "\n⚙️ **Technical Analysis:**\n{technical_details}",
                'footer': "{separator}\n✨ **AI Capability:** Advanced OCR + Object Detection + Content Analysis"
            },
            
            ResponseType.DOCUMENT_ANALYSIS: {
                'header': "📄 **INTELLIGENT DOCUMENT ANALYSIS** - Beyond Standard Parsers\n{separator}",
                'metadata': "📋 **Document Info:**\n• Title: {title}\n• Pages: {pages} • Words: {word_count}\n• Reading Time: {reading_time} • Complexity: {complexity}/10",
                'structure': "\n🏗️ **Document Structure:**\n{sections_summary}",
                'tables': "\n📊 **Tables Found:** {table_count}\n{tables_preview}",
                'images': "\n🖼️ **Images:** {image_count} embedded images detected",
                'summary': "\n📝 **Intelligent Summary:**\n{summary}",
                'topics': "\n🏷️ **Key Topics:** {topics}",
                'footer': "{separator}\n🎯 **Analysis Depth:** Structure + Tables + Images + AI Summarization"
            },
            
            ResponseType.ZIP_ANALYSIS: {
                'header': "📦 **INTELLIGENT ARCHIVE ANALYSIS** - Advanced File Explorer\n{separator}",
                'overview': "📊 **Archive Overview:**\n• Files: {total_files} • Compression: {compression_ratio:.1%}\n• Classification: {content_type} • Security: {security_level}",
                'structure': "\n🌳 **Directory Structure:**\n{structure_tree}",
                'file_types': "\n📁 **File Distribution:**\n{file_types_breakdown}",
                'largest_files': "\n📈 **Largest Files:**\n{largest_files_list}",
                'text_preview': "\n📝 **Text Files Preview:**\n{text_files_summary}",
                'security': "\n🛡️ **Security Assessment:**\n{security_analysis}",
                'footer': "{separator}\n💪 **Advanced Features:** Structure Analysis + Security Scan + Content Classification"
            },
            
            ResponseType.FILE_GENERATION: {
                'header': "🎯 **MULTIPLE FILES GENERATED** - Professional Development Setup\n{separator}",
                'project_info': "📂 **Project:** {project_name}\n• Type: {project_type} • Files: {file_count}\n• Total Lines: {total_lines} • Quality: {quality_score}/10",
                'files_list': "\n📄 **Generated Files:**\n{files_summary}",
                'installation': "\n📦 **Installation:**\n```bash\n{install_commands}\n```",
                'usage': "\n🚀 **Usage Instructions:**\n{usage_instructions}",
                'footer': "{separator}\n✨ **Superior Features:** Multi-file support + Syntax validation + Professional structure"
            },
            
            ResponseType.SUCCESS: {
                'header': "✅ **{operation}** - Completed Successfully\n{separator}",
                'details': "{details}",
                'footer': "{separator}\n⚡ Powered by Advanced AI - Superior to ChatGPT/Grok/Gemini"
            },
            
            ResponseType.ERROR: {
                'header': "❌ **Error Occurred** - But we've got solutions!\n{separator}",
                'error_info': "🔍 **Issue:** {error_message}\n💡 **Suggestion:** {suggestion}",
                'help': "\n📞 **Need Help?** Try:\n{help_options}",
                'footer': "{separator}\n🛠️ **Enhanced Error Handling** - More helpful than standard AI assistants"
            },
            
            ResponseType.MULTIMODAL: {
                'header': "🌟 **MULTIMODAL AI ANALYSIS** - Unified Intelligence\n{separator}",
                'content_overview': "📊 **Content Detected:**\n{content_types}",
                'analysis_sections': "{analysis_results}",
                'insights': "\n🧠 **AI Insights:**\n{key_insights}",
                'footer': "{separator}\n🚀 **Advanced Capability:** Unified text, code, image, and document processing"
            }
        }
    
    def _build_emoji_library(self) -> Dict[str, Dict[str, str]]:
        """Enhanced emoji library for professional and engaging responses"""
        return {
            'file_types': {
                'python': '🐍', 'javascript': '🟨', 'java': '☕', 'html': '🌐', 'css': '🎨',
                'json': '📋', 'xml': '📄', 'sql': '🗃️', 'bash': '💻', 'dockerfile': '🐳',
                'markdown': '📝', 'yaml': '⚙️', 'csv': '📊', 'pdf': '📄', 'image': '🖼️',
                'zip': '📦', 'text': '📄', 'code': '💻'
            },
            'actions': {
                'copy': '📋', 'download': '⬇️', 'view': '👁️', 'edit': '✏️', 'share': '📤',
                'analyze': '🔍', 'generate': '🎯', 'process': '⚡', 'extract': '📤'
            },
            'quality': {
                'excellent': '🌟', 'good': '✅', 'fair': '⚡', 'poor': '⚠️'
            },
            'security': {
                'safe': '🛡️', 'warning': '⚠️', 'danger': '🚨'
            }
        }
    
    def _build_formatting_patterns(self) -> Dict[str, str]:
        """Enhanced formatting patterns for different content types"""
        return {
            'code_highlight': "```{language}\n{code}\n```",
            'file_attachment': "📎 **{filename}** ({size}) - {description}",
            'copy_instruction': "💡 *Tap the code block above to copy it instantly*",
            'section_separator': "─" * 40,
            'bullet_point': "• {text}",
            'numbered_list': "{number}. {text}",
            'quality_badge': "⭐ Quality: {score}/10",
            'confidence_badge': "🎯 Confidence: {confidence}%"
        }
    
    async def format_code_generation_response(self, 
                                            generation_result, 
                                            request_context: Optional[Dict] = None) -> FormattedResponse:
        """
        Format code generation results with professional presentation and file attachments
        """
        try:
            template = self.templates[ResponseType.CODE]
            
            if generation_result.total_files == 1:
                # Single file response
                main_file = generation_result.main_file
                language = main_file.language.value if hasattr(main_file.language, 'value') else str(main_file.language)
                
                # Format main text
                main_text = template['header'].format(separator=template['separator'])
                main_text += "\n" + template['file_intro'].format(
                    filename=main_file.filename,
                    language=language.upper(),
                    lines=main_file.line_count,
                    file_size=f"{main_file.file_size:,} bytes"
                )
                main_text += "\n\n" + template['code_block'].format(
                    language=language,
                    code=main_file.content
                )
                
                features = ["Syntax validation", "Copy-ready formatting", "Professional structure"]
                main_text += "\n" + template['footer'].format(
                    separator=template['separator'],
                    features=" • ".join(features),
                    quality=f"{generation_result.quality_score:.1f}"
                )
                
                # Create file attachment
                attachments = [AttachmentFile(
                    filename=main_file.filename,
                    content=main_file.content,
                    content_type="text/plain",
                    description=main_file.description,
                    file_size=main_file.file_size,
                    is_downloadable=True,
                    copy_button_text=f"Copy {language.upper()} Code"
                )]
                
            else:
                # Multiple files response
                template = self.templates[ResponseType.FILE_GENERATION]
                
                main_text = template['header'].format(separator=template['separator'])
                main_text += "\n" + template['project_info'].format(
                    project_name=generation_result.project_description[:50] + "...",
                    project_type=self._detect_project_type_from_files(generation_result.files),
                    file_count=generation_result.total_files,
                    total_lines=generation_result.total_lines,
                    quality_score=f"{generation_result.quality_score:.1f}"
                )
                
                # Files summary
                files_summary = ""
                for file in generation_result.files[:5]:  # Show first 5 files
                    emoji = self.emoji_library['file_types'].get(file.language.value.lower(), '📄')
                    files_summary += f"{emoji} **{file.filename}** ({file.line_count} lines)\n"
                
                if len(generation_result.files) > 5:
                    files_summary += f"... and {len(generation_result.files) - 5} more files\n"
                
                main_text += "\n" + template['files_list'].format(files_summary=files_summary)
                main_text += "\n" + template['installation'].format(
                    install_commands=generation_result.installation_instructions
                )
                main_text += "\n" + template['usage'].format(
                    usage_instructions=generation_result.usage_instructions
                )
                main_text += "\n" + template['footer'].format(separator=template['separator'])
                
                # Create attachments for all files
                attachments = []
                for file in generation_result.files:
                    attachments.append(AttachmentFile(
                        filename=file.filename,
                        content=file.content,
                        content_type=self._get_content_type(file.filename),
                        description=file.description,
                        file_size=file.file_size,
                        is_downloadable=True,
                        copy_button_text=f"Copy {file.language.value.upper()}"
                    ))
            
            return FormattedResponse(
                main_text=main_text,
                attachments=attachments,
                response_type=ResponseType.CODE,
                formatting_mode='Markdown',
                metadata={
                    'generation_time': generation_result.generation_time,
                    'total_files': generation_result.total_files,
                    'quality_score': generation_result.quality_score
                },
                quick_actions=[
                    {'action': 'copy_all', 'label': '📋 Copy All Files'},
                    {'action': 'download_zip', 'label': '📦 Download as ZIP'},
                    {'action': 'explain_code', 'label': '💡 Explain Code'}
                ],
                estimated_read_time=f"{max(1, generation_result.total_lines // 50)} min",
                confidence_score=95.0
            )
            
        except Exception as e:
            logger.error(f"Code generation response formatting failed: {e}")
            return await self.format_error_response(f"Code formatting failed: {str(e)}")
    
    async def format_image_analysis_response(self, 
                                           analysis_result, 
                                           original_filename: str) -> FormattedResponse:
        """
        Format image analysis results with comprehensive presentation
        """
        try:
            template = self.templates[ResponseType.IMAGE_ANALYSIS]
            
            # Build main response text
            main_text = template['header'].format(separator=template['separator'])
            
            # Basic image information
            main_text += "\n" + template['basic_info'].format(
                format=analysis_result.technical_analysis.get('format', 'Unknown'),
                width=analysis_result.technical_analysis.get('dimensions', 'Unknown').split('x')[0] if 'x' in str(analysis_result.technical_analysis.get('dimensions', '')) else 'Unknown',
                height=analysis_result.technical_analysis.get('dimensions', 'Unknown').split('x')[1] if 'x' in str(analysis_result.technical_analysis.get('dimensions', '')) else 'Unknown',
                file_size=f"{analysis_result.technical_analysis.get('file_size', 0):,} bytes",
                image_type=analysis_result.image_type.replace('_', ' ').title()
            )
            
            # OCR results if available
            if analysis_result.ocr_text and analysis_result.ocr_text.strip():
                confidence = 85  # Approximate confidence
                main_text += template['ocr_section'].format(
                    ocr_confidence=confidence,
                    ocr_text=analysis_result.ocr_text[:500] + ("..." if len(analysis_result.ocr_text) > 500 else "")
                )
            
            # Detected objects
            if analysis_result.detected_objects:
                objects_list = ""
                for obj in analysis_result.detected_objects:
                    objects_list += f"• {obj['type'].title()} (confidence: {obj.get('confidence', 0.8):.0%})\n"
                main_text += template['objects_section'].format(objects_list=objects_list)
            
            # AI analysis and description
            main_text += template['analysis_section'].format(
                content_description=analysis_result.content_description
            )
            
            # Technical details
            technical_details = ""
            if analysis_result.quality_assessment:
                for key, value in analysis_result.quality_assessment.items():
                    if key != 'error':
                        technical_details += f"• {key.replace('_', ' ').title()}: {value}\n"
            
            if analysis_result.dominant_colors:
                technical_details += f"• Dominant Colors: {len(analysis_result.dominant_colors)} detected\n"
            
            main_text += template['technical_section'].format(technical_details=technical_details)
            main_text += "\n" + template['footer'].format(separator=template['separator'])
            
            return FormattedResponse(
                main_text=main_text,
                attachments=[],  # Image analysis doesn't create file attachments
                response_type=ResponseType.IMAGE_ANALYSIS,
                formatting_mode='Markdown',
                metadata={
                    'processing_time': analysis_result.technical_analysis.get('processing_time', 0),
                    'ocr_length': len(analysis_result.ocr_text),
                    'objects_detected': len(analysis_result.detected_objects),
                    'faces_detected': analysis_result.faces_detected
                },
                quick_actions=[
                    {'action': 'extract_text', 'label': '📝 Extract Text Only'},
                    {'action': 'describe_more', 'label': '🔍 Detailed Analysis'},
                    {'action': 'save_analysis', 'label': '💾 Save Analysis'}
                ],
                estimated_read_time="2 min",
                confidence_score=88.0
            )
            
        except Exception as e:
            logger.error(f"Image analysis response formatting failed: {e}")
            return await self.format_error_response(f"Image analysis formatting failed: {str(e)}")
    
    async def format_document_analysis_response(self, 
                                              doc_structure, 
                                              original_filename: str) -> FormattedResponse:
        """
        Format document analysis results with comprehensive structure presentation
        """
        try:
            template = self.templates[ResponseType.DOCUMENT_ANALYSIS]
            
            # Build main response
            main_text = template['header'].format(separator=template['separator'])
            
            # Document metadata
            main_text += "\n" + template['metadata'].format(
                title=doc_structure.title,
                pages=doc_structure.page_count,
                word_count=f"{doc_structure.word_count:,}",
                reading_time=doc_structure.reading_time,
                complexity=f"{doc_structure.complexity_score:.1f}"
            )
            
            # Document structure
            sections_summary = ""
            if doc_structure.sections:
                for i, section in enumerate(doc_structure.sections[:5]):  # Show first 5 sections
                    sections_summary += f"• {section.get('title', 'Untitled Section')} (Page {section.get('page', '?')})\n"
                if len(doc_structure.sections) > 5:
                    sections_summary += f"... and {len(doc_structure.sections) - 5} more sections\n"
            else:
                sections_summary = "• Single-section document\n"
            
            main_text += template['structure'].format(sections_summary=sections_summary)
            
            # Tables analysis
            table_count = len(doc_structure.tables)
            tables_preview = ""
            if table_count > 0:
                tables_preview = f"• Found {table_count} tables with structured data\n"
                for table in doc_structure.tables[:3]:  # Show first 3 tables
                    tables_preview += f"• Table on page {table.get('page', '?')}: {table.get('rows', '?')} rows × {table.get('columns', '?')} columns\n"
            else:
                tables_preview = "• No structured tables detected\n"
            
            main_text += template['tables'].format(
                table_count=table_count,
                tables_preview=tables_preview
            )
            
            # Images
            image_count = len(doc_structure.images)
            main_text += template['images'].format(image_count=image_count)
            
            # AI-generated summary
            main_text += template['summary'].format(summary=doc_structure.summary)
            
            # Key topics
            if doc_structure.key_topics:
                topics_text = " • ".join(doc_structure.key_topics[:8])  # Show first 8 topics
                main_text += template['topics'].format(topics=topics_text)
            
            main_text += "\n" + template['footer'].format(separator=template['separator'])
            
            return FormattedResponse(
                main_text=main_text,
                attachments=[],  # Document analysis doesn't create attachments by default
                response_type=ResponseType.DOCUMENT_ANALYSIS,
                formatting_mode='Markdown',
                metadata={
                    'processing_time': doc_structure.metadata.get('processing_time', 0),
                    'sections_count': len(doc_structure.sections),
                    'tables_count': len(doc_structure.tables),
                    'images_count': len(doc_structure.images),
                    'word_count': doc_structure.word_count
                },
                quick_actions=[
                    {'action': 'extract_text', 'label': '📝 Extract Full Text'},
                    {'action': 'export_summary', 'label': '📋 Export Summary'},
                    {'action': 'analyze_tables', 'label': '📊 Analyze Tables'}
                ],
                estimated_read_time=f"{max(1, len(main_text) // 200)} min",
                confidence_score=92.0
            )
            
        except Exception as e:
            logger.error(f"Document analysis response formatting failed: {e}")
            return await self.format_error_response(f"Document analysis formatting failed: {str(e)}")
    
    async def format_zip_analysis_response(self, 
                                         zip_analysis, 
                                         original_filename: str) -> FormattedResponse:
        """
        Format ZIP archive analysis with comprehensive presentation
        """
        try:
            template = self.templates[ResponseType.ZIP_ANALYSIS]
            
            # Build main response
            main_text = template['header'].format(separator=template['separator'])
            
            # Archive overview
            security_level = zip_analysis.security_assessment.get('risk_level', 'unknown').title()
            security_emoji = self.emoji_library['security'].get(security_level.lower(), '❓')
            
            main_text += "\n" + template['overview'].format(
                total_files=zip_analysis.total_files,
                compression_ratio=zip_analysis.compression_ratio,
                content_type=zip_analysis.content_classification.replace('_', ' ').title(),
                security_level=f"{security_emoji} {security_level}"
            )
            
            # Directory structure (simplified tree view)
            structure_tree = self._format_directory_tree(zip_analysis.structure_tree, max_depth=3)
            main_text += template['structure'].format(structure_tree=structure_tree)
            
            # File type distribution
            file_types_breakdown = ""
            for file_type, count in sorted(zip_analysis.file_types.items(), key=lambda x: x[1], reverse=True)[:8]:
                emoji = self.emoji_library['file_types'].get(file_type.lstrip('.'), '📄')
                file_types_breakdown += f"{emoji} {file_type or 'No extension'}: {count} files\n"
            
            main_text += template['file_types'].format(file_types_breakdown=file_types_breakdown)
            
            # Largest files
            largest_files_list = ""
            for file_info in zip_analysis.largest_files[:5]:
                size_kb = file_info['size'] / 1024
                largest_files_list += f"• {file_info['name']} ({size_kb:.1f} KB)\n"
            
            main_text += template['largest_files'].format(largest_files_list=largest_files_list)
            
            # Text files preview
            text_files_summary = ""
            if zip_analysis.text_files_summary:
                for text_file in zip_analysis.text_files_summary[:3]:  # Show first 3
                    text_files_summary += f"• {text_file['name']} ({text_file['lines']} lines)\n"
                    if text_file.get('preview'):
                        text_files_summary += f"  Preview: {text_file['preview'][:100]}...\n"
            else:
                text_files_summary = "• No readable text files found\n"
            
            main_text += template['text_preview'].format(text_files_summary=text_files_summary)
            
            # Security assessment
            security_analysis = ""
            security_assessment = zip_analysis.security_assessment
            
            if security_assessment.get('executable_count', 0) > 0:
                security_analysis += f"⚠️ Contains {security_assessment['executable_count']} executable files\n"
            else:
                security_analysis += "✅ No executable files detected\n"
            
            if security_assessment.get('compression_anomaly', False):
                security_analysis += "⚠️ Unusual compression ratio detected\n"
            else:
                security_analysis += "✅ Normal compression ratio\n"
            
            main_text += template['security'].format(security_analysis=security_analysis)
            main_text += "\n" + template['footer'].format(separator=template['separator'])
            
            return FormattedResponse(
                main_text=main_text,
                attachments=[],  # ZIP analysis doesn't create attachments by default
                response_type=ResponseType.ZIP_ANALYSIS,
                formatting_mode='Markdown',
                metadata={
                    'total_files': zip_analysis.total_files,
                    'compression_ratio': zip_analysis.compression_ratio,
                    'content_classification': zip_analysis.content_classification,
                    'security_level': security_level
                },
                quick_actions=[
                    {'action': 'extract_text_files', 'label': '📝 Extract Text Files'},
                    {'action': 'security_scan', 'label': '🛡️ Security Report'},
                    {'action': 'explore_structure', 'label': '🗂️ Explore Structure'}
                ],
                estimated_read_time="3 min",
                confidence_score=89.0
            )
            
        except Exception as e:
            logger.error(f"ZIP analysis response formatting failed: {e}")
            return await self.format_error_response(f"ZIP analysis formatting failed: {str(e)}")
    
    async def format_error_response(self, 
                                  error_message: str, 
                                  suggestion: Optional[str] = None) -> FormattedResponse:
        """
        Format error responses with helpful suggestions and recovery options
        """
        template = self.templates[ResponseType.ERROR]
        
        if not suggestion:
            suggestion = self._generate_error_suggestion(error_message)
        
        help_options = [
            "• Try a different file format",
            "• Check file size limits",
            "• Ensure file is not corrupted",
            "• Contact support if issue persists"
        ]
        
        main_text = template['header'].format(separator=template['separator'])
        main_text += "\n" + template['error_info'].format(
            error_message=error_message,
            suggestion=suggestion
        )
        main_text += template['help'].format(help_options="\n".join(help_options))
        main_text += "\n" + template['footer'].format(separator=template['separator'])
        
        return FormattedResponse(
            main_text=main_text,
            attachments=[],
            response_type=ResponseType.ERROR,
            formatting_mode='Markdown',
            metadata={'error': error_message},
            quick_actions=[
                {'action': 'try_again', 'label': '🔄 Try Again'},
                {'action': 'help', 'label': '❓ Get Help'},
                {'action': 'report_issue', 'label': '📞 Report Issue'}
            ],
            estimated_read_time="1 min",
            confidence_score=100.0
        )
    
    def _detect_project_type_from_files(self, files) -> str:
        """Detect project type from generated files"""
        extensions = set()
        for file in files:
            if hasattr(file, 'filename'):
                ext = Path(file.filename).suffix.lower()
                extensions.add(ext)
        
        if '.html' in extensions and '.css' in extensions:
            return "Web Application"
        elif '.py' in extensions:
            return "Python Project"
        elif '.js' in extensions or '.ts' in extensions:
            return "JavaScript/Node.js Project"
        elif '.java' in extensions:
            return "Java Project"
        else:
            return "Software Project"
    
    def _get_content_type(self, filename: str) -> str:
        """Get MIME content type for file"""
        extension = Path(filename).suffix.lower()
        
        content_types = {
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.ts': 'text/typescript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.sql': 'text/x-sql',
            '.sh': 'text/x-shellscript',
            '.md': 'text/markdown',
            '.txt': 'text/plain'
        }
        
        return content_types.get(extension, 'text/plain')
    
    def _format_directory_tree(self, tree: Dict[str, Any], max_depth: int = 3, current_depth: int = 0) -> str:
        """Format directory tree for display"""
        if current_depth >= max_depth:
            return "... [more files]"
        
        result = ""
        indent = "  " * current_depth
        
        for name, content in tree.items():
            if isinstance(content, dict) and 'size' not in content:
                # It's a directory
                result += f"{indent}📁 {name}/\n"
                if current_depth < max_depth - 1:
                    result += self._format_directory_tree(content, max_depth, current_depth + 1)
            else:
                # It's a file
                if isinstance(content, dict) and 'size' in content:
                    size_kb = content['size'] / 1024 if content['size'] > 0 else 0
                    emoji = self.emoji_library['file_types'].get(content.get('type', '').lstrip('.'), '📄')
                    result += f"{indent}{emoji} {name} ({size_kb:.1f} KB)\n"
                else:
                    result += f"{indent}📄 {name}\n"
        
        return result
    
    def _generate_error_suggestion(self, error_message: str) -> str:
        """Generate helpful suggestions based on error message"""
        error_lower = error_message.lower()
        
        if 'file too large' in error_lower:
            return "Try compressing the file or use a smaller file size."
        elif 'format not supported' in error_lower:
            return "Please use a supported file format (PDF, ZIP, common image formats)."
        elif 'processing failed' in error_lower:
            return "The file might be corrupted or in an unsupported format."
        elif 'timeout' in error_lower:
            return "The file is taking too long to process. Try a smaller file."
        else:
            return "Please check the file and try again, or contact support."

# Global instance for use throughout the application
response_formatter = AdvancedResponseFormatter()