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
from bot.security_utils import escape_markdown, safe_markdown_format

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Enhanced response types for different content"""
    TEXT = "text"
    CODE = "code"
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_GENERATION = "image_generation"
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
                'copy_instruction': "💡 **TAP THE CODE BLOCK ABOVE TO COPY** ⬆️\n📱 *One-tap copy works perfectly in Telegram*",
                'download_instruction': "⬇️ **DOWNLOAD FILE** below for direct file access",
                'footer': "{separator}\n💡 **Features:** {features}\n⚡ **Quality Score:** {quality}/10\n\n🎯 **Perfect Copy Experience:**\n📋 **Mobile:** Tap code block → Copy\n💻 **Desktop:** Click code block → Copy\n⬇️ **Files:** Use download attachments",
                'separator': "─" * 45
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
            
            ResponseType.IMAGE_GENERATION: {
                'header': "🎨 **ENHANCED IMAGE CREATION** - Professional Artistic Guidance\n{separator}",
                'description_header': "✨ **Detailed Artistic Vision:**\n{detailed_description}",
                'alternative_services_header': "\n🚀 **Free AI Art Generators** - Create Your Image Now:",
                'service_item': "\n🌟 **{name}**\n   🔗 {url}\n   📝 {description}\n   ✅ Pros: {pros}\n   💡 Tip: {tips}\n   📋 Optimized Prompt:\n   ```\n{optimized_prompt}\n   ```",
                'optimized_prompts_header': "\n🎯 **Platform-Specific Prompts** - Copy & Use:",
                'prompt_item': "\n**{platform}:**\n```\n{prompt}\n```",
                'creation_steps_header': "\n🛠️ **Manual Creation Guide** - Step by Step:",
                'step_item': "{step}",
                'technical_specs_header': "\n⚙️ **Technical Specifications:**",
                'spec_item': "• **{spec_name}:** {spec_value}",
                'footer': "{separator}\n🎨 **Enhanced Value:** Detailed descriptions + Free tools + Copy-ready prompts\n⏱️ **Estimated Creation Time:** {creation_time}\n🎯 **Confidence:** {confidence}%",
                'separator': "─" * 40
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
            'copy_instruction': "💡 **TAP CODE BLOCK ABOVE TO COPY** ⬆️\n📱 *Perfect one-tap copy in Telegram*",
            'enhanced_copy_instruction': "🎯 **COPY INSTRUCTIONS:**\n📱 **Mobile:** Tap code block → Select All → Copy\n💻 **Desktop:** Click code block → Ctrl+A → Ctrl+C\n⬇️ **Alternative:** Download file attachment below",
            'copy_prompt_instruction': "📋 **TAP ANY CODE BLOCK TO COPY** - Works perfectly in Telegram!",
            'download_instruction': "⬇️ **DOWNLOAD OPTION:** File attachments available below",
            'dual_option_header': "🎯 **TWO WAYS TO GET YOUR CODE:**\n📋 **Option A:** Copy from messages (tap code blocks)\n⬇️ **Option B:** Download files (use attachments)",
            'section_separator': "─" * 45,
            'bullet_point': "• {text}",
            'numbered_list': "{number}. {text}",
            'quality_badge': "⭐ Quality: {score}/10",
            'confidence_badge': "🎯 Confidence: {confidence}%",
            'file_info_badge': "📊 {lines} lines • {size} • {type}",
            'copy_success_tip': "✅ *Code copied successfully!*"
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
                
                # Add enhanced copy instructions
                main_text += "\n" + template['copy_instruction']
                main_text += "\n" + template['download_instruction']
                
                features = ["One-tap copy", "Syntax highlighting", "Download ready", "Professional structure"]
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
                    # Use enhanced MIME type detection
                    enhanced_mime_type = self._get_enhanced_content_type(file.filename)
                    attachments.append(AttachmentFile(
                        filename=file.filename,
                        content=file.content,
                        content_type=enhanced_mime_type,
                        description=f"{file.description} - {file.line_count} lines, {file.file_size:,} bytes",
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
    
    async def format_image_generation_response(self, 
                                             generation_result: Dict,
                                             original_prompt: str) -> FormattedResponse:
        """
        Format enhanced image generation results with comprehensive artistic guidance
        
        Args:
            generation_result (Dict): Enhanced image description data from model_caller
            original_prompt (str): Original user prompt
            
        Returns:
            FormattedResponse: Formatted response with copy functionality and alternatives
        """
        try:
            template = self.templates[ResponseType.IMAGE_GENERATION]
            
            # Build main response text
            main_text = template['header'].format(separator=template['separator'])
            
            # Add detailed artistic description with safe escaping
            detailed_description = generation_result.get('detailed_description', '')
            safe_detailed_description = escape_markdown(detailed_description)
            main_text += "\n" + template['description_header'].format(
                detailed_description=safe_detailed_description
            )
            
            # Add alternative services section
            alternative_services = generation_result.get('alternative_services', [])
            if alternative_services:
                main_text += template['alternative_services_header']
                
                for service in alternative_services[:3]:  # Show top 3 services
                    pros_text = " • ".join(service.get('pros', []))
                    # Safely escape all dynamic content
                    safe_name = escape_markdown(service.get('name', ''))
                    safe_url = service.get('url', '')  # URLs don't need escaping
                    safe_description = escape_markdown(service.get('description', ''))
                    safe_pros = escape_markdown(pros_text)
                    safe_tips = escape_markdown(service.get('tips', ''))
                    safe_prompt = service.get('optimized_prompt', '')  # Will be in code block
                    
                    main_text += template['service_item'].format(
                        name=safe_name,
                        url=safe_url,
                        description=safe_description,
                        pros=safe_pros,
                        tips=safe_tips,
                        optimized_prompt=safe_prompt
                    )
            
            # Add platform-specific prompts
            optimized_prompts = generation_result.get('optimized_prompts', {})
            if optimized_prompts:
                main_text += template['optimized_prompts_header']
                
                # Show most popular platforms
                priority_platforms = ['dalle3_optimized', 'stable_diffusion', 'leonardo_ai', 'playground_ai']
                for platform in priority_platforms:
                    if platform in optimized_prompts:
                        platform_display = platform.replace('_', ' ').title().replace('Dalle3', 'DALL-E 3')
                        safe_platform_display = escape_markdown(platform_display)
                        prompt_content = optimized_prompts[platform]  # Will be in code block, no escaping needed
                        main_text += template['prompt_item'].format(
                            platform=safe_platform_display,
                            prompt=prompt_content
                        )
            
            # Add creation steps
            creation_steps = generation_result.get('creation_steps', [])
            if creation_steps:
                main_text += template['creation_steps_header']
                for step in creation_steps:
                    safe_step = escape_markdown(str(step))
                    main_text += "\n" + template['step_item'].format(step=safe_step)
            
            # Add technical specifications
            technical_specs = generation_result.get('technical_specs', {})
            if technical_specs:
                main_text += template['technical_specs_header']
                for spec_name, spec_value in technical_specs.items():
                    display_name = spec_name.replace('_', ' ').title()
                    safe_display_name = escape_markdown(display_name)
                    safe_spec_value = escape_markdown(str(spec_value))
                    main_text += "\n" + template['spec_item'].format(
                        spec_name=safe_display_name,
                        spec_value=safe_spec_value
                    )
            
            # Add footer with safe escaping
            creation_time = generation_result.get('estimated_creation_time', 'Varies by complexity')
            confidence = int(generation_result.get('confidence_score', 0.9) * 100)
            safe_creation_time = escape_markdown(str(creation_time))
            
            main_text += "\n" + template['footer'].format(
                separator=template['separator'],
                creation_time=safe_creation_time,
                confidence=confidence
            )
            
            # Add copy instructions
            main_text += "\n\n" + self.formatting_patterns['copy_prompt_instruction']
            
            # Create quick actions for easy access
            quick_actions = [
                {"text": "🎨 Try Bing Image Creator", "url": "https://www.bing.com/images/create"},
                {"text": "✨ Try Leonardo.ai", "url": "https://leonardo.ai"},
                {"text": "🚀 Try Playground AI", "url": "https://playground.com"},
                {"text": "📋 Copy DALL-E Prompt", "data": optimized_prompts.get('dalle3_optimized', original_prompt)}
            ]
            
            # Create attachments with copy-ready prompts
            attachments = []
            
            # Main prompt attachment
            if optimized_prompts.get('dalle3_optimized'):
                attachments.append(AttachmentFile(
                    filename="dalle3_prompt.txt",
                    content=optimized_prompts['dalle3_optimized'],
                    content_type="text/plain",
                    description="Optimized prompt for DALL-E 3 (Bing Image Creator)",
                    file_size=len(optimized_prompts['dalle3_optimized'].encode('utf-8')),
                    is_downloadable=True,
                    copy_button_text="Copy DALL-E Prompt"
                ))
            
            # Stable Diffusion prompt attachment
            if optimized_prompts.get('stable_diffusion'):
                attachments.append(AttachmentFile(
                    filename="stable_diffusion_prompt.txt",
                    content=optimized_prompts['stable_diffusion'],
                    content_type="text/plain",
                    description="Optimized prompt for Stable Diffusion models",
                    file_size=len(optimized_prompts['stable_diffusion'].encode('utf-8')),
                    is_downloadable=True,
                    copy_button_text="Copy Stable Diffusion Prompt"
                ))
            
            # Detailed description attachment
            if detailed_description:
                attachments.append(AttachmentFile(
                    filename="artistic_description.txt",
                    content=detailed_description,
                    content_type="text/plain",
                    description="Detailed artistic description for manual creation",
                    file_size=len(detailed_description.encode('utf-8')),
                    is_downloadable=True,
                    copy_button_text="Copy Full Description"
                ))
            
            return FormattedResponse(
                main_text=main_text,
                attachments=attachments,
                response_type=ResponseType.IMAGE_GENERATION,
                formatting_mode='Markdown',
                metadata={
                    'original_prompt': original_prompt,
                    'generation_type': generation_result.get('type', 'enhanced_description'),
                    'alternative_services_count': len(alternative_services),
                    'optimized_prompts_count': len(optimized_prompts),
                    'confidence_score': generation_result.get('confidence_score', 0.9),
                    'estimated_creation_time': creation_time
                },
                quick_actions=quick_actions,
                estimated_read_time=f"{max(2, len(main_text.split()) // 200)} min",
                confidence_score=generation_result.get('confidence_score', 0.9) * 100
            )
            
        except Exception as e:
            logger.error(f"Image generation response formatting failed: {e}")
            return await self.format_error_response(f"Image generation formatting failed: {str(e)}")
    
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
        return self._get_enhanced_content_type(filename)
    
    def _get_enhanced_content_type(self, filename: str) -> str:
        """Get enhanced MIME content type with comprehensive file support"""
        extension = Path(filename).suffix.lower()
        
        # Comprehensive MIME type mapping matching the message handler
        content_types = {
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.jsx': 'text/javascript',
            '.ts': 'application/typescript',
            '.tsx': 'application/typescript',
            '.java': 'text/x-java-source',
            '.cs': 'text/x-csharp',
            '.cpp': 'text/x-c++src',
            '.cxx': 'text/x-c++src',
            '.cc': 'text/x-c++src',
            '.c': 'text/x-csrc',
            '.h': 'text/x-chdr',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.css': 'text/css',
            '.scss': 'text/x-scss',
            '.sass': 'text/x-sass',
            '.less': 'text/x-less',
            '.sql': 'application/sql',
            '.sh': 'application/x-shellscript',
            '.bash': 'application/x-shellscript',
            '.zsh': 'application/x-shellscript',
            '.fish': 'application/x-shellscript',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.md': 'text/markdown',
            '.markdown': 'text/markdown',
            '.go': 'text/x-go',
            '.rs': 'text/x-rust',
            '.php': 'text/x-php',
            '.rb': 'text/x-ruby',
            '.swift': 'text/x-swift',
            '.kt': 'text/x-kotlin',
            '.scala': 'text/x-scala',
            '.r': 'text/x-r',
            '.m': 'text/x-matlab',
            '.dockerfile': 'text/x-dockerfile',
            '.txt': 'text/plain',
            '.log': 'text/plain',
            '.conf': 'text/plain',
            '.config': 'text/plain'
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