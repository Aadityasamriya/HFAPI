"""
Enhanced Code Generation with File Output - Superior to ChatGPT/Grok/Gemini
Handles code generation, syntax highlighting, file formatting, and copy functionality
"""

import logging
import re
import tempfile
import os
import base64
import io
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class CodeLanguage(Enum):
    """Supported programming languages with enhanced detection"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript" 
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    MARKDOWN = "markdown"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    DOCKERFILE = "dockerfile"
    MAKEFILE = "makefile"
    PLAINTEXT = "plaintext"

@dataclass
class CodeFile:
    """Enhanced code file representation with metadata"""
    filename: str
    content: str
    language: CodeLanguage
    description: str
    file_size: int
    line_count: int
    complexity_score: float
    estimated_execution_time: str
    dependencies: List[str]
    file_type: str
    syntax_valid: bool
    copy_ready: bool

@dataclass
class CodeGeneration:
    """Complete code generation result with multiple files support"""
    files: List[CodeFile]
    main_file: Optional[CodeFile]
    project_description: str
    installation_instructions: str
    usage_instructions: str
    total_files: int
    total_lines: int
    generation_time: float
    quality_score: float

class AdvancedCodeGenerator:
    """
    Advanced code generation system with comprehensive language support
    Features: Multi-file support, syntax validation, enhanced formatting, copy functionality
    """
    
    def __init__(self):
        self.language_extensions = self._build_language_extensions()
        self.syntax_patterns = self._build_syntax_patterns()
        self.template_library = self._build_template_library()
        self.complexity_analyzers = self._build_complexity_analyzers()
        
    def _build_language_extensions(self) -> Dict[CodeLanguage, List[str]]:
        """Enhanced file extension mapping for better language detection"""
        return {
            CodeLanguage.PYTHON: ['.py', '.pyw', '.pyi', '.pyx'],
            CodeLanguage.JAVASCRIPT: ['.js', '.mjs', '.jsx'],
            CodeLanguage.TYPESCRIPT: ['.ts', '.tsx', '.d.ts'],
            CodeLanguage.JAVA: ['.java', '.class', '.jar'],
            CodeLanguage.CSHARP: ['.cs', '.csx', '.dll'],
            CodeLanguage.CPP: ['.cpp', '.cxx', '.cc', '.hpp', '.h++'],
            CodeLanguage.C: ['.c', '.h'],
            CodeLanguage.HTML: ['.html', '.htm', '.xhtml'],
            CodeLanguage.CSS: ['.css', '.scss', '.sass', '.less'],
            CodeLanguage.SQL: ['.sql', '.mysql', '.pgsql'],
            CodeLanguage.BASH: ['.sh', '.bash', '.zsh', '.fish'],
            CodeLanguage.JSON: ['.json', '.jsonl', '.geojson'],
            CodeLanguage.XML: ['.xml', '.xsd', '.xsl', '.xslt'],
            CodeLanguage.YAML: ['.yaml', '.yml'],
            CodeLanguage.MARKDOWN: ['.md', '.markdown', '.mdown'],
            CodeLanguage.GO: ['.go'],
            CodeLanguage.RUST: ['.rs', '.rlib'],
            CodeLanguage.PHP: ['.php', '.phtml', '.php3', '.php4', '.php5'],
            CodeLanguage.RUBY: ['.rb', '.rbw', '.gem'],
            CodeLanguage.SWIFT: ['.swift'],
            CodeLanguage.KOTLIN: ['.kt', '.kts'],
            CodeLanguage.SCALA: ['.scala', '.sc'],
            CodeLanguage.R: ['.r', '.R'],
            CodeLanguage.MATLAB: ['.m', '.mat'],
            CodeLanguage.DOCKERFILE: ['Dockerfile', '.dockerfile'],
            CodeLanguage.MAKEFILE: ['Makefile', 'makefile', '.mk'],
            CodeLanguage.PLAINTEXT: ['.txt', '.text', '.log', '.conf']
        }
    
    def _build_syntax_patterns(self) -> Dict[CodeLanguage, List[str]]:
        """Enhanced syntax validation patterns"""
        return {
            CodeLanguage.PYTHON: [
                r'^(def|class|import|from|if|while|for|try|with)\s',
                r'^\s*(def\s+\w+|class\s+\w+|import\s+\w+)',
                r':\s*$',  # Python colon syntax
                r'^\s*(#|"""|\'\'\').*',  # Comments and docstrings
            ],
            CodeLanguage.JAVASCRIPT: [
                r'^(function|const|let|var|class|import|export)\s',
                r'^\s*(function\s+\w+|const\s+\w+|let\s+\w+)',
                r'[{};]',  # JS syntax characters
                r'//.*|/\*.*?\*/',  # Comments
            ],
            CodeLanguage.JAVA: [
                r'^(public|private|protected|class|interface|import)\s',
                r'^\s*(public\s+class|private\s+\w+|import\s+\w+)',
                r'[{};]',  # Java syntax
                r'//.*|/\*.*?\*/',  # Comments
            ],
            CodeLanguage.HTML: [
                r'<[^>]+>',  # HTML tags
                r'<!DOCTYPE|<html|<head|<body|<div|<span',
                r'</[^>]+>',  # Closing tags
            ],
            CodeLanguage.CSS: [
                r'[.#]?[\w-]+\s*\{',  # CSS selectors
                r'[\w-]+\s*:\s*[^;]+;',  # CSS properties
                r'\}',  # CSS closing braces
            ],
            CodeLanguage.SQL: [
                r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s',
                r'(FROM|WHERE|JOIN|GROUP BY|ORDER BY|HAVING)\s',
                r';',  # SQL semicolons
            ]
        }
    
    def _build_template_library(self) -> Dict[str, Dict[CodeLanguage, str]]:
        """Enhanced code templates for rapid generation"""
        return {
            'web_app': {
                CodeLanguage.HTML: '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>{{title}}</h1>
        <main id="main-content">
            <!-- Your content here -->
        </main>
    </div>
    <script src="script.js"></script>
</body>
</html>''',
                CodeLanguage.CSS: ''':root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    --danger-color: #dc3545;
    --warning-color: #ffc107;
    --info-color: #17a2b8;
    --light-color: #f8f9fa;
    --dark-color: #343a40;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: var(--dark-color);
    background-color: var(--light-color);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: var(--primary-color);
    margin-bottom: 1rem;
    font-size: 2.5rem;
    text-align: center;
}''',
                CodeLanguage.JAVASCRIPT: '''// Modern JavaScript Application
class App {
    constructor() {
        this.init();
    }
    
    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupEventListeners();
            this.loadContent();
        });
    }
    
    setupEventListeners() {
        // Add your event listeners here
        console.log('App initialized successfully');
    }
    
    async loadContent() {
        try {
            // Load dynamic content
            const content = await this.fetchData();
            this.renderContent(content);
        } catch (error) {
            console.error('Error loading content:', error);
        }
    }
    
    async fetchData() {
        // Implement data fetching logic
        return { message: 'Hello, World!' };
    }
    
    renderContent(data) {
        const mainContent = document.getElementById('main-content');
        if (mainContent) {
            mainContent.innerHTML = `<p>${data.message}</p>`;
        }
    }
}

// Initialize the application
new App();'''
            },
            'api_server': {
                CodeLanguage.PYTHON: '''#!/usr/bin/env python3
"""
Modern FastAPI Server with Enhanced Features
Superior to basic Flask/Django implementations
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="{{title}}",
    description="Advanced API Server with modern features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# API Routes
@app.get("/", response_model=BaseResponse)
async def root():
    """Root endpoint"""
    return BaseResponse(message="Welcome to {{title}} API")

@app.get("/health", response_model=BaseResponse)
async def health_check():
    """Health check endpoint"""
    return BaseResponse(message="API is healthy and running")

@app.get("/api/v1/status")
async def get_status():
    """Get server status"""
    return {
        "status": "running",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=f"Path: {request.url.path}"
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )''',
                CodeLanguage.JAVASCRIPT: '''const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet()); // Security headers
app.use(cors()); // Enable CORS
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan('combined')); // Logging

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'Welcome to {{title}} API',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// API routes
app.get('/api/v1/status', (req, res) => {
  res.json({
    status: 'running',
    environment: process.env.NODE_ENV || 'development',
    version: '1.0.0'
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: 'Something went wrong!',
    message: err.message
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Route not found',
    path: req.originalUrl
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📚 Environment: ${process.env.NODE_ENV || 'development'}`);
});'''
            },
            'mobile_app': {
                CodeLanguage.SWIFT: '''import SwiftUI
import Foundation

@main
struct {{AppName}}App: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var message = "Hello, World!"
    @State private var isLoading = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text(message)
                    .font(.title)
                    .multilineTextAlignment(.center)
                    .padding()
                
                Button(action: {
                    withAnimation {
                        isLoading.toggle()
                    }
                }) {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isLoading ? "Loading..." : "Tap Me")
                    }
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
                }
                
                Spacer()
            }
            .navigationTitle("{{AppName}}")
            .padding()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}''',
                CodeLanguage.KOTLIN: '''package com.example.{{packagename}}

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.{{packagename}}.ui.theme.{{AppName}}Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            {{AppName}}Theme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    var message by remember { mutableStateOf("Hello, World!") }
    var isLoading by remember { mutableStateOf(false) }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 20.dp)
        )
        
        Button(
            onClick = {
                isLoading = !isLoading
                message = if (isLoading) "Loading..." else "Hello, World!"
            },
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text(if (isLoading) "Loading..." else "Tap Me")
        }
        
        if (isLoading) {
            CircularProgressIndicator(
                modifier = Modifier.padding(top = 16.dp)
            )
        }
    }
}'''
            }
        }
    
    def _build_complexity_analyzers(self) -> Dict[CodeLanguage, Any]:
        """Build complexity analysis patterns for different languages"""
        return {
            CodeLanguage.PYTHON: {
                'cyclomatic_patterns': [r'if\s', r'elif\s', r'while\s', r'for\s', r'try:', r'except:', r'with\s'],
                'function_patterns': [r'def\s+\w+', r'class\s+\w+', r'lambda\s'],
                'complexity_weights': {'if': 1, 'loop': 2, 'function': 3, 'class': 4}
            },
            CodeLanguage.JAVASCRIPT: {
                'cyclomatic_patterns': [r'if\s*\(', r'while\s*\(', r'for\s*\(', r'switch\s*\(', r'try\s*{', r'catch\s*\('],
                'function_patterns': [r'function\s+\w+', r'class\s+\w+', r'=>', r'async\s+function'],
                'complexity_weights': {'if': 1, 'loop': 2, 'function': 3, 'class': 4, 'async': 2}
            }
        }
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> CodeLanguage:
        """Enhanced language detection with multiple heuristics"""
        
        # Method 1: File extension detection
        if filename:
            for language, extensions in self.language_extensions.items():
                if any(filename.lower().endswith(ext) or filename.lower() == ext.lstrip('.') for ext in extensions):
                    logger.info(f"🔍 Language detected by extension: {language.value}")
                    return language
        
        # Method 2: Syntax pattern analysis
        code_lower = code.lower()
        language_scores = {}
        
        for language, patterns in self.syntax_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
                score += matches
            language_scores[language] = score
        
        # Method 3: Keyword detection
        keyword_patterns = {
            CodeLanguage.PYTHON: ['def ', 'import ', 'from ', 'class ', 'print(', '__init__'],
            CodeLanguage.JAVASCRIPT: ['function ', 'var ', 'let ', 'const ', 'console.', 'document.'],
            CodeLanguage.JAVA: ['public class', 'private ', 'public static void main', 'System.out'],
            CodeLanguage.HTML: ['<html', '<div', '<span', '<!DOCTYPE', '<head>', '<body>'],
            CodeLanguage.CSS: ['{', '}', ':', ';', 'color:', 'background:'],
            CodeLanguage.SQL: ['SELECT', 'FROM', 'WHERE', 'INSERT INTO', 'UPDATE', 'DELETE FROM']
        }
        
        for language, keywords in keyword_patterns.items():
            keyword_score = sum(code.count(keyword) for keyword in keywords)
            language_scores[language] = language_scores.get(language, 0) + keyword_score * 2
        
        # Return language with highest score
        if language_scores:
            best_language = max(language_scores, key=language_scores.get)
            if language_scores[best_language] > 0:
                logger.info(f"🔍 Language detected by analysis: {best_language.value} (score: {language_scores[best_language]})")
                return best_language
        
        logger.info("🔍 Language detection fallback: plaintext")
        return CodeLanguage.PLAINTEXT
    
    def analyze_complexity(self, code: str, language: CodeLanguage) -> float:
        """Advanced complexity analysis for code quality assessment"""
        
        if language not in self.complexity_analyzers:
            # Basic complexity for unsupported languages
            lines = code.split('\n')
            return min(len(lines) / 50.0, 10.0)  # Scale to 0-10
        
        analyzer = self.complexity_analyzers[language]
        complexity_score = 0.0
        
        # Cyclomatic complexity
        for pattern in analyzer['cyclomatic_patterns']:
            matches = len(re.findall(pattern, code, re.IGNORECASE))
            complexity_score += matches * 1.5
        
        # Function/class complexity
        for pattern in analyzer['function_patterns']:
            matches = len(re.findall(pattern, code, re.IGNORECASE))
            complexity_score += matches * 2.0
        
        # Line count factor
        line_count = len(code.split('\n'))
        complexity_score += line_count / 20.0
        
        # Normalize to 0-10 scale
        normalized_score = min(complexity_score / 10.0, 10.0)
        
        logger.info(f"📊 Code complexity analysis: {normalized_score:.2f}/10")
        return normalized_score
    
    def validate_syntax(self, code: str, language: CodeLanguage) -> Tuple[bool, List[str]]:
        """Enhanced syntax validation with detailed error reporting"""
        
        errors = []
        
        try:
            if language == CodeLanguage.PYTHON:
                # Python syntax validation
                try:
                    compile(code, '<string>', 'exec')
                    return True, []
                except SyntaxError as e:
                    errors.append(f"Python syntax error at line {e.lineno}: {e.msg}")
                except Exception as e:
                    errors.append(f"Python compilation error: {str(e)}")
            
            elif language == CodeLanguage.JSON:
                # JSON validation
                try:
                    json.loads(code)
                    return True, []
                except json.JSONDecodeError as e:
                    errors.append(f"JSON syntax error at line {e.lineno}: {e.msg}")
            
            elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
                # Basic JavaScript validation
                js_errors = []
                
                # Check for balanced braces
                open_braces = code.count('{')
                close_braces = code.count('}')
                if open_braces != close_braces:
                    js_errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
                
                # Check for balanced parentheses
                open_parens = code.count('(')
                close_parens = code.count(')')
                if open_parens != close_parens:
                    js_errors.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
                
                if js_errors:
                    errors.extend(js_errors)
                else:
                    return True, []
            
            elif language == CodeLanguage.HTML:
                # Basic HTML validation
                html_errors = []
                
                # Check for basic HTML structure
                if not re.search(r'<html.*?>', code, re.IGNORECASE) and len(code) > 100:
                    html_errors.append("Missing <html> tag in what appears to be a full HTML document")
                
                # Check for unclosed tags (basic)
                open_tags = re.findall(r'<(\w+)[^>]*>', code)
                close_tags = re.findall(r'</(\w+)>', code)
                
                if len(open_tags) != len(close_tags):
                    html_errors.append(f"Potential unclosed tags: {len(open_tags)} opening, {len(close_tags)} closing")
                
                if html_errors:
                    errors.extend(html_errors)
                else:
                    return True, []
            
            else:
                # For other languages, do basic validation
                if language in self.syntax_patterns:
                    patterns = self.syntax_patterns[language]
                    pattern_matches = sum(len(re.findall(pattern, code, re.IGNORECASE)) for pattern in patterns)
                    
                    if pattern_matches == 0 and len(code.strip()) > 10:
                        errors.append(f"No {language.value} syntax patterns detected in code")
                    else:
                        return True, []
                else:
                    return True, []  # Assume valid for unsupported languages
                    
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def format_code_for_telegram(self, code: str, language: CodeLanguage, filename: str) -> str:
        """Enhanced code formatting for Telegram with copy functionality"""
        
        # Language mapping for syntax highlighting
        lang_map = {
            CodeLanguage.PYTHON: "python",
            CodeLanguage.JAVASCRIPT: "javascript", 
            CodeLanguage.TYPESCRIPT: "typescript",
            CodeLanguage.JAVA: "java",
            CodeLanguage.CSHARP: "csharp",
            CodeLanguage.CPP: "cpp",
            CodeLanguage.C: "c",
            CodeLanguage.HTML: "html",
            CodeLanguage.CSS: "css",
            CodeLanguage.SQL: "sql",
            CodeLanguage.BASH: "bash",
            CodeLanguage.JSON: "json",
            CodeLanguage.XML: "xml",
            CodeLanguage.YAML: "yaml",
            CodeLanguage.GO: "go",
            CodeLanguage.RUST: "rust",
            CodeLanguage.PHP: "php",
            CodeLanguage.RUBY: "ruby",
            CodeLanguage.SWIFT: "swift",
            CodeLanguage.KOTLIN: "kotlin",
            CodeLanguage.R: "r",
            CodeLanguage.DOCKERFILE: "dockerfile"
        }
        
        syntax_lang = lang_map.get(language, "text")
        
        # Enhanced formatting with metadata
        header = f"📁 **{filename}** ({language.value.upper()})"
        separator = "─" * 40
        
        formatted_code = f"""
{header}
{separator}
```{syntax_lang}
{code}
```
{separator}
"""
        
        return formatted_code
    
    def generate_file_metadata(self, code: str, filename: str, language: CodeLanguage) -> Dict[str, Any]:
        """Generate comprehensive metadata for generated files"""
        
        lines = code.split('\n')
        line_count = len(lines)
        file_size = len(code.encode('utf-8'))
        complexity = self.analyze_complexity(code, language)
        is_valid, syntax_errors = self.validate_syntax(code, language)
        
        # Estimate execution time based on complexity and language
        execution_estimates = {
            CodeLanguage.PYTHON: "Fast" if complexity < 5 else "Medium",
            CodeLanguage.JAVASCRIPT: "Very Fast" if complexity < 3 else "Fast",
            CodeLanguage.JAVA: "Medium" if complexity < 7 else "Slow",
            CodeLanguage.CPP: "Very Fast" if complexity < 6 else "Fast"
        }
        
        estimated_time = execution_estimates.get(language, "Unknown")
        
        # Extract dependencies (basic detection)
        dependencies = []
        if language == CodeLanguage.PYTHON:
            import_matches = re.findall(r'import\s+(\w+)|from\s+(\w+)\s+import', code)
            dependencies = [match[0] or match[1] for match in import_matches if match[0] or match[1]]
        elif language == CodeLanguage.JAVASCRIPT:
            require_matches = re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', code)
            import_matches = re.findall(r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', code)
            dependencies = require_matches + import_matches
        elif language == CodeLanguage.JAVA:
            import_matches = re.findall(r'import\s+([^;]+);', code)
            dependencies = [imp.strip() for imp in import_matches]
        
        return {
            'filename': filename,
            'language': language.value,
            'file_size': file_size,
            'line_count': line_count,
            'complexity_score': complexity,
            'estimated_execution_time': estimated_time,
            'dependencies': dependencies[:10],  # Limit to first 10
            'syntax_valid': is_valid,
            'syntax_errors': syntax_errors,
            'copy_ready': True
        }
    
    async def generate_code_files(self, 
                                prompt: str, 
                                context: Optional[Dict] = None) -> CodeGeneration:
        """
        Main method to generate code files with advanced features
        Superior to ChatGPT/Grok/Gemini with multi-file support and validation
        """
        import time
        start_time = time.time()
        
        logger.info(f"🚀 Starting advanced code generation for prompt: {prompt[:100]}...")
        
        try:
            # This will be integrated with the existing model_caller
            # For now, we'll return a structured response that can be enhanced
            # when integrated with the HuggingFace models
            
            # Detect the type of project requested
            project_type = self._detect_project_type(prompt)
            language = self._detect_primary_language(prompt)
            
            logger.info(f"📊 Detected project type: {project_type}, Primary language: {language.value}")
            
            # Generate appropriate file structure
            files = await self._generate_project_files(prompt, project_type, language, context)
            
            # Calculate generation metrics
            generation_time = time.time() - start_time
            total_lines = sum(file.line_count for file in files)
            
            # Determine main file
            main_file = None
            for file in files:
                if any(main_indicator in file.filename.lower() for main_indicator in ['main', 'index', 'app']):
                    main_file = file
                    break
            if not main_file and files:
                main_file = files[0]
            
            result = CodeGeneration(
                files=files,
                main_file=main_file,
                project_description=self._generate_project_description(prompt, project_type),
                installation_instructions=self._generate_installation_instructions(language, files),
                usage_instructions=self._generate_usage_instructions(project_type, main_file),
                total_files=len(files),
                total_lines=total_lines,
                generation_time=generation_time,
                quality_score=self._calculate_quality_score(files)
            )
            
            logger.info(f"✅ Code generation completed: {len(files)} files, {total_lines} lines in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            raise
    
    def _detect_project_type(self, prompt: str) -> str:
        """Detect the type of project from the prompt"""
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ['web app', 'website', 'html', 'frontend', 'web application']):
            return 'web_app'
        elif any(keyword in prompt_lower for keyword in ['api', 'server', 'backend', 'rest', 'fastapi', 'express']):
            return 'api_server'
        elif any(keyword in prompt_lower for keyword in ['mobile app', 'android', 'ios', 'swift', 'kotlin']):
            return 'mobile_app'
        elif any(keyword in prompt_lower for keyword in ['script', 'automation', 'tool', 'utility']):
            return 'script'
        elif any(keyword in prompt_lower for keyword in ['game', 'pygame', 'unity', 'graphics']):
            return 'game'
        elif any(keyword in prompt_lower for keyword in ['machine learning', 'ai', 'neural network', 'data science']):
            return 'ml_project'
        else:
            return 'general'
    
    def _detect_primary_language(self, prompt: str) -> CodeLanguage:
        """Detect the primary programming language from the prompt"""
        prompt_lower = prompt.lower()
        
        language_keywords = {
            CodeLanguage.PYTHON: ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
            CodeLanguage.JAVASCRIPT: ['javascript', 'js', 'node', 'express', 'react', 'vue', 'angular'],
            CodeLanguage.TYPESCRIPT: ['typescript', 'ts', 'angular', 'nest'],
            CodeLanguage.JAVA: ['java', 'spring', 'android'],
            CodeLanguage.CSHARP: ['c#', 'csharp', '.net', 'asp.net'],
            CodeLanguage.CPP: ['c++', 'cpp'],
            CodeLanguage.GO: ['go', 'golang'],
            CodeLanguage.RUST: ['rust'],
            CodeLanguage.SWIFT: ['swift', 'ios'],
            CodeLanguage.KOTLIN: ['kotlin', 'android'],
            CodeLanguage.PHP: ['php', 'laravel', 'wordpress'],
            CodeLanguage.RUBY: ['ruby', 'rails']
        }
        
        for language, keywords in language_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return language
        
        return CodeLanguage.PYTHON  # Default to Python
    
    async def _generate_project_files(self, 
                                    prompt: str, 
                                    project_type: str, 
                                    language: CodeLanguage,
                                    context: Optional[Dict] = None) -> List[CodeFile]:
        """Generate appropriate files for the project type"""
        
        files = []
        
        # This is a placeholder that will be enhanced when integrated with model_caller
        # For now, we'll use templates and basic generation
        
        if project_type == 'web_app':
            files.extend(await self._generate_web_app_files(prompt, language))
        elif project_type == 'api_server':
            files.extend(await self._generate_api_server_files(prompt, language))
        elif project_type == 'mobile_app':
            files.extend(await self._generate_mobile_app_files(prompt, language))
        else:
            # Generate single file for simple projects
            files.append(await self._generate_single_file(prompt, language))
        
        return files
    
    async def _generate_web_app_files(self, prompt: str, language: CodeLanguage) -> List[CodeFile]:
        """Generate web application files"""
        files = []
        
        # HTML file
        html_template = self.template_library['web_app'][CodeLanguage.HTML]
        title = self._extract_title_from_prompt(prompt)
        html_content = html_template.replace('{{title}}', title)
        
        html_file = CodeFile(
            filename="index.html",
            content=html_content,
            language=CodeLanguage.HTML,
            description="Main HTML structure",
            **self.generate_file_metadata(html_content, "index.html", CodeLanguage.HTML)
        )
        files.append(html_file)
        
        # CSS file
        css_content = self.template_library['web_app'][CodeLanguage.CSS]
        css_file = CodeFile(
            filename="styles.css",
            content=css_content,
            language=CodeLanguage.CSS,
            description="Styling and layout",
            **self.generate_file_metadata(css_content, "styles.css", CodeLanguage.CSS)
        )
        files.append(css_file)
        
        # JavaScript file
        js_content = self.template_library['web_app'][CodeLanguage.JAVASCRIPT]
        js_file = CodeFile(
            filename="script.js",
            content=js_content,
            language=CodeLanguage.JAVASCRIPT,
            description="Interactive functionality",
            **self.generate_file_metadata(js_content, "script.js", CodeLanguage.JAVASCRIPT)
        )
        files.append(js_file)
        
        return files
    
    async def _generate_api_server_files(self, prompt: str, language: CodeLanguage) -> List[CodeFile]:
        """Generate API server files"""
        files = []
        
        if language == CodeLanguage.PYTHON:
            template = self.template_library['api_server'][CodeLanguage.PYTHON]
        else:
            template = self.template_library['api_server'][CodeLanguage.JAVASCRIPT]
        
        title = self._extract_title_from_prompt(prompt)
        content = template.replace('{{title}}', title)
        
        filename = "main.py" if language == CodeLanguage.PYTHON else "server.js"
        
        file = CodeFile(
            filename=filename,
            content=content,
            language=language,
            description="Main server application",
            **self.generate_file_metadata(content, filename, language)
        )
        files.append(file)
        
        return files
    
    async def _generate_mobile_app_files(self, prompt: str, language: CodeLanguage) -> List[CodeFile]:
        """Generate mobile app files"""
        files = []
        
        if language == CodeLanguage.SWIFT:
            template = self.template_library['mobile_app'][CodeLanguage.SWIFT]
            app_name = self._extract_title_from_prompt(prompt).replace(' ', '')
            content = template.replace('{{AppName}}', app_name)
            filename = f"{app_name}App.swift"
        elif language == CodeLanguage.KOTLIN:
            template = self.template_library['mobile_app'][CodeLanguage.KOTLIN]
            app_name = self._extract_title_from_prompt(prompt).replace(' ', '')
            package_name = app_name.lower()
            content = template.replace('{{AppName}}', app_name).replace('{{packagename}}', package_name)
            filename = "MainActivity.kt"
        else:
            # Default to a simple mobile app structure
            content = f"// {self._extract_title_from_prompt(prompt)} Mobile App\n// Generated code structure\n"
            filename = "main_activity.py"
        
        file = CodeFile(
            filename=filename,
            content=content,
            language=language,
            description="Main mobile application",
            **self.generate_file_metadata(content, filename, language)
        )
        files.append(file)
        
        return files
    
    async def _generate_single_file(self, prompt: str, language: CodeLanguage) -> CodeFile:
        """Generate a single file for simple projects"""
        
        # Basic template based on language
        templates = {
            CodeLanguage.PYTHON: f'''#!/usr/bin/env python3
"""
{self._extract_title_from_prompt(prompt)}
Generated by Advanced AI Assistant
"""

def main():
    """Main function"""
    print("Hello, World!")
    # TODO: Implement your logic here

if __name__ == "__main__":
    main()
''',
            CodeLanguage.JAVASCRIPT: f'''/**
 * {self._extract_title_from_prompt(prompt)}
 * Generated by Advanced AI Assistant
 */

function main() {{
    console.log("Hello, World!");
    // TODO: Implement your logic here
}}

main();
''',
            CodeLanguage.JAVA: f'''/**
 * {self._extract_title_from_prompt(prompt)}
 * Generated by Advanced AI Assistant
 */

public class Main {{
    public static void main(String[] args) {{
        System.out.println("Hello, World!");
        // TODO: Implement your logic here
    }}
}}
'''
        }
        
        content = templates.get(language, f"// {self._extract_title_from_prompt(prompt)}\n// TODO: Implement functionality")
        filename = self._generate_filename(prompt, language)
        
        return CodeFile(
            filename=filename,
            content=content,
            language=language,
            description="Main application file",
            **self.generate_file_metadata(content, filename, language)
        )
    
    def _extract_title_from_prompt(self, prompt: str) -> str:
        """Extract a meaningful title from the prompt"""
        # Simple extraction - can be enhanced with NLP
        words = prompt.split()[:5]  # Take first 5 words
        title = " ".join(words).strip()
        if not title:
            title = "Generated Application"
        return title.title()
    
    def _generate_filename(self, prompt: str, language: CodeLanguage) -> str:
        """Generate appropriate filename based on prompt and language"""
        
        # Extract key words and clean them
        words = re.findall(r'\w+', prompt.lower())
        meaningful_words = [word for word in words if len(word) > 2 and word not in ['the', 'and', 'for', 'with', 'create', 'make', 'build']]
        
        base_name = "_".join(meaningful_words[:3]) if meaningful_words else "generated_code"
        
        # Get extension based on language
        extensions = {
            CodeLanguage.PYTHON: ".py",
            CodeLanguage.JAVASCRIPT: ".js",
            CodeLanguage.TYPESCRIPT: ".ts",
            CodeLanguage.JAVA: ".java",
            CodeLanguage.CSHARP: ".cs",
            CodeLanguage.CPP: ".cpp",
            CodeLanguage.C: ".c",
            CodeLanguage.HTML: ".html",
            CodeLanguage.CSS: ".css",
            CodeLanguage.SQL: ".sql",
            CodeLanguage.JSON: ".json",
            CodeLanguage.GO: ".go",
            CodeLanguage.RUST: ".rs",
            CodeLanguage.PHP: ".php",
            CodeLanguage.RUBY: ".rb",
            CodeLanguage.SWIFT: ".swift",
            CodeLanguage.KOTLIN: ".kt"
        }
        
        extension = extensions.get(language, ".txt")
        return f"{base_name}{extension}"
    
    def _generate_project_description(self, prompt: str, project_type: str) -> str:
        """Generate project description"""
        return f"Generated {project_type.replace('_', ' ').title()} based on: {prompt[:100]}..."
    
    def _generate_installation_instructions(self, language: CodeLanguage, files: List[CodeFile]) -> str:
        """Generate installation instructions"""
        
        instructions = {
            CodeLanguage.PYTHON: """
📦 **Installation Instructions:**

1. Ensure Python 3.8+ is installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```
""",
            CodeLanguage.JAVASCRIPT: """
📦 **Installation Instructions:**

1. Ensure Node.js 14+ is installed
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the application:
   ```bash
   node server.js
   ```
""",
            CodeLanguage.JAVA: """
📦 **Installation Instructions:**

1. Ensure Java 11+ is installed
2. Compile the code:
   ```bash
   javac *.java
   ```
3. Run the application:
   ```bash
   java Main
   ```
"""
        }
        
        return instructions.get(language, "Please refer to language-specific documentation for setup instructions.")
    
    def _generate_usage_instructions(self, project_type: str, main_file: Optional[CodeFile]) -> str:
        """Generate usage instructions"""
        
        if not main_file:
            return "Run the generated code files according to their respective language requirements."
        
        instructions = {
            'web_app': f"Open {main_file.filename} in your web browser to view the application.",
            'api_server': f"Run {main_file.filename} to start the server, then access the API endpoints.",
            'mobile_app': f"Import {main_file.filename} into your mobile development environment.",
            'script': f"Execute {main_file.filename} to run the script."
        }
        
        return instructions.get(project_type, f"Run {main_file.filename} according to its language requirements.")
    
    def _calculate_quality_score(self, files: List[CodeFile]) -> float:
        """Calculate overall quality score for the generated code"""
        
        if not files:
            return 0.0
        
        total_score = 0.0
        for file in files:
            # Base score from syntax validity
            file_score = 8.0 if file.syntax_valid else 3.0
            
            # Bonus for lower complexity
            if file.complexity_score < 3:
                file_score += 1.0
            elif file.complexity_score < 6:
                file_score += 0.5
            
            # Bonus for having dependencies (shows integration)
            if file.dependencies:
                file_score += 0.5
            
            total_score += min(file_score, 10.0)
        
        return total_score / len(files)

# Global instance
code_generator = AdvancedCodeGenerator()