"""
Intelligent AI model routing system
Analyzes user prompts to determine the best Hugging Face model to use
"""

import re
import logging
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """2025 Enhanced intent types for model routing with specialized AI models"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUESTION_ANSWERING = "question_answering"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"          # 2025: New intent
    DOCUMENT_PROCESSING = "document_processing"  # 2025: New intent
    MULTI_MODAL = "multi_modal"              # 2025: New intent
    CONVERSATION = "conversation"            # 2025: New intent
    PDF_PROCESSING = "pdf_processing"        # 2025: New P1 feature - PDF analysis
    ZIP_ANALYSIS = "zip_analysis"            # 2025: New P1 feature - ZIP file analysis  
    IMAGE_ANALYSIS = "image_analysis"        # 2025: New P1 feature - Image content analysis
    FILE_GENERATION = "file_generation"      # 2025: New P1 feature - File generation and delivery
    
    # 2025: NEW SPECIALIZED INTENTS for superior AI routing
    MATHEMATICAL_REASONING = "mathematical_reasoning"   # Math problems, calculations, proofs
    ADVANCED_REASONING = "advanced_reasoning"           # Complex logical reasoning, QwQ model
    ALGORITHM_DESIGN = "algorithm_design"               # Complex coding algorithms, CodeLlama
    SCIENTIFIC_ANALYSIS = "scientific_analysis"         # Scientific research, data analysis
    MEDICAL_ANALYSIS = "medical_analysis"               # Medical image/text analysis
    CREATIVE_DESIGN = "creative_design"                 # UI/UX, graphic design, artistic creation
    EDUCATIONAL_CONTENT = "educational_content"         # Teaching, explanations, tutorials
    BUSINESS_ANALYSIS = "business_analysis"             # Business intelligence, market analysis
    TECHNICAL_DOCUMENTATION = "technical_documentation" # API docs, technical writing
    MULTILINGUAL_PROCESSING = "multilingual_processing" # Advanced translation, cultural context
    
    # 2025: BREAKTHROUGH CAPABILITIES - Revolutionary new intent types
    GUI_AUTOMATION = "gui_automation"                   # BREAKTHROUGH: UI-TARS GUI automation tasks
    TOOL_USE = "tool_use"                               # BREAKTHROUGH: Function calling, API integration
    PREMIUM_VISION = "premium_vision"                   # BREAKTHROUGH: Advanced vision with MiniCPM-V
    SYSTEM_INTERACTION = "system_interaction"           # Advanced system-level interactions

class IntelligentRouter:
    """
    Advanced AI model router with intent detection
    Analyzes user prompts and selects optimal models
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_patterns()
        self.intent_priorities = self._initialize_priorities()
        # 2025: Expanded programming languages and frameworks
        self.programming_languages = {
            # Core languages
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'php', 
            'ruby', 'go', 'rust', 'swift', 'kotlin', 'dart', 'scala', 'r',
            'sql', 'html', 'css', 'bash', 'shell', 'powershell', 'lua', 'perl',
            # 2025: New languages
            'zig', 'v', 'nim', 'crystal', 'elixir', 'haskell', 'ocaml', 'f#',
            'julia', 'fortran', 'cobol', 'assembly', 'wasm', 'webassembly',
            # Frameworks & libraries  
            'react', 'vue', 'angular', 'svelte', 'node', 'deno', 'bun',
            'django', 'flask', 'fastapi', 'laravel', 'spring', 'express',
            'next', 'nuxt', 'gatsby', 'remix', 'astro', 'solid', 'qwik',
            # Mobile & platforms
            'android', 'ios', 'flutter', 'react-native', 'xamarin', 'ionic',
            # 2025: AI/ML frameworks
            'pytorch', 'tensorflow', 'jax', 'huggingface', 'transformers',
            'langchain', 'llamaindex', 'openai', 'anthropic', 'cohere'
        }
        self.intent_cache = {}  # 2025: Cache for performance
        self.complexity_weights = {
            'technical_terms': 2.0,
            'code_snippets': 3.0,
            'specific_frameworks': 2.5,
            'creative_words': 1.5,
            'emotional_words': 1.8
        }
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize regex patterns for intent detection"""
        return {
            # 2025: Enhanced image generation patterns with latest AI models
            IntentType.IMAGE_GENERATION: [
                r'\b(?:draw|paint|sketch|illustrate|create|generate|make|design|show|visualize|render|produce)\s+(?:a|an|some)?\s*(?:beautiful|stunning|amazing|gorgeous|lovely|pretty|professional|artistic|detailed|realistic|abstract|minimalist|modern|vintage|futuristic)?\s*(?:sunset|sunrise|landscape|portrait|scene|picture|image|drawing|artwork|illustration|photo|graphic|visual|logo|icon|banner|poster|wallpaper|avatar|meme|infographic|diagram|chart|mockup)',
                r'\b(?:create|generate|make|design|produce|craft|build)\s+(?:a|an|some)?\s*(?:image|picture|photo|artwork|drawing|illustration|graphic|logo|icon|visual|art|design|mockup|prototype|concept)',
                r'\b(?:show|visualize|picture|display|demonstrate)\s+(?:me)?\s*(?:a|an|some)?\s*(?:image|picture|visual|artwork|design|mockup|example)',
                r'\b(?:draw|paint|sketch|illustrate|design)\b(?!.*(?:function|code|program|script|class|method|algorithm|database|query))',
                r'(?:can you|could you|please|i want|i need|help me).*(?:draw|paint|sketch|create|make|generate|design|visualize).*(?:image|picture|artwork|visual|art|graphic|logo|icon|design)',
                r'\b(?:logo|icon|banner|poster|wallpaper|avatar|art|artistic|visual|painting|drawing|sketch|illustration|graphic|artwork|photography|render|composition|mockup|wireframe|diagram|infographic|meme|sticker|emoji)\b',
                r'(?:dalle|midjourney|stable.?diffusion|text.?to.?image|flux|ai.?art|image.?generation|firefly|leonardo|runway)',
                r'(?:digital.?art|concept.?art|fantasy.?art|photorealistic|hyperrealistic|stylized|pixel.?art|vector.?art|3d.?render|ui.?mockup|app.?design)',
                # 2025: New patterns for modern image generation
                r'(?:design|create).*(?:ui|ux|interface|dashboard|website|app|mobile)',
                r'\b(?:meme|funny|cute|cartoon|anime|manga|realistic|portrait)\b.*(?:image|picture|art)',
                r'(?:product.?shot|brand.?identity|social.?media.?post|thumbnail|cover.?image)',
                r'(?:architectural|interior|fashion|food|nature|abstract).*(?:design|image|photo)',
            ],
            
            # 2025: Enhanced code generation patterns with latest frameworks
            IntentType.CODE_GENERATION: [
                r'\b(?:write|create|generate|code|program|implement|build|develop)\s+(?:a|an|some)?\s*(?:function|class|method|script|program|application|app|module|component|service|api|library|microservice|webhook|middleware|plugin|extension)',
                r'\b(?:python|javascript|java|c\+\+|typescript|php|ruby|go|rust|swift|kotlin|c#|dart|scala|r|zig|nim|elixir|haskell|julia)\s+(?:code|function|script|program|class|method|application|api)',
                r'\b(?:code|program|implement|write|build|create).*(?:algorithm|function|class|method|API|library|framework|module|component)',
                r'(?:how to|help me|show me|teach me|guide me|walk me through).*(?:code|program|implement|write|build|develop|deploy|setup|configure)',
                r'\b(?:debug|fix|error|bug|issue|optimize|refactor|improve|test|unit.?test|integration.?test).*(?:code|function|script|program|application)',
                r'\b(?:react|vue|angular|svelte|solid|qwik|django|flask|fastapi|laravel|spring|rails|express|nest|next|nuxt|gatsby|remix|astro)',
                r'(?:github|gitlab|stackoverflow|programming|coding|development|software|engineering|devops|ci.?cd)',
                r'\b(?:sql|database|query|table|schema|mongodb|postgresql|mysql|sqlite|redis|elasticsearch|graphql)',
                r'(?:rest.?api|graphql|microservices|web.?service|backend|full.?stack|serverless|lambda|cloud.?function)',
                r'(?:machine.?learning|ai|neural.?network|data.?science|automation|pytorch|tensorflow|huggingface|langchain)',
                # 2025: New patterns for modern development
                r'(?:docker|kubernetes|terraform|ansible|jenkins|github.?actions|aws|azure|gcp|vercel|netlify)',
                r'(?:websocket|realtime|streaming|event.?driven|pub.?sub|message.?queue|kafka|rabbitmq)',
                r'(?:blockchain|web3|smart.?contract|defi|nft|cryptocurrency|ethereum|solidity)',
                r'(?:mobile.?app|ios|android|flutter|react.?native|expo|xamarin|ionic)',
                r'(?:game.?dev|unity|unreal|godot|pygame|phaser|three.js|webgl)',
                r'(?:data.?pipeline|etl|batch.?processing|stream.?processing|data.?lake|data.?warehouse)',
            ],
            
            IntentType.SENTIMENT_ANALYSIS: [
                r'\b(?:what\'?s|what\s+is|analyze|check|determine|find)\s+(?:the\s+)?(?:sentiment|mood|tone|emotion|feeling)',
                r'(?:sentiment|emotion|feeling|mood|tone)\s+(?:of|analysis|detection|recognition)',
                r'(?:is this|how does this sound|what do you think about).*(?:positive|negative|neutral)',
                r'(?:analyze|check)\s+(?:this|the)?\s*(?:text|message|comment|review)',
                r'(?:positive|negative|neutral|happy|sad|angry|excited|disappointed)\s+(?:sentiment|feeling|emotion)',
                r'\b(?:sentiment)\b.*(?:of|about|in)\b',
            ],
            
            IntentType.CREATIVE_WRITING: [
                r'\b(?:write|create|compose|draft)\s+(?:a|an)?\s*(?:story|poem|song|lyrics|novel|tale|narrative)',
                r'\b(?:creative|fiction|poetry|storytelling|narrative|literature)',
                r'(?:once upon a time|tell me a story|write a story about)',
                r'\b(?:character|plot|dialogue|scene|chapter)',
                r'(?:rhyme|verse|stanza|ballad|sonnet|haiku)',
            ],
            
            IntentType.QUESTION_ANSWERING: [
                r'^\s*(?:what|who|when|where|why|how|which|whose)\s+(?:is|are|was|were|does|do|did|will|would|can|could|should)',
                r'\b(?:what\s+is)\s+(?:artificial\s+intelligence|machine\s+learning|deep\s+learning|AI|ML|computer\s+science|programming|technology)',
                r'(?:explain|tell me|describe|clarify|elaborate)\s+(?:what|how|why|the)',
                r'(?:question|answer|information|knowledge|facts)(?!.*(?:draw|create|generate|image|picture))',
                r'(?:do you know|can you tell me|help me understand)\s+(?:what|how|why)',
                r'\b(?:definition|meaning|concept)\s+(?:of|for)',
            ],
            
            IntentType.TRANSLATION: [
                r'\b(?:translate|translation|convert)\s+(?:this|the|from|to)',
                r'(?:from|to)\s+(?:english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|italian|russian|hindi|bengali|urdu|turkish|vietnamese|thai|dutch|swedish|norwegian|finnish|polish)',
                r'\b(?:language|linguistic|multilingual)',
                r'(?:what does.*mean in|how do you say.*in)',
            ],
            
            # 2025: BREAKTHROUGH GUI Automation patterns - UI-TARS capabilities
            IntentType.GUI_AUTOMATION: [
                r'\b(?:automate|automation|automatic|auto).*(?:gui|ui|interface|window|screen|desktop|application|app)',
                r'(?:click|tap|press|select|choose).*(?:button|link|menu|icon|element)',
                r'(?:navigate|browse|scroll|swipe).*(?:website|page|app|interface|screen)',
                r'(?:fill|enter|type|input).*(?:form|field|textbox|input)',
                r'(?:test|testing|qa|quality.?assurance).*(?:ui|gui|interface|application|app)',
                r'(?:screenshot|capture|image).*(?:screen|desktop|window|application)',
                r'(?:interact|control|operate|manipulate).*(?:interface|gui|ui|application|software)',
                r'\b(?:selenium|playwright|puppeteer|cypress|appium)\b',
                r'(?:web.?scraping|data.?extraction|automation.?script)',
                r'(?:browser.?automation|web.?automation|desktop.?automation)',
                r'(?:rpa|robotic.?process.?automation)',
                r'(?:ui.?testing|gui.?testing|end.?to.?end|e2e)',
            ],
            
            # 2025: BREAKTHROUGH Tool Use patterns - Groq function calling excellence
            IntentType.TOOL_USE: [
                r'\b(?:function|api|tool|service|integration|webhook).*(?:call|calling|invoke|execute|run)',
                r'(?:use|call|invoke|integrate|connect).*(?:api|service|tool|function|external)',
                r'(?:external|third.?party|remote).*(?:api|service|integration|tool)',
                r'(?:rest|graphql|soap|webhook|endpoint).*(?:call|request|integration)',
                r'(?:database|sql|query).*(?:execute|run|call)',
                r'(?:plugin|extension|addon|module).*(?:use|call|integrate)',
                r'(?:json|xml|yaml).*(?:parse|process|transform)',
                r'(?:http|https|request|response|api.?call)',
                r'(?:authentication|auth|token|key).*(?:api|service)',
                r'(?:payment|stripe|paypal|billing).*(?:integration|api)',
                r'(?:email|sms|notification).*(?:send|api|service)',
                r'(?:calendar|scheduling|booking).*(?:api|integration)',
                r'(?:file|upload|download|storage).*(?:api|service)',
                r'(?:search|elasticsearch|solr).*(?:api|query)',
                r'(?:machine.?learning|ai|model).*(?:api|inference|prediction)',
            ],
            
            # 2025: BREAKTHROUGH Premium Vision patterns - MiniCPM-V excellence
            IntentType.PREMIUM_VISION: [
                r'(?:analyze|examine|read|extract|understand).*(?:image|picture|photo|visual|diagram)',
                r'(?:ocr|optical.?character.?recognition|text.?extraction).*(?:image|document|photo)',
                r'(?:chart|graph|plot|diagram|visualization).*(?:analyze|read|interpret)',
                r'(?:medical|x.?ray|mri|ct.?scan|radiology).*(?:image|analysis)',
                r'(?:document|pdf|receipt|invoice|form).*(?:analysis|extraction|processing)',
                r'(?:handwriting|handwritten|cursive).*(?:recognition|read|transcribe)',
                r'(?:table|spreadsheet|data).*(?:extract|parse|analyze).*(?:image|photo)',
                r'(?:face|facial|person|human).*(?:recognition|detection|analysis)',
                r'(?:object|item|product).*(?:detection|recognition|identification)',
                r'(?:scene|environment|location).*(?:understanding|analysis|description)',
                r'(?:quality|defect|inspection).*(?:image|visual|analysis)',
                r'(?:compare|similarity|matching).*(?:image|photo|visual)',
                r'(?:caption|describe|explain).*(?:image|picture|photo|visual)',
                r'(?:logo|brand|trademark).*(?:recognition|detection|identification)',
                r'(?:barcode|qr.?code|code).*(?:scan|read|decode)',
            ],
            
            # 2025: New intent patterns
            IntentType.DATA_ANALYSIS: [
                r'\b(?:analyze|analysis|examine|study|investigate|process|parse)\s+(?:data|dataset|csv|excel|json|statistics|metrics|trends)',
                r'(?:data.?science|data.?analysis|statistical.?analysis|exploratory.?data.?analysis|eda)',
                r'\b(?:visualize|chart|graph|plot|dashboard|report)\s+(?:data|statistics|metrics|results)',
                r'(?:correlation|regression|clustering|classification|prediction|forecast)',
                r'\b(?:pandas|numpy|matplotlib|seaborn|plotly|tableau|powerbi)',
                r'(?:sql.?query|database.?analysis|data.?mining|business.?intelligence)',
            ],
            
            IntentType.DOCUMENT_PROCESSING: [
                r'\b(?:read|parse|extract|process|analyze)\s+(?:document|pdf|doc|docx|text|file|report)',
                r'(?:document.?analysis|text.?extraction|content.?parsing|file.?processing)',
                r'\b(?:summarize|summary|abstract|extract.?key.?points)\s+(?:from|this|document|text|article)',
                r'(?:ocr|optical.?character.?recognition|text.?from.?image)',
                r'\b(?:contract|invoice|receipt|form|application)\s+(?:processing|analysis|extraction)',
            ],
            
            IntentType.MULTI_MODAL: [
                r'(?:image.?and.?text|text.?and.?image|visual.?and.?text|multimodal|multi.?modal)',
                r'(?:describe.?this.?image|what.?do.?you.?see|analyze.?this.?picture)',
                r'(?:combine|merge|integrate)\s+(?:image|visual|text|code|data)',
                r'(?:vision.?language|vlm|visual.?question.?answering|image.?captioning)',
            ],
            
            IntentType.CONVERSATION: [
                r'(?:let\'s.?chat|casual.?conversation|just.?talking|friendly.?chat)',
                r'(?:how.?are.?you|what\'s.?up|tell.?me.?about.?yourself)',
                r'(?:opinion|thoughts|what.?do.?you.?think|personal.?view)',
                r'(?:conversation|chat|talk|discuss)(?!.*(?:code|image|analysis))',
            ],
            
            # 2025: P1 Features - Advanced file processing capabilities
            IntentType.PDF_PROCESSING: [
                r'\b(?:analyze|read|extract|process|parse|summarize)\s+(?:this\s+)?(?:pdf|document|file)\b',
                r'(?:pdf.?analysis|document.?processing|text.?extraction|content.?from.?pdf)',
                r'\b(?:tables?|charts?|graphs?)\s+(?:from|in)\s+(?:pdf|document)',
                r'(?:extract.?text|get.?content|read.?document|parse.?pdf)',
                r'\b(?:summarize|summary|key.?points|main.?ideas)\s+(?:from|of|in)\s+(?:this\s+)?(?:pdf|document|file)',
                r'(?:ocr|optical.?character.?recognition)\s+(?:pdf|document|scanned)',
                r'(?:metadata|properties|info)\s+(?:from|of|in)\s+(?:pdf|document)',
                r'(?:analyze|process)\s+(?:this\s+)?pdf\s+(?:document|file)?\s+(?:and|to)',
            ],
            
            IntentType.ZIP_ANALYSIS: [
                r'\b(?:analyze|examine|process|extract|explore)\s+(?:zip|archive|compressed)',
                r'(?:zip.?analysis|archive.?processing|compressed.?file|contents.?of.?zip)',
                r'\b(?:unzip|extract|decompress)\s+(?:and\s+)?(?:analyze|examine|process)',
                r'(?:file.?structure|directory.?listing|contents.?overview)',
                r'\b(?:scan|check|inspect)\s+(?:zip|archive)\s+(?:contents|files)',
                r'(?:bulk.?file.?analysis|multiple.?files.?processing)',
            ],
            
            IntentType.IMAGE_ANALYSIS: [
                r'\b(?:analyze|examine|describe|identify|recognize)\s+(?:image|photo|picture)',
                r'(?:image.?analysis|photo.?recognition|visual.?analysis|computer.?vision)',
                r'(?:what.?is.?in|what.?do.?you.?see|describe.?this)\s+(?:image|photo|picture)',
                r'\b(?:ocr|text.?recognition|read.?text)\s+(?:from|in)\s+(?:image|photo|picture)',
                r'(?:object.?detection|face.?recognition|scene.?analysis)',
                r'(?:extract.?text|get.?text|read)\s+(?:from|in)\s+(?:image|photo|screenshot)',
                r'\b(?:identify|classify|categorize)\s+(?:objects?|people|scenes?)\s+(?:in|from)\s+(?:image|photo)',
                r'(?:visual.?content|image.?content|picture.?content)\s+(?:analysis|recognition|description)',
            ],
            
            # 2025: P1 Feature - File Generation with AI
            IntentType.FILE_GENERATION: [
                r'\b(?:create|generate|make|build|produce|write)\s+(?:a|an|some)?\s*(?:file|script|document|config|configuration)',
                r'\b(?:create|generate|make|build|write)\s+(?:a|an|some)?\s*(?:python|py|javascript|js|json|txt|text|csv|xml|md|markdown)\s+(?:file|script|document)',
                r'(?:generate|create|make|build|produce)\s+(?:a|an|some)?\s*(?:\.py|\.js|\.json|\.txt|\.csv|\.xml|\.md)\s+(?:file|script)',
                r'\b(?:save|export|download|deliver)\s+(?:as|to|in)\s+(?:file|document|script)',
                r'(?:can you|could you|please|help me)\s+(?:create|generate|make|build|write)\s+(?:a|an|some)?\s*(?:file|script|document)',
                r'(?:output|save|export|write)\s+(?:this|that|the result|the response)\s+(?:to|as|in)\s+(?:a|an)?\s*(?:file|document|script)',
                r'\b(?:code|script|program)\s+(?:file|document)\s+(?:for|to|that|which)',
                r'(?:configuration|config|settings)\s+(?:file|document|json|xml|yaml)',
                r'(?:data|content|text|information)\s+(?:file|document|csv|json|txt)',
                r'(?:put|save|store|write|output)\s+(?:this|that|the|it)\s+(?:in|to|as)\s+(?:a|an)?\s*(?:file|document|script)',
                r'(?:send|give|provide|deliver)\s+(?:me|us)?\s*(?:a|an|some)?\s*(?:file|document|script|attachment)',
                r'(?:download|attachment|file.?download|document.?download)',
                # File format specific patterns
                r'(?:python.?script|py.?file|\.py\s+file)',
                r'(?:json.?file|json.?config|\.json\s+file|configuration.?json)',
                r'(?:text.?file|txt.?file|\.txt\s+file|plain.?text)',
                r'(?:markdown.?file|md.?file|\.md\s+file|documentation.?md)',
                r'(?:csv.?file|\.csv\s+file|comma.?separated|spreadsheet.?data)',
                r'(?:xml.?file|\.xml\s+file|xml.?document|xml.?config)',
            ],
            
            # 2025: NEW SPECIALIZED INTENT PATTERNS for SUPERIOR AI routing
            IntentType.MATHEMATICAL_REASONING: [
                r'\b(?:solve|calculate|compute|find)\s+(?:the|this)?\s*(?:equation|integral|derivative|limit|sum|product)',
                r'(?:math|mathematics|mathematical)\s+(?:problem|equation|proof|theorem|calculation)',
                r'\b(?:algebra|calculus|geometry|trigonometry|statistics|probability|linear.?algebra)',
                r'(?:differential|integral|partial.?derivative|matrix|vector|eigenvalue|polynomial)',
                r'\b(?:prove|theorem|lemma|corollary|axiom|postulate)',
                r'(?:\d+\s*[\+\-\*/\^]\s*\d+|\b(?:sin|cos|tan|log|ln|sqrt)\s*\()',
                r'(?:optimization|minimize|maximize|constraint|linear.?programming)',
                r'(?:graph.?theory|combinatorics|discrete.?math|number.?theory)',
            ],
            
            IntentType.ADVANCED_REASONING: [
                r'\b(?:analyze|reason|logic|logical|reasoning|think|thinking)\s+(?:about|through|step.?by.?step)',
                r'(?:complex|advanced|sophisticated|deep)\s+(?:reasoning|analysis|thinking|logic)',
                r'(?:step.?by.?step|systematically|methodically|logically)\s+(?:analyze|approach|solve)',
                r'(?:philosophical|epistemological|ontological|metaphysical)\s+(?:question|analysis)',
                r'(?:cause.?and.?effect|causal|inference|deduction|induction)',
                r'\b(?:pros.?and.?cons|trade.?offs|advantages.?disadvantages|benefits.?risks)',
                r'(?:critical.?thinking|problem.?solving|decision.?making|strategic.?analysis)',
                r'(?:what.?if|scenario|hypothetical|counterfactual)\s+(?:analysis|reasoning)',
            ],
            
            IntentType.ALGORITHM_DESIGN: [
                r'\b(?:design|create|implement|develop)\s+(?:an?\s+)?(?:algorithm|data.?structure|efficient)',
                r'(?:time.?complexity|space.?complexity|big.?o|algorithm.?analysis)',
                r'\b(?:sort|search|graph|tree|dynamic.?programming|greedy|divide.?conquer)',
                r'(?:optimization|optimize|efficient|performance)\s+(?:algorithm|solution|approach)',
                r'\b(?:recursion|recursive|backtracking|memoization|dp|dijkstra|bfs|dfs)',
                r'(?:leetcode|competitive.?programming|coding.?interview|algorithm.?problem)',
                r'(?:hash.?table|binary.?tree|heap|queue|stack|linked.?list|trie)',
                r'(?:shortest.?path|minimum.?spanning.?tree|topological.?sort)',
            ],
            
            IntentType.SCIENTIFIC_ANALYSIS: [
                r'(?:scientific|research|academic|scholarly)\s+(?:analysis|study|investigation)',
                r'\b(?:hypothesis|experiment|methodology|peer.?review|publication)',
                r'(?:data.?analysis|statistical.?analysis|correlation|regression|significance)',
                r'\b(?:chemistry|physics|biology|medicine|engineering|astronomy|geology)',
                r'(?:research.?paper|scientific.?method|literature.?review|meta.?analysis)',
                r'(?:quantum|molecular|genetic|cellular|neural|biochemical)',
                r'(?:journal|doi|citation|reference|bibliography|peer.?reviewed)',
                r'(?:clinical.?trial|study|sample.?size|control.?group|placebo)',
            ],
            
            IntentType.MEDICAL_ANALYSIS: [
                r'\b(?:medical|health|clinical|diagnostic|therapeutic)\s+(?:analysis|diagnosis|treatment)',
                r'(?:symptom|disease|condition|disorder|syndrome|pathology)',
                r'\b(?:x.?ray|mri|ct.?scan|ultrasound|mammogram|ekg|ecg)',
                r'(?:pharmaceutical|drug|medication|dosage|side.?effect|contraindication)',
                r'(?:patient|diagnosis|prognosis|treatment|therapy|intervention)',
                r'\b(?:anatomy|physiology|pathophysiology|epidemiology|pharmacology)',
                r'(?:medical.?image|radiology|histology|biopsy|lab.?result)',
                r'(?:icd|cpt|medical.?code|healthcare|hospital|clinic)',
            ],
            
            IntentType.CREATIVE_DESIGN: [
                r'\b(?:design|create|mockup|prototype|wireframe)\s+(?:ui|ux|interface|dashboard|app|website)',
                r'(?:user.?experience|user.?interface|interaction.?design|visual.?design)',
                r'(?:brand|branding|logo|identity|style.?guide|color.?palette)',
                r'\b(?:typography|font|layout|composition|visual.?hierarchy)',
                r'(?:responsive|mobile|desktop|tablet)\s+(?:design|layout|interface)',
                r'(?:figma|sketch|adobe|photoshop|illustrator|design.?tool)',
                r'(?:accessibility|usability|user.?research|persona|journey.?map)',
                r'(?:material.?design|bootstrap|tailwind|design.?system|component.?library)',
            ],
            
            IntentType.EDUCATIONAL_CONTENT: [
                r'\b(?:explain|teach|tutorial|lesson|course|learning)\s+(?:about|how|what|why)',
                r'(?:educational|pedagogical|instructional)\s+(?:content|material|resource)',
                r'(?:step.?by.?step|beginner|intermediate|advanced)\s+(?:guide|tutorial|explanation)',
                r'\b(?:curriculum|syllabus|lesson.?plan|learning.?objective|assessment)',
                r'(?:understand|comprehend|grasp|learn|study)\s+(?:concept|principle|theory)',
                r'(?:example|demonstration|illustration|case.?study|practice)',
                r'(?:quiz|test|exercise|homework|assignment|project)',
                r'(?:knowledge|skill|competency|proficiency|mastery)',
            ],
            
            IntentType.BUSINESS_ANALYSIS: [
                r'(?:business|market|commercial|financial)\s+(?:analysis|strategy|intelligence)',
                r'\b(?:roi|kpi|metrics|performance|revenue|profit|loss)',
                r'(?:market.?research|competitive.?analysis|swot|pest)',
                r'(?:financial.?modeling|forecasting|budgeting|valuation)',
                r'(?:customer.?analysis|segmentation|demographics|behavior)',
                r'(?:sales|marketing|operations|supply.?chain)\s+(?:analysis|optimization)',
                r'(?:dashboard|report|visualization|analytics|bi)',
                r'(?:growth|expansion|scalability|efficiency|productivity)',
            ],
            
            IntentType.TECHNICAL_DOCUMENTATION: [
                r'(?:technical|api|software)\s+(?:documentation|docs|manual|guide)',
                r'\b(?:api.?reference|developer.?guide|sdk|library.?docs)',
                r'(?:installation|setup|configuration|deployment)\s+(?:guide|instructions)',
                r'(?:troubleshooting|faq|error.?handling|debugging)\s+(?:guide|documentation)',
                r'(?:changelog|release.?notes|version|update)\s+(?:documentation|history)',
                r'(?:architecture|design|specification|requirements)\s+(?:document|spec)',
                r'(?:readme|documentation|wiki|knowledge.?base)',
                r'(?:comment|docstring|annotation|inline.?docs)',
            ],
            
            IntentType.MULTILINGUAL_PROCESSING: [
                r'(?:multilingual|cross.?lingual|multi.?language)\s+(?:processing|analysis|translation)',
                r'(?:cultural|localization|internationalization|i18n|l10n)',
                r'(?:language.?pair|source.?language|target.?language)',
                r'(?:dialect|accent|regional|native.?speaker)',
                r'(?:translation.?quality|fluency|accuracy|naturalness)',
                r'(?:context.?aware|semantic|pragmatic)\s+(?:translation|understanding)',
                r'(?:machine.?translation|neural.?translation|statistical.?translation)',
                r'(?:bilingual|polyglot|language.?model|cross.?cultural)',
            ]
        }
    
    def _initialize_priorities(self) -> Dict[IntentType, int]:
        """2025 Enhanced priority weights for intent types with specialized AI model routing"""
        return {
            # P1 High Priority - File and data processing
            IntentType.FILE_GENERATION: 10,          # File creation requests
            IntentType.PDF_PROCESSING: 10,           # PDF file uploads
            IntentType.ZIP_ANALYSIS: 10,             # ZIP file uploads
            
            # Specialized High Priority - Advanced AI capabilities  
            IntentType.MATHEMATICAL_REASONING: 9,    # Math problems require specialized models
            IntentType.ADVANCED_REASONING: 9,        # Complex reasoning tasks (QwQ model)
            IntentType.ALGORITHM_DESIGN: 9,          # Complex coding algorithms (CodeLlama)
            IntentType.MEDICAL_ANALYSIS: 9,          # Medical analysis requires specialized models
            
            # High Priority - Core AI tasks
            IntentType.IMAGE_GENERATION: 8,          # Visual generation requests
            IntentType.CODE_GENERATION: 8,           # Programming tasks
            IntentType.IMAGE_ANALYSIS: 8,            # Image analysis
            IntentType.SCIENTIFIC_ANALYSIS: 8,       # Scientific research tasks
            IntentType.DATA_ANALYSIS: 8,             # Data processing
            
            # Medium-High Priority - Specialized content
            IntentType.CREATIVE_DESIGN: 7,           # UI/UX and design tasks
            IntentType.TECHNICAL_DOCUMENTATION: 7,   # Technical writing
            IntentType.MULTILINGUAL_PROCESSING: 7,   # Advanced translation
            IntentType.SENTIMENT_ANALYSIS: 7,        # Analysis requests
            IntentType.CREATIVE_WRITING: 7,          # Creative requests
            IntentType.EDUCATIONAL_CONTENT: 7,       # Teaching and tutorials
            
            # Medium Priority - General processing
            IntentType.DOCUMENT_PROCESSING: 6,       # Document tasks
            IntentType.BUSINESS_ANALYSIS: 6,         # Business intelligence
            IntentType.MULTI_MODAL: 6,               # Complex multimodal
            IntentType.TRANSLATION: 6,               # Basic language requests
            IntentType.QUESTION_ANSWERING: 5,        # Broad Q&A category
            
            # Lower Priority - General conversation
            IntentType.CONVERSATION: 4,              # Conversational AI
            IntentType.TEXT_GENERATION: 3,           # General text fallback
        }
    
    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """
        Enhanced prompt complexity analysis for better model selection
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            Dict: Analysis results with complexity metrics
        """
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        analysis = {
            'length': len(prompt),
            'word_count': word_count,
            'has_code_blocks': bool(re.search(r'```|`[^`]+`', prompt)),
            'has_technical_terms': any(term in prompt_lower for term in list(self.programming_languages)),
            'has_questions': bool(re.search(r'\?', prompt)),
            'complexity_score': 0,
            'requires_context': False,
            'language_detected': self._detect_programming_language(prompt)
        }
        
        # Enhanced complexity scoring system
        complexity = 0
        
        # Base length score (more reasonable scaling)
        complexity += min(word_count / 10, 3)  # Up to 3 points for length
        
        # Technical complexity indicators (balanced weights)
        if analysis['has_code_blocks']:
            complexity += 2.5  # Reduced from 3
        if analysis['has_technical_terms']:
            complexity += 1.5  # Reduced from 2.5
        
        # Advanced complexity keywords (reduced scoring)
        advanced_keywords = ['complex', 'advanced', 'sophisticated', 'detailed', 'comprehensive', 
                           'algorithm', 'architecture', 'optimization', 'distributed', 'microservices',
                           'reasoning', 'philosophical', 'analysis', 'framework', 'principles']
        if any(keyword in prompt_lower for keyword in advanced_keywords):
            complexity += 2  # Reduced from 3 to 2
            
        # Creative complexity indicators
        creative_keywords = ['epic', 'novel', 'intricate', 'character development', 'plot twists',
                           'fantasy', 'creative', 'artistic', 'story', 'narrative']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            complexity += 2
            
        # Scientific/academic complexity (reduced scoring)
        academic_keywords = ['quantum', 'superconductivity', 'cooper pairs', 'principles',
                          'mechanical', 'theoretical', 'research', 'scientific']
        if any(keyword in prompt_lower for keyword in academic_keywords):
            complexity += 2.5  # Reduced from 4 to 2.5
        
        # Question complexity
        if analysis['has_questions']:
            complexity += 1.5
            
        # Multiple concepts/requirements
        if ' and ' in prompt_lower or ' with ' in prompt_lower:
            complexity += 1
            
        analysis['complexity_score'] = round(complexity, 1)
        
        # Context requirements
        context_indicators = ['this', 'that', 'previous', 'above', 'earlier', 'before', 'continue', 'also']
        analysis['requires_context'] = any(indicator in prompt_lower for indicator in context_indicators)
        
        return analysis
    
    def _detect_programming_language(self, prompt: str) -> str:
        """Detect programming language mentioned in prompt"""
        prompt_lower = prompt.lower()
        
        language_indicators = {
            'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular', 'express'],
            'typescript': ['typescript', 'ts'],
            'java': ['java', 'spring', 'android'],
            'c++': ['c++', 'cpp', 'c plus plus'],
            'c#': ['c#', 'csharp', 'dotnet', '.net'],
            'php': ['php', 'laravel', 'symfony'],
            'ruby': ['ruby', 'rails'],
            'go': ['golang', 'go'],
            'rust': ['rust'],
            'swift': ['swift', 'ios'],
            'kotlin': ['kotlin'],
            'sql': ['sql', 'database', 'query', 'mysql', 'postgresql', 'sqlite']
        }
        
        for language, indicators in language_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                return language
        
        return 'unknown'
    
    def route_prompt(self, prompt: str) -> Tuple[IntentType, Dict]:
        """
        Analyze prompt and determine the best intent and model to use
        
        Args:
            prompt (str): User prompt to analyze
            
        Returns:
            Tuple[IntentType, Dict]: (detected_intent, routing_info)
        """
        prompt_lower = prompt.lower()
        intent_scores = {}
        
        # Calculate weighted scores for each intent type
        for intent_type, patterns in self.intent_patterns.items():
            raw_score = 0
            matches = []
            
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    raw_score += 1
                    matches.append(pattern)
            
            if raw_score > 0:
                # Apply priority weighting
                priority_weight = self.intent_priorities.get(intent_type, 1)
                weighted_score = raw_score * priority_weight
                
                intent_scores[intent_type] = {
                    'raw_score': raw_score,
                    'weighted_score': weighted_score,
                    'priority': priority_weight,
                    'matches': matches
                }
        
        # Get prompt analysis
        analysis = self.analyze_prompt_complexity(prompt)
        
        # Determine primary intent using weighted scores
        if intent_scores:
            # Use weighted score for primary intent selection
            primary_intent = max(intent_scores.keys(), key=lambda x: intent_scores[x]['weighted_score'])
            confidence = intent_scores[primary_intent]['weighted_score']
        else:
            # Default to text generation if no specific intent detected
            primary_intent = IntentType.TEXT_GENERATION
            confidence = 0
        
        # Apply additional heuristics for edge cases
        primary_intent = self._apply_intent_heuristics(prompt, primary_intent, intent_scores, analysis)
        
        # Build routing information
        routing_info = {
            'primary_intent': primary_intent,
            'confidence': confidence,
            'all_intents': intent_scores,
            'analysis': analysis,
            'recommended_model': self._get_recommended_model(primary_intent, analysis, prompt),
            'special_parameters': self._get_special_parameters(primary_intent, analysis)
        }
        
        logger.info(f"Routed prompt to {primary_intent.value} with confidence {routing_info['confidence']}")
        
        return primary_intent, routing_info
    
    def _validate_model_selection(self, model_name: str, intent: IntentType) -> str:
        """Validate that the selected model exists in Config and return fallback if needed"""
        from bot.config import Config
        
        # List of all valid model names from Config - 2025 Enhanced with breakthrough specialized models
        valid_models = {
            # Text models - Enhanced with specialized reasoning models
            Config.DEFAULT_TEXT_MODEL, Config.ADVANCED_TEXT_MODEL, Config.FAST_TEXT_MODEL, 
            Config.FALLBACK_TEXT_MODEL, Config.LIGHTWEIGHT_TEXT_MODEL,
            getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            getattr(Config, 'MATH_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            getattr(Config, 'FLAGSHIP_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL),
            getattr(Config, 'EFFICIENT_TEXT_MODEL', Config.DEFAULT_TEXT_MODEL),
            getattr(Config, 'COMPACT_TEXT_MODEL', Config.FAST_TEXT_MODEL),
            
            # Code models - Enhanced with specialized coding and tool use models
            Config.DEFAULT_CODE_MODEL, Config.ADVANCED_CODE_MODEL, Config.FAST_CODE_MODEL,
            Config.FALLBACK_CODE_MODEL, Config.LIGHTWEIGHT_CODE_MODEL,
            getattr(Config, 'SPECIALIZED_CODE_MODEL', Config.ADVANCED_CODE_MODEL),
            getattr(Config, 'TOOL_USE_CODE_MODEL', Config.ADVANCED_CODE_MODEL),
            getattr(Config, 'EFFICIENT_CODE_MODEL', Config.FAST_CODE_MODEL),
            
            # Vision models - Enhanced with premium, GUI, and specialized vision
            Config.DEFAULT_VISION_MODEL, Config.ADVANCED_VISION_MODEL, Config.FAST_VISION_MODEL,
            Config.FALLBACK_VISION_MODEL, Config.DOCUMENT_VISION_MODEL, Config.LIGHTWEIGHT_VISION_MODEL,
            getattr(Config, 'REASONING_VISION_MODEL', Config.ADVANCED_VISION_MODEL),
            getattr(Config, 'MEDICAL_VISION_MODEL', Config.ADVANCED_VISION_MODEL),
            getattr(Config, 'GUI_AUTOMATION_MODEL', Config.DEFAULT_VISION_MODEL),
            getattr(Config, 'PREMIUM_VISION_MODEL', Config.DEFAULT_VISION_MODEL),
            getattr(Config, 'QUANTIZED_VISION_MODEL', Config.LIGHTWEIGHT_VISION_MODEL),
            
            # Image generation models - Enhanced with breakthrough 2025 models
            Config.DEFAULT_IMAGE_MODEL, Config.COMMERCIAL_IMAGE_MODEL, Config.ADVANCED_IMAGE_MODEL,
            Config.FALLBACK_IMAGE_MODEL, Config.ARTISTIC_IMAGE_MODEL,
            getattr(Config, 'FLAGSHIP_IMAGE_MODEL', Config.DEFAULT_IMAGE_MODEL),
            getattr(Config, 'PROFESSIONAL_IMAGE_MODEL', Config.ADVANCED_IMAGE_MODEL),
            getattr(Config, 'TURBO_IMAGE_MODEL', Config.ADVANCED_IMAGE_MODEL),
            getattr(Config, 'REALISTIC_IMAGE_MODEL', Config.DEFAULT_IMAGE_MODEL),
            getattr(Config, 'EDITING_IMAGE_MODEL', Config.ADVANCED_IMAGE_MODEL),
            
            # 2025: BREAKTHROUGH SPECIALIZED MODELS
            # GUI Automation models
            getattr(Config, 'DEFAULT_GUI_MODEL', Config.DEFAULT_VISION_MODEL),
            getattr(Config, 'ADVANCED_GUI_MODEL', Config.ADVANCED_VISION_MODEL),
            getattr(Config, 'LIGHTWEIGHT_GUI_MODEL', Config.LIGHTWEIGHT_VISION_MODEL),
            
            # Tool Use & Function Calling models
            getattr(Config, 'DEFAULT_TOOL_MODEL', Config.ADVANCED_CODE_MODEL),
            getattr(Config, 'EFFICIENT_TOOL_MODEL', Config.DEFAULT_CODE_MODEL),
            
            # Sentiment & NLP models
            Config.DEFAULT_SENTIMENT_MODEL, Config.ADVANCED_SENTIMENT_MODEL, Config.EMOTION_MODEL,
            Config.MULTILINGUAL_SENTIMENT_MODEL, Config.FALLBACK_SENTIMENT_MODEL,
            
            # Translation models
            Config.DEFAULT_TRANSLATION_MODEL, Config.ADVANCED_TRANSLATION_MODEL, Config.FALLBACK_TRANSLATION_MODEL
        }
        
        # Check if the selected model is valid
        if model_name in valid_models:
            return model_name
        
        # Fallback to appropriate model based on intent category  
        logger.warning(f"⚠️ MODEL_VALIDATION: {model_name} not found in Config, using fallback")
        
        fallback_map = {
            IntentType.CODE_GENERATION: Config.DEFAULT_CODE_MODEL,
            IntentType.IMAGE_GENERATION: Config.DEFAULT_IMAGE_MODEL,
            IntentType.IMAGE_ANALYSIS: Config.DEFAULT_VISION_MODEL,
            IntentType.MULTI_MODAL: Config.DEFAULT_VISION_MODEL,
            IntentType.SENTIMENT_ANALYSIS: Config.DEFAULT_SENTIMENT_MODEL,
            IntentType.TRANSLATION: Config.DEFAULT_TRANSLATION_MODEL,
            # 2025: BREAKTHROUGH intent fallbacks
            IntentType.GUI_AUTOMATION: Config.DEFAULT_VISION_MODEL,
            IntentType.TOOL_USE: Config.DEFAULT_CODE_MODEL,
            IntentType.PREMIUM_VISION: Config.DEFAULT_VISION_MODEL,
            IntentType.SYSTEM_INTERACTION: Config.DEFAULT_TEXT_MODEL,
        }
        
        return fallback_map.get(intent, Config.DEFAULT_TEXT_MODEL)
    
    def _get_recommended_model(self, intent: IntentType, analysis: Dict, original_prompt: str = "") -> str:
        """Get recommended model based on intent and analysis with 2024-2025 STATE-OF-THE-ART models"""
        from bot.config import Config
        
        complexity = analysis.get('complexity_score', 0)
        word_count = analysis.get('word_count', 0)
        has_technical = analysis.get('has_technical_terms', False)
        language = analysis.get('language_detected', 'unknown')
        
        # 🚀 ENHANCED ROUTER DECISION LOGGING - Proving Superior AI Selection
        logger.info(f"\n🎯 ═══════════════════════ INTELLIGENT ROUTER DECISION ═══════════════════════")
        logger.info(f"🔍 PROMPT_ANALYSIS: intent={intent.value.upper()}, complexity={complexity}/10")
        logger.info(f"📊 METRICS: words={word_count}, technical_terms={has_technical}, language={language}")
        logger.info(f"📝 PROMPT_SAMPLE: '{original_prompt[:80]}{'...' if len(original_prompt) > 80 else ''}'")
        
        # Advanced model selection based on complexity and context
        if intent == IntentType.TEXT_GENERATION:
            # Enhanced model selection with latest 2025 models
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            
            # Advanced reasoning tasks - Use latest DeepSeek-R1-0528
            if complexity > 8 or 'reasoning' in original_prompt.lower() or 'logic' in original_prompt.lower() or 'philosophy' in original_prompt.lower():
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-0528 (87.5% AIME 2025)
            # High complexity - Use flagship MoE model
            elif complexity > 6 or word_count > 500:
                selected_model = Config.FLAGSHIP_TEXT_MODEL  # Qwen3-235B-A22B (competitive with O1/O3-mini)
            # Balanced performance - Use efficient 80B/3B model
            elif complexity > 4 or analysis.get('requires_context'):
                selected_model = Config.EFFICIENT_TEXT_MODEL  # Qwen3-Next-80B-A3B (10x speed improvement)
            # General tasks - Use Qwen3-32B
            elif complexity > 2:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen3-32B (latest architecture)
            # Simple tasks - Use compact high-performance model
            else:
                selected_model = Config.COMPACT_TEXT_MODEL  # DeepSeek-R1-0528-Qwen3-8B (SOTA 8B)
            
            # Enhanced decision logging for text generation with latest models
            if complexity > 8:
                reasoning = "ADVANCED (DeepSeek-R1-0528)"
            elif complexity > 6:
                reasoning = "FLAGSHIP (Qwen3-235B-MoE)"
            elif complexity > 4:
                reasoning = "EFFICIENT (Qwen3-Next-80B/3B)"
            elif complexity > 2:
                reasoning = "DEFAULT (Qwen3-32B)"
            else:
                reasoning = "COMPACT (R1-Qwen3-8B)"
                
            logger.info(f"🤖 TEXT_GENERATION: complexity={complexity} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (2025 SOTA, Superior to GPT-4o/Claude/Gemini)")
            logger.info(f"⚡ REASONING: {'Complex reasoning (DeepSeek-R1-0528)' if complexity > 8 else 'High complexity (Flagship MoE)' if complexity > 6 else 'Balanced performance (Efficient)' if complexity > 4 else 'General task (Qwen3)' if complexity > 2 else 'Simple query (Compact SOTA)'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CODE_GENERATION:
            # Enhanced coding model selection with latest 2025 models
            complexity = analysis.get('complexity_score', 0)
            language = analysis.get('language_detected', 'python')
            
            # Select optimal coding model based on complexity and language
            selected_model = None
            if complexity > 7 or language in ['rust', 'go', 'c++', 'java', 'assembly'] or 'algorithm' in original_prompt.lower():
                selected_model = Config.ADVANCED_CODE_MODEL  # DeepSeek-Coder-V2-Instruct for complex algorithms
            elif complexity > 5 or word_count > 300:
                selected_model = Config.DEFAULT_CODE_MODEL  # Qwen2.5-Coder-32B (matches GPT-4o)
            elif complexity > 3:
                selected_model = Config.FAST_CODE_MODEL  # Qwen2.5-Coder-14B for balanced performance
            elif complexity > 1:
                selected_model = Config.EFFICIENT_CODE_MODEL  # Qwen2.5-Coder-7B (outperforms larger models)
            else:
                selected_model = Config.FALLBACK_CODE_MODEL  # StarCoder2-7B for basic tasks
            
            # Enhanced decision logging for code generation with latest models
            if complexity > 7:
                reasoning = "ADVANCED (DeepSeek-Coder-V2)"
            elif complexity > 5:
                reasoning = "DEFAULT (Qwen2.5-Coder-32B)"
            elif complexity > 3:
                reasoning = "FAST (Qwen2.5-Coder-14B)"
            elif complexity > 1:
                reasoning = "EFFICIENT (Qwen2.5-Coder-7B)"
            else:
                reasoning = "FALLBACK (StarCoder2-7B)"
                
            logger.info(f"💻 CODE_GENERATION: complexity={complexity}, language={language} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (2025 SOTA, Superior to GitHub Copilot/GPT-4o Code)")
            logger.info(f"⚡ REASONING: {'Complex algorithms (DeepSeek-V2)' if complexity > 7 else 'Production-ready code (Qwen-32B)' if complexity > 5 else 'Standard development (Qwen-14B)' if complexity > 3 else 'Efficient coding (Qwen-7B)' if complexity > 1 else 'Basic code snippets'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.IMAGE_ANALYSIS:
            # Vision models for image understanding and analysis
            prompt_lower = original_prompt.lower()
            selected_model = None
            if 'document' in prompt_lower or 'text' in prompt_lower or 'ocr' in prompt_lower:
                selected_model = Config.DOCUMENT_VISION_MODEL  # Florence-2 for OCR/documents
            elif 'complex' in prompt_lower or 'detailed' in prompt_lower:
                selected_model = Config.ADVANCED_VISION_MODEL  # Qwen2.5-VL-72B for complex vision
            elif 'fast' in prompt_lower or 'quick' in prompt_lower:
                selected_model = Config.FAST_VISION_MODEL  # Qwen2.5-VL-3B for fast inference
            else:
                selected_model = Config.DEFAULT_VISION_MODEL  # Qwen2.5-VL-7B (balanced)
            
            # Enhanced decision logging for image analysis
            is_document = 'document' in prompt_lower or 'text' in prompt_lower or 'ocr' in prompt_lower
            is_complex = 'complex' in prompt_lower or 'detailed' in prompt_lower
            reasoning = "DOCUMENT (Florence-2)" if is_document else "ADVANCED (Qwen2.5-VL-72B)" if is_complex else "DEFAULT (Qwen2.5-VL-7B)"
            logger.info(f"👁️ IMAGE_ANALYSIS: document={is_document}, complex={is_complex} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Beats GPT-4V/Claude Sonnet Vision)")
            logger.info(f"⚡ REASONING: {'OCR/Document processing' if is_document else 'Complex visual analysis' if is_complex else 'Standard image understanding'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.MULTI_MODAL:
            # Multimodal tasks combining text and vision
            selected_model = Config.ADVANCED_VISION_MODEL  # Qwen2.5-VL-72B (best multimodal)
            logger.info(f"🔄 MULTI_MODAL: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.IMAGE_GENERATION:
            # Enhanced image generation with Qwen-Image and latest FLUX models
            prompt_lower = original_prompt.lower()
            selected_model = None
            
            # Advanced model selection based on prompt requirements
            if 'text' in prompt_lower or 'chinese' in prompt_lower or 'typography' in prompt_lower:
                selected_model = Config.FLAGSHIP_IMAGE_MODEL  # Qwen-Image (20B, superior text rendering)
            elif 'edit' in prompt_lower or 'modify' in prompt_lower or 'change' in prompt_lower:
                selected_model = Config.EDITING_IMAGE_MODEL  # Qwen-Image-Edit (specialized editing)
            elif 'commercial' in prompt_lower or 'business' in prompt_lower or 'fast' in prompt_lower:
                selected_model = Config.COMMERCIAL_IMAGE_MODEL  # FLUX.1-schnell (1-4 steps, commercial)
            elif 'artistic' in prompt_lower or 'creative' in prompt_lower:
                selected_model = Config.ARTISTIC_IMAGE_MODEL  # SD3.5-Medium (artistic styles)
            elif 'control' in prompt_lower or 'precise' in prompt_lower:
                selected_model = Config.KONTEXT_IMAGE_MODEL  # FLUX.1-Kontext (in-context control)
            else:
                selected_model = Config.DEFAULT_IMAGE_MODEL  # FLUX.1-dev (best overall quality)
            
            # Enhanced decision logging for image generation with latest models
            has_text = 'text' in prompt_lower or 'chinese' in prompt_lower or 'typography' in prompt_lower
            is_editing = 'edit' in prompt_lower or 'modify' in prompt_lower
            is_commercial = 'commercial' in prompt_lower or 'business' in prompt_lower or 'fast' in prompt_lower
            is_artistic = 'artistic' in prompt_lower or 'creative' in prompt_lower
            is_controlled = 'control' in prompt_lower or 'precise' in prompt_lower
            
            if has_text:
                reasoning = "FLAGSHIP (Qwen-Image-20B)"
            elif is_editing:
                reasoning = "EDITING (Qwen-Image-Edit)"
            elif is_commercial:
                reasoning = "COMMERCIAL (FLUX.1-schnell)"
            elif is_artistic:
                reasoning = "ARTISTIC (SD3.5-Medium)"
            elif is_controlled:
                reasoning = "KONTEXT (FLUX.1-Kontext)"
            else:
                reasoning = "DEFAULT (FLUX.1-dev)"
                
            logger.info(f"🎨 IMAGE_GENERATION: text={has_text}, edit={is_editing}, commercial={is_commercial} → {reasoning}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (2025 SOTA, Superior to DALL-E 3/Midjourney/Firefly)")
            logger.info(f"⚡ REASONING: {'Superior text rendering (Qwen-Image)' if has_text else 'Advanced editing capabilities' if is_editing else 'Fast commercial generation' if is_commercial else 'Artistic style generation' if is_artistic else 'Precise control generation' if is_controlled else 'Best overall quality (FLUX.1-dev)'}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            # Enhanced sentiment analysis with emotion detection
            prompt_lower = original_prompt.lower()
            emotion_keywords = ['emotion', 'feeling', 'mood', 'angry', 'happy', 'sad', 'excited', 'fear', 'joy', 'love']
            
            selected_model = None
            if any(keyword in prompt_lower for keyword in emotion_keywords):
                selected_model = Config.EMOTION_MODEL  # Multi-class emotion detection (7 emotions)
            elif any(lang in prompt_lower for lang in ['spanish', 'french', 'german', 'italian', 'portuguese']):
                selected_model = Config.MULTILINGUAL_SENTIMENT_MODEL  # Multilingual sentiment (8+ languages)
            else:
                selected_model = Config.DEFAULT_SENTIMENT_MODEL  # SOTA sentiment model (124M tweets)
            
            logger.info(f"💭 SENTIMENT_ANALYSIS: emotion_detected={any(keyword in prompt_lower for keyword in emotion_keywords)}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.TRANSLATION:
            # Advanced translation models for 200+ languages
            prompt_lower = original_prompt.lower()
            selected_model = None
            if 'complex' in prompt_lower or 'document' in prompt_lower:
                selected_model = Config.DEFAULT_TRANSLATION_MODEL  # NLLB-200 (200+ languages, SOTA)
            elif 'to english' in prompt_lower:
                selected_model = Config.ADVANCED_TRANSLATION_MODEL  # Multilingual to English specialist
            else:
                selected_model = Config.DEFAULT_TRANSLATION_MODEL  # NLLB-200 (best overall)
            
            logger.info(f"🌍 TRANSLATION: complex_doc={('complex' in prompt_lower or 'document' in prompt_lower)}, to_english={'to english' in prompt_lower}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CREATIVE_WRITING:
            # Use large models for creative tasks requiring high quality
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            if complexity > 5:
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for complex creative reasoning
            else:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for creative writing
            
            logger.info(f"✍️ CREATIVE_WRITING: complexity={complexity}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        # 2025: NEW SPECIALIZED INTENT ROUTING - Superior to ChatGPT/Grok/Gemini
        elif intent == IntentType.MATHEMATICAL_REASONING:
            # Use specialized math model for calculations, proofs, equations
            selected_model = getattr(Config, 'MATH_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL)
            logger.info(f"📐 MATHEMATICAL_REASONING: model={selected_model} (Specialized math reasoning)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Superior to GPT-4 for mathematical problems)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.ADVANCED_REASONING:
            # Use QwQ model for complex logical reasoning tasks
            selected_model = getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL)
            logger.info(f"🧠 ADVANCED_REASONING: model={selected_model} (QwQ-32B reasoning specialist)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Beats o1-preview in reasoning benchmarks)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.ALGORITHM_DESIGN:
            # Use specialized coding model for complex algorithms
            selected_model = getattr(Config, 'SPECIALIZED_CODE_MODEL', Config.ADVANCED_CODE_MODEL)
            logger.info(f"⚡ ALGORITHM_DESIGN: model={selected_model} (CodeLlama-34B specialist)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Superior for competitive programming)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.SCIENTIFIC_ANALYSIS:
            # Use advanced reasoning model for scientific research tasks
            complexity = analysis.get('complexity_score', 0)
            selected_model = getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL) if complexity > 5 else Config.DEFAULT_TEXT_MODEL
            logger.info(f"🔬 SCIENTIFIC_ANALYSIS: complexity={complexity}, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Research-grade analysis)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.MEDICAL_ANALYSIS:
            # Use medical vision model for medical images, or advanced text for medical text
            prompt_lower = original_prompt.lower()
            if any(term in prompt_lower for term in ['image', 'x-ray', 'mri', 'ct scan', 'ultrasound']):
                selected_model = getattr(Config, 'MEDICAL_VISION_MODEL', Config.ADVANCED_VISION_MODEL)
                logger.info(f"🏥 MEDICAL_ANALYSIS: type=image, model={selected_model}")
            else:
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1 for medical text analysis
                logger.info(f"🏥 MEDICAL_ANALYSIS: type=text, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Medical specialist model)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CREATIVE_DESIGN:
            # Use artistic image generation for visual design, text model for design concepts
            prompt_lower = original_prompt.lower()
            if any(term in prompt_lower for term in ['ui', 'ux', 'mockup', 'wireframe', 'logo', 'visual']):
                selected_model = getattr(Config, 'ARTISTIC_IMAGE_MODEL', Config.DEFAULT_IMAGE_MODEL)
                logger.info(f"🎨 CREATIVE_DESIGN: type=visual, model={selected_model}")
            else:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for design concepts
                logger.info(f"🎨 CREATIVE_DESIGN: type=conceptual, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Design specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.EDUCATIONAL_CONTENT:
            # Use balanced model for teaching and tutorials
            complexity = analysis.get('complexity_score', 0)
            selected_model = Config.ADVANCED_TEXT_MODEL if complexity > 6 else Config.DEFAULT_TEXT_MODEL
            logger.info(f"📚 EDUCATIONAL_CONTENT: complexity={complexity}, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Optimized for teaching)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.BUSINESS_ANALYSIS:
            # Use advanced reasoning for business intelligence and analysis
            selected_model = getattr(Config, 'REASONING_TEXT_MODEL', Config.ADVANCED_TEXT_MODEL)
            logger.info(f"📊 BUSINESS_ANALYSIS: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Business intelligence specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.TECHNICAL_DOCUMENTATION:
            # Use coding-specialized model for technical docs
            selected_model = Config.ADVANCED_CODE_MODEL  # DeepSeek-Coder-V2 for technical writing
            logger.info(f"📝 TECHNICAL_DOCUMENTATION: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Technical writing specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.MULTILINGUAL_PROCESSING:
            # Use advanced translation model with cultural context
            selected_model = Config.DEFAULT_TRANSLATION_MODEL  # NLLB-200 for 200+ languages
            logger.info(f"🌐 MULTILINGUAL_PROCESSING: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (200+ language specialist)")
            return self._validate_model_selection(selected_model, intent)
        
        # 2025: BREAKTHROUGH SPECIALIZED INTENT ROUTING - Revolutionary capabilities
        elif intent == IntentType.GUI_AUTOMATION:
            # BREAKTHROUGH: UI-TARS native GUI automation - Superior to GPT-4V for GUI tasks
            selected_model = getattr(Config, 'DEFAULT_GUI_MODEL', Config.DEFAULT_VISION_MODEL)
            logger.info(f"🖱️ GUI_AUTOMATION: model={selected_model} (UI-TARS DPO trained)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (BREAKTHROUGH: Outperforms GPT-4V on GUI benchmarks)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.TOOL_USE:
            # BREAKTHROUGH: Groq function calling excellence - #1 on BFCL leaderboard
            complexity = analysis.get('complexity_score', 0)
            if complexity > 6 or 'complex' in original_prompt.lower():
                selected_model = getattr(Config, 'DEFAULT_TOOL_MODEL', Config.ADVANCED_CODE_MODEL)  # 70B Tool Use
            else:
                selected_model = getattr(Config, 'EFFICIENT_TOOL_MODEL', Config.DEFAULT_CODE_MODEL)  # 8B Tool Use
            logger.info(f"🔧 TOOL_USE: complexity={complexity}, model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (BREAKTHROUGH: #1 on BFCL leaderboard, 90.76% accuracy)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.PREMIUM_VISION:
            # BREAKTHROUGH: MiniCPM-V excellence - Beats GPT-4V on OCRBench
            selected_model = getattr(Config, 'PREMIUM_VISION_MODEL', Config.DEFAULT_VISION_MODEL)
            logger.info(f"👁️ PREMIUM_VISION: model={selected_model} (MiniCPM-Llama3-V-2.5)")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (BREAKTHROUGH: 700+ OCRBench, beats GPT-4V)")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.SYSTEM_INTERACTION:
            # Advanced system-level interactions
            selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-0528 for complex system tasks
            logger.info(f"⚙️ SYSTEM_INTERACTION: model={selected_model}")
            logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Advanced reasoning for system tasks)")
            return self._validate_model_selection(selected_model, intent)

        elif intent == IntentType.QUESTION_ANSWERING:
            # Use large models for complex Q&A requiring deep knowledge
            complexity = analysis.get('complexity_score', 0)
            selected_model = None
            if complexity > 6 or 'complex' in original_prompt.lower():
                selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for complex reasoning
            else:
                selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B (131K context)
            
            logger.info(f"❓ QUESTION_ANSWERING: complexity={complexity}, complex_keyword={'complex' in original_prompt.lower()}, model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.DATA_ANALYSIS:
            # Data analysis and processing tasks
            selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for analytical reasoning
            logger.info(f"📈 DATA_ANALYSIS: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.DOCUMENT_PROCESSING:
            # Document processing with vision capabilities
            selected_model = Config.DOCUMENT_VISION_MODEL  # Florence-2 (excellent OCR)
            logger.info(f"📄 DOCUMENT_PROCESSING: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.PDF_PROCESSING:
            # PDF analysis and processing
            selected_model = Config.DOCUMENT_VISION_MODEL  # Florence-2 for document understanding
            logger.info(f"📁 PDF_PROCESSING: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.ZIP_ANALYSIS:
            # ZIP file analysis
            selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill for file structure analysis
            logger.info(f"🗜️ ZIP_ANALYSIS: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.FILE_GENERATION:
            # File generation tasks
            selected_model = Config.ADVANCED_CODE_MODEL  # DeepSeek-Coder-V2 for generating files
            logger.info(f"📁 FILE_GENERATION: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        elif intent == IntentType.CONVERSATION:
            # Conversational AI
            selected_model = Config.DEFAULT_TEXT_MODEL  # Qwen2.5-14B for responsive conversation
            logger.info(f"💬 CONVERSATION: model={selected_model}")
            return self._validate_model_selection(selected_model, intent)
        
        # Default fallback to our most powerful reasoning model
        selected_model = Config.ADVANCED_TEXT_MODEL  # DeepSeek-R1-Distill (matches o1 performance)
        logger.info(f"⚠️ FALLBACK: intent={intent.value}, model={selected_model}")
        
        # Enhanced fallback logging
        logger.info(f"⚠️ FALLBACK: intent={intent.value} → ADVANCED (DeepSeek-R1-Distill)")
        logger.info(f"🎯 MODEL_SELECTED: {selected_model} (Superior to OpenAI o1/Claude Sonnet)")
        logger.info(f"⚡ REASONING: Fallback to most powerful reasoning model for unknown intent")
        
        # Validate model selection and return
        validated_model = self._validate_model_selection(selected_model, intent)
        logger.info(f"✅ FINAL_VALIDATION: model={validated_model} for intent={intent.value}")
        logger.info(f"🏆 ═══════════════════════ ROUTER DECISION COMPLETE ═══════════════════════\n")
        return validated_model
    
    def _get_special_parameters(self, intent: IntentType, analysis: Dict) -> Dict:
        """Get optimized parameters for 2024-2025 SOTA models"""
        from bot.config import Config
        base_params = {}
        complexity = analysis.get('complexity_score', 0)
        
        if intent == IntentType.CODE_GENERATION:
            # Optimized for DeepSeek-Coder-V2 and StarCoder2 series
            base_params.update({
                'temperature': Config.STARCODER_TEMPERATURE,  # 0.2 for precise code
                'max_new_tokens': 1500 if complexity > 5 else 1200,
                'top_p': 0.95,
                'do_sample': True,
                'language': analysis.get('language_detected', 'python'),
                'use_flash_attention': True,  # DeepSeek optimization
                'use_advanced_model': True
            })
        
        elif intent == IntentType.IMAGE_ANALYSIS or intent == IntentType.MULTI_MODAL:
            # Optimized for Qwen2.5-VL and PaliGemma2 series
            base_params.update({
                'temperature': Config.VISION_TEMPERATURE,  # 0.5 for vision tasks
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'image_size': Config.QWEN_VL_IMAGE_SIZE,  # 448 optimal for Qwen2.5-VL
                'dynamic_resolution': Config.QWEN_VL_DYNAMIC_RESOLUTION,
                'vision_mode': True
            })
        
        elif intent == IntentType.IMAGE_GENERATION:
            # Optimized for FLUX.1 and SD3.5 series (SUPERIOR to DALL-E)
            base_params.update({
                'guidance_scale': 7.5,
                'num_inference_steps': Config.FLUX_INFERENCE_STEPS,  # 4 for FLUX.1-schnell
                'width': Config.FLUX_MAX_RESOLUTION,  # 1024
                'height': Config.FLUX_MAX_RESOLUTION,  # 1024
                'enhanced_prompt': True,
                'flux_turbo_mode': Config.FLUX_SCHNELL_TURBO_MODE,
                'use_flux_dev': True  # Best quality model
            })
        
        elif intent == IntentType.CREATIVE_WRITING:
            # Optimized for Qwen2.5-72B and DeepSeek-R1 creative tasks
            base_params.update({
                'temperature': 0.8 if complexity < 5 else Config.QWEN_TEMPERATURE,  # 0.7
                'max_new_tokens': 2000 if complexity > 5 else 1500,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'creative_mode': True,
                'use_long_context': True  # Leverage 131K context
            })
        
        elif intent == IntentType.SENTIMENT_ANALYSIS:
            # Optimized for latest sentiment and emotion models
            base_params.update({
                'return_all_scores': True,  # Get confidence for all emotions
                'use_emotion_detection': 'emotion' in str(analysis).lower(),
                'multilingual_support': True,
                'confidence_threshold': 0.7
            })
        
        elif intent == IntentType.TRANSLATION:
            # Optimized for NLLB-200 and multilingual models
            base_params.update({
                'temperature': 0.3,  # Lower for accurate translation
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'beam_search': True,  # Better translation quality
                'num_beams': 4,
                'multilingual_mode': True
            })
        
        elif intent in [IntentType.TEXT_GENERATION, IntentType.QUESTION_ANSWERING]:
            # Optimized for DeepSeek-R1 and Qwen2.5 series
            base_params.update({
                'temperature': Config.DEEPSEEK_TEMPERATURE if complexity > 6 else Config.QWEN_TEMPERATURE,
                'max_new_tokens': Config.DEEPSEEK_MAX_TOKENS if complexity > 6 else 1500,
                'top_p': 0.9,
                'repetition_penalty': 1.05,
                'advanced_reasoning': complexity > 3,
                'use_flash_attention': Config.DEEPSEEK_USE_FLASH_ATTENTION,
                'long_context_support': True  # 131K for Qwen2.5
            })
        
        elif intent == IntentType.DATA_ANALYSIS:
            # Optimized for analytical reasoning tasks
            base_params.update({
                'temperature': Config.DEEPSEEK_TEMPERATURE,  # 0.8 for reasoning
                'max_new_tokens': 2000,
                'top_p': 0.9,
                'analytical_mode': True,
                'use_advanced_reasoning': True
            })
        
        elif intent in [IntentType.DOCUMENT_PROCESSING, IntentType.PDF_PROCESSING]:
            # Optimized for PaliGemma2 document understanding
            base_params.update({
                'temperature': 0.3,  # Lower for accurate document analysis
                'max_new_tokens': 1500,
                'top_p': 0.9,
                'ocr_mode': True,
                'document_understanding': True,
                'resolution_mode': 896  # PaliGemma2 max resolution
            })
        
        elif intent == IntentType.CONVERSATION:
            # Optimized for conversational AI
            base_params.update({
                'temperature': 0.7,
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'conversational_mode': True,
                'response_speed_optimized': True
            })
        
        else:
            # Default parameters for DeepSeek-R1 fallback
            base_params.update({
                'temperature': Config.DEEPSEEK_TEMPERATURE,  # 0.8
                'max_new_tokens': 1000,
                'top_p': 0.9,
                'use_advanced_model': True
            })
        
        return base_params
    
    def _apply_intent_heuristics(self, prompt: str, primary_intent: IntentType, intent_scores: Dict, analysis: Dict) -> IntentType:
        """Apply additional heuristics to refine intent detection"""
        prompt_lower = prompt.lower()
        
        # Special case: "Explain" + philosophical/theoretical concepts should be TEXT_GENERATION, not CODE
        if (prompt_lower.strip().startswith('explain') and 
            any(term in prompt_lower for term in ['philosophical', 'implications', 'consciousness', 'theoretical', 'principles'])):
            return IntentType.TEXT_GENERATION
        
        # Special case: If prompt starts with "what" and mentions technical concepts, prioritize Q&A
        if (prompt_lower.strip().startswith('what') and 
            any(term in prompt_lower for term in ['artificial intelligence', 'machine learning', 'ai', 'technology', 'computer', 'programming'])):
            return IntentType.QUESTION_ANSWERING
        
        # Special case: "Draw/paint" + descriptive words should be image generation
        if (re.search(r'\b(?:draw|paint|sketch)\s+(?:a|an)?\s*(?:beautiful|stunning|amazing|gorgeous|lovely|pretty)', prompt_lower) or
            re.search(r'\b(?:draw|paint|sketch)\s+(?:a|an)?\s*(?:sunset|sunrise|landscape|portrait|scene)', prompt_lower)):
            return IntentType.IMAGE_GENERATION
        
        # Special case: "sentiment of" should always be sentiment analysis
        if re.search(r'\b(?:sentiment|feeling|emotion)\s+of\b', prompt_lower):
            return IntentType.SENTIMENT_ANALYSIS
        
        # Special case: Explicit PDF analysis should be PDF_PROCESSING
        if re.search(r'\b(?:analyze|process)\s+(?:this\s+)?pdf\s+(?:document|file)?', prompt_lower):
            return IntentType.PDF_PROCESSING
        
        # Special case: Code-specific terms WITH programming verbs should prioritize code generation
        if (analysis.get('has_technical_terms') and 
            re.search(r'\b(?:write|create|implement|build|develop|generate)\s+(?:a|an)?\s*(?:function|class|method|algorithm|code|script|program)\b', prompt_lower)):
            return IntentType.CODE_GENERATION
        
        return primary_intent
    
    def get_model_performance_feedback(self, intent: IntentType, success: bool, response_quality: str) -> Dict:
        """
        Collect performance feedback for model improvement
        
        Args:
            intent (IntentType): The intent that was processed
            success (bool): Whether the request was successful
            response_quality (str): Quality assessment (good/average/poor)
            
        Returns:
            Dict: Performance feedback data
        """
        return {
            'intent': intent.value,
            'success': success,
            'quality': response_quality,
            'timestamp': None,  # Will be set by caller
            'model_used': self._get_recommended_model(intent, {})
        }

# Global router instance
router = IntelligentRouter()