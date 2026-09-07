# HFAPI Agent Capability Matrix

HFAPI is being evolved into a Hugging Face–native, open-source agent platform with a small core and capability-on-demand skills.

## Design goals

1. **Maximum capability with minimum local footprint.** Heavy inference and specialized workloads should prefer remote Hugging Face resources when practical.
2. **Hugging Face first.** Models and Spaces are first-class capability providers; provider adapters remain replaceable.
3. **Dynamic skills.** Capabilities are discoverable and composable rather than hard-coded into the chat interface.
4. **Agentic execution.** Plan → act → observe → verify → recover/re-plan.
5. **Persistent memory.** Useful user preferences, task outcomes, and reusable experience can improve future execution while remaining user-controllable.
6. **Open source.** Core interfaces should use permissive, documented contracts and avoid proprietary runtime dependencies.

## Capability families

### Core intelligence
- conversational reasoning
- task decomposition and planning
- context management
- model selection and routing
- structured outputs
- tool calling
- retry and recovery
- verification and evaluation
- task checkpoints and resumability

### Knowledge and research
- web research adapters
- document understanding
- summarization
- extraction and transformation
- source-aware synthesis
- comparison and fact checking

### Coding and software engineering
- repository inspection
- code generation and editing
- debugging
- test generation
- test execution adapters
- static-analysis adapters
- dependency analysis
- Git/GitHub workflows
- documentation generation
- release assistance

### Files and data
- file discovery and inspection
- text/JSON/CSV transformation
- PDF/document processing
- image understanding
- archive processing with safety limits
- structured data analysis

### Automation and computer interaction
- browser adapters
- HTTP/API adapters
- local process adapters (explicitly sandboxed)
- scheduled jobs
- workflow composition
- event-driven triggers

### Media and multimodal AI
- image generation/editing adapters
- image understanding
- speech-to-text
- text-to-speech
- audio processing
- video processing adapters
- OCR

### Hugging Face ecosystem
- token-based authentication
- model discovery
- Space discovery
- capability metadata inspection
- inference routing
- Space/API adapters
- model compatibility checks
- resource-aware selection
- reusable HF capability registration

### Agent ecosystem
- skill registry
- versioned skill manifests
- capability discovery
- skill permissions
- sandbox policies
- human approval gates
- agent events/telemetry
- multi-agent orchestration
- reusable workflows
- experience/memory store

## Capability-on-demand policy

HFAPI should not require every capability at installation time. A task should resolve capabilities in this order:

1. Reuse a verified installed skill.
2. Use a lightweight built-in capability.
3. Discover an authorized Hugging Face model or Space.
4. Use an approved remote/open-source adapter.
5. Request/install an optional dependency only when required and permitted.
6. Execute in a constrained environment.
7. Verify the result.
8. Persist successful capability metadata for future use.

Destructive actions, arbitrary code execution, secret access, and untrusted software installation must never be silently enabled by capability discovery.

## Quality bar

A capability is considered production-ready only when it has a clear input/output contract, permission scope, bounded execution, failure handling, verification strategy, tests, and documentation.

This matrix is a roadmap and does not claim every capability is currently implemented.
