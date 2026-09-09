# HFAPI

A production-oriented Telegram AI assistant built around Hugging Face inference, intelligent request routing, secure user data handling, file processing, persistent conversations, and operational health monitoring.

## What it provides

- **Telegram-first AI experience** using `python-telegram-bot`.
- **Hugging Face inference integration** with a provider abstraction and standardized responses.
- **Intent-aware routing** and prompt-complexity analysis for model selection.
- **Dynamic model selection** with performance signals, health awareness, conversation context, and fallback strategies.
- **Persistent conversation storage** through the project's storage abstraction.
- **Encrypted user-sensitive data handling** with authenticated encryption and per-user derivation context.
- **File processing** for supported documents, PDFs, images, archives, and OCR workflows.
- **Admin tooling** for operational management and controlled bot administration.
- **Production health checks** and Railway-friendly deployment support.
- **Automated quality gates** for dependency consistency, Python compilation, and the deterministic test suite.

## Architecture

```text
Telegram user
    │
    ▼
Telegram handlers
    │
    ├── authentication / rate limiting
    ├── conversation context
    ├── file processing
    └── intent classification
    │
    ▼
Intelligent router
    │
    ├── complexity analysis
    ├── domain / reasoning signals
    ├── model health
    └── fallback strategy
    │
    ▼
Dynamic model selector
    │
    ▼
Provider abstraction
    │
    ▼
Hugging Face inference
    │
    ▼
Response processing → Telegram
```

## Requirements

- Python **3.12**
- A Telegram bot token
- A Hugging Face access token for AI inference
- A supported persistent storage configuration
- `ENCRYPTION_SEED` for production deployments

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Start from `.env.example` and configure the required secrets in your deployment environment. **Never commit real credentials, tokens, database URLs, or encryption seeds.**

For production, configure a stable `ENCRYPTION_SEED`; do not rely on generated development values.

## Run locally

```bash
python main.py
```

The application validates dependencies, configuration, security requirements, and database connectivity during startup. A health server is also started for deployment monitoring.

## Testing

Run the complete test suite:

```bash
python -m pytest -q
```

Compile-check both application and test Python files:

```bash
python -m compileall -q bot tests *.py
```

Check dependency consistency:

```bash
python -m pip check
```

GitHub Actions runs these quality checks on pushes to `main` and pull requests targeting `main`.

## Deployment

The repository includes Railway-oriented deployment configuration and health monitoring. Keep production secrets in the deployment platform's secret/environment-variable store rather than in Git.

## Security

Security-sensitive behavior includes encrypted user data handling, secret redaction in logging utilities, production configuration validation, and admin controls. Security issues should **not** be disclosed publicly in an issue before maintainers have had an opportunity to assess them.

See [`SECURITY.md`](SECURITY.md) for the reporting policy.

## Development principles

HFAPI is actively improved as a long-lived project. Changes should:

1. Fix correctness and security issues before cosmetic work.
2. Preserve existing behavior unless a change is intentional and documented.
3. Add regression tests for important bug fixes.
4. Avoid leaking secrets or sensitive user data into logs and diagnostics.
5. Keep provider and storage boundaries explicit.
6. Prefer small, reviewable changes over speculative rewrites.
7. Verify changes with tests and CI before calling them complete.

## Project documentation

The repository contains additional operational, API, security, architecture, and verification documentation. Treat executable code and automated tests as the source of truth when older reports disagree with current behavior.
