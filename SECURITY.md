# Security Policy

## Supported versions

The `main` branch is the actively maintained version of HFAPI.

## Reporting a vulnerability

Please do **not** publish sensitive vulnerability details, credentials, tokens, database URLs, encryption material, or exploitable proof-of-concept payloads in a public issue.

When private security reporting is available for this repository, use GitHub's private vulnerability reporting flow. Otherwise, contact the repository maintainer privately before public disclosure.

Include:

- A concise description of the vulnerability.
- The affected component or file.
- Reproduction steps that do not expose real secrets or user data.
- Security impact and likely attack surface.
- Any suggested mitigation, if known.

## Secret handling

Never commit:

- Telegram bot tokens
- Hugging Face tokens
- Database credentials or connection strings containing credentials
- Encryption seeds or keys
- User personal data
- Production logs containing sensitive information

Use environment variables or the deployment platform's secret manager instead.

## Response expectations

Security reports are prioritized above feature requests. Maintainers should validate the issue, assess impact, create a minimal safe fix, add regression coverage where practical, and verify the fix before considering the incident resolved.
