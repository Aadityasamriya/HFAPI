# Contributing to HFAPI

Thanks for helping improve HFAPI.

## Before changing code

- Read the relevant module and its existing tests first.
- Check the current GitHub Actions status.
- Prefer fixing the smallest root cause rather than adding a workaround.
- Do not commit secrets, private data, generated credentials, or local runtime artifacts.

## Pull requests

A good pull request should:

- Explain the problem and the intended behavior.
- Keep unrelated refactors out of the change.
- Include regression tests for bug fixes where practical.
- Update documentation when public behavior or configuration changes.
- Keep error handling and logging free of secrets.
- Pass the repository's automated quality gate.

## Local verification

```bash
python -m compileall -q bot *.py
python -m pytest -q
```

## Architecture guidance

Keep the major boundaries clear:

- Telegram handlers handle transport and user interaction.
- Routing and model selection make AI-selection decisions.
- Providers perform model-service calls behind the provider interface.
- Storage providers own persistence details.
- Security utilities own secret-safe logging and cryptographic helpers.
- Health checks should report real operational state without exposing sensitive internals.

Avoid adding more responsibilities to already-large modules when a focused component can provide a cleaner boundary.

## Security

For suspected vulnerabilities, follow [`SECURITY.md`](SECURITY.md) instead of publishing sensitive details in a normal issue.
