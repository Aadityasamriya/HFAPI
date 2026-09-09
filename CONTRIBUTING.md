# Contributing to HFAPI

Thanks for helping improve HFAPI. The project values correctness, security, privacy, reliability, and a smooth Telegram + web experience.

## Before changing code

- Read the relevant module and its existing tests first.
- Check the current GitHub Actions status.
- Prefer fixing the smallest root cause rather than adding a workaround.
- Do not commit Telegram tokens, Hugging Face tokens, database URLs, encryption seeds, private data, generated credentials, screenshots containing user data, or local runtime artifacts.

## Development workflow

```bash
python -m venv .venv
# Activate the environment for your platform.
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
python -m compileall -q bot tests *.py
python -m pip check
```

Prefer deterministic tests that do not call Telegram or Hugging Face services. For configuration, routing, security, health-server, and UI changes, add regression coverage for the important contract and failure path.

## Pull requests

A good pull request should:

- explain the problem and the intended behavior;
- keep unrelated refactors out of the change;
- include regression tests for bug fixes where practical;
- update documentation when public behavior or configuration changes;
- keep error handling and logging free of secrets;
- include the verification commands used and the resulting CI status.

Do not claim a fix is complete until the relevant checks pass. If a check cannot run locally, state that clearly and rely on the GitHub Actions quality gate for the final repository verification.

## Architecture guidance

Keep the major boundaries clear:

- Telegram handlers handle transport and user interaction.
- Routing and model selection make AI-selection decisions.
- Providers perform model-service calls behind the provider interface.
- Storage providers own persistence details.
- Security utilities own secret-safe logging and cryptographic helpers.
- Health checks should report real operational state without exposing sensitive internals.
- The web control center and Telegram bot should continue to use the same deployed process and platform port unless a change is intentional and documented.

Avoid adding more responsibilities to already-large modules when a focused component can provide a cleaner boundary. Preserve compatibility unless a behavior change is clearly justified.

## Commit guidance

Use short, imperative commit messages such as:

- `fix(server): reject invalid platform ports`
- `test(router): cover fallback selection`
- `docs: clarify local verification`

Avoid unrelated formatting churn and generated artifacts.

## Security

For suspected vulnerabilities, follow [`SECURITY.md`](SECURITY.md) instead of publishing sensitive details in a normal issue.
