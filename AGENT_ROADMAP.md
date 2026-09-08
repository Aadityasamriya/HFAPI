# HFAPI Agent Roadmap

HFAPI is evolving from a Telegram AI bot into a lightweight, Hugging Face-native agent platform.

## Current direction

- Telegram-first control plane for task status, cancellation, pause/resume, and approvals.
- Bounded multi-step execution: every task has step, time, and tool-call budgets.
- Human approval before destructive operations, including deletion of temporary Hugging Face Spaces.
- Provider-agnostic core with Hugging Face as the first-class open-source capability ecosystem.
- Persistent, user-controlled memory with view, edit, delete, and forget operations.
- Capability-on-demand: prefer lightweight local tools, otherwise use remote Hugging Face resources.

## Near-term implementation order

1. Add a small agent runtime with explicit state transitions and resumable task records.
2. Add a typed tool registry with permission scopes and deterministic verification hooks.
3. Add approval events and Telegram inline actions for destructive operations.
4. Add memory CRUD with provenance, confidence, timestamps, and retention controls.
5. Add Hugging Face model/Space discovery adapters with secret-safe logging.
6. Add end-to-end tests and deployment guidance for the lightweight profile.

## Non-goals

HFAPI will not silently delete user resources, execute arbitrary code without policy controls, or rewrite its own production code without reviewable changes.
