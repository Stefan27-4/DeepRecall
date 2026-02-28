# DeepRecall — OpenClaw Skill Definition

## Description
Recursive memory recall for persistent AI agents using RLM (Recursive Language Models).
Implements the Anamnesis Architecture: "The soul stays small, the mind scales forever."

## When to Use
- Agent needs to recall information from a large/growing memory store
- Memory files exceed the context window (~128K tokens)
- Agent needs to cross-reference across many daily log files
- Deep reasoning about historical decisions, patterns, or events
- Finding specific information across weeks/months of conversation history

## Requirements

### System Dependencies
- **[Deno](https://deno.com) 2+** — runtime for fast-rlm (install via your package manager or `curl -fsSL https://deno.land/install.sh | sh`)
- **[fast-rlm](https://github.com/avbiswas/fast-rlm)** — the RLM engine. Clone it locally and set `FAST_RLM_DIR=/path/to/fast-rlm`

### Python Dependencies
- **PyYAML** (`pip install pyyaml`) — for RLM config generation

### OpenClaw Configuration (Required)
- A configured LLM provider in OpenClaw (Anthropic, OpenAI, Google, OpenRouter, Ollama, etc.)
- The skill **reads your OpenClaw config and credential files** (`~/.openclaw/`) to auto-detect your provider and API key
- No separate API key configuration needed — it reuses your existing OpenClaw setup

### Environment Variables (Optional)
- `FAST_RLM_DIR` — path to your fast-rlm clone (auto-detected if in common locations)
- `OPENCLAW_WORKSPACE` — override workspace path (defaults to `~/.openclaw/workspace`)

## What This Skill Accesses

**Be aware of what DeepRecall reads and where data goes:**

| What | Why | Where It Goes |
|------|-----|---------------|
| `~/.openclaw/openclaw.json` | Find your LLM provider + model | Stays local |
| `~/.openclaw/credentials/` | Read API keys (e.g. GitHub Copilot token) | Passed to fast-rlm as env var |
| Workspace files (scope-dependent) | Build memory context for the query | Sent to your configured LLM provider via fast-rlm |

- **Scopes `identity` and `memory`**: only reads soul/mind files and daily logs
- **Scopes `project` and `all`**: reads all readable text files in your workspace (up to 100KB each)
- **All assembled context is sent to your LLM provider** — the same provider you already use with OpenClaw

### Privacy Recommendations
- Use **`identity` or `memory` scope** unless you need broader search
- For maximum privacy, configure a **local model provider** (Ollama) so data never leaves your machine
- **Audit [fast-rlm](https://github.com/avbiswas/fast-rlm)** before running — it executes locally and receives your API key

## Files
- `deep_recall.py` — Main entry point (recall, recall_quick, recall_deep)
- `provider_bridge.py` — Reads OpenClaw config to resolve API keys + models
- `model_pairs.py` — Auto-selects cheap sub-agent model
- `memory_scanner.py` — Discovers and indexes agent memory files
- `memory_indexer.py` — Generates MEMORY_INDEX.md for efficient navigation
- `rlm_config_builder.py` — Generates fast-rlm config

## Quick Start
```python
from deep_recall import recall
result = recall("What did we discuss last week about the project?")
```

## Configuration
No additional config needed — DeepRecall reads your existing OpenClaw setup.
Override RLM settings via `config_overrides` parameter.
