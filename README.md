# DeepRecall — Recursive Memory for AI Agents

> **v1.0.4** — Pure Python RLM. No Deno, no fast-rlm, no external runtimes.

DeepRecall is an [OpenClaw](https://github.com/openclaw/openclaw) skill that gives AI agents **infinite memory** using a Recursive Language Model (RLM) loop. Instead of cramming everything into the context window, the agent recursively queries its own memory files through a manager → workers → synthesis pipeline — entirely in Python.

**Architecture:** Anamnesis Architecture
**Principle:** *"The soul stays small, the mind scales forever."*

## The Problem

AI agents face an impossible tradeoff: **remember who they are** (personality, identity) or **remember what happened** (conversations, decisions). The more they remember, the less room for identity. Agents literally forget who they are.

## The Solution

DeepRecall separates the agent into **Soul** (small, always in context), **Index** (topic map), and **Mind** (infinite, queried recursively):

```
┌─────────────────────────────────────────────┐
│              SOUL (Small, Fixed)            │
│  Identity, values, personality, core rules  │
│  Always in context. Never grows. ~2-5K tkns │
├─────────────────────────────────────────────┤
│          INDEX (Small, Auto-loaded)         │
│  MEMORY.md — orientation + topic index      │
│  ~2-4K tokens                               │
├─────────────────────────────────────────────┤
│         WORKING MEMORY (Context Window)     │
│  Current conversation + recall results      │
├─────────────────────────────────────────────┤
│             MIND (Infinite, External)       │
│  LONG_TERM.md + daily logs + project files  │
│  Queried via RLM — never fully loaded       │
└─────────────────────────────────────────────┘
         ↕ Pure Python RLM Bridge
```

## Installation

### Prerequisites

- Python 3.10+
- [OpenClaw](https://github.com/openclaw/openclaw) installed and configured
- Any supported LLM provider (Anthropic, OpenAI, Google, GitHub Copilot, etc.)

### Install

```bash
clawhub install deep-recall
```

That's it. No Deno, no TypeScript, no external runtimes.

## Usage

### From Python

```python
from deep_recall import recall, recall_quick, recall_deep

# Basic memory query
result = recall("What did we decide about the project architecture?")

# Quick recall — identity scope, cheapest
result = recall_quick("What is my human's name?")

# Deep recall — searches all workspace files
result = recall_deep("Summarize all decisions we made in the last month")

# Custom options
result = recall(
    "Find all mentions of budget discussions",
    scope="all",              # "memory", "identity", "project", "all"
    verbose=True,
    config_overrides={
        "max_files": 5,       # max files the manager can select (default: 3)
    }
)
```

### Scopes

| Scope | Files Included | Speed | Cost | Use Case |
|-------|---------------|-------|------|----------|
| `identity` | Soul + mind files | ⚡ Fastest | 💰 Cheapest | "What's my name?" |
| `memory` | Identity + daily logs | 🔄 Fast | 💰💰 Low | "What did we do last week?" |
| `project` | All workspace files | 🐢 Slow | 💰💰💰 Medium | "Find that config change" |
| `all` | Everything | 🐌 Slowest | 💰💰💰💰 High | "Search everything for X" |

## Architecture

DeepRecall v1.0.4 runs a **pure Python RLM loop** with three stages:

```
┌──────────────────────────────────────────────────────────────┐
│  1. MANAGER (LLM call)                                       │
│     Receives: memory index + user query                      │
│     Returns:  list of files most likely to contain the answer │
├──────────────────────────────────────────────────────────────┤
│  2. WORKERS (parallel LLM calls via ThreadPoolExecutor)      │
│     Each worker reads one file and extracts exact quotes      │
│     Anti-hallucination: must return verbatim text only        │
├──────────────────────────────────────────────────────────────┤
│  3. SYNTHESIS (LLM call)                                     │
│     Composes a coherent answer from all extracted quotes      │
│     Cites sources (filename:line), notes contradictions       │
└──────────────────────────────────────────────────────────────┘
```

### How a query flows

1. **Scan** — `MemoryScanner` discovers workspace files and categorizes them (soul, mind, daily-log, long-term)
2. **Index** — `MemoryIndexer` builds topic → file mappings
3. **Manager** — LLM selects the most relevant files for the query
4. **Workers** — Parallel LLM calls extract exact quotes from each file (anti-hallucination prompts enforce verbatim extraction)
5. **Synthesis** — LLM composes a cited answer from all quotes
6. **Return** — Human-readable response with `(filename:line)` references

### RLM vs Traditional Approaches

| Feature | RAG / Vector Search | DeepRecall (RLM) |
|---------|-------------------|------------------|
| Method | Keyword/vector match | Agent reasons about file structure |
| Cross-reference | Limited | Connects dots across files |
| Structure-aware | No | Reads headers, sections, dates |
| Infrastructure | Vector DB + embeddings | None — just files |
| Privacy | Data may leave your machine | Can be fully local |
| Git-trackable | No | Yes — it's all markdown |

## Provider Auto-Detection

DeepRecall reads your OpenClaw config and **automatically detects** your LLM provider. No extra API keys to configure.

### Resolution order

1. Read primary model from `~/.openclaw/openclaw.json`
2. Detect provider from model prefix (e.g. `anthropic/claude-opus-4` → Anthropic)
3. Resolve API key: OpenClaw config → environment variable → credential files
4. For GitHub Copilot: reads token from `~/.openclaw/credentials/github-copilot.token.json` with expiry checking

### Supported Providers (20+)

| Provider | Env Variable | Notes |
|----------|-------------|-------|
| **OpenClaw** | *(from config)* | Primary config source |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude models |
| **Google / Gemini** | `GOOGLE_API_KEY` | Gemini models |
| **OpenAI** | `OPENAI_API_KEY` | GPT models |
| **GitHub Copilot** | *(token file)* | Auto-token with expiry check |
| OpenRouter | `OPENROUTER_API_KEY` | Multi-provider gateway |
| DeepSeek | `DEEPSEEK_API_KEY` | |
| Mistral | `MISTRAL_API_KEY` | |
| Together | `TOGETHER_API_KEY` | |
| Groq | `GROQ_API_KEY` | |
| Fireworks | `FIREWORKS_API_KEY` | |
| Ollama | *(none)* | Local models on `localhost:11434` |
| + 8 more | | Cohere, Perplexity, SambaNova, Cerebras, xAI, Minimax, Zhipu, Moonshot |

### Auto Model Pairing

Your primary model orchestrates. A cheaper model handles file reading:

| Your Primary Model | Sub-agent (Worker) Model |
|---|---|
| Claude Opus 4 / Opus 4.6 | Claude Sonnet 4 |
| Claude Sonnet 4 / 4.5 | Claude Haiku 3.5 |
| GPT-4o / GPT-4 / GPT-4-Turbo | GPT-4o-mini |
| GPT-5 | GPT-5 mini |
| Gemini 2.5 Pro | Gemini 2.0 Flash |
| DeepSeek Reasoner | DeepSeek Chat |
| Llama 3.1 70B | Llama 3.1 8B |

20+ model pairs supported. Override via `config_overrides`.

## Configuration

### Default Settings

```yaml
max_files: 3              # Max files the manager can select per query
timeout: 120              # HTTP timeout in seconds per LLM call
```

Override any setting via `config_overrides`:

```python
result = recall("query", config_overrides={"max_files": 5})
```

## Memory File Structure

DeepRecall understands the standard OpenClaw workspace layout:

```
~/.openclaw/workspace/
├── SOUL.md            # Agent identity — always in context
├── IDENTITY.md        # Core facts about the agent
├── MEMORY.md          # Compact orientation + topic index (~100 lines)
├── USER.md            # About the human
├── AGENTS.md          # Agent behavior rules
├── TOOLS.md           # Tool-specific notes
└── memory/
    ├── LONG_TERM.md   # Full detailed memories — grows forever
    ├── 2026-02-24.md  # Daily log
    ├── 2026-02-25.md
    └── ...
```

### Three-Tier Memory Model

1. **MEMORY.md (Index)** — Auto-loaded every session. Topic → file index. Stays under ~120 lines.
2. **memory/LONG_TERM.md (Full Memories)** — Everything important. Grows forever, never deleted. Searched via RLM.
3. **memory/YYYY-MM-DD.md (Daily Logs)** — Raw notes per day. End of day, distill into LONG_TERM.md.

```
Session starts → MEMORY.md auto-loads (small, orientation)
Need specifics → DeepRecall searches LONG_TERM.md + daily files
During the day → Write raw logs to daily file
End of day    → Distill → LONG_TERM.md, update index if needed
```

## What It Can Do

- **Recall specific facts** from months of conversation logs
- **Synthesize across files** — connects information from daily logs, project docs, and memory files
- **Navigate large document collections** — tested on 800-page textbook (59 files, 2.9MB) in 7–15 seconds
- **Auto-detect your LLM provider** — works with 20+ providers out of the box
- **Anti-hallucination** — workers extract verbatim quotes; synthesis cites sources
- **Cost-efficient** — cheap sub-agent models for file reading ($0.005–$0.15 per query)
- **Zero infrastructure** — no vector database, no embeddings, no external runtime. Just markdown + an LLM

## Limitations

- **Query specificity matters** — broad queries may pull from the wrong section. Specific queries nail it
- **Not instant** — 7–90 seconds per query depending on depth and provider
- **Costs tokens** — cheap ($0.005–$0.15), but not free
- **LLMs can still hallucinate** — better prompts = better accuracy
- **Best with unique content** — personal memory files work great; generic/academic content is harder

## The Anamnesis Architecture

> *Anamnesis (Greek: ἀνάμνησις)* — "recollection." In Platonic philosophy, the
> idea that learning is really the process of remembering what the soul already knows.

1. **The Soul** (small, fixed) — Who the agent IS. Always in context, never sacrificed.
2. **The Index** (small, auto-loaded) — MEMORY.md. A table of contents for the mind.
3. **The Mind** (infinite, external) — What the agent KNOWS. Grows forever.
4. **The Bridge** (RLM) — The agent *reasons* about where to find memories and synthesizes answers.

See `docs/anamnesis-architecture.md` for the full theoretical framework.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

Key areas: provider support, memory navigation prompts, performance, new scope strategies.

## Citation

```bibtex
@software{deeprecall2026,
  title={DeepRecall: Recursive Memory for Persistent AI Agents},
  author={Chitez, Stefan and Crick},
  year={2026},
  url={https://github.com/Stefan27-4/DeepRecall},
  note={Implements the Anamnesis Architecture for AI agent memory persistence}
}
```

## ⚠️ Privacy

DeepRecall sends workspace file contents to your configured LLM provider for recall. This includes memory files, daily logs, and potentially project files (depending on scope). API keys and credentials are read locally for authentication and are **never included in prompts**.

See [SKILL.md](skill/SKILL.md#-privacy-notice) for full details.

## 🧠 Recommended Memory Architecture

DeepRecall works best with a **two-tier memory system**:

| Tier | File | Purpose | Auto-loaded? |
|---|---|---|---|
| Index | `MEMORY.md` | Compact orientation (~100 lines), table of contents pointing to details | ✅ Yes |
| Encyclopedia | `memory/LONG_TERM.md` | Full detailed memories — decisions, reasoning, timestamps | ❌ Searched via DeepRecall |
| Daily logs | `memory/YYYY-MM-DD.md` | Raw notes, distilled nightly into LONG_TERM.md | ❌ Searched via DeepRecall |

**The philosophy:** MEMORY.md is the Wikipedia summary. LONG_TERM.md is the diary entry. DeepRecall prefers the diary — *the devil is in the details.*

**Nightly sync:** At the end of each day, distill your daily log into `LONG_TERM.md`. Keep it detailed — preserve the story, don't just summarize.

> 💡 **Tip for new agents:** Ask your human before restructuring existing memory files.
> Show them this recommendation and let them decide how to organize their agent's memory.

## License

MIT License © 2026 Stefan Chitez & Crick — see [LICENSE](LICENSE).

## Acknowledgments

- [RLM](https://github.com/alexzhang13/rlm) by Alex Zhang (MIT OASYS Lab) — the recursive language model framework
- [OpenClaw](https://github.com/openclaw/openclaw) — the AI agent platform
- Built by a human and his AI cat, proving that the best partnerships don't require the same species 🐱
- and last but not least, - [fast-rlm](https://github.com/avbiswas/fast-rlm) by avbiswas — thank you for the deep dive on RLM implementation!
