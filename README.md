# DeepRecall — Recursive Memory for AI Agents

## Overview

DeepRecall is an OpenClaw skill that gives AI agents **infinite memory** using 
RLM (Recursive Language Models). Instead of cramming all memory into the context 
window, the agent recursively queries its own memory files.

**Architecture:** Anamnesis Architecture  
**Principle:** *"The soul stays small, the mind scales forever."*

## The Problem

Current AI agents face an impossible tradeoff:
- **Remember who they are** (personality, identity, values) → less room for memory
- **Remember what happened** (conversations, decisions, history) → identity gets pushed out

The more an agent remembers, the less room for personality. Agents literally 
**forget who they are** to remember what happened.

## The Solution

DeepRecall separates the agent into **Soul** and **Mind**:

```
┌─────────────────────────────────────────────┐
│              SOUL (Small, Fixed)             │
│  Identity, values, personality, core rules  │
│  Always in context. Never grows.            │
│  ~2-5K tokens                               │
├─────────────────────────────────────────────┤
│         WORKING MEMORY (Context Window)     │
│  Current conversation + recall results      │
│  Finite but sufficient                      │
├─────────────────────────────────────────────┤
│             MIND (Infinite, External)       │
│  Long-term memory, daily logs, project      │
│  files, conversation history                │
│  Queried via RLM — never fully loaded       │
│  ∞ tokens                                   │
└─────────────────────────────────────────────┘
         ↕ RLM (Recursive Bridge)
  Agent writes code to search memory
  Sub-agents read specific files/sections
  Only answers enter working memory
```

## What It Can Do

- **Recall specific facts** from months of conversation logs ("What did we decide about the budget?")
- **Synthesize across files** — connects information from multiple daily logs, project docs, and memory files
- **Navigate large document collections** — tested on 800-page textbook (59 files, 2.9MB), found specific tables and sections in 7-15 seconds
- **Auto-detect your LLM provider** — reads OpenClaw config, works with 20+ providers out of the box
- **Cost-efficient** — uses cheap sub-agent models for file reading ($0.005-$0.15 per query)
- **Zero infrastructure** — no vector database, no embeddings server, no cloud API. Just markdown files + an LLM

## Limitations (Honest)

- **Accuracy depends on query specificity** — broad queries ("tell me about power") may find the wrong section when topics overlap across files. Specific queries ("PepsiCo's top 10 products") nail it every time
- **Not instant** — 7-90 seconds per query depending on depth and provider. This is recall, not search
- **Costs tokens** — every query runs an LLM. Cheap ($0.005-$0.15), but not free
- **Requires fast-rlm + Deno** — extra dependencies to install
- **LLMs can still hallucinate** — if the answer isn't clearly in the files, the model may synthesize from training data. Better prompts = better accuracy
- **Best with unique content** — personal memory files (decisions, conversations) work great because the LLM can't hallucinate personal facts. Generic/academic content is harder

## When Is It Actually Useful?

DeepRecall becomes valuable when your agent's memory **exceeds what fits in context**:

| Workspace Size | Need DeepRecall? | Why |
|---|---|---|
| < 50KB | ❌ Not yet | Just paste files in the prompt |
| 50-200KB | ⚠️ Maybe | Starting to drop older files from context |
| 200KB-1MB | ✅ Yes | Agent is "forgetting" important history |
| > 1MB | ✅ Absolutely | Impossible to fit in any context window |

**Sweet spot:** An agent that's been running for weeks/months with daily logs, project docs, and accumulated decisions.

### Tips for Better Queries

- **Be specific:** "What are PepsiCo's top 10 products?" ✅ vs "Tell me about PepsiCo" ❌
- **Name the framework/author:** "Lynch's three levels of diversification" ✅ vs "diversification types" ❌
- **Include section references if known:** "Section 9.2 on corporate options" ✅
- **Ask about unique content:** Personal decisions, project history, and conversations work best

## Test Results

Tested against a 800-page textbook (Lynch, Strategic Management 7th Ed) — 59 chunks, 2.9MB:

| Test | Query | Provider | Time | Cost | Result |
|---|---|---|---|---|---|
| 1 | PepsiCo top 10 products | Gemini 3 Flash | 7.5s | ~$0.005 | ✅ All 10 products with prices |
| 2 | PepsiCo top 10 products | GitHub Copilot | 15s | ~$0.01 | ✅ All 10 with prices, Table 12.1 |
| 3 | Elements of power in organisations | GitHub Copilot | 58s | ~$0.05 | ⚠️ Found power content from multiple chapters |
| 4 | Degrees of diversification | GitHub Copilot | ~20s | ~$0.02 | ⚠️ Found correct topic, different classification |

**Key finding:** Specific, targeted queries work best. Broad queries find relevant content but may pull from the wrong section when topics appear in multiple places.

## How It Works

1. Agent needs to recall something from its past
2. DeepRecall scans the workspace for memory files
3. Reads OpenClaw's config to find the user's LLM provider
4. Auto-pairs a cheap sub-agent model for cost efficiency
5. Builds a Memory Index — maps topics/people/dates to files
6. Runs RLM: root agent reads the index first, then navigates to specific files
7. Returns the answer with source citations

### RLM vs Traditional Approaches

| Feature | RAG / Vector Search | DeepRecall (RLM) |
|---------|-------------------|------------------|
| Method | Keyword/vector match | Agent writes code to navigate |
| Intelligence | Fetch chunks | **Reason** about where memories are |
| Cross-reference | Limited | Connects dots across files |
| Structure-aware | No | Reads headers, sections, dates |
| Infrastructure | Vector DB + embeddings | None — just files |
| Privacy | Data leaves your machine | 100% local |
| Git-trackable | No | Yes — it's all markdown |

## Installation

### Prerequisites
- [OpenClaw](https://github.com/openclaw/openclaw) installed and configured
- [fast-rlm](https://github.com/avbiswas/fast-rlm) cloned locally
- [Deno](https://deno.com) 2+ installed (brew install deno - for Mac users)
- Any LLM provider configured in OpenClaw (Anthropic, OpenAI, Google, OpenRouter, Ollama, etc.)

### Install the Skill

```bash
# Clone DeepRecall
git clone https://github.com/<org>/deep-recall
cp -r deep-recall/skill ~/.openclaw/workspace/skills/deep-recall

# Clone fast-rlm (the RLM engine)
git clone https://github.com/avbiswas/fast-rlm.git
export FAST_RLM_DIR=/path/to/fast-rlm
```

## Usage

### From Python
```python
from deep_recall import recall

# Basic memory query
result = recall("What did we decide about the project architecture?")

# Quick recall (minimal scope, cheapest)
result = recall_quick("What is my human's name?")

# Deep recall (all workspace files, most thorough)
result = recall_deep("Summarize all decisions we made in the last month")

# Custom options
result = recall(
    "Find all mentions of budget discussions",
    scope="all",              # "memory", "identity", "project", "all"
    verbose=True,             # Show RLM execution
    config_overrides={
        "max_depth": 3,       # Deeper recursion
        "max_money_spent": 0.50,  # Higher budget
    }
)
```

### From Command Line
```bash
python deep_recall.py "What was the first project we worked on?" memory
```

## Supported Providers (20+)

DeepRecall auto-detects your provider from OpenClaw config. No extra API keys needed.

Anthropic, OpenAI, Google (Gemini), GitHub Copilot, OpenRouter, Ollama, DeepSeek, 
Mistral, Together, Groq, Fireworks, Cohere, Perplexity, SambaNova, Cerebras, xAI, 
Minimax, Zhipu (GLM), Moonshot (Kimi), Qwen.

### Auto Model Pairing

Your primary model orchestrates. A cheaper model handles file reading:

| Your Primary Model | Sub-agent Model |
|---|---|
| Claude Opus 4 | Claude Sonnet 4 |
| Claude Sonnet 4 | Claude Haiku 3.5 |
| GPT-4o / GPT-4.1 | GPT-4o-mini / GPT-4.1-mini |
| Gemini 2.5 Pro | Gemini 2.5 Flash |
| Llama 3.3 70B | Llama 3.3 8B |
| DeepSeek R1 | DeepSeek V3 |

Override both models via `config_overrides`.

## Configuration

### Default Settings
```yaml
max_depth: 2              # Memory → File → Section
max_calls_per_subagent: 10
max_money_spent: 0.25     # 25 cents per query
max_completion_tokens: 30000
max_prompt_tokens: 200000
api_timeout_ms: 120000    # 2 minutes
```

## Memory File Structure

DeepRecall understands the standard OpenClaw workspace layout:

```
~/.openclaw/workspace/
├── SOUL.md            # [soul] Agent identity — always in context
├── IDENTITY.md        # [soul] Core facts about the agent
├── MEMORY_INDEX.md    # [index] Auto-generated navigation map
├── MEMORY.md          # [mind] Long-term curated memory
├── USER.md            # [mind] About the human
├── AGENTS.md          # [mind] Agent behavior rules
├── TOOLS.md           # [mind] Tool-specific notes
└── memory/            # [daily-log] One file per day
    ├── 2026-02-24.md
    ├── 2026-02-25.md
    └── ...
```

### Memory Index (the secret weapon)

`MEMORY_INDEX.md` is auto-generated and maps topics, people, dates, and projects 
to specific files. RLM reads the index FIRST, then jumps directly to the right file.

Without it, RLM searches through every file blindly. After a year (365+ daily logs), 
the index is the difference between 2 file reads and 200.

```python
from memory_indexer import update_memory_index
update_memory_index()  # Regenerate MEMORY_INDEX.md
```

## Scopes

| Scope | Files Included | Speed | Cost | Use Case |
|-------|---------------|-------|------|----------|
| `identity` | Soul + mind files | ⚡ Fastest | 💰 Cheapest | "What's my name?" |
| `memory` | Identity + daily logs | 🔄 Fast | 💰💰 Low | "What did we do last week?" |
| `project` | All workspace files | 🐢 Slow | 💰💰💰 Medium | "Find that config change" |
| `all` | Everything | 🐌 Slowest | 💰💰💰💰 High | "Search everything for X" |

## The Anamnesis Architecture

> *Anamnesis (Greek: ἀνάμνησις)* — "recollection" or "remembering." In Platonic 
> philosophy, the idea that the soul possesses knowledge from before birth, 
> and learning is really the process of remembering what the soul already knows.

DeepRecall implements the Anamnesis Architecture for AI agents:

1. **The Soul** (small, fixed) — Who the agent IS. Always present in context. 
   Never sacrificed for memory.

2. **The Mind** (infinite, external) — What the agent KNOWS. Stored in files. 
   Grows forever. Accessed through recursive queries.

3. **The Bridge** (RLM) — How Soul accesses Mind. Not a database lookup — 
   the agent *reasons* about where to find memories and synthesizes answers.

## The Compounding Intelligence Hypothesis

> *"Does persistent memory make AI agents objectively better at their job?"*

We hypothesize that an AI agent with accumulated personal context — decisions, 
preferences, work history, communication style — produces **higher-quality, more 
relevant outputs** than the same agent without that context.

This means:
- The agent gets **better the longer it works with you**
- Every interaction adds to its understanding of your preferences and patterns
- After months of collaboration, the agent doesn't just remember facts — it 
  understands your tendencies, anticipates your needs, and produces work that 
  matches YOUR style

This isn't just a feature — it's a **compounding advantage** that grows over time.
Current AI tools treat every session as a fresh start. DeepRecall enables the 
agent-human relationship to actually develop.

See `docs/anamnesis-architecture.md` for the full theoretical framework.

## Contributing

Contributions welcome! Key areas:
- Adding provider support
- Improving memory navigation prompts
- Performance optimization
- New scope strategies
- Documentation and examples

## Citation

If you use DeepRecall or the Anamnesis Architecture in academic work:

```bibtex
@software{deeprecall2026,
  title={DeepRecall: Recursive Memory for Persistent AI Agents},
  author={Chitez, Daniel-Stefan and Crick},
  year={2026},
  url={https://github.com/Stefan27-4/DeepRecall},
  note={Implements the Anamnesis Architecture for AI agent memory persistence}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- [RLM](https://github.com/alexzhang13/rlm) by Alex Zhang (MIT OASYS Lab) — the recursive language model framework
- [fast-rlm](https://github.com/avbiswas/fast-rlm) by avbiswas — sandboxed RLM implementation
- [OpenClaw](https://github.com/openclaw/openclaw) — the AI agent platform
- Built by a human and his AI cat, proving that the best partnerships don't require the same species 🐱
