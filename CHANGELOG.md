# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-05

### Changed

- Rewrote core engine: replaced Deno/fast-rlm subprocess with pure Python RLM loop
- Switched to OpenAI SDK for LLM calls with urllib fallback
- Anti-hallucination prompts: workers quote exactly, synthesis cites sources
- Parallel worker execution via ThreadPoolExecutor
- Updated model catalog (Claude 4.5/4.6, GPT-5, Gemini 2.5/3)
- Fixed API key leak in `__repr__`
- Fixed Copilot token expiry check
- Removed silent OpenRouter fallback
- Prefix matching for model pairs (replaces ambiguous substring matching)

### Removed

- Deno/fast-rlm dependency
- TypeScript patches (skill/patches/)
- YAML config builder

## [0.1.0] - 2026-02-28

### Added

- Initial release with RLM-based recursive memory recall
- OpenClaw skill integration
- Multi-provider support
