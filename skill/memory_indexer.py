"""
Memory Indexer — Auto-generates a structured index of agent memories.

Creates MEMORY_INDEX.md — a lightweight table of contents that maps:
- Topics → which files contain them
- Timeline → what happened when
- People → where they're mentioned
- Projects → related files

RLM reads this FIRST, then jumps directly to the right file/section.
Without this, RLM would blindly search through every daily log.

The more memory you have, the more this matters.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict


def extract_topics(content: str, filename: str) -> dict:
    """Extract structured info from a memory file."""
    info = {
        "headers": [],
        "people": set(),
        "projects": set(),
        "keywords": set(),
        "summary_lines": [],
    }

    # Common people patterns: **Name** or "Name" in context
    # Only match short standalone names, not multi-word technical terms
    people_pattern = re.compile(
        r'\*\*([A-Z][a-z]{1,12}(?:\s[A-Z][a-z]{1,12})?)\*\*'
    )

    # Project patterns: things that look like project names
    project_keywords = {
        "fractal", "deeprecall", "deep recall", "ide", "rlm",
        "tauri", "fast-rlm", "whisboo", "anamnesis",
    }

    # Words that look like names but aren't
    NOT_PEOPLE = {
        "done", "note", "important", "warning", "error",
        "status", "update", "pending", "blocked", "todo",
        "the", "this", "that", "overview", "key", "mode",
        "build", "plan", "test", "review", "settings",
        "editor", "context", "smart", "pinned", "live",
        "agent", "model", "budget", "cost", "file",
        "phase", "layer", "index", "scope", "query",
        "soul", "mind", "bridge", "core", "setup",
    }

    for line in content.split("\n"):
        stripped = line.strip()

        # Headers
        if stripped.startswith("#"):
            header = stripped.lstrip("#").strip()
            if header and len(header) > 1:
                info["headers"].append(header)

        # People (bold names)
        for match in people_pattern.finditer(stripped):
            name = match.group(1)
            # Filter out common non-names
            if name.lower() not in NOT_PEOPLE and name.split()[0].lower() not in NOT_PEOPLE:
                info["people"].add(name)

        # Projects
        lower = stripped.lower()
        for kw in project_keywords:
            if kw in lower:
                info["projects"].add(kw)

        # Key action lines (checkmarks, decisions)
        if any(marker in stripped for marker in ["✅", "- [x]", "→", "Decision:", "Key"]):
            clean = stripped.lstrip("- [x]✅ ").strip()
            if len(clean) > 10 and len(clean) < 200:
                info["summary_lines"].append(clean)

        # Bold keywords as topics
        for match in re.finditer(r'\*\*(.+?)\*\*', stripped):
            term = match.group(1).strip()
            if 2 < len(term) < 40:
                info["keywords"].add(term.lower())

    return info


def _format_index(
    daily_logs: dict,
    other_memory: dict,
    memory_topics: Optional[dict],
    topic_map: dict,
    people_map: dict,
    project_map: dict,
    *,
    soul_mind_files: Optional[dict] = None,
    workspace_files: Optional[dict] = None,
) -> str:
    """Render the memory index as a markdown string.

    Shared by both the filesystem-scanning and the file-list-based paths.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Memory Index",
        f"Auto-generated: {now}",
        "",
        "---",
        "",
    ]

    # Soul / Mind files (only present when built from scanner file list)
    if soul_mind_files:
        lines.append("## Identity & Mind Files")
        for path, info in sorted(soul_mind_files.items()):
            t = info["topics"]
            hdrs = ", ".join(t["headers"][:8]) if t["headers"] else "no headers"
            lines.append(f"- `{info['path']}` ({info['size']:,} chars)")
            lines.append(f"  Sections: {hdrs}")
        lines.append("")

    # Timeline
    lines.append("## Timeline")
    for date_str in sorted(daily_logs.keys(), reverse=True):
        log = daily_logs[date_str]
        t = log["topics"]
        summary = ", ".join(t["headers"][:4]) if t["headers"] else "No headers"
        lines.append(f"- **{date_str}** → `{log['path']}` ({log['size']:,} chars)")
        lines.append(f"  Topics: {summary}")
    lines.append("")

    # Long-term memory files (non-daily)
    if other_memory:
        lines.append("## Long-Term Memory Files")
        for name, info in sorted(other_memory.items()):
            t = info["topics"]
            hdrs = ", ".join(t["headers"][:8]) if t["headers"] else "no headers"
            lines.append(f"- `{info['path']}` ({info['size']:,} chars)")
            lines.append(f"  Sections: {hdrs}")
        lines.append("")

    # Workspace files (only present when built from scanner file list)
    if workspace_files:
        lines.append("## Workspace Files")
        for path, info in sorted(workspace_files.items()):
            t = info["topics"]
            hdrs = ", ".join(t["headers"][:4]) if t["headers"] else "no headers"
            lines.append(f"- `{info['path']}` ({info['size']:,} chars)")
            lines.append(f"  Sections: {hdrs}")
        lines.append("")

    # People
    if people_map:
        lines.append("## People")
        for person in sorted(people_map.keys()):
            file_list = sorted(people_map[person])
            files_str = ", ".join(f"`{f}`" for f in file_list)
            lines.append(f"- **{person}** → {files_str}")
        lines.append("")

    # Projects
    if project_map:
        lines.append("## Projects")
        for proj in sorted(project_map.keys()):
            file_list = sorted(project_map[proj])
            files_str = ", ".join(f"`{f}`" for f in file_list)
            lines.append(f"- **{proj}** → {files_str}")
        lines.append("")

    # Topic map (top 30 most referenced topics)
    if topic_map:
        lines.append("## Topics")
        sorted_topics = sorted(
            topic_map.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:30]
        for topic, file_list in sorted_topics:
            files_str = ", ".join(f"`{f}`" for f in sorted(file_list))
            lines.append(f"- {topic} → {files_str}")
        lines.append("")

    # Key events from daily logs (most recent first)
    lines.append("## Key Events")
    for date_str in sorted(daily_logs.keys(), reverse=True):
        log = daily_logs[date_str]
        summaries = log["topics"]["summary_lines"][:5]
        if summaries:
            lines.append(f"### {date_str}")
            for s in summaries:
                lines.append(f"- {s}")
            lines.append("")

    # MEMORY.md summary
    if memory_topics and memory_topics["headers"]:
        lines.append("## MEMORY.md Sections")
        for h in memory_topics["headers"]:
            lines.append(f"- {h}")
        lines.append("")

    lines.append("---")
    lines.append("*This index is auto-generated. RLM reads this first to navigate memory efficiently.*")

    return "\n".join(lines)


def _build_index_from_files(files: list) -> str:
    """Build a memory index from pre-scanned MemoryFile objects.

    Categorises each file by its ``category`` attribute and produces
    the same structured markdown as the filesystem-scanning path.
    """
    daily_logs: dict[str, dict] = {}
    other_memory: dict[str, dict] = {}
    memory_topics: Optional[dict] = None
    soul_mind_files: dict[str, dict] = {}
    workspace_files: dict[str, dict] = {}

    for mf in files:
        topics = extract_topics(mf.content, Path(mf.rel_path).name)
        info = {
            "path": mf.rel_path,
            "size": mf.size,
            "topics": topics,
        }

        if mf.category == "mind" and Path(mf.rel_path).name == "MEMORY.md":
            memory_topics = topics

        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", Path(mf.rel_path).name)

        if mf.category == "daily-log" and date_match:
            daily_logs[date_match.group(1)] = info
        elif mf.category in ("long-term", "daily-log"):
            other_memory[Path(mf.rel_path).name] = info
        elif mf.category in ("soul", "mind"):
            soul_mind_files[mf.rel_path] = info
        elif mf.category == "workspace":
            workspace_files[mf.rel_path] = info

    # Build topic/people/project maps from ALL files
    topic_map: dict[str, set] = defaultdict(set)
    people_map: dict[str, set] = defaultdict(set)
    project_map: dict[str, set] = defaultdict(set)

    all_entries = (
        [(info["path"], info["topics"]) for info in daily_logs.values()]
        + [(info["path"], info["topics"]) for info in other_memory.values()]
        + [(info["path"], info["topics"]) for info in soul_mind_files.values()]
        + [(info["path"], info["topics"]) for info in workspace_files.values()]
    )
    if memory_topics:
        all_entries.append(("MEMORY.md", memory_topics))

    for path, t in all_entries:
        for kw in t["keywords"]:
            topic_map[kw].add(path)
        for p in t["people"]:
            people_map[p].add(path)
        for proj in t["projects"]:
            project_map[proj].add(path)

    return _format_index(
        daily_logs, other_memory, memory_topics,
        topic_map, people_map, project_map,
        soul_mind_files=soul_mind_files,
        workspace_files=workspace_files,
    )


def build_memory_index(workspace: Optional[Path] = None,
                       files: Optional[list] = None) -> str:
    """
    Build MEMORY_INDEX.md content from memory files.

    When *files* is provided (a list of ``MemoryFile`` objects from
    ``MemoryScanner``), the index is built from those files — honouring
    the scanner's scope filtering.  Otherwise falls back to scanning
    ``memory/`` and ``MEMORY.md`` directly from *workspace*.

    Returns the index as a markdown string.
    """
    if files is not None:
        return _build_index_from_files(files)

    workspace = workspace or Path(
        os.environ.get("OPENCLAW_WORKSPACE",
                       os.path.expanduser("~/.openclaw/workspace"))
    )

    memory_dir = workspace / "memory"
    memory_md = workspace / "MEMORY.md"

    # Collect all daily logs
    daily_logs = {}
    if memory_dir.exists():
        for f in sorted(memory_dir.glob("*.md")):
            # Extract date from filename
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})", f.name)
            if date_match:
                date_str = date_match.group(1)
                content = f.read_text(errors="replace")
                topics = extract_topics(content, f.name)
                daily_logs[date_str] = {
                    "path": f"memory/{f.name}",
                    "size": len(content),
                    "topics": topics,
                }

    # Non-dated memory files (e.g. LONG_TERM.md, heartbeat-state.json)
    other_memory = {}
    if memory_dir.exists():
        for f in sorted(memory_dir.glob("*.md")):
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})", f.name)
            if not date_match:  # Not a daily log
                content = f.read_text(errors="replace")
                topics = extract_topics(content, f.name)
                other_memory[f.name] = {
                    "path": f"memory/{f.name}",
                    "size": len(content),
                    "topics": topics,
                }

    # Analyze MEMORY.md
    memory_topics = None
    if memory_md.exists():
        content = memory_md.read_text(errors="replace")
        memory_topics = extract_topics(content, "MEMORY.md")

    # Build topic → files mapping
    topic_map = defaultdict(set)
    people_map = defaultdict(set)
    project_map = defaultdict(set)

    if memory_topics:
        for kw in memory_topics["keywords"]:
            topic_map[kw].add("MEMORY.md")
        for p in memory_topics["people"]:
            people_map[p].add("MEMORY.md")
        for proj in memory_topics["projects"]:
            project_map[proj].add("MEMORY.md")

    for date_str, log in daily_logs.items():
        t = log["topics"]
        for kw in t["keywords"]:
            topic_map[kw].add(log["path"])
        for p in t["people"]:
            people_map[p].add(log["path"])
        for proj in t["projects"]:
            project_map[proj].add(log["path"])

    # Include non-dated memory files (e.g. LONG_TERM.md)
    for name, info in other_memory.items():
        t = info["topics"]
        for kw in t["keywords"]:
            topic_map[kw].add(info["path"])
        for p in t["people"]:
            people_map[p].add(info["path"])
        for proj in t["projects"]:
            project_map[proj].add(info["path"])

    return _format_index(
        daily_logs, other_memory, memory_topics,
        topic_map, people_map, project_map,
    )


def update_memory_index(workspace: Optional[Path] = None) -> Path:
    """Build and write MEMORY_INDEX.md to the workspace."""
    workspace = workspace or Path(
        os.environ.get("OPENCLAW_WORKSPACE",
                       os.path.expanduser("~/.openclaw/workspace"))
    )

    index_content = build_memory_index(workspace)
    index_path = workspace / "MEMORY_INDEX.md"
    index_path.write_text(index_content)

    return index_path


if __name__ == "__main__":
    path = update_memory_index()
    print(f"✅ Memory index written to: {path}")
    print()
    print(path.read_text())
