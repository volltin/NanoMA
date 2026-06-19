"""Runtime state: IDs, agents, messages, and tool context."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nanoma.llm import Message

_NATO = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
    "victor", "whiskey", "xray", "yankee", "zulu",
]


class IdGenerator:
    """Sequential NATO-phonetic agent IDs (alpha, bravo, …, alpha-1, …)."""

    def __init__(self):
        self._counter = 0

    def next(self) -> str:
        name = _NATO[self._counter % len(_NATO)]
        suffix = self._counter // len(_NATO)
        self._counter += 1
        return f"{name}-{suffix}" if suffix else name


@dataclass
class Envelope:
    """A message between agents (or system→agent)."""
    from_id: str
    to_id: str
    content: str
    tokens: int
    timestamp: float
    priority: int = 0
    mode: Literal["immediate", "steer", "queue"] = "queue"


@dataclass
class ResourceQuota:
    budget: float = 10.0
    time_limit: float = 0.0      # 0 = unlimited
    max_turns: int = 200


@dataclass
class Artifact:
    path: str
    absolute_path: Path
    description: str = ""
    agent_id: str = ""


@dataclass
class ToolContext:
    """Shared, read-only-ish context passed to every work-tool handler."""
    shared_dir: Path
    workspace_root: Path
    shell_max_output: int = 10000
    file_read_max_chars: int = 50000
    file_list_max_entries: int = 500
    grep_max_results: int = 100


@dataclass
class Agent:
    id: str
    task: str
    model: str

    status: Literal["running", "idle", "done", "failed"] = "running"
    history: list[Message] = field(default_factory=list)
    children: set[str] = field(default_factory=set)
    parent: str | None = None
    depth: int = 0
    result: str | None = None

    # Resources
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    context_tokens: int = 0
    context_limit: int = 128_000  # overridden from the model registry at creation
    tokens_consumed: int = 0
    _created_at: float = field(default_factory=time.time)

    # Workspace
    workspace: Path = field(default_factory=lambda: Path("."))
    artifacts: list[Artifact] = field(default_factory=list)

    # Inboxes (three priority levels)
    _queue_inbox: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)
    _steer_inbox: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)
    _immediate_inbox: asyncio.Queue[Envelope] = field(default_factory=asyncio.Queue)

    # Internal
    _task: asyncio.Task | None = field(default=None, repr=False)
    _turns: int = 0
    _last_active: float = field(default_factory=time.time)
    _notified_thresholds: set = field(default_factory=set)
