"""Public core API — re-exports the runtime types from their focused modules.

Internals live in: ``state`` (Agent/Envelope/…), ``config`` (RuntimeConfig),
``prompts`` (system-prompt construction), and ``runtime`` (the Runtime/ReAct loop).
This module is the stable import surface: ``from nanoma.core import Runtime, …``.
"""

from __future__ import annotations

from nanoma.config import RuntimeConfig
from nanoma.runtime import Runtime
from nanoma.state import (
    Agent,
    Artifact,
    Envelope,
    IdGenerator,
    ResourceQuota,
    ToolContext,
)

__all__ = [
    "Agent", "Artifact", "Envelope", "IdGenerator", "ResourceQuota",
    "ToolContext", "RuntimeConfig", "Runtime",
]
