"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nanoma.llm import RetryConfig


@dataclass
class RuntimeConfig:
    # Limits
    max_agents: int = 1000
    max_depth: int = 100
    max_concurrent_llm: int = 50
    budget: float = 10.0
    time_limit: float = 0.0
    max_turns: int = 200

    # Models
    default_model: str = "mini"          # tier alias; resolved via models.yaml
    allowed_models: list[str] | None = None
    temperature: float | None = None     # None = omit (some models reject it)
    # Route Claude-family models through the native Anthropic /messages endpoint so
    # prompt caching (cache_control) works — OpenAI-compatible proxies usually strip it.
    anthropic_native: bool = False

    # Paths
    log_dir: Path | None = field(default_factory=lambda: Path("./logs"))
    workspace_root: Path = field(default_factory=lambda: Path("./workspace"))
    shared_dir: str = "shared"

    # LLM retry
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Resource notification thresholds (fraction consumed)
    notify_thresholds: list[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.70, 0.80, 0.90, 0.95])

    # Context compression
    context_compress_ratio: float = 0.8
    compress_keep_recent: int = 6
    compress_max_messages: int = 40
    compress_max_chars: int = 300

    # Tool output caps (0 = unlimited)
    shell_max_output: int = 10000
    file_read_max_chars: int = 50000
    file_list_max_entries: int = 500
    grep_max_results: int = 100
