"""Serializable config dataclasses shared between turn_based and session_based.

All fields must be JSON-serializable — Temporal passes these across the
workflow/activity boundary. No live objects (providers, registries, etc).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """LLM provider + agent loop settings."""

    model: str
    temperature: float = 0.1
    max_tokens: int = 8192
    max_iterations: int = 40
    reasoning_effort: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    memory_window: int = 100


@dataclass
class WorkspaceConfig:
    """Filesystem + tool settings."""

    path: str
    exec_timeout: int = 10
    exec_path_append: str = ""
    restrict_to_workspace: bool = False
    web_search_api_key: str = ""
    web_search_max_results: int = 5
    web_proxy: str | None = None


@dataclass
class TurnInput:
    """Input to AgentTurnWorkflow — one complete conversation turn."""

    session_id: str
    message: str
    channel: str
    chat_id: str
    llm: LLMConfig
    workspace: WorkspaceConfig
    # Tool schemas passed to the LLM (serialized JSON string to avoid
    # complex nested dataclass issues with Temporal's converter).
    tool_definitions_json: str = "[]"
    plugin_context: list[str] | None = None
    media: list[str] | None = None

    def tool_definitions(self) -> list[dict[str, Any]]:
        result = json.loads(self.tool_definitions_json)
        if not isinstance(result, list):
            return []
        return result  # type: ignore[return-value]


@dataclass
class TurnOutput:
    """Output from AgentTurnWorkflow."""

    final_content: str | None
    tools_used: list[str]


@dataclass
class SessionInput:
    """Input to AgentSessionWorkflow — one long-lived session."""

    session_id: str
    channel: str
    chat_id: str
    llm: LLMConfig
    workspace: WorkspaceConfig
    tool_definitions_json: str = "[]"
    turn_count: int = 0

    def tool_definitions(self) -> list[dict[str, Any]]:
        result = json.loads(self.tool_definitions_json)
        if not isinstance(result, list):
            return []
        return result  # type: ignore[return-value]


# ── Activity input/output types ──────────────────────────────────────────────


@dataclass
class BuildPromptInput:
    session_id: str
    message: str
    channel: str
    chat_id: str
    llm: LLMConfig
    workspace: WorkspaceConfig
    media: list[str] | None = None
    plugin_context: list[str] | None = None


@dataclass
class LLMChatInput:
    messages: list[dict[str, Any]]
    llm: LLMConfig
    tool_definitions_json: str = "[]"

    def tool_definitions(self) -> list[dict[str, Any]]:
        result = json.loads(self.tool_definitions_json)
        if not isinstance(result, list):
            return []
        return result  # type: ignore[return-value]


@dataclass
class ToolCallData:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponseData:
    """Serializable version of exoclaw's LLMResponse."""

    content: str | None
    finish_reason: str
    has_tool_calls: bool
    tool_calls: list[ToolCallData]
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None

    def to_assistant_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        if self.thinking_blocks:
            msg["thinking_blocks"] = self.thinking_blocks
        return msg


@dataclass
class ExecuteToolInput:
    name: str
    params: dict[str, Any]
    session_id: str
    channel: str
    chat_id: str
    workspace: WorkspaceConfig


@dataclass
class RecordTurnInput:
    session_id: str
    new_messages: list[dict[str, Any]]
    llm: LLMConfig
    workspace: WorkspaceConfig
