"""Conversation activities — build prompts and record turn history.

Uses DefaultConversation from exoclaw-conversation, same as nanobot.
The conversation JSONL files live on the shared workspace volume so any
worker pod can read/write them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalio import activity

from exoclaw_conversation.conversation import DefaultConversation
from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_temporal.config import BuildPromptInput, LLMConfig, RecordTurnInput, WorkspaceConfig


def _build_conversation(llm: LLMConfig, ws: WorkspaceConfig) -> DefaultConversation:
    provider = LiteLLMProvider(
        api_key=llm.api_key,
        api_base=llm.api_base,
        default_model=llm.model,
        extra_headers=llm.extra_headers or None,
    )
    return DefaultConversation.create(
        workspace=Path(ws.path),
        provider=provider,
        model=llm.model,
        memory_window=llm.memory_window,
    )


@activity.defn
async def build_prompt(input: BuildPromptInput) -> list[dict[str, Any]]:
    """Build the full messages list for this turn (system prompt + history + user message)."""
    conv = _build_conversation(input.llm, input.workspace)
    result = await conv.build_prompt(
        input.session_id,
        input.message,
        channel=input.channel,
        chat_id=input.chat_id,
        media=input.media,
        plugin_context=input.plugin_context,
    )
    return result  # type: ignore[return-value]


@activity.defn
async def record_turn(input: RecordTurnInput) -> None:
    """Persist the new messages from this turn to the conversation store."""
    conv = _build_conversation(input.llm, input.workspace)
    await conv.record(input.session_id, input.new_messages)
