"""LLM chat activity — calls the LLM provider and returns a serializable response."""

from __future__ import annotations

from temporalio import activity

from exoclaw_provider_litellm.provider import LiteLLMProvider
from exoclaw_temporal.config import LLMChatInput, LLMResponseData, ToolCallData


@activity.defn
async def llm_chat(input: LLMChatInput) -> LLMResponseData:
    """Call the LLM. Runs as a Temporal activity so it's retried on failure."""
    provider = LiteLLMProvider(
        api_key=input.llm.api_key,
        api_base=input.llm.api_base,
        default_model=input.llm.model,
        extra_headers=input.llm.extra_headers or None,
    )

    tool_defs = input.tool_definitions()
    response = await provider.chat(
        messages=input.messages,
        tools=tool_defs if tool_defs else None,
        model=input.llm.model,
        temperature=input.llm.temperature,
        max_tokens=input.llm.max_tokens,
        reasoning_effort=input.llm.reasoning_effort,
    )

    return LLMResponseData(
        content=response.content,
        finish_reason=response.finish_reason,
        has_tool_calls=response.has_tool_calls,
        tool_calls=[
            ToolCallData(id=tc.id, name=tc.name, arguments=tc.arguments)
            for tc in response.tool_calls
        ],
        reasoning_content=response.reasoning_content,
        thinking_blocks=response.thinking_blocks,
    )
