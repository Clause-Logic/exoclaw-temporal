"""AgentTurnWorkflow — the agent loop as a durable Temporal workflow.

This mirrors AgentLoop._run_agent_loop() from exoclaw core, but every
LLM call and tool call is a Temporal activity. The messages list accumulates
in workflow local state (Temporal history). Worker bounces are transparent:
Temporal replays completed activities from history and resumes from the
last checkpoint.

  User message
      │
      ▼
  AgentTurnWorkflow.run()
      │
      ├── activity: build_prompt  ──► initial messages
      │
      └── loop (up to max_iterations):
              │
              ├── activity: llm_chat  ──► LLMResponseData
              │
              ├── if tool calls:
              │     ├── activity: execute_tool (with heartbeat)  ──► str
              │     ├── activity: execute_tool ...
              │     └── append results, loop again
              │
              └── if final: activity: record_turn, return content
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from typing import Any
    from exoclaw_temporal.activities.conversation import build_prompt, record_turn
    from exoclaw_temporal.activities.llm import llm_chat
    from exoclaw_temporal.activities.tools import execute_tool
    from exoclaw_temporal.config import (
        BuildPromptInput,
        ExecuteToolInput,
        LLMChatInput,
        RecordTurnInput,
        TurnInput,
        TurnOutput,
    )

# Activity options for fast operations (LLM call, conversation ops).
_LLM_OPTIONS = {
    "start_to_close_timeout": timedelta(minutes=5),
    "retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)),
}

# Activity options for tool execution — heartbeat keeps slow tools alive.
# If a worker dies during a tool call, Temporal waits heartbeat_timeout
# before rescheduling on another worker.
_TOOL_OPTIONS = {
    "start_to_close_timeout": timedelta(minutes=10),
    "heartbeat_timeout": timedelta(seconds=30),
    "retry_policy": RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=1)),
}

_CONV_OPTIONS = {
    "start_to_close_timeout": timedelta(minutes=2),
    "retry_policy": RetryPolicy(maximum_attempts=5, initial_interval=timedelta(seconds=1)),
}


@workflow.defn
class AgentTurnWorkflow:
    """One durable execution of a single conversation turn."""

    @workflow.run
    async def run(self, input: TurnInput) -> TurnOutput:
        # ── 1. Build full prompt (system prompt + history + user message) ──
        messages: list[dict[str, Any]] = await workflow.execute_activity(
            build_prompt,
            BuildPromptInput(
                session_id=input.session_id,
                message=input.message,
                channel=input.channel,
                chat_id=input.chat_id,
                llm=input.llm,
                workspace=input.workspace,
                media=input.media,
                plugin_context=input.plugin_context,
            ),
            **_CONV_OPTIONS,  # type: ignore[arg-type]
        )
        initial_len = len(messages)

        # ── 2. Agent loop ──────────────────────────────────────────────────
        iteration = 0
        final_content: str | None = None
        tools_used: list[str] = []

        while iteration < input.llm.max_iterations:
            iteration += 1

            response = await workflow.execute_activity(
                llm_chat,
                LLMChatInput(
                    messages=messages,
                    llm=input.llm,
                    tool_definitions_json=input.tool_definitions_json,
                ),
                **_LLM_OPTIONS,  # type: ignore[arg-type]
            )

            if response.has_tool_calls:
                # Append assistant message with tool_calls
                messages = [*messages, response.to_assistant_message()]

                for tc in response.tool_calls:
                    tools_used.append(tc.name)

                    result = await workflow.execute_activity(
                        execute_tool,
                        ExecuteToolInput(
                            name=tc.name,
                            params=tc.arguments,
                            session_id=input.session_id,
                            channel=input.channel,
                            chat_id=input.chat_id,
                            workspace=input.workspace,
                        ),
                        **_TOOL_OPTIONS,  # type: ignore[arg-type]
                    )

                    messages = [
                        *messages,
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        },
                    ]
            else:
                # Final response — strip think blocks, record, return
                final_content = _strip_think(response.content)
                msg: dict[str, Any] = {"role": "assistant", "content": final_content}
                if response.reasoning_content is not None:
                    msg["reasoning_content"] = response.reasoning_content
                if response.thinking_blocks:
                    msg["thinking_blocks"] = response.thinking_blocks
                messages = [*messages, msg]
                break

        if final_content is None:
            final_content = (
                f"Reached maximum tool call iterations ({input.llm.max_iterations}) "
                "without completing. Try breaking the task into smaller steps."
            )

        # ── 3. Record new messages to conversation store ───────────────────
        await workflow.execute_activity(
            record_turn,
            RecordTurnInput(
                session_id=input.session_id,
                new_messages=messages[initial_len:],
                llm=input.llm,
                workspace=input.workspace,
            ),
            **_CONV_OPTIONS,  # type: ignore[arg-type]
        )

        return TurnOutput(final_content=final_content, tools_used=tools_used)


def _strip_think(content: str | None) -> str | None:
    """Remove <think>...</think> blocks from LLM output."""
    if not content:
        return content
    import re

    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() or content
