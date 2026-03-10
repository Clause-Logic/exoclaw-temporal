"""AgentSessionWorkflow — one long-running workflow per conversation session.

Messages arrive via Temporal Signals. The workflow processes them one at a
time and delivers responses via a result query. When workflow history grows
too large, continue_as_new resets the history while the conversation
persists in the external store (JSONL on the shared volume).

Signal flow:
  CLI / channel  →  workflow.signal(send_message)  →  workflow processes turn
  CLI / channel  ←  workflow.query(get_last_response)  ←  reads response

Why this matters vs turn_based:
  - The session is alive and can receive concurrent messages
  - A heartbeat or scheduled trigger can signal the same session workflow
  - You can query current status without waiting for a turn to complete
  - Temporal tracks the full session lifecycle as a single workflow
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
        SessionInput,
        TurnOutput,
    )

# Rebuild history after this many turns to keep workflow history bounded
_CONTINUE_AS_NEW_AFTER_TURNS = 50

_LLM_OPTIONS = {
    "start_to_close_timeout": timedelta(minutes=5),
    "retry_policy": RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2)),
}
_TOOL_OPTIONS = {
    "start_to_close_timeout": timedelta(minutes=10),
    "heartbeat_timeout": timedelta(seconds=30),
    "retry_policy": RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=1)),
}
_CONV_OPTIONS = {
    "start_to_close_timeout": timedelta(minutes=2),
    "retry_policy": RetryPolicy(maximum_attempts=5, initial_interval=timedelta(seconds=1)),
}

# Session idle timeout — if no message for this long, the workflow exits cleanly
_IDLE_TIMEOUT = timedelta(hours=24)


@dataclass
class PendingMessage:
    message: str
    media: list[str] | None = None
    plugin_context: list[str] | None = None


@workflow.defn
class AgentSessionWorkflow:
    """Long-running session workflow. Accepts messages via signals."""

    def __init__(self) -> None:
        self._pending: list[PendingMessage] = []
        self._last_response: str | None = None
        self._processing = False

    @workflow.signal
    async def send_message(
        self,
        message: str,
        media: list[str] | None = None,
        plugin_context: list[str] | None = None,
    ) -> None:
        """Signal a new message into the session."""
        self._pending.append(
            PendingMessage(message=message, media=media, plugin_context=plugin_context)
        )

    @workflow.query
    def get_last_response(self) -> str | None:
        """Query the last agent response without waiting for a turn."""
        return self._last_response

    @workflow.query
    def is_processing(self) -> bool:
        """True if a turn is currently in flight."""
        return self._processing

    @workflow.run
    async def run(self, input: SessionInput) -> None:
        turn_count = input.turn_count

        while True:
            # Wait for a message or idle timeout
            try:
                await workflow.wait_condition(
                    lambda: len(self._pending) > 0,
                    timeout=_IDLE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # No message for 24 hours — exit cleanly
                workflow.logger.info("Session idle timeout, exiting")
                return

            # Process all pending messages sequentially
            while self._pending:
                msg = self._pending.pop(0)
                self._processing = True

                try:
                    output = await self._run_turn(input, msg)
                    self._last_response = output.final_content
                    turn_count += 1
                finally:
                    self._processing = False

                # continue_as_new to keep history bounded
                if turn_count >= _CONTINUE_AS_NEW_AFTER_TURNS and not self._pending:
                    workflow.logger.info(
                        "Reached {} turns, continuing as new", _CONTINUE_AS_NEW_AFTER_TURNS
                    )
                    workflow.continue_as_new(
                        SessionInput(
                            session_id=input.session_id,
                            channel=input.channel,
                            chat_id=input.chat_id,
                            llm=input.llm,
                            workspace=input.workspace,
                            tool_definitions_json=input.tool_definitions_json,
                            turn_count=turn_count,
                        )
                    )

    async def _run_turn(self, input: SessionInput, msg: PendingMessage) -> TurnOutput:
        """Run one agent turn inside the session workflow."""
        messages: list[dict[str, Any]] = await workflow.execute_activity(
            build_prompt,
            BuildPromptInput(
                session_id=input.session_id,
                message=msg.message,
                channel=input.channel,
                chat_id=input.chat_id,
                llm=input.llm,
                workspace=input.workspace,
                media=msg.media,
                plugin_context=msg.plugin_context,
            ),
            **_CONV_OPTIONS,  # type: ignore[arg-type]
        )
        initial_len = len(messages)

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
                        {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result},
                    ]
            else:
                final_content = response.content
                msg_dict: dict[str, Any] = {"role": "assistant", "content": final_content}
                if response.reasoning_content is not None:
                    msg_dict["reasoning_content"] = response.reasoning_content
                if response.thinking_blocks:
                    msg_dict["thinking_blocks"] = response.thinking_blocks
                messages = [*messages, msg_dict]
                break

        if final_content is None:
            final_content = f"Reached max iterations ({input.llm.max_iterations})."

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
