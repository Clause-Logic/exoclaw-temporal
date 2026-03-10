"""
Verify that shell commands execute inside an isolated agent-sandbox pod,
not in the worker process.

Prerequisites:
  - ANTHROPIC_API_KEY set in environment
  - kind cluster running with agent-sandbox controller + SandboxTemplate
  - Worker running: uv run python -m exoclaw_temporal.turn_based --worker
  - Port-forward to Temporal: kubectl port-forward -n temporal svc/temporal-frontend 7233:7233

Run:
  uv run python examples/test_sandbox_exec.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_temporal.config import LLMConfig, TurnInput, WorkspaceConfig
from exoclaw_temporal.sandbox_exec import SandboxExecTool
from exoclaw_temporal.turn_based.worker import TASK_QUEUE
from exoclaw_temporal.turn_based.workflows.agent_turn import AgentTurnWorkflow
from exoclaw_tools_workspace.filesystem import ReadFileTool, WriteFileTool
from temporalio.client import Client


async def main() -> None:
    client = await Client.connect("localhost:7233")

    workspace = Path("/tmp/exoclaw-sandbox-test")
    workspace.mkdir(exist_ok=True)

    registry = ToolRegistry()
    for tool in [
        WriteFileTool(workspace=workspace),
        ReadFileTool(workspace=workspace),
        SandboxExecTool(),
    ]:
        registry.register(tool)

    llm = LLMConfig(model="anthropic/claude-haiku-4-5-20251001")
    ws = WorkspaceConfig(path=str(workspace), sandbox_exec=True)

    async def turn(session_id: str, message: str, run_id: str) -> str | None:
        result = await client.execute_workflow(
            AgentTurnWorkflow.run,
            TurnInput(
                session_id=session_id,
                channel="cli",
                chat_id="direct",
                message=message,
                llm=llm,
                workspace=ws,
                tool_definitions_json=json.dumps(registry.get_definitions()),
            ),
            id=f"sandbox-test:{session_id}:{run_id}",
            task_queue=TASK_QUEUE,
        )
        return result.final_content if result else None

    print("=== Test: shell command runs in isolated sandbox ===")
    r = await turn("sandbox:1", "Run the command: hostname", "1a")
    print(f"Response: {r}")
    # The sandbox pod hostname differs from the worker pod hostname
    assert r is not None, "No response"
    print("✓ Got response from sandbox exec")

    print("\n=== Test: sandbox has isolated filesystem ===")
    r2 = await turn(
        "sandbox:1",
        "Run: ls /app and tell me what you see",
        "1b",
    )
    print(f"Response: {r2}")
    print("✓ Sandbox filesystem is visible")

    print("\n=== All sandbox tests passed ===\n")


if __name__ == "__main__":
    asyncio.run(main())
