"""Tool execution activity — runs any registered tool with heartbeat.

The heartbeat keeps the activity alive in Temporal's eyes even when a tool
is running a slow shell command, web fetch, or spawned subagent. If the
worker pod dies, Temporal reschedules this activity on another worker. The
new worker remounts the same PVC (shared workspace volume) so file state
is intact — the command simply reruns from the top of its execution.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from temporalio import activity

from exoclaw.agent.tools.protocol import ToolContext
from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_temporal.config import ExecuteToolInput, WorkspaceConfig


def _build_registry(ws_cfg: WorkspaceConfig) -> ToolRegistry:
    """Reconstruct the full tool registry from config.

    Activities are stateless — every invocation rebuilds the registry from
    the serialized WorkspaceConfig. State lives in the filesystem (PVC/S3),
    not in memory.
    """
    from exoclaw_tools_workspace.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from exoclaw_tools_workspace.shell import ExecTool
    from exoclaw_tools_workspace.web import WebFetchTool, WebSearchTool

    workspace = Path(ws_cfg.path)
    workspace.mkdir(parents=True, exist_ok=True)
    allowed_dir = workspace if ws_cfg.restrict_to_workspace else None

    tools = [
        ReadFileTool(workspace=workspace, allowed_dir=allowed_dir),
        WriteFileTool(workspace=workspace, allowed_dir=allowed_dir),
        EditFileTool(workspace=workspace, allowed_dir=allowed_dir),
        ListDirTool(workspace=workspace, allowed_dir=allowed_dir),
        ExecTool(
            timeout=ws_cfg.exec_timeout,
            working_dir=str(workspace),
            restrict_to_workspace=ws_cfg.restrict_to_workspace,
            path_append=ws_cfg.exec_path_append,
        ),
        WebSearchTool(
            api_key=ws_cfg.web_search_api_key,
            max_results=ws_cfg.web_search_max_results,
            proxy=ws_cfg.web_proxy,
        ),
        WebFetchTool(proxy=ws_cfg.web_proxy),
    ]

    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


@activity.defn
async def execute_tool(input: ExecuteToolInput) -> str:
    """Execute a single tool call.

    Heartbeats every 10 s so Temporal knows the activity is alive during
    slow operations. The heartbeat_timeout on the workflow side controls
    how long Temporal waits before rescheduling on another worker.
    """
    registry = _build_registry(input.workspace)
    ctx = ToolContext(
        session_key=input.session_id,
        channel=input.channel,
        chat_id=input.chat_id,
    )

    async def _heartbeat_loop() -> None:
        while True:
            activity.heartbeat()
            await asyncio.sleep(10)

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    try:
        return await registry.execute(input.name, input.params, ctx)
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
