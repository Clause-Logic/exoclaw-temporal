"""Turn-based app — wires config, tool registry, and Temporal client.

Mirrors exoclaw-nanobot's create() factory: loads config, builds the
tool definitions list, and returns an ExoclawTemporal instance ready to run.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from temporalio.client import Client

from exoclaw.agent.tools.registry import ToolRegistry
from exoclaw_nanobot import Config, load_config
from exoclaw_temporal.config import LLMConfig, TurnInput, TurnOutput, WorkspaceConfig
from exoclaw_temporal.turn_based.worker import TASK_QUEUE
from exoclaw_temporal.turn_based.workflows.agent_turn import AgentTurnWorkflow


def _build_tool_registry(config: Config) -> ToolRegistry:
    """Build the tool registry from nanobot config (same tools as nanobot)."""
    from exoclaw_tools_workspace.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
    from exoclaw_tools_workspace.shell import ExecTool
    from exoclaw_tools_workspace.web import WebFetchTool, WebSearchTool
    from exoclaw_tools_cron import CronService, CronTool

    workspace = config.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)
    allowed_dir = workspace if config.tools.restrict_to_workspace else None

    tools = [
        ReadFileTool(workspace=workspace, allowed_dir=allowed_dir),
        WriteFileTool(workspace=workspace, allowed_dir=allowed_dir),
        EditFileTool(workspace=workspace, allowed_dir=allowed_dir),
        ListDirTool(workspace=workspace, allowed_dir=allowed_dir),
        ExecTool(
            timeout=config.tools.exec.timeout,
            working_dir=str(workspace),
            restrict_to_workspace=config.tools.restrict_to_workspace,
            path_append=config.tools.exec.path_append,
        ),
        WebSearchTool(
            api_key=config.tools.web.search.api_key,
            max_results=config.tools.web.search.max_results,
            proxy=config.tools.web.proxy,
        ),
        WebFetchTool(proxy=config.tools.web.proxy),
    ]

    # Cron — activities execute cron jobs as Temporal workflows (see cron_job.py)
    cron_service = CronService(store_path=workspace / "cron.json")
    tools.append(CronTool(cron_service=cron_service))

    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _llm_config(config: Config) -> LLMConfig:
    model = config.agents.defaults.model
    prov = config.get_provider(model)
    return LLMConfig(
        model=model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        api_key=prov.api_key if prov else None,
        api_base=config.get_api_base(model),
        extra_headers=dict(prov.extra_headers) if prov and prov.extra_headers else {},
        memory_window=config.agents.defaults.memory_window,
    )


def _workspace_config(config: Config) -> WorkspaceConfig:
    return WorkspaceConfig(
        path=str(config.workspace_path),
        exec_timeout=config.tools.exec.timeout,
        exec_path_append=config.tools.exec.path_append,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        web_search_api_key=config.tools.web.search.api_key,
        web_search_max_results=config.tools.web.search.max_results,
        web_proxy=config.tools.web.proxy,
    )


class ExoclawTemporal:
    """Turn-based durable agent. Each message is one Temporal workflow run."""

    def __init__(
        self,
        client: Client,
        llm: LLMConfig,
        workspace: WorkspaceConfig,
        tool_definitions_json: str,
    ) -> None:
        self._client = client
        self._llm = llm
        self._workspace = workspace
        self._tool_definitions_json = tool_definitions_json

    async def chat(
        self,
        message: str,
        *,
        session_id: str,
        channel: str = "cli",
        chat_id: str = "direct",
        plugin_context: list[str] | None = None,
        media: list[str] | None = None,
    ) -> TurnOutput:
        """Submit one message turn as a Temporal workflow and wait for the result."""
        input = TurnInput(
            session_id=session_id,
            message=message,
            channel=channel,
            chat_id=chat_id,
            llm=self._llm,
            workspace=self._workspace,
            tool_definitions_json=self._tool_definitions_json,
            plugin_context=plugin_context,
            media=media,
        )
        result = await self._client.execute_workflow(
            AgentTurnWorkflow.run,
            input,
            id=f"turn-{session_id}-{_short_hash(message)}",
            task_queue=TASK_QUEUE,
        )
        return result

    async def run_cli(self) -> None:
        """Simple interactive REPL — one workflow per message."""
        import uuid

        session_id = f"cli:{uuid.uuid4().hex[:8]}"
        logger.info("Session: {}  (Ctrl+C to exit)", session_id)
        print(f"\nExoclaw Temporal (turn-based) — session {session_id}\n")

        while True:
            try:
                message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not message:
                continue
            if message in ("/new", "/clear"):
                session_id = f"cli:{uuid.uuid4().hex[:8]}"
                print(f"New session: {session_id}\n")
                continue

            try:
                output = await self.chat(message, session_id=session_id)
                print(f"\nAgent: {output.final_content}\n")
            except Exception as exc:
                logger.error("Workflow error: {}", exc)
                print(f"\nError: {exc}\n")


async def create(
    config: Config | None = None,
    *,
    config_path: Path | None = None,
    temporal_url: str = "localhost:7233",
) -> ExoclawTemporal:
    """Create an ExoclawTemporal instance from nanobot config."""
    if config is None:
        config = load_config(config_path)

    client = await Client.connect(temporal_url)
    registry = _build_tool_registry(config)

    tool_definitions_json = json.dumps(registry.get_definitions())
    llm = _llm_config(config)
    workspace = _workspace_config(config)

    logger.info("ExoclawTemporal ready — model={} temporal={}", llm.model, temporal_url)
    return ExoclawTemporal(
        client=client,
        llm=llm,
        workspace=workspace,
        tool_definitions_json=tool_definitions_json,
    )


def _short_hash(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode()).hexdigest()[:8]
