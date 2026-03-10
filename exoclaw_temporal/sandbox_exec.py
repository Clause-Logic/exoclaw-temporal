"""
SandboxExecTool — replaces ExecTool with isolated per-session sandbox pods.

Each agent session gets a SandboxClaim (named sandbox-{slug}).  The controller
creates the Sandbox and headless Service.  Inside the cluster the worker hits
the sandbox directly via K8s DNS — no router needed.

External access (local dev / tests) uses the SandboxClient in tunnel mode via
the sandbox-router-svc port-forward.

Requires (cluster-side):
  - agent-sandbox controller + extensions installed
  - SandboxTemplate "exoclaw-sandbox" in default namespace
  - Worker ServiceAccount with create/get on SandboxClaims
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from kubernetes import client as k8s_client  # type: ignore[import-untyped]
from kubernetes import config as k8s_config  # type: ignore[import-untyped]
from loguru import logger


SANDBOX_TEMPLATE = "exoclaw-sandbox"
SANDBOX_NAMESPACE = "default"
SANDBOX_PORT = 8888
SANDBOX_READY_TIMEOUT = 60  # seconds


def _session_slug(session_key: str) -> str:
    """Convert a session key to a valid DNS label (≤63 chars)."""
    slug = re.sub(r"[^a-z0-9-]", "-", session_key.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"sandbox-{slug}"[:63]


def _load_k8s() -> k8s_client.CustomObjectsApi:
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config(context="kind-exoclaw-temporal")
    return k8s_client.CustomObjectsApi()


def _sandbox_url(claim_name: str) -> str:
    """Direct in-cluster URL — bypasses the router."""
    return f"http://{claim_name}.{SANDBOX_NAMESPACE}.svc.cluster.local:{SANDBOX_PORT}"


async def ensure_sandbox(session_key: str) -> str:
    """
    Ensure a SandboxClaim exists for this session and the sandbox pod is ready.
    Returns the claim name (== sandbox DNS name).
    """
    name = _session_slug(session_key)
    api = _load_k8s()

    # Check if SandboxClaim already exists
    try:
        api.get_namespaced_custom_object(
            group="extensions.agents.x-k8s.io",
            version="v1alpha1",
            namespace=SANDBOX_NAMESPACE,
            plural="sandboxclaims",
            name=name,
        )
        logger.debug(f"SandboxClaim {name} already exists")
    except k8s_client.exceptions.ApiException as e:
        if e.status != 404:
            raise
        # Create it
        claim = {
            "apiVersion": "extensions.agents.x-k8s.io/v1alpha1",
            "kind": "SandboxClaim",
            "metadata": {"name": name, "namespace": SANDBOX_NAMESPACE},
            "spec": {"sandboxTemplateRef": {"name": SANDBOX_TEMPLATE}},
        }
        api.create_namespaced_custom_object(
            group="extensions.agents.x-k8s.io",
            version="v1alpha1",
            namespace=SANDBOX_NAMESPACE,
            plural="sandboxclaims",
            body=claim,
        )
        logger.info(f"Created SandboxClaim {name}")

    # Wait for the sandbox HTTP server to be reachable
    url = _sandbox_url(name)
    deadline = asyncio.get_event_loop().time() + SANDBOX_READY_TIMEOUT
    async with httpx.AsyncClient() as http:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await http.get(f"{url}/", timeout=2.0)
                if r.status_code == 200:
                    logger.info(f"Sandbox {name} is ready at {url}")
                    return name
            except Exception:
                pass
            await asyncio.sleep(1)

    raise TimeoutError(f"Sandbox {name} did not become ready within {SANDBOX_READY_TIMEOUT}s")


async def sandbox_exec(session_key: str, command: str) -> str:
    """
    Run a shell command in the sandbox for this session.
    Creates the sandbox on first call for a new session.
    """
    name = await ensure_sandbox(session_key)
    url = _sandbox_url(name)

    async with httpx.AsyncClient() as http:
        r = await http.post(
            f"{url}/execute",
            json={"command": command},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()

    stdout: str = data.get("stdout", "")
    stderr: str = data.get("stderr", "")
    exit_code: int = data.get("exit_code", 0)

    output = stdout
    if stderr:
        output += f"\nstderr: {stderr}"
    if exit_code != 0:
        output += f"\nexit_code: {exit_code}"
    return output.strip()


class SandboxExecTool:
    """
    Drop-in replacement for ExecTool that routes shell commands to an isolated
    agent-sandbox pod.  Same tool name ("exec") — transparent to the LLM.

    Each session gets its own sandbox pod.  Commands run in the sandbox's /app
    directory, isolated from the worker and other tenants.
    """

    name = "exec"
    description = (
        "Execute a shell command in an isolated sandbox environment. "
        "Each session has its own sandbox with a clean filesystem. "
        "Use for running scripts, installing packages, or any shell operations."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    }

    async def execute(self, *, command: str, **ctx: Any) -> str:
        session_key: str = ctx.get("session_key", "default")
        return await sandbox_exec(session_key, command)
