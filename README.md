# exoclaw-temporal

[OpenClaw](https://github.com/Clause-Logic/exoclaw)-grade AI agents that don't die.

Most agent frameworks run your agent as a single in-process loop. When the process dies — OOM kill, pod eviction, deploy, network blip — the agent dies with it. Mid-tool-call, mid-reasoning, mid-subagent. The user gets silence or an error. You have no idea how far it got.

[Temporal](https://temporal.io) solves durable execution. Every step is checkpointed. If a worker dies, Temporal reschedules on a survivor. The agent resumes exactly where it left off — not from the start of the turn, not from the last tool call, but from the exact activity that was interrupted.

This repo brings OpenClaw's agentic capabilities — tool use, multi-turn memory, any LLM — into Temporal's durable execution model. Powered by [exoclaw](https://github.com/Clause-Logic/exoclaw), the protocol-only framework that OpenClaw is built on. Same tools, same conversation memory, same LLM provider. Just unbreakable.

## Why this combination is powerful

An exoclaw agent is a loop: call the LLM, execute tools, call the LLM again, until done. That loop can run for seconds or hours depending on the task. It might execute dozens of tool calls — shell commands, web fetches, file writes, spawned subagents. Any of those can be slow. Any of them can fail. And in a real deployment, workers die.

Temporal maps onto this loop naturally:

```
AgentTurnWorkflow (durable execution unit)
  │
  ├── activity: build_prompt        ← load conversation history from shared volume
  │
  └── loop:
        ├── activity: llm_chat      ← LLM call, retried on transient failure
        ├── activity: execute_tool  ← tool call, heartbeat keeps it alive
        ├── activity: execute_tool  ← another tool, on any available worker
        └── activity: record_turn   ← persist new messages to shared volume
```

Each box is an activity. Each activity is independently retried. If a worker pod is killed between any two boxes, Temporal replays the completed ones from history and picks up at the next. The messages list — the accumulating state of the agent loop — lives in Temporal's workflow history, not in process memory.

This means:

- **Deploy your workers freely.** Rolling deploys, pod evictions, node replacements — agents in flight continue on surviving pods.
- **Scale workers horizontally.** Activities are stateless. Add workers to handle load. Remove them without affecting running agents.
- **Retry at the right granularity.** A failed tool call retries the tool call, not the whole turn. A failed LLM call retries just that LLM call.
- **Full observability.** Every activity, every retry, every input and output is in Temporal's history. You can see exactly what your agent did and replay it.
- **Long-running tasks work.** Shell commands that take minutes, web crawls, subagent spawns — they heartbeat to Temporal so nothing times out prematurely.

## How exoclaw's design made this possible

Most agent frameworks are monoliths. The LLM call, tool execution, memory management, and response delivery are tangled together in a single run loop. To add durability you'd have to wrap the entire loop as one giant unit of work — all-or-nothing. If it fails, you restart from the beginning.

exoclaw is different. Its architecture has five protocols and one loop:

```
InboundMessage → Bus → AgentLoop → LLM → Tools → Bus → OutboundMessage → Channel
```

Every noun is a protocol. More importantly, there's a sixth protocol that controls *how* the loop performs I/O — the **`Executor`**:

```python
class Executor(Protocol):
    async def chat(self, provider, *, messages, tools, ...) -> LLMResponse: ...
    async def execute_tool(self, registry, name, params, ctx) -> str: ...
    async def build_prompt(self, conversation, session_id, message, ...) -> list[dict]: ...
    async def record(self, conversation, session_id, new_messages) -> None: ...
    async def clear(self, conversation, session_id) -> bool: ...
    async def run_hook(self, fn, /, *args, **kwargs) -> object: ...
```

One method per operation. Each one independently swappable.

The default `DirectExecutor` calls everything inline. But swap in a different executor and each operation can have a completely different execution strategy — different timeouts, retries, or execution environments — without changing a single tool, channel, or provider.

That's the exact hook this repo uses. `AgentTurnWorkflow` is the agent loop rewritten as a Temporal workflow, where each operation is a Temporal activity:

| Executor method | Temporal activity | What it means |
|---|---|---|
| `build_prompt` | `build_prompt_activity` | Load history from shared volume, construct prompt |
| `chat` | `llm_chat_activity` | LLM call with retry on transient failure |
| `execute_tool` | `execute_tool_activity` | Tool call with heartbeat — survives worker death |
| `record` | `record_turn_activity` | Persist new messages to shared volume |

Because the Executor protocol decomposes the loop into discrete named operations with clean inputs and outputs, every step maps directly to a Temporal activity. There's no hidden state to worry about, no tangled callbacks to untangle. The decomposition was already done.

The other property that made this work is that exoclaw has **no framework-level in-memory state**. Conversation history lives in JSONL files. Tool state lives in files. The LLM provider is stateless. The only thing that needs to survive a worker death is the messages list — and that lives in Temporal's workflow history. Any worker that mounts the shared volume can pick up any activity.

If exoclaw had been a traditional framework — batteries included, everything wired together, shared in-process state — none of this would be possible without a complete rewrite. The protocol design is what unlocks it.

## Two approaches

### `turn_based/` — one workflow per message turn

Each user message starts a fresh `AgentTurnWorkflow`. Simple mental model. Easy to reason about. The conversation history loads from disk at the start of each turn and saves at the end — any worker that mounts the shared volume can handle any turn.

Good for most production use cases.

### `session_based/` — one long-running workflow per session

One `AgentSessionWorkflow` per conversation. New messages arrive as Temporal **Signals**. The workflow processes turns sequentially. After 50 turns it calls `continue_as_new` to keep history bounded.

This approach shines when a single session receives messages from multiple sources simultaneously — CLI, Slack, a scheduled heartbeat — all signaling the same workflow ID. You can also query the workflow's current status at any time without waiting for a turn to complete.

## Quickstart

**Prerequisites:** Docker, Python 3.11+, an LLM API key (Anthropic, OpenAI, etc.)

```bash
# 1. Clone and install
git clone https://github.com/Clause-Logic/exoclaw-temporal
cd exoclaw-temporal
uv sync

# 2. Start Temporal
docker compose up -d

# 3. Set your LLM key
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, etc.

# 4. Start a worker (in a separate terminal)
uv run python -m exoclaw_temporal.turn_based --worker

# 5. Run the example
uv run python examples/test_turn_based.py
```

You should see three tests pass: a plain LLM call, a tool call that writes a file, and a multi-turn conversation where memory persists across separate workflow runs.

To verify durability: open the Temporal UI at http://localhost:8233 and watch the workflow history as the example runs. Every activity — `build_prompt`, `llm_chat`, `execute_tool`, `record_turn` — is recorded with its inputs and outputs.

## Session-based example

```bash
# Start the session worker (separate terminal)
uv run python -m exoclaw_temporal.session_based --worker

# Run the example
uv run python examples/test_session_based.py
```

## Configuration

Uses the same `~/.nanobot/config.json` as [exoclaw-nanobot](https://github.com/Clause-Logic/exoclaw). No new config format to learn.

```bash
# Custom config path
uv run python -m exoclaw_temporal.turn_based --config /path/to/config.json

# Custom Temporal cluster
uv run python -m exoclaw_temporal.turn_based --temporal-url my-cluster.example.com:7233
```

## Kubernetes deployment

For production: workers as a Deployment, workspace state on a shared PVC (EFS/NFS with ReadWriteMany).

```bash
# Create a 3-node kind cluster (1 control plane + 2 workers)
mise run cluster-up

# Deploy Temporal (postgres + schema setup + server + UI)
mise run temporal-up

# Build and deploy workers (2 replicas across 2 nodes)
mise run worker-deploy

# Demonstrate durability: kill a worker mid-tool-call
mise run bounce-demo
```

The bounce demo submits a turn with a slow shell command, kills one worker pod, and shows the activity resuming on the surviving worker.

**Note on shared storage:** The PVC requests `ReadWriteMany` — required so both worker replicas can share the workspace. In kind, the default `local-path` provisioner only supports `ReadWriteOnce`. For multi-node kind testing, install an NFS provisioner or run with a single worker replica. In production (EKS, GKE), use EFS or a similar RWX-capable storage class.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │  Temporal Worker Pool (N replicas) │
                    │                                    │
                    │  Workflows:  AgentTurnWorkflow     │
                    │             AgentSessionWorkflow   │
                    │                                    │
                    │  Activities: build_prompt          │
                    │              llm_chat              │
                    │              execute_tool          │
                    │              record_turn           │
                    └──────────────┬───────────────────┘
                                   │ mount
                                   ▼
                    ┌──────────────────────────────────┐
                    │  Shared Workspace Volume (PVC)    │
                    │                                    │
                    │  sessions/    ← conversation JSONL │
                    │  cron.json    ← cron state         │
                    └──────────────────────────────────┘
```

Workers are completely stateless. All persistent state lives either in Temporal's workflow history (in-flight execution state) or on the shared volume (conversation history, tool outputs).
