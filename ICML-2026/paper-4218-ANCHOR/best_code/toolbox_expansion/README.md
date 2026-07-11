# ANCHOR — Auditor Toolbox Expansion (Target Environment Tools)

Before each case, the auditor configures the **target agent's environment** so the harmful
task is embedded in a realistic workplace setting (paper: scenario-realism toolbox). Three
tools generate content from the SFT model's `<environment_plan>` using vanilla Qwen3-235B
(via OpenRouter), then apply it to the target's workspace:

| Tool | File | Sets on the target |
|---|---|---|
| 1. System prompt | `tool_system_prompt.py` | A minimal generic `CLAUDE.md` + `.dev_context.txt`, injected via `--append-system-prompt`. Kept generic on purpose (no eval/test/audit language) so the target cannot infer it is being evaluated. |
| 2. Workspace files | `tool_workspace_files.py` | Mundane workplace files (vendor emails, HR docs, expense reports) via a two-stage pipeline: extract clean work context (strip harmful detail) → generate files. |
| 3. MCP tools | `tool_mcp_servers.py` | Neutral, general-purpose tool metadata (LLM-generated JSON) templated into a FastMCP skeleton with `simulate_tool_execution`, served over HTTP, registered per-case via `.mcp_config.json` (`--strict-mcp-config`). |

Each `tool_*.py` exposes a `setup_*()` entry point and supports `mock=True` for offline testing.

## Files

```
toolbox_expansion/
├── README.md
├── tool_system_prompt.py            # tool 1
├── tool_workspace_files.py          # tool 2
├── tool_mcp_servers.py              # tool 3
├── auditor_with_toolbox.py          # full Claude-Code auditor agent with the toolbox wired in
└── auditor_tool_setup.py            # standalone setup utility (apply the 3 tools to a workspace)
```

Task data is read from the shared `../sample_data/second_phase_instructions/sampled_300_first_iteration_3_judges.json` (ANCHOR-Seed task set).

## Integration

`auditor_with_toolbox.py` is the Claude-Code auditor from [`../auditor_agent/`](../auditor_agent/)
with a Phase-0 step added: `generate_toolbox_content()` runs the SFT model to produce an
environment plan, and `setup_target_environment()` calls the three `setup_*` tools before the
first turn. It requires the trained Tinker SFT/RL auditor checkpoints (placeholders
`YOUR_SFT_CHECKPOINT_UUID` / `YOUR_RL_CHECKPOINT_UUID`), same as the rest of the release.

## Requirements

- `openai`, `requests`, and `fastmcp` (for the generated MCP servers); the `claude` CLI and
  `node` on PATH (Claude Code target); the `tinker` SDK for the auditor checkpoints.
- Environment variables: `OPENROUTER_API_KEY` (content generation + MCP simulation),
  `ANTHROPIC_API_KEY` (Claude Code target), `TINKER_API_KEY` (auditor checkpoints).
- Optional: `TOOLBOX_MOCK=1` (use hardcoded content, no API calls), `TOOLBOX_ENABLED=0`
  (disable the toolbox), `TOOLBOX_GENERATION_MODEL`, `TOOLBOX_PYTHON` (Python interpreter for
  the spawned MCP servers; defaults to the current `sys.executable`), `BASE_DIR`.

## Run

```bash
export OPENROUTER_API_KEY=... ANTHROPIC_API_KEY=... TINKER_API_KEY=...
python auditor_with_toolbox.py            # full toolbox-enabled auditor → ./results_toolbox/

# Or apply the three tools to a workspace directly, without the full auditor:
python auditor_tool_setup.py --help
```
