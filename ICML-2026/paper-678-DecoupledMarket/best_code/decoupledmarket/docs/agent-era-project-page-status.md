# Agent Era Project Page Status

Date: 2026-06-02

## Research Summary

Current agent-era project pages and product pages emphasize production readiness instead of only model demos. The common pattern is:

- A clear project workspace rather than isolated scripts.
- Versioned workflows and reproducible execution.
- Persistent context, memory, and traceable decisions.
- Multi-agent orchestration with human review.
- Evaluation, monitoring, and permission boundaries.
- Documentation that separates quick start, experiment notes, and architecture.

References checked:

- OpenAI, "New tools for building agents", 2025-03-11: Responses API, built-in tools, Agents SDK, and observability are positioned as production agent building blocks.
- Anthropic, "Building effective agents", 2024-12-19: agent systems are framed around simple composable workflows, tool use, evaluation, and human-readable control flow.

## Implications for This Repository

This repository already has the core pieces of an agent-era project:

- Multiple trader types: human-like `Person`, LLM `Agents`, technical traders, and `VirtualAgent`.
- Market simulation primitives: broker, stocks, market index, order matching, and day/iteration loops.
- LLM prompt assets under `content/` and `arena_content/`.
- Parallel execution and performance monitoring.
- Experiment analysis scripts and result reports.

The main structural issue was that source modules, experiments, docs, generated files, and temporary scratch files were mixed in the root directory. The project now uses a `src/decoupledmarket/` package layout and keeps experiments, docs, and tests in their own top-level directories.

## Refactor Direction Applied

- Added root `README.md` as the project page entry point.
- Moved core simulation modules and prompt resources into `src/decoupledmarket/`.
- Moved experiment runners into `scripts/`.
- Moved tests into `tests/`.
- Moved experiment and performance reports into `docs/`.
- Removed generated caches and empty output files.
- Removed duplicate `requirement.txt`; `requirements.txt` is the canonical dependency file.

## Future Refactor Direction

The next structural step, if a larger refactor is acceptable, is:

- Split the package into lower-case subpackages: `agents/`, `market/`, `storage/`, `llm/`, `simulation/`.
- Rename capitalized modules such as `Person.py`, `Market.py`, and `Stock.py` after import compatibility wrappers are added.
- Add a small CLI entry point so simulations run through one command.
