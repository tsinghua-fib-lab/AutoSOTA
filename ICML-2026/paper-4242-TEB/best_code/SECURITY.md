# Security Policy

This repository is the research code companion to the paper
*Threshold-Based Exclusive Batching for Memory-Bandwidth-Constrained LLM
Inference* (ICML 2026). It is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm).

## Reporting issues in this fork

Please use one of the following, depending on where the issue lives:

- **A vulnerability in code added by this fork** — the EB / MB scheduler
  in `vllm/v1/core/sched/` or any of the reproduction scripts under
  `reproduce/`: report privately via
  [this repository's security advisory form](https://github.com/weifang231/eb-vllm/security/advisories/new).

- **A vulnerability in upstream vLLM** (anything outside the directories
  listed above): please report it to the upstream project at
  [vllm-project/vllm security advisories](https://github.com/vllm-project/vllm/security/advisories/new)
  so it can be triaged by the vLLM security team. Upstream's full
  vulnerability-management process, severity rubric, and prenotification
  policy are documented in
  [their SECURITY.md](https://github.com/vllm-project/vllm/blob/main/SECURITY.md).

For non-security bugs and reproduction questions, open a regular issue
on [this repository](https://github.com/weifang231/eb-vllm/issues).
