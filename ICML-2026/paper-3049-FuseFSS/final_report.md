# Final Report: paper-3049

- Title: FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing
- Primary metric: `Online time` (lower)
- Records: 8
- Generated: 2026-07-17T23:34:50Z

## Best Result

- Iteration: 6
- Idea: CODE-CPU-AFFINITY — Auto CPU affinity for NUMA-aware process pinning
- Primary metric: 56.659
- Commit: `1cf15d1fe2a63b01474c09b7a971a443fcded7ed`
- Notes: Enabled --auto-cpu-affinity by default, binding party 0 and party 1 to disjoint CPU sets via taskset. Online time: 56.66ms (30.1% below baseline 81.08ms). Keygen also improved: 0.062s (best across all iterations). Eliminates NUMA-crossing and CPU contention between the two protocol parties on the same machine.
