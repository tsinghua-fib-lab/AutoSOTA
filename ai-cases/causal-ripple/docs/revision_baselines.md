# Revision controls

The additional controls use the same checkpoint, public prompt files, response
caps, greedy token predictions, cleanup, and official scorers as the paper runs.
Their settings are fixed before full-set scoring.

| Control | Selection signal | Revision schedule | Sequence-forwards |
| :-- | :-- | :-- | --: |
| Margin Remask | Smallest top-two margin after the candidate-mask check | Every 8 steps in steps 32–95; at most one revision | Up to 136 |
| ReMDM-conf plug-in | Sample with weights `exp(-cached probability)` | Every 8 steps in steps 32–95; one eligible token | 128 |

Margin Remask follows the [CoRe official control](https://github.com/UCF-CRCV/CoRe)
and checks up to 32 low-margin generated candidates. It applies no proposal-change
or minimum-confidence gate. Eight fixture trajectories match the original
implementation exactly, including tokens and forward batch sizes.

The ReMDM-conf control is the **fixed-schedule LLaDA plug-in** described in
[CoRe Appendix B.1](https://arxiv.org/html/2602.04096v1), using the cached-confidence
weighting in [ReMDM Section 4.1](https://arxiv.org/html/2503.00307v3).
It caches a token's probability when that position is last decoded, samples one
previously generated position from the normalized confidence weights, masks it
before the ordinary model call, and adds one transfer slot. The cache is updated
only when a position is decoded. The prompt and final 16 response positions are
protected in both controls.

This control is distinct from the ReMDM authors' standalone categorical LLaDA
sampler, which runs an extra revision phase. The explicit label and cost counters
make the adaptation and its compute allocation visible.

ReMDM remasking is stochastic, even though token predictions are greedy. Each
task's Torch seed is derived from the first eight bytes of
`sha256(f"{seed}:{task_id}")`, modulo `2**63`. NumPy receives that seed modulo
`2**32`. A task therefore has the same remasking stream whether generated in a
fresh run or resumed after earlier tasks.

```bash
python -m causal_ripple.run --benchmark mbppplus --method margin-remask \
  --prompts data/mbppplus/public.jsonl --output results/mbppplus_margin.jsonl
python -m causal_ripple.run --benchmark mbppplus --method remdm-conf \
  --prompts data/mbppplus/public.jsonl --output results/mbppplus_remdm.jsonl
```

The experiment manifest stores source hashes, prompt hashes, the seed convention,
revision settings, generated token IDs, and actual serial/sequence forward counts.
Only complete official task sets enter result tables.
