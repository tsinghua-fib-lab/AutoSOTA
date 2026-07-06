# API reference

The public surface of the `engram` package, generated from the source docstrings.

```python
from engram import (
    EditorConfig, EngramEditor, edit_llm, get_engram, apply_engram,
    CovarianceCollector, Statistics,
    EngramResult, LayerScaleInfo,
    count_ratio, weight_norm, effective_rank, uniform, compose,
    LayerHandler, LinearHandler, Conv1DHandler,
)
```

::: engram.EditorConfig

::: engram.EngramEditor

::: engram.edit_llm

::: engram.get_engram

::: engram.apply_engram

::: engram.LinearHandler

::: engram.Conv1DHandler

::: engram.LayerHandler

::: engram.CovarianceCollector

## Statistics & scaling

`collect_statistics` returns a `Statistics` (mean covariances + sample counts);
`compute_engram_weights` returns an `EngramResult` of per-layer projections. The
per-layer edit weighting is a pluggable scaling function — see
[Guide → Scaling](guide.md#scaling).

::: engram.stats.Statistics

::: engram.scaling.EngramResult

::: engram.scaling.LayerScaleInfo

::: engram.scaling.count_ratio

::: engram.scaling.weight_norm

::: engram.scaling.effective_rank

::: engram.scaling.uniform

::: engram.scaling.compose

## MoE (optional)

Support for transformers&nbsp;≥5 **fused-expert** MoE layers lives in a separate,
detachable module — import it explicitly; the core never depends on it:

```python
from engram import EngramEditor
from engram.moe import FusedExpertAdapter, apply_engram_weights

editor = EngramEditor(model, adapters=[FusedExpertAdapter()])
```

See [Guide → Mixture-of-experts](guide.md#mixture-of-experts).

::: engram.moe.FusedExpertAdapter

::: engram.moe.apply_engram_weights
