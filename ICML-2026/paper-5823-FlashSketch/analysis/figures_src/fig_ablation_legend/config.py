from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from dataclasses import replace

from dataclasses import replace

from analysis.figures_src.ablation.common import MetricSpec, build_config
from bench.ablation.config_gram import CONFIG as BENCH_CONFIG


GRAM_FRO_REL_METRIC = MetricSpec(
    name="gram_fro_rel",
    label="Gram Frobenius error (relative)",
    higher_is_better=False,
    requires=["gram_fro_rel"],
)


_CONFIG = build_config(
    BENCH_CONFIG,
    GRAM_FRO_REL_METRIC,
    g.TASK_GRAM_APPROX,
    "fig_ablation_legend",
)

_PLOT = replace(
    _CONFIG.plots[0],
    name="ablation_legend",
    output_name="fig_ablation_legend_universal",
)
CONFIG = replace(_CONFIG, plots=[_PLOT])
CONFIG = replace(CONFIG, min_font_size=12)
