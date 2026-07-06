from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from analysis.figures_src.ablation.common import MetricSpec, build_config
from bench.ablation.config_gram import CONFIG as BENCH_CONFIG


GRAM_FRO_REL_METRIC = MetricSpec(
    name="gram_fro_rel",
    label="Gram Frobenius error (relative)",
    higher_is_better=False,
    requires=["gram_fro_rel"],
)


CONFIG = build_config(
    BENCH_CONFIG,
    GRAM_FRO_REL_METRIC,
    g.TASK_GRAM_APPROX,
    "fig_ablation_gram_nolegend",
    include_legends=False,
)
