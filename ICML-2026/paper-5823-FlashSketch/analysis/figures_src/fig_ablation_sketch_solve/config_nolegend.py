from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from analysis.figures_src.ablation.common import MetricSpec, build_config
from bench.ablation.config_sketch_solve import CONFIG as BENCH_CONFIG


RESIDUAL_METRIC = MetricSpec(
    name="residual",
    label="Relative residual",
    higher_is_better=False,
    requires=["residual"],
)


CONFIG = build_config(
    BENCH_CONFIG,
    RESIDUAL_METRIC,
    g.TASK_SKETCH_SOLVE_LS,
    "fig_ablation_sketch_solve_nolegend",
    include_legends=False,
)
