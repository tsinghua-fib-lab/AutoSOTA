from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from analysis.figures_src.ablation.common import MetricSpec, build_config
from bench.ablation.config_ose_error import CONFIG as BENCH_CONFIG


OSE_SPEC_ERR_METRIC = MetricSpec(
    name="ose_spec_err",
    label="OSE spectral error",
    higher_is_better=False,
    requires=["ose_spec_err"],
)


CONFIG = build_config(
    BENCH_CONFIG,
    OSE_SPEC_ERR_METRIC,
    g.TASK_OSE_ERROR,
    "fig_ablation_ose_nolegend",
    include_legends=False,
)
