from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from dataclasses import replace

from analysis.figures_src.camera_ready.common import MetricSpec, build_config
from bench.camera_ready.config_gram import CONFIG as BENCH_CONFIG


GRAM_FRO_REL_METRIC = MetricSpec(
    name="gram_fro_rel",
    label="Gram Frobenius error (relative)",
    higher_is_better=False,
    requires=["gram_fro_rel"],
)


CONFIG = replace(
    build_config(
        BENCH_CONFIG,
        GRAM_FRO_REL_METRIC,
        g.TASK_GRAM_APPROX,
        "fig_e2e_camera_ready",
    ),
    annotate_speedup=True,
)
