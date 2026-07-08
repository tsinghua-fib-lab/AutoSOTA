import os
from internal.util.data_adapter import read_csv_to_fairness_input
from substantive.faircp.fairness.set.plots import (
    plot_set_size_by_group,
    plot_set_size_distribution,
    plot_shape_heatmap,
    plot_shape_heatmap_by_group,
    plot_grouped_bar_by_label,
)
from substantive.faircp.structs.enums import ConformalMethod
from internal.util.writer import get_writer


def chart_heatmap(cfg: dict):
    writer = get_writer(cfg["dataset"], cfg=cfg)
    dataset_path = os.path.join(
        cfg["logdir_root"],
        cfg["conformal_result_dataset"],
        cfg["dataset"] + ".csv",
    )
    fairness_input = read_csv_to_fairness_input(dataset_path)

    image_dir = writer.logdir
    plot_set_size_distribution(
        fairness_input, os.path.join(image_dir, "set_size_distribution.pdf")
    )
    plot_set_size_by_group(
        fairness_input, os.path.join(image_dir, "set_size_by_group.pdf")
    )
    if cfg["chart_heatmap_label_ordered"]:
        for method in [
            #ConformalMethod.TOP_K,
            #ConformalMethod.AVG_K,
            ConformalMethod.MARGINAL,
            ConformalMethod.CONDITIONAL,
            ConformalMethod.BACKWARD,
            ConformalMethod.CLUSTERED_LABEL,
            ConformalMethod.CLUSTERED_GROUP
        ]:
            plot_shape_heatmap(
                fairness_input,
                method,
                os.path.join(image_dir, f"shape_heatmap_{method.value}.pdf"),
            )
            plot_shape_heatmap_by_group(
                fairness_input,
                method,
                os.path.join(image_dir, f"shape_heatmap_by_group_{method.value}.pdf"),
            )
    else:
        for method in [
            #ConformalMethod.TOP_K,
            #ConformalMethod.AVG_K,
            ConformalMethod.MARGINAL,
            ConformalMethod.CONDITIONAL,
            ConformalMethod.BACKWARD,
            ConformalMethod.CLUSTERED_LABEL,
            ConformalMethod.CLUSTERED_GROUP
        ]:
            plot_grouped_bar_by_label(
                fairness_input,
                method,
                os.path.join(image_dir, f"grouped_bar_by_label_{method.value}.pdf"),
            )
