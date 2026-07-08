import os
from internal.dataset.llm_message_loader_factory import get_llm_message_builder
from internal.util.data_adapter import read_csv_to_fairness_input
from substantive.faircp.fairness.llm.llm_in_loop import run_llm_prediction
from substantive.faircp.fairness.llm.llm_result_stats import (
    calculate_accuracy_per_method,
    compute_comprehensive_fairness_statistics,
)
from internal.util.writer import Writer, get_writer
from substantive.faircp.structs.fairness_input import FairnessInput


def run_llm_in_loop(
    cfg: dict, writer: Writer | None, fairness_input: FairnessInput | None
):
    if fairness_input is None:
        dataset_path = None
        if writer is None:
            writer = get_writer(cfg["dataset"], cfg=cfg)
            dataset_path = os.path.join(
                cfg["logdir_root"],
                cfg["conformal_result_dataset"],
                cfg["dataset"] + ".csv",
            )
        else:
            dataset_path = os.path.join(writer.logdir, cfg["dataset"] + ".csv")

        fairness_input = read_csv_to_fairness_input(dataset_path)

    if writer is None:
        writer = get_writer(cfg["dataset"], cfg=cfg)

    message_builder = get_llm_message_builder(cfg, fairness_input.label_map)
    llm_result, df = run_llm_prediction(fairness_input, cfg, message_builder)
    writer.write_pandas("llm_individual_result", df)

    calculate_accuracy_per_method(llm_result)

    dataset_name = cfg.get("dataset", "Dataset")
    output_dir = writer.logdir

    comprehensive_stats = compute_comprehensive_fairness_statistics(
        predictions=llm_result,
        label_map=fairness_input.label_map,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    print(f"All results and plots saved to: {output_dir}")

    return llm_result, comprehensive_stats
