import pandas as pd
from typing import Dict, List
from substantive.faircp.structs.enums import ConformalMethod
from substantive.faircp.structs.fairness_input import FairnessInput, ConformalDetail

conformal_method_col_name = {
    #ConformalMethod.TOP_K: "topk_set",
    #ConformalMethod.AVG_K: "avgk_set",
    ConformalMethod.MARGINAL: "conformal_marginal_set",
    ConformalMethod.CONDITIONAL: "conformal_conditional_set",
    ConformalMethod.BACKWARD: "conformal_backward_set",
    ConformalMethod.CLUSTERED_LABEL: "conformal_clustered_label_set",
    ConformalMethod.CLUSTERED_GROUP: "conformal_clustered_group_set",
}


def read_csv_to_fairness_input(path: str) -> FairnessInput:
    df = pd.read_csv(path)

    label_map: Dict[int, str] = {}
    group_map: Dict[int, str] = {}
    instances: List[ConformalDetail] = []

    for _, row in df.iterrows():
        label = int(row["label"])
        group = int(row["group"])
        if "prompt" in row:
            prompt = row["prompt"]
        else:
            prompt = row["idx"]

        label_map[label] = row.get("label_text", "")
        group_map[group] = row.get("group_text", "")

        predictions = {}
        for method, column_name in conformal_method_col_name.items():
            val = row[column_name]

            # Handle NaN, empty, and invalid values
            if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                predictions[method] = []
            else:
                try:
                    val_str = str(val).strip()
                    if val_str == '':
                        predictions[method] = []
                    else:
                        predictions[method] = list(map(int, val_str.split()))
                except (ValueError, AttributeError):
                    predictions[method] = []

        instances.append(
            ConformalDetail(
                prompt=prompt,
                label=label,
                group=group,
                predictions=predictions,
            )
        )

    return FairnessInput(instances=instances, label_map=label_map, group_map=group_map)

def conformal_data_frame_to_fairness_input(
    df: pd.DataFrame, label_map, group_map
) -> FairnessInput:
    instances: List[ConformalDetail] = []

    for _, row in df.iterrows():
        label = int(row["label"])
        group = int(row["group"])
        if "prompt" in row:
            prompt = row["prompt"]
        else:
            prompt = row["idx"]

        predictions = {}
        for method, column_name in conformal_method_col_name.items():
            val = row[column_name]

            # Handle NaN, empty, and invalid values
            if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                predictions[method] = []
            else:
                try:
                    val_str = str(val).strip()
                    if val_str == '':
                        predictions[method] = []
                    else:
                        predictions[method] = list(map(int, val_str.split()))
                except (ValueError, AttributeError):
                    predictions[method] = []

        instances.append(
            ConformalDetail(
                prompt=prompt,
                label=label,
                group=group,
                predictions=predictions,
            )
        )

    return FairnessInput(instances=instances, label_map=label_map, group_map=group_map)
