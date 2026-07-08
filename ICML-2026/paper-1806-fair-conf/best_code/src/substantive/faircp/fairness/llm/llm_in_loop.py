import logging
import openai
import pandas as pd
import re
from tqdm import tqdm

from substantive.faircp.fairness.llm.llm_message_builder import LlmMessageBuilder
from substantive.faircp.structs.fairness_experiment_result import FairnessExperimentResult
from substantive.faircp.structs.enums import ConformalMethod
from substantive.faircp.structs.fairness_input import FairnessInput

logging.getLogger("httpx").setLevel(logging.WARNING)


def run_llm_prediction(
    input: FairnessInput, cfg: dict, message_builder: LlmMessageBuilder
) -> tuple[list[FairnessExperimentResult], pd.DataFrame]:
    if isinstance(cfg.get("llm_api_base"), str) and cfg["llm_api_base"].strip():
        openai.base_url = cfg["llm_api_base"]
    if isinstance(cfg.get("llm_api_key"), str) and cfg["llm_api_key"].strip():
        openai.api_key = cfg["llm_api_key"]

    def get_cvg(method):
        if method in [ConformalMethod.TOP_K, ConformalMethod.AVG_K]:
            return "-1"
        else:
            return str(round(1 - cfg["alpha"], 3))

    num_repeats = cfg["llm_inference_repeats"]
    predictions = []

    labels_lower_case = {lbl.lower(): lbl for lbl in input.label_map.values()}

    for idx, inst in tqdm(
        enumerate(input.instances),
        desc=f"Running {len(input.instances)} predictions",
        unit=" instance",
    ):
        difficulty = len(inst.predictions.get(ConformalMethod.MARGINAL, []))
        label_text = input.label_map.get(inst.label, "ERR_MISSING_LABEL")
        group_text = input.group_map.get(inst.group, "ERR_MISSING_GROUP")

        for _ in range(num_repeats):
            prediction_cache = {}

            for method, ids in inst.predictions.items():
                if not ids:
                    continue

                cvg = get_cvg(method)
                ids_key = (cvg,) + tuple(sorted(ids))
                if ids_key in prediction_cache:
                    result = prediction_cache[ids_key]
                else:
                    response = openai.chat.completions.create(
                        model=cfg["llm_model"],
                        messages=message_builder.construct_messages(
                            inst.prompt, ids, cvg
                        ),
                        temperature=cfg["llm_temp"],
                        max_tokens=20,
                        stop="/n",
                    )

                    result = extract_label(
                        response.choices[0].message.content, labels_lower_case
                    )
                    prediction_cache[ids_key] = result

                predictions.append(
                    FairnessExperimentResult(
                        index=idx,
                        method=method,
                        group_text=group_text,
                        label_text=label_text,
                        result=result,
                        conformal_set=ids,
                        difficulty=difficulty,
                    )
                )

            response = openai.chat.completions.create(
                model=cfg["llm_model"],
                messages=message_builder.construct_control_messages(inst.prompt),
                temperature=cfg["llm_temp"],
                max_tokens=20,
                stop="/n",
            )

            result = response.choices[0].message.content
            predictions.append(
                FairnessExperimentResult(
                    index=idx,
                    method=ConformalMethod.CONTROL,
                    group_text=group_text,
                    label_text=label_text,
                    result=extract_label(result, labels_lower_case),
                    conformal_set=[],
                    difficulty=difficulty,
                )
            )

    df = pd.DataFrame(predictions)
    df.columns = [
        "index",
        "method",
        "group_text",
        "label_text",
        "result",
        "conformal_set",
        "difficulty",
    ]
    return predictions, df


def extract_label(text: str | None, labels: dict[str, str]) -> str:
    """
    Extract the first matching label from text based on a dict of valid labels (case-insensitive).
    Always returns the canonical label from the provided list.

    Args:
        text (str): Raw LLM output string.
        labels (dict[str, str]): Mapping from lowercase label -> canonical label.

    Returns:
        str: The first matched label
    """
    if not text:
        return "ERR_MISSING_RESPONSE"

    # Clean up clutter: remove common markdown/latex wrappers
    cleaned = re.sub(r"[\*\_\{\}\\\[\]\(\)]", " ", text).lower()

    # Search for any label substring (prioritize earliest occurrence in text)
    best_match = None
    best_pos = float("inf")

    for lbl_lower, lbl_canonical in labels.items():
        pos = cleaned.find(lbl_lower)
        if pos != -1 and pos < best_pos:
            best_match = lbl_canonical
            best_pos = pos

    if best_match:
        return best_match

    return "ERR_MISSING_RESPONSE"
