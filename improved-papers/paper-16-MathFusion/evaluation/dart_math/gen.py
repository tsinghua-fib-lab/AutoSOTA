import logging
import os
from typing import Callable

import orjson
from pebble import ProcessPool
from tqdm import tqdm
from vllm import LLM, RequestOutput, SamplingParams

from .data import DS_ID2N_SHOTS, ICL_EGS, QueryDataPoint, RespSampleBase, RespSampleVLLM
from .eval import EvaluatorBatchBase
from .exec import CodeExecCfg, exec_cells
from .utils import BASE_MODEL_IDS, get_pathname_from_name_or_path

# %% ../nbs/02_gen.ipynb 0

# Data Preprocessing


def get_n_shots(
    dataset: str,
    model_dirname: str,
) -> int:
    """Get the number of ICL examples adaptive to the dataset and model."""
    if model_dirname in BASE_MODEL_IDS or "mammoth2" in model_dirname.lower():
        n_shots = DS_ID2N_SHOTS.get(dataset)
    else:
        n_shots = 0

    return n_shots


def get_icl_egs(
    dataset: str, n_shots: int = None, model_dirname: str | None = None
) -> list[tuple[str, str]]:
    """Get the ICL examples for the dataset.

    Parameters
    ----------
    dataset : str
        Preset dataset ID.
    n_shots : int, default: None
        Number of examples in the few-shot prompt. `None` / Negative means adaptive to the datasets.
    model_dirname : str | None, default: None
        HF ID or path to the model.

    Returns
    -------
    list[tuple[str, str]]
        ICL examples adaptive to the dataset (and model).
    """
    n_shots = (
        get_n_shots(dataset, model_dirname)
        if n_shots is None or n_shots < 0
        else n_shots
    )
    return [] if n_shots == 0 else ICL_EGS[dataset][:n_shots]


# Stopping criteria


def is_dp_dars_finished(dp: QueryDataPoint) -> str | None:
    """Judge whether DARS for a data point is finished and return the stopping reason or `None` if not finished.

    Parameters
    ----------
    dp : QueryDataPoint
        Query data point having at least the following attributes: `max_n_trials` (and `n_trials`), `min_n_corrects` (and `n_corrects`).

    Returns
    -------
    str | None
        The stopping reason or `None` if not finished.
    """
    if dp.max_n_trials > 0 and dp.n_trials >= dp.max_n_trials:
        return "max_n_trials"
    elif dp.min_n_corrects > 0 and dp.n_corrects > dp.min_n_corrects:
        return "max_n_corrects"
    else:
        return None


# IO


def get_res_fname(
    model_name_or_path: str,  # HF ID or path to the model.
    max_new_toks: int,  # Maximum length of the model output in token.
    temperature: float,  # Temperature for sampling.
    top_p: float,  # Top-p for sampling.
    prompt_template: str,  # ID or path to the prompt template.
    dataset: str,  # Name of the dataset to generate on.
    n_shots: int,  # Number of egs in few-shot prompt.
    tag: str,  # Tag describing sample number informantion for the result file.
    inf_seed: int,  # Seed for randomness.
) -> str:  # Path to the result file.
    """Get the JSONL file name to save results to."""
    # LLM-specific
    res_filename = get_pathname_from_name_or_path(model_name_or_path)
    res_filename += "-" + f"outlen{max_new_toks}"
    if temperature <= 0:
        temperature = 0
        top_p = 1
    res_filename += "-" + f"t{temperature}"
    res_filename += "-" + f"p{top_p}"
    res_filename += "-" + f"prompt-{get_pathname_from_name_or_path(prompt_template)}"

    # Dataset-specific
    res_filename += "-" + get_pathname_from_name_or_path(dataset)
    res_filename += "-" + f"{n_shots}shot"

    # Run specific
    res_filename += "-" + f"tag-{tag}"
    res_filename += "-" + f"seed{inf_seed}"

    res_filename += ".jsonl"

    return res_filename


# Pipeline


class Generator:
    """Generator with various features such as stopping criteria and code execution.

    Parameters
    ----------
    llm : LLM
        The `vllm` model to generate with (or other objects with compatible `generate` interfaces).
    sampling_params : SamplingParams
        The sampling parameters for the `llm` (or other objects with compatible interfaces).
        NOTE: `n > 1` might cause bugs in `vllm` for now (0.4.2).
    batch_evaluator : EvaluatorBatchBase | None, default: None
        The batch evaluator to evaluate the generated responses. `None` means no evaluation.
    resp_sample_cls : type, default: RespSampleVLLM
        The class to collect the generated response as.
    code_exec_cfg : str | None, default: None
        The tool using configuration.
    """

    def __init__(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        resp_sample_cls: type = RespSampleVLLM,
        batch_evaluator: EvaluatorBatchBase | None = None,
        code_exec_cfg: CodeExecCfg | str | None = None,
    ):
        self.llm = llm
        self.sampling_params = sampling_params
        self.resp_sample_cls = resp_sample_cls
        self.batch_evaluator = batch_evaluator

        self.code_exec_cfg = (
            (
                code_exec_cfg
                if isinstance(code_exec_cfg, CodeExecCfg)
                else CodeExecCfg.load_from_id_or_path(code_exec_cfg)
            )
            if code_exec_cfg is not None
            else None
        )

        if (
            self.code_exec_cfg is not None
            and self.code_exec_cfg.output_begin.strip() not in self.sampling_params.stop
        ):
            self.sampling_params.stop.append(self.code_exec_cfg.output_begin.strip())

    def gen_pure(
        self,
        input_strs: list[str],
    ) -> list[RequestOutput]:
        """Code execution only supports one-path generation for now.

        Parameters
        ----------
        input_strs : list[str]
            The input strings as direct input to the model.

        Returns
        -------
        list[RequestOutput]
            The generated responses grouped by input strings.
        """
        # rand_idx = random.choice(range(len(batch_input_strs)))
        logging.info(f"{self.sampling_params=}")
        logging.info(f"{input_strs[0]=}")
        if self.code_exec_cfg is None:
            req_outputs = self.llm.generate(input_strs, self.sampling_params)
        else:
            # With code execution
            assert (
                self.sampling_params.n == 1
            ), "Support one-path generation only for now."
            req_outputs = [None] * len(input_strs)
            remain_ids = list(range(len(input_strs)))
            while True:
                remain_input_strs = [
                    input_strs[i]
                    + (
                        req_outputs[i].outputs[0].text
                        if req_outputs[i] is not None
                        else ""
                    )
                    for i in remain_ids
                ]
                remain_req_outputs = self.llm.generate(
                    remain_input_strs, self.sampling_params
                )
                for i, new_req_output in zip(remain_ids, remain_req_outputs):
                    if req_outputs[i] is None:
                        new_req_output.outputs[0].cumulative_logprob = [
                            new_req_output.outputs[0].cumulative_logprob
                        ]
                        req_outputs[i] = new_req_output
                    else:  # Align with RespSampleVLLM.collect
                        gen_path = req_outputs[i].outputs[0]
                        new_gen_path = new_req_output.outputs[0]
                        gen_path.text += new_gen_path.text
                        gen_path.finish_reason = new_gen_path.finish_reason
                        gen_path.stop_reason = new_gen_path.stop_reason
                        # Non-sense if simply adding up
                        gen_path.cumulative_logprob.append(
                            new_gen_path.cumulative_logprob
                        )
                new_remain_ids = []
                for i in remain_ids:
                    req_output = req_outputs[i]
                    gen_path = req_output.outputs[0]
                    code_extract_no = self.code_exec_cfg.no_cells_todo(gen_path.text)
                    if code_extract_no == 1 or (
                        self.batch_evaluator.extract_explicit_ans(gen_path.text)
                        is not None
                    ):  # Finish
                        continue
                    if code_extract_no == 2:
                        req_output.finish_reason = "over-output"
                        continue
                    if isinstance(self.code_exec_cfg.max_n_calls, int) and (
                        gen_path.text.count(self.code_exec_cfg.output_begin)
                        >= self.code_exec_cfg.max_n_calls
                        > 0
                    ):
                        req_output.finish_reason = "call"
                        continue
                    sched_cfg = (  # vLLM version back-compatibility
                        self.llm.llm_engine.scheduler.scheduler_config
                        if (hasattr(self.llm.llm_engine, "scheduler"))
                        else self.llm.llm_engine.scheduler_config
                    )
                    # All tokens
                    if len(gen_path.token_ids) + len(req_output.prompt_token_ids) >= (
                        0.8 * sched_cfg.max_model_len
                    ):
                        req_output.finish_reason = "total-length"
                        continue
                    new_remain_ids.append(i)

                remain_ids = new_remain_ids
                logging.info(f"len(remain_ids): {len(remain_ids)}")
                if len(remain_ids) == 0:
                    break
                logging.info(f"{req_outputs[remain_ids[0]].outputs[0].text=}")
                cells_list = [
                    self.code_exec_cfg.extract_cells(req_outputs[i].outputs[0].text)
                    for i in remain_ids
                ]
                logging.info(
                    f"cells_list: (#{len(cells_list)})[(#{len(cells_list[0])}){cells_list[0]},...]"
                )
                assert len(cells_list) == len(
                    remain_ids
                ), f"Mismatched cells(#{len(cells_list)}) and ids(#{len(remain_ids)})"

                # DEFUNCT: co-routine would get stuck
                # import asyncio
                # pbar = tqdm(total=len(cells_list), desc="Executing")
                # idxed_results = asyncio.run(
                #     seq_consume_preset_queue_w_each_timeout(
                #         exec_cells,
                #         idxed_kwargs_queue=enumerate(
                #             [{"cells": cells} for cells in cells_list]
                #         ),
                #         timeout=self.code_exec_cfg.timeout,
                #         pbar=pbar,
                #     )
                # )
                # pbar.close()
                # for idx, exec_res in idxed_results:

                results = []
                with ProcessPool(
                    max_workers=min(self.code_exec_cfg.max_n_workers, len(cells_list))
                ) as pool:
                    iterator = pool.map(
                        exec_cells, cells_list, timeout=self.code_exec_cfg.timeout
                    ).result()
                    pbar = tqdm(total=len(cells_list), desc="Executing")
                    while True:
                        try:
                            result = next(iterator)
                            results.append(result)
                        except StopIteration:
                            break
                        except Exception as e:
                            results.append(e)
                        pbar.update(1)
                    pbar.close()
                for idx, exec_res in enumerate(results):
                    if isinstance(exec_res, tuple):
                        stdout, stderr = exec_res
                        output = stdout
                        if stderr:
                            output += "\n" + stderr

                    else:  # e.g. `asyncio.TimeoutError`
                        output = str(exec_res)
                    if (
                        isinstance(self.code_exec_cfg.trunc_len, tuple)
                        and len(output) > sum(self.code_exec_cfg.trunc_len) > 0
                    ):
                        len_begin, len_end = self.code_exec_cfg.trunc_len
                        output = (
                            output[:len_begin]
                            + self.code_exec_cfg.elipsis
                            + output[-len_end:]
                        )
                    req_id = remain_ids[idx]
                    req_output = req_outputs[req_id]
                    req_output.outputs[0].text += (
                        "\n"  # Assure the output is separated from the input
                        + self.code_exec_cfg.wrap_output(output)
                        + "\n"  # Neccessary for preventing LLM from genearting the output itself
                    )

        logging.info(f"req_outputs[0].outputs[0]: {req_outputs[0].outputs[0]}")
        return req_outputs

    def gen(
        self,
        query_dps: list[QueryDataPoint],
        dp_stop_criteria: Callable[[QueryDataPoint], bool],
        save_path: str | None = None,
        n_paths_per_save: int | None = None,
    ) -> list[RespSampleBase] | None:
        """Generate responses on the given query data points with specified stopping criteria.

        Parameters
        ----------

        query_dps : list[QueryDataPoint]
            The query-level data points to generate responses on.
        dp_stop_criteria : Callable[[QueryDataPoint], bool]
            The function to check if a query data point should be stopped generating on.
        save_path : str | None, default: "auto"
            Path to save the generated reponses to. `None` or `""` means no saving.
        n_paths_per_save : int | None, default: None
            Response-level samples or `None` if saving.


        Returns
        -------
        list[RespSampleBase] | None
            The generated responses or `None` if saving.
        """
        if save_path and "seed" not in save_path.lower():
            seed = self.sampling_params.seed
            prefix, ext = os.path.splitext(save_path)
            save_path = f"{prefix}-seed{seed}{ext}"  # Add seed to the file name

        model_name_or_path = self.llm.llm_engine.model_config.model
        model_dirname = get_pathname_from_name_or_path(model_name_or_path)

        all_new_samples = []
        sched_finished = False

        sample_cnt = 0
        while not sched_finished:  # Loop on batches
            batch_dps = []
            batch_input_strs = []

            # Collect input strings batch
            batch_collected = False
            while not (batch_collected or sched_finished):

                # Loop on `query_samples` repeatedly
                sched_finished = True  # Speculate that all data points are finished
                for dp in query_dps:  # Loop on data points
                    stop_reason = dp_stop_criteria(dp)
                    if stop_reason is not None:
                        continue  # Skip the stopped data point
                    # Still have data points to generate
                    sched_finished = False
                    # Count # of trials for `dp_stop_criteria`
                    dp.n_trials += self.sampling_params.best_of
                    batch_dps.append(dp)

                    # Make the input string
                    icl_egs = get_icl_egs(dp.dataset, dp.n_shots, model_dirname)

                    input_str = dp.prompt_template.make_full_prompt(dp.query, icl_egs)
                    batch_input_strs.append(input_str)

                    # Check if batch is full
                    batch_collected = (
                        n_paths_per_save is not None
                        and n_paths_per_save > 0
                        and len(batch_input_strs) * self.sampling_params.n
                        >= n_paths_per_save
                    )
                    if batch_collected or sched_finished:
                        break

            # Generate responses batch
            batch_req_outputs = self.gen_pure(batch_input_strs)

            # Collect outputs batch
            batch_new_samples = []
            for dp, req_output in zip(batch_dps, batch_req_outputs):
                for gen_path in req_output.outputs:
                    batch_new_samples.append(
                        self.resp_sample_cls.collect(
                            dp, gen_path, agent=model_name_or_path
                        )
                    )

            if self.batch_evaluator is not None:
                self.batch_evaluator.batch_eval(batch_new_samples)

                sample_idx = 0
                for dp, req_output in zip(batch_dps, batch_req_outputs):
                    for _ in req_output.outputs:
                        dp.n_corrects += batch_new_samples[sample_idx].correct
                        sample_idx += 1

            # Save responses batch incrementally
            if save_path is None or save_path == "":  # No saving, return
                all_new_samples += batch_new_samples
            else:  # Save to file
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with open(save_path, "a", encoding="utf-8") as f:
                    for sample in batch_new_samples:
                        f.write(orjson.dumps(sample.to_dict()).decode() + "\n")

                sample_cnt += len(batch_new_samples)

            n_all_dps = len(query_dps)
            quit_cnt = 0
            achieve_cnt = 0
            for dp in query_dps:
                stop_reason = dp_stop_criteria(dp)
                quit_cnt += stop_reason == "max_n_trials"
                achieve_cnt += stop_reason == "max_n_corrects"

            logging.info(
                f"""# of new samples: {sample_cnt}
                Rate achieving `max_n_corrects` : {achieve_cnt / n_all_dps:.1%} (= {achieve_cnt}/{n_all_dps}) (not Acc. here)
                Rate running out `max_n_trials` : {quit_cnt / n_all_dps:.1%} (= {quit_cnt}/{n_all_dps})"""
            )

        if save_path is None:
            return all_new_samples
        else:
            return None
