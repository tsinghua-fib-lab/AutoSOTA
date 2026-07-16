""" VLLM Server """

from typing import Any, Dict, List, Optional

import torch
from vllm import (
    LLM,
    DefaultLogitProcessor,
    LogitProcessor,
    SamplingMetadata,
    SamplingParams,
)
from vllm.executor.gpu_executor import GPUExecutor
from vllm.executor.ray_gpu_executor import RayGPUExecutor

from watermark_benchmark.utils.classes import (
    ConfigSpec,
    Generation,
    WatermarkSpec,
)
from watermark_benchmark.utils.stats import Stats

from .server import Server


class VLLMServer(Server, LogitProcessor):
    """
    A VLLM based watermarking server

    Attributes:
    - config (Dict[str, Any]): A dictionary containing the configuration of the model.
    - model (Any): The LLM model.
    - ray (bool): A boolean indicating whether the model is distributed or not.
    - server (LLM): The LLM server.
    - devices (List[int]): A list of integers representing the available devices.
    - watermark_engine (Any): The watermark engine.
    - max_idx (int): The maximum index.
    """

    def __init__(self, config: Dict[str, Any], **kwargs) -> None:
        """
        Initializes the VLLMServer.

        Args:
        - config (Dict[str, Any]): A dictionary containing the configuration of the model.
        - **kwargs: Additional keyword arguments.
        """
        model = config.model
        self.ray = config.distributed
        if self.ray:
            self.server = LLM(
                model, tensor_parallel_size=torch.cuda.device_count(), **kwargs
            )
        else:
            self.server = LLM(model, **kwargs)

        self.devices = [i for i in range(torch.cuda.device_count())]
        self.watermark_engine = None
        self._activate_processor()
        self.max_idx = 0

    def _activate_processor(self) -> None:
        """
        Activates the logit processor.
        """
        if isinstance(self.server.llm_engine.model_executor, GPUExecutor):
            self.server.llm_engine.model_executor.driver_worker.model_runner.model.sampler.processor = (
                self
            )
        elif isinstance(self.server.llm_engine.model_executor, RayGPUExecutor):
            self.server.llm_engine.model_executor._run_workers(
                "update_logit_processor", get_all_outputs=True, processor=self
            )
        else:
            raise NotImplementedError(
                "We only currently support GPU and Ray backends for vLLM."
            )

    def _deactivate_processor(self) -> None:
        """
        Deactivates the logit processor.
        """
        if isinstance(self.server.llm_engine.model_executor, GPUExecutor):
            self.server.llm_engine.model_executor.driver_worker.model_runner.model.sampler.processor = (
                DefaultLogitProcessor()
            )
        elif isinstance(self.server.llm_engine.model_executor, RayGPUExecutor):
            self.server.llm_engine.model_executor._run_workers(
                "update_logit_processor",
                get_all_outputs=True,
                processor=DefaultLogitProcessor(),
            )
        else:
            raise NotImplementedError(
                "We only currently support GPU and Ray backends for vLLM."
            )

    def install(self, watermark_engine) -> None:
        """
        Installs the watermark engine.

        Args:
        - watermark_engine (Any): The watermark engine.
        """
        self.watermark_engine = watermark_engine

    def process(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        """
        Processes the logits.

        Args:
        - logits (torch.Tensor): The logits.
        - input_metadata (InputMetadata): The input metadata.

        Returns:
        - torch.Tensor: The processed logits.
        """
        prev_token_ids = [
            seq_group.seq_data[seq_id].get_token_ids()
            for seq_group in sampling_metadata.seq_groups
            for seq_id in seq_group.seq_ids
        ]
        ids = [
            seq_id - self.max_idx
            for seq_group in sampling_metadata.seq_groups
            for seq_id in seq_group.seq_ids
        ]

        logits_copy = logits.clone()

        if self.watermark_engine is not None:
            logits = self.watermark_engine.process(logits, prev_token_ids, ids)

        if logits_copy.shape[0] != len(ids):
            prompt_embeddings = {
                id: v.prompt_token_ids[1:]
                for seq_group in sampling_metadata.seq_groups
                for id, v in seq_group.seq_data.items()
                if v.prompt_token_ids is not None
            }
            self.stats.update(
                logits_copy,
                ids,
                logits.argmax(dim=-1),
                prompt_embeddings=prompt_embeddings,
            )
        else:
            self.stats.update(logits_copy, ids, logits.argmax(dim=-1))

        LogitProcessor._apply_temperatures(logits, sampling_metadata)
        return logits

    def run(
        self,
        inputs: List[str],
        config: ConfigSpec,
        temp: float,
        keys: Optional[List[int]] = None,
        watermark_spec: Optional[WatermarkSpec] = None,
        use_tqdm=False,
        **kwargs
    ) -> List[Generation]:
        """
        Runs the server.

        Args:
        - inputs (List[str]): A list of input strings.
        - config (ConfigSpec): The configuration.
        - temp (float): The temperature.
        - keys (Optional[List[int]]): A list of keys.
        - watermark_spec (Optional[WatermarkSpec]): The watermark specification.
        - use_tqdm (bool): A boolean indicating whether to use tqdm.

        Returns:
        - List[Generation]: A list of generations.
        """
        params = SamplingParams(
            temperature=temp,
            max_tokens=config.max_new_tokens,
            n=config.num_return_sequences,
            **kwargs,
        )
        self.stats = Stats(len(inputs), temp, config.logprobs)
        outputs = self.server.generate(inputs, params, use_tqdm=use_tqdm)

        if len(outputs) != len(inputs):
            raise RuntimeError(
                "Number of inputs and outputs do not match: {} inputs versus {} outputs".format(
                    len(inputs), len(outputs)
                )
            )

        generations = [
            Generation(
                watermark_spec if watermark_spec is not None else None,
                keys[i] if keys is not None else None,
                None,
                int(output.request_id) - self.max_idx,
                output.prompt,
                output.outputs[0].text.strip(),
                None,
                None,
                None,
                *self.stats[int(output.request_id) - self.max_idx],
                temp,
                (
                    [
                        (t, lp[t])
                        for lp, t in zip(
                            output.prompt_token_ids[1:],
                            output.prompt_logprobs[1:],
                        )
                    ]
                    if output.prompt_logprobs is not None
                    else None
                ),
                (
                    self.stats.logprobs[int(output.request_id) - self.max_idx]
                    if config.logprobs
                    else None
                ),
                original_tokens=output.outputs[0].token_ids,
            )
            for i, output in enumerate(outputs)
        ]
        self.max_idx = 1 + max([g.id + self.max_idx for g in generations])
        return generations

    def tokenizer(self):
        """
        Returns the tokenizer.
        """
        return self.server.get_tokenizer()
