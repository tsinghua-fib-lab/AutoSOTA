""" VLLM Server """

from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
)

from watermark_benchmark.utils.classes import (
    ConfigSpec,
    Generation,
    WatermarkSpec,
)
from watermark_benchmark.utils.stats import Stats

from .server import Server


class HFServer(Server, LogitsProcessor):
    """
    A Hugging Face based watermarking server
    """

    def __init__(self, config: Dict[str, Any], **kwargs) -> None:
        """
        Initializes the HF server.

        Args:
        - config (Dict[str, Any]): A dictionary containing the configuration of the model.
        - **kwargs: Additional keyword arguments.
        """
        model = config.model
        self.server = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.devices = [[i for i in range(torch.cuda.device_count())][0]]
        self.server = self.server.to(self.devices[0])
        self.watermark_engine = None
        self.batch_size = config.hf_batch_size
        self.current_batch = 0
        self.current_offset = 0

    def install(self, watermark_engine) -> None:
        """
        Installs the watermark engine.

        Args:
        - watermark_engine (Any): The watermark engine.
        """
        self.watermark_engine = watermark_engine

    def __call__(self, input_ids, scores):
        # Apply watermarking
        ids = [
            self.current_offset + (self.current_batch * self.batch_size) + i
            for i in range(self.batch_size)
        ]
        self.stats.update(scores, ids)
        if self.watermark_engine is not None:
            scores = self.watermark_engine.process(scores, input_ids, ids)
        return scores

    def run(
        self,
        inputs: List[str],
        config: ConfigSpec,
        temp: float,
        keys: Optional[List[int]] = None,
        watermark_spec: Optional[WatermarkSpec] = None,
        use_tqdm=False,
        **kwargs,
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
        # Setup logit processor
        processors = LogitsProcessorList()
        processors.append(self)

        # Run
        generations = []
        self.stats = Stats(len(inputs), temp)
        while True:
            try:
                self.current_offset = len(generations)
                for i in tqdm(
                    range(0, len(inputs) - len(generations), self.batch_size),
                    total=(len(inputs) - len(generations)) // self.batch_size,
                    description=f"Generating text (batch size {self.batch_size})",
                    disable=not use_tqdm,
                ):
                    self.current_batch = i
                    batch = self.tokenizer(
                        inputs[
                            self.current_offset
                            + (i * self.batch_size) : self.current_offset
                            + ((i + 1) * self.batch_size)
                        ],
                        return_tensors="pt",
                        padding=True,
                    ).to(self.devices[0])
                    outputs = self.server.generate(
                        batch,
                        temperature=temp,
                        max_length=config.max_new_tokens,
                        num_return_sequences=config.num_return_sequences,
                        do_sample=(temp > 0),
                        logits_processor=processors,
                    )
                    generations.extend(
                        [
                            Generation(
                                (
                                    watermark_spec
                                    if watermark_spec is not None
                                    else None
                                ),
                                (
                                    keys[
                                        self.current_offset
                                        + (i * self.batch_size)
                                        + j
                                    ]
                                    if keys is not None
                                    else None
                                ),
                                None,
                                self.current_offset + (i * self.batch_size) + j,
                                output.prompt,
                                output.outputs[0].text.strip(),
                                None,
                                None,
                                None,
                                *self.stats[
                                    self.current_offset
                                    + (i * self.batch_size)
                                    + j
                                ],
                                temp,
                                [],
                            )
                            for j, output in enumerate(outputs)
                        ]
                    )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.batch_size > 1:
                    torch.cuda.empty_cache()
                    self.batch_size = self.batch_size // 2
                    continue
                else:
                    raise e
            break

        self.current_batch = 0
        self.current_offset = 0
        return generations

    def tokenizer(self):
        """
        Returns the tokenizer.
        """
        return self.tokenizer
