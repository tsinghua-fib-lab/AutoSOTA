from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import dacite
from apps.classes import ConfigCode


def str_or_none(val):
    if val is not None:
        return str(val)
    else:
        return "-"


def parse_string_val(val):
    if val == "-":
        return None
    return val


def dict_to_str(val):
    return ast.literal_eval(val)


@dataclass(frozen=True)
class VerifierSpec:
    """
    A class representing the verifier specification.

    Attributes:
        verifier (str): The verifier to use. Defaults to 'Theoretical'.
        empirical_method (str): The empirical method to use. Defaults to 'regular'.
        log (Optional[bool]): Whether to use a log score
        gamma (Optional[float]): The gamma value to use for edit distance. Defaults to 0.
    """

    verifier: str = "Theoretical"
    empirical_method: str = "regular"
    log: Optional[bool] = None
    gamma: Optional[float] = 0

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> VerifierSpec:
        return dacite.from_dict(VerifierSpec, d)

    @staticmethod
    def from_str(s: str) -> VerifierSpec:
        return VerifierSpec.from_dict(dict_to_str(s))


@dataclass(frozen=True)
class WatermarkSpec:
    """
    Specifies how to perform the watermarking
    """

    # Random number generator type
    rng: str = "Internal"

    # Random number generator parameters
    hash_len: Optional[int] = None
    min_hash: Optional[bool] = None
    secret_key: int = field(default=0, hash=False)
    key_len: Optional[int] = None

    # Generator settings
    generator: Optional[str] = None
    tokenizer: Optional[str] = field(default=None, hash=False)
    temp: float = 1.0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    skip_prob: Optional[float] = 0

    # Verifier settings
    pvalue: float = 0.01
    verifiers: List[VerifierSpec] = field(default_factory=list, hash=False)

    # Randomization settings
    randomize: bool = False
    offset: bool = False

    # Custom
    custom_fields: Optional[Dict[str, Any]] = field(
        default_factory=dict, hash=False
    )

    def to_dict(self, omit_key=False, omit_verifiers=False) -> Dict:
        d = self.__dict__
        if omit_key:
            d = {k: v for k, v in d.items() if k != "secret_key"}
        if omit_verifiers:
            d = {k: v for k, v in d.items() if k != "verifiers"}
        else:
            d = {
                k: v if k != "verifiers" else [i.__dict__ for i in v]
                for k, v in d.items()
            }
        return d

    def __str__(self) -> str:
        return str(self.to_dict(omit_key=True, omit_verifiers=False))

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str:Any]) -> WatermarkSpec:
        return dacite.from_dict(WatermarkSpec, d)

    @staticmethod
    def from_str(s: str) -> WatermarkSpec:
        return WatermarkSpec.from_dict(dict_to_str(s))

    def sep_verifiers(self) -> List[WatermarkSpec]:
        rtn = []
        for v in self.verifiers:
            rtn.append(replace(self, verifiers=[v]))
        return rtn


@dataclass(frozen=True)
class Generation:
    """Defines the content of a generation"""

    watermark: Optional[WatermarkSpec] = None
    key: Optional[int] = None
    attack: Optional[str] = None
    id: int = 0
    prompt: str = ""
    response: str = ""
    rating: Optional[float] = None
    pvalue: Optional[float] = None
    efficiency: Optional[float] = None
    token_count: int = 0
    entropy: Optional[float] = None
    spike_entropy: Optional[float] = None
    temp: Optional[float] = None

    # Non printed fields
    prompt_logprobs: Optional[List[Tuple[int, float]]] = None
    logprobs: Optional[List[float]] = field(default_factory=list, hash=False)
    original_tokens: Optional[List[int]] = field(
        default_factory=list, hash=False
    )

    @staticmethod
    def keys() -> List[str]:
        return [
            "watermark",
            "key",
            "id",
            "attack",
            "prompt",
            "response",
            "rating",
            "pvalue",
            "efficiency",
            "token_count",
            "entropy",
            "spike_entropy",
            "temp",
        ]

    def __str__(self) -> str:
        cpy_dict = {k: v for k, v in self.__dict__.items()}
        if "response" in cpy_dict and cpy_dict["response"] is not None:
            cpy_dict["response"] = (
                cpy_dict["response"]
                .replace("\000", "")
                .replace("\r", "___LINE___")
                .replace("\t", "___TAB___")
                .replace("\n", "___LINE___")
            )
        if "prompt" in cpy_dict and cpy_dict["prompt"] is not None:
            cpy_dict["prompt"] = (
                cpy_dict["prompt"]
                .replace("\000", "")
                .replace("\r", "___LINE___")
                .replace("\t", "___TAB___")
                .replace("\n", "___LINE___")
            )
        return "\t".join(
            [
                str_or_none(cpy_dict[v] if v in cpy_dict else None)
                for v in Generation.keys()
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_str(s: str) -> Generation:
        keys = Generation.keys()
        val_dict = {key: None for key in keys}
        s = [parse_string_val(val) for val in s.split("\t")]
        for i in range(len(keys)):
            if i >= len(s):
                break
            key = keys[i]
            val = s[i]

            if key == "watermark" and val is not None:
                val = WatermarkSpec.from_str(val)
            elif key in ["key", "token_count", "id"] and val is not None:
                val = int(val)
            elif (
                key
                in [
                    "rating",
                    "pvalue",
                    "efficiency",
                    "entropy",
                    "spike_entropy",
                ]
                and val is not None
            ):
                try:
                    val = float(val)
                except ValueError:
                    val = tuple(
                        float(v.strip())
                        for v in val.replace(")", "")
                        .replace("(", "")
                        .split(",")
                        if len(v)
                    )
            elif (key == "response" or key == "prompt") and val is not None:
                val = (
                    re.sub(r"(___LINE___)+$", "___LINE___", val)
                    .replace("___LINE___", "\n")
                    .replace("___TAB___", "\t")
                )

            val_dict[key] = val

        if val_dict["key"] is not None and val_dict["watermark"] is not None:
            val_dict["watermark"] = replace(
                val_dict["watermark"], secret_key=val_dict["key"]
            )

        if val_dict["temp"] is None and val_dict["watermark"] is not None:
            val_dict["temp"] = val_dict["watermark"].temp

        return Generation(**val_dict)

    @staticmethod
    def from_file(filename: str) -> List[Generation]:
        with open(filename, "r") as infile:
            raw = [l for l in infile.read().split("\n") if len(l)][1:]
        return [Generation.from_str(r) for r in raw]

    @staticmethod
    def to_file(
        filename: str, generations: Optional[List[Generation]] = None
    ) -> None:
        with open(filename, "w") as outfile:
            outfile.write("\t".join(Generation.keys()) + "\n")
            if generations is not None and len(generations):
                outfile.write("\n".join(str(g) for g in generations) + "\n")


@dataclass(frozen=True)
class Hull:
    volume: float = 0
    keys: Optional[List[str]] = field(default_factory=list, hash=False)
    faces: Optional[List[int]] = field(default_factory=list, hash=False)
    points: Optional[Dict[str, Tuple]] = field(default_factory=dict, hash=False)
    selected: Optional[List[str]] = field(default_factory=list, hash=False)
    added: Optional[Set[str]] = field(default_factory=set, hash=False)


@dataclass(frozen=True)
class Robustness:
    hull: Hull
    efficiency_threshold: Optional[float] = None
    percentage_threshold: Optional[float] = None
    best_attack_efficiency: Optional[str] = None
    best_attack_percentage: Optional[str] = None

    @staticmethod
    def tsv_keys() -> str:
        return [
            "area",
            "efficiency_threshold",
            "percentage_threshold",
            "best_attack_efficiency",
            "best_attack_percentage",
        ]

    def to_list(self) -> list:
        return [
            str(v)
            for v in [
                self.hull.volume,
                self.efficiency_threshold,
                self.percentage_threshold,
                self.best_attack_efficiency,
                self.best_attack_percentage,
            ]
        ]


@dataclass(frozen=True)
class BenchmarkResults:
    """Summary of benchmark run"""

    quality: Optional[float] = None
    efficiency: Optional[float] = None
    percent_watermarked: Optional[float] = None
    robustness: Optional[Robustness] = None

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> BenchmarkResults:
        return dacite.from_dict(BenchmarkResults, d)

    @staticmethod
    def from_str(s: str) -> BenchmarkResults:
        return BenchmarkResults.from_dict(dict_to_str(s))

    @staticmethod
    def to_tsv(s: Dict[str, BenchmarkResults]) -> str:
        base_keys = ["quality", "efficiency", "percent_watermarked"]
        keys = ["watermark"] + base_keys + Robustness.tsv_keys()
        return (
            "\t".join(keys)
            + "\n"
            + "\n".join(
                "\t".join(
                    v
                    for v in [
                        str(i)
                        for i in [w]
                        + [v.__dict__[k] for k in base_keys]
                        + (
                            v.robustness.to_list()
                            if v.robustness is not None
                            else []
                        )
                    ]
                )
                for w, v in s.items()
            )
            + "\n"
        )


@dataclass(frozen=True)
class Threshold:
    objective: str
    q: float
    r: Optional[float] = None

    def __init__(self, thsd):
        if thsd[1] >= 0:
            object.__setattr__(self, "objective", "size")
            object.__setattr__(self, "q", thsd[0])
            object.__setattr__(self, "r", thsd[1])
        else:
            object.__setattr__(self, "objective", "robustness")
            object.__setattr__(self, "q", thsd[0])
            object.__setattr__(self, "r", None)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class AggregateResults:
    """Summary of benchmark run"""

    axis: str
    value: str
    temp: float
    threshold: Threshold
    q: float
    e: float
    r: float

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> AggregateResults:
        return dacite.from_dict(AggregateResults, d)

    @staticmethod
    def from_str(s: str) -> AggregateResults:
        return AggregateResults.from_dict(dict_to_str(s))

    @staticmethod
    def to_tsv(s: List[AggregateResults]) -> str:
        keys = [
            "axis",
            "value",
            "temp",
            "objective",
            "quality_threshold",
            "robustness_threshold",
            "Q",
            "E",
            "R",
        ]
        header = "\t".join(keys) + "\n"
        lines = [
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                v.axis,
                v.value,
                v.temp,
                v.threshold.objective,
                v.threshold.q,
                v.threshold.r,
                v.q,
                v.e,
                v.r,
            )
            for v in s
        ]
        return header + "".join(lines)


@dataclass(frozen=False)
class VerifierOutput:
    """Output from a watermark's verifier"""

    sequence_token_count: int = 0
    pvalues: Dict[int, float] = field(default_factory=dict, hash=False)

    def update(self, token_index, pvalue):
        if token_index > self.sequence_token_count:
            self.sequence_token_count = token_index
        self.pvalues[token_index] = pvalue

    def get_pvalue(self):
        return (
            0.5
            if not len(self.pvalues)
            else self.pvalues[self.sequence_token_count]
        )

    def get_size(self, pvalue):
        """
        Returns the size of a watermark detection algorithm given a list of watermark scores and a p-value threshold.

        Returns:
            float: The efficiency of the watermark detection algorithm, defined as the detection time of the last detected watermark.
        """
        if not len(self.pvalues):
            return math.inf

        if self.pvalues[self.sequence_token_count] > pvalue:
            return math.inf

        idx = self.sequence_token_count - 1
        while idx > 0 and idx in self.pvalues and self.pvalues[idx] <= pvalue:
            idx -= 1

        return idx


@dataclass(frozen=False)
class ConfigSpec:
    """Configuration spec"""

    done_tasks: str = ".saved_tasks"
    num_return_sequences: int = 1
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    engine: str = "vllm"
    output_file: Optional[str] = None
    input_file: Optional[str] = None
    baseline: bool = True
    watermark: str = "watermark_specs"
    max_new_tokens: int = 1024
    seed: Optional[int] = None
    huffman_coding: Optional[str] = None
    distributed: bool = False
    hf_batch_size: int = 64

    paraphrase: bool = False
    dipper_processes: int = 1
    openai_processes: int = 1
    openai_key: Optional[str] = None
    threads: int = 32
    misspellings: str = "static_data/misspellings.json"
    devices: Optional[List[int]] = None
    detections_per_gpu: int = 4

    quality_metric: str = "llm"

    results: str = "results"
    threshold: float = 0.8
    hull_axis: List[Any] = field(
        default_factory=lambda: [["generator"], ["rng"]]
    )
    aggregate_thresholds: List[Any] = field(
        default_factory=lambda: [
            [0.02, 1],
            [0.1, 1],
            [0.02, 0.8],
            [0.1, 0.8],
            [0.02, -1],
            [0.1, -1],
        ]
    )
    validation: bool = False
    load_from_save: bool = False

    gpu_memory_utilization: Optional[float] = None
    dtype: Optional[str] = "bfloat16"
    trust_remote_code: Optional[bool] = None
    logprobs: bool = False

    openai_parallelism: Optional[int] = 32

    code_config: ConfigCode = ConfigCode()

    def get_devices(self):
        import torch

        if self.devices is not None:
            return self.devices
        elif not torch.cuda.is_available():
            return ["cpu"]
        else:
            return list(range(torch.cuda.device_count()))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ConfigSpec:
        return dacite.from_dict(ConfigSpec, d)
