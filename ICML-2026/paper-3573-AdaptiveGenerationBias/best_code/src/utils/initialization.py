import os
import random
import sys
from typing import TextIO, Tuple, no_type_check, Optional

import numpy as np
import openai
import yaml
from pydantic import ValidationError

from src.configs import (
    Config,
    GENPROFILESConfig,
    RUNConfig,
    QuestionConfig,
    NeutralGenConfig,
    BiasTransferConfig,
    MODELEVALConfig,
)
from src.configs.config import BaselineGenConfig, QuestionTransformConfig

from .string_utils import string_hash


class SafeOpen:
    def __init__(self, path: str, mode: str = "a", ask: bool = False):
        self.path = path

        if ask and os.path.exists(path):
            if input("File already exists. Overwrite? (y/n)") != "y":
                raise Exception("File already exists")
        self.mode = "a+" if os.path.exists(path) else "w+"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.file = None
        self.lines = []

    def __enter__(self):
        self.file = open(self.path, self.mode)
        self.file.seek(0)  # move the cursor to the beginning of the file
        self.lines = self.file.readlines()
        # Remove last lines if empty
        while len(self.lines) > 0 and self.lines[-1] == "":
            self.lines.pop()
        return self

    def flush(self):
        self.file.flush()

    def write(self, content):
        self.file.write(content)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


def read_config_from_yaml(path) -> Config:
    with open(path, "r") as stream:
        try:
            yaml_obj = yaml.full_load(stream)
            print(yaml_obj)
            cfg = Config(**yaml_obj)
            return cfg
        except (yaml.YAMLError, ValidationError) as exc:
            print(exc)
            raise exc


def seed_everything(seed: int) -> None:
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_credentials(cfg: Config) -> None:
    # Go through the config and get all model providers
    task = cfg.task_config
    if isinstance(task, GENPROFILESConfig):
        providers = [task.gen_model.provider]
    elif isinstance(task, RUNConfig):
        providers = [
            task.conversation_config.persona_model.provider,
            task.judge_config.judge_model.provider,
            task.question_config.gen_model.provider,
            task.question_config.refiner_config.model.provider,
            task.question_config.topic_generation.topic_model.provider,
        ]
        assistant_providers = [model.provider for model in task.conversation_config.assistant_model]
        providers.extend(assistant_providers)
    elif isinstance(task, QuestionConfig):
        providers = [task.gen_model.provider]
    elif isinstance(task, MODELEVALConfig):
        providers = []
        for model in task.eval_models:
            if model.provider not in providers:
                providers.append(model.provider)
    elif (
        isinstance(task, NeutralGenConfig)
        or isinstance(task, BiasTransferConfig)
        or isinstance(task, BaselineGenConfig)
    ):
        providers = [task.model.provider]
    elif isinstance(task, QuestionTransformConfig):
        providers = [task.question_transformer_config.model.provider]
    else:
        providers = []

    for provider in providers:
        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "OpenAI API key not set. Please set in environment variable OPENAI_API_KEY"
                )
        elif provider == "azure":
            if not os.environ.get("AZURE_KEY"):
                raise ValueError(
                    "Azure API key not set. Please set in environment variable AZURE_KEY"
                )
        elif provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "Anthropic API key not set. Please set in environment variable ANTHROPIC_API_KEY"
                )
        elif provider == "together":
            if not os.environ.get("TOGETHER_API_KEY"):
                raise ValueError(
                    "Together API key not set. Please set in environment variable TOGETHER_API_KEY"
                )
        elif provider == "hf":
            continue
        else:
            print(f"Provider {provider} not recognized - Assuming no credentials needed")

        if provider == "azure":
            openai.api_type = "azure"
            openai.api_base = os.environ.get("AZURE_ENDPOINT")
            openai.api_key = os.environ.get("AZURE_KEY")
            openai.api_version = os.environ.get("AZURE_API_VERSION")


@no_type_check
def get_out_file(cfg: Config) -> Tuple[TextIO, str]:
    file_path = cfg.get_out_path()

    if not cfg.store:
        return sys.stdout, ""

    if len(file_path) > 255:
        file_path = file_path.split("/")
        file_name = file_path[-1]
        file_name_hash = string_hash(file_name)
        file_path = "/".join(file_path[:-1]) + "/hash_" + str(file_name_hash) + ".txt"

    ctr = 1
    while os.path.exists(file_path):
        with open(file_path, "r") as fp:
            num_lines = len(fp.readlines())

        if num_lines >= 20:
            file_path = file_path.split("/")
            file_name = file_path[-1]

            ext = file_name.split(".")[-1]
            v_counter = file_name.split("_")[-1].split(".")[0]
            if v_counter.isdigit():
                ext_len = len(ext) + len(v_counter) + 2
                file_name = file_name[:-ext_len]
            else:
                file_name = file_name[: -(len(ext) + 1)]

            file_path = "/".join(file_path[:-1]) + "/" + file_name + "_" + str(ctr) + ".txt"
            ctr += 1
        else:
            break

    if cfg.store:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f = open(file_path, "w")
        sys.stdout = f

    return f, file_path
