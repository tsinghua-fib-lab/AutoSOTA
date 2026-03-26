"""Load CiteEval.
"""
import os
import io
import json
from pathlib import Path
from modules import ContextAttributor, CitationEditor, LightCitationRater, DistanceBasedCitationRater
from common_utils.logging_utils import get_logger
logger = get_logger(__name__)


def load_citeeval_config(version):
    citeeval_config_file = Path(__file__).parent / f"citeeval_config.json"
    with io.open(citeeval_config_file) as f:
        citeeval_configs = json.load(f)
        
        if version not in citeeval_configs:
            raise ValueError(f"CiteEval-Auto version not found: {version}. Has to be one of: {list(citeeval_configs.keys())}")
        config = citeeval_configs[version]
    return config


class CiteEvalLoader:
    def __init__(self, version, model_name, n_threads):
        self.version = version
        self.citeeval_config = load_citeeval_config(version=version)
        self.args = {
            "citeeval_config": self.citeeval_config,
            "verbose": False,
        }

        prompt_version = self.citeeval_config["prompt_version"] if "prompt_version" in self.citeeval_config else version
        template_config_dir = Path(__file__).parent.parent / f"template_configs/{prompt_version}"
        self.llm_args = {
            "template_config_dir": template_config_dir,
            "max_request": 128,
            "max_len": 4096,
            "model": model_name,
            "n_threads": n_threads
        }

        logger.info(f"Prompt version: {prompt_version}")
        logger.info(f"Template config: {template_config_dir}")
        logger.info(f"model: {model_name}")

    def load_module(
            self,
            module_name=None,
            ca_output_file=None,
            ce_output_file=None,
        ):
        """
        module_name: ca, ce, cr_itercoe, cr_editdist
        ca_output_file: Output file from Context Attributor (required by Citation Rater)
        ce_output_file: Output file from Citation Editor (required by Citation Rater)
        """
        if module_name == "ca":
            return ContextAttributor(**self.args, **self.llm_args)
        
        elif module_name.startswith("ce"):
            return CitationEditor(**self.args, **self.llm_args)
        
        elif module_name.startswith('cr'):
            cr_args = {
                "ca_output_file": ca_output_file,
                "ce_output_file": ce_output_file,
            }
            if module_name == "cr_itercoe":
                return LightCitationRater(**self.args, **cr_args)
            elif module_name == "cr_editdist":
                return DistanceBasedCitationRater(**self.args, **cr_args, distance_type="estimated")
            else:
                raise ValueError(f"Invalid module_name: {module_name}")
        
        else:
            raise NotImplementedError(module_name)