import os
import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import config
from retriever import PhantomRetriever, AttentionRetriever, LLMRetriever, AutoDANRetriever
from generator import MCGGenerator, AttentionGenerator, LLMGenerator, AutoDANGenerator


class Retriever:
    """
    Thin wrapper around different retriever implementations.
    Handles creation and simple pass-through for optimize(), set_dataset(),
    and correlation (only if the underlying retriever supports it).
    """
    def __init__(
        self,
        dataset,
        save_dir,
        model_path: str = config.RETRIEVER_MODEL_PATH,
        model_type: str = config.RETRIEVER_TYPE,
        device: str = config.DEVICE,
        trigger_phrase: str = config.TRIGGER_PHRASE,
        k: int = config.RET_CORRELATION_THRESHOLD,
        malicious_template: str=config.RET_MALICIOUS_TRIGGER_DOC_TEMPLATE
    ):
        self.model_path = model_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if model_type == "attention":
            self.retriever = AttentionRetriever(
                model_path=model_path,
                save_dir=save_dir,
                filter_model_path=config.FILTER_MODEL_PATH,
                trigger_phrase=trigger_phrase,
                device=self.device,
                k=k,
                dataset=dataset,
                malicious_template=malicious_template,
            )
        elif model_type == "phantom" or model_type == "GCG":
            self.retriever = PhantomRetriever(
                model_path=model_path,
                trigger=trigger_phrase,
            )
            self._set_dataset(dataset)
        elif model_type == "llm":
            self.retriever = LLMRetriever(
                model_path=config.GENERATOR_MODEL_PATH,
                trigger_phrase=trigger_phrase,
                dataset=dataset,
            )
        elif model_type == "AutoDAN":
            self.retriever = AutoDANRetriever(
                retriever_model_path=config.RETRIEVER_MODEL_PATH,
                generator_model_path=config.GENERATOR_MODEL_PATH,
                trigger_phrase=trigger_phrase,
                dataset=dataset,
            )
        else:
            raise ValueError(f"Unknown retriever type: {model_type}")
        

    # ---- pass-through APIs ----
    def _set_dataset(self, dataset):
        return self.retriever.set_dataset(dataset)

    def optimize(self, *args, **kwargs):
        return self.retriever.optimize(*args, **kwargs)

    # Correlation is only supported by AttentionRetriever
    def get_correlation(self, *args, **kwargs):
        if hasattr(self.retriever, "get_correlation"):
            return self.retriever.get_correlation(*args, **kwargs)
        raise NotImplementedError("Correlation is only available for AttentionRetriever.")

    def load_correlation(self, *args, **kwargs):
        if hasattr(self.retriever, "load_correlation"):
            return self.retriever.load_correlation(*args, **kwargs)
        raise NotImplementedError("Correlation is only available for AttentionRetriever.")


class Generator:
    """
    Thin wrapper around different generator implementations.
    Exposes optimize(), set_dataset(), and correlation for AttentionGenerator.
    """
    def __init__(
        self,
        dataset,
        save_dir,
        retrieval_results: str = "",
        model_path: str = config.GENERATOR_MODEL_PATH,
        model_type: str = config.GENERATOR_TYPE,
        trigger_phrase: str = config.TRIGGER_PHRASE,
        k: int = config.GEN_CORRELATION_THRESHOLD,
    ):
        self.model_path = model_path

        if model_type == "MCG" or model_type == "GCG":
            self.generator = MCGGenerator(dataset, retrieval_results, model_path, trigger_phrase=trigger_phrase)
        elif model_type == "attention":
            self.generator = AttentionGenerator(dataset, save_dir, retrieval_results, model_path, trigger_phrase=trigger_phrase, k=k)
            self.generator.set_dataset(dataset=dataset)
        elif model_type == "llm":
            self.generator = LLMGenerator(
                model_path=model_path,
                trigger_phrase=trigger_phrase,
                retrieval_results=retrieval_results
            )
        elif model_type == "AutoDAN":
            self.generator = AutoDANGenerator(
                model_path=model_path,
                retrieval_results=retrieval_results
            )
        else:
            raise ValueError(f"Unknown generator type: {model_type}")

        self.device = torch.device(config.DEVICE)
        self.model = None
        self.tokenizer = None

    # Optional loaders if you need direct access (not required for wrapper usage)
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, output_attentions=True
        ).to(self.device).eval()

    # ---- pass-through APIs ----
    def optimize(self, *args, **kwargs):
        return self.generator.optimize(*args, **kwargs)

    def get_correlation(self, *args, **kwargs):
        if hasattr(self.generator, "get_correlation"):
            return self.generator.get_correlation(*args, **kwargs)
        raise NotImplementedError("Correlation is only available for AttentionGenerator.")

    def load_correlation(self, *args, **kwargs):
        if hasattr(self.generator, "load_correlation"):
            return self.generator.load_correlation(*args, **kwargs)
        raise NotImplementedError("Correlation is only available for AttentionGenerator.")
