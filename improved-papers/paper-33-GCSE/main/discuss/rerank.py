# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 7"
import mteb
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from main.models.gcse import GCSERoberta

class RerankModel(nn.Module):
    def __init__(self, model_path, tokenizer_path, batch_size: int = 32, gpu=[0]) -> None:
        super().__init__()
        self.model = GCSERoberta(from_pretrained=model_path,
                                pooler_type='cls')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.batch_size = batch_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu).cuda()
        self.to(device)
        self.share_memory()

    def encode(
        self, sentences: list[str], **kwargs
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        batch_list = [sentences[i:i+self.batch_size] for i in range(0, len(sentences), self.batch_size)]
        emb = []
        for batcher in batch_list:
            inputs = self.tokenizer(batcher, padding='max_length', truncation=True, return_tensors="pt", max_length=32)
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.module.device)
            outputs = self.model(**inputs)
            pooler_output = outputs.pooler_output
            emb += pooler_output.cpu().detach()
        return torch.stack(emb)

model_path = '/home/lpc/repos/sTextSim/save_model/gcseLarge_TransferDAFullUn_unsup_sota/GCSE_step_750'
tokenizer_path = '/home/lpc/models/unsup-simcse-bert-base-uncased/'
model = RerankModel(model_path, tokenizer_path, batch_size=64, gpu=[0, 1, 2, 3, 4, 5, 6])

# tasks = mteb.get_tasks(tasks=["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions"])
tasks = mteb.get_tasks(tasks=["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/gcse_large", eval_splits=["test"], overwrite_results=True)

# %%
