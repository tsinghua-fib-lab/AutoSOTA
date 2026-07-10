import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set your model path here, or set the MODEL_DIR environment variable
MODEL_PATH = str(config.MODEL_DIR / "bce-embedding-base_v1")

# Replace with the two strings you want to compare
text1 = "sldisldkfjldskfualsdifk"
text2 = "Hello world"


def mean_pool(model_output, attention_mask):
    token_embs = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
    return torch.sum(token_embs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def compute_similarity(t1: str, t2: str, model_path: str = MODEL_PATH) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    encoded = tokenizer([t1, t2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pool(output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return float(F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item())


if __name__ == "__main__":
    score = compute_similarity(text1, text2)
    print(f"Cosine similarity: {score:.4f}")
