"""
Minimal OpenAI-compatible embedding server for gte-Qwen2-7B-instruct.
"""
import argparse
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import torch
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

model = None
tokenizer = None
device = None


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "gte-Qwen2-7B-instruct"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str = "gte-Qwen2-7B-instruct"
    usage: UsageInfo


def last_token_pool(last_hidden_states, attention_mask):
    """Pool embeddings using the last token (EOS token)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint."""
    global model, tokenizer, device
    texts = request.input if isinstance(request.input, list) else [request.input]

    prompt_tokens = 0
    truncated = []
    for t in texts:
        tok = tokenizer.encode(t, add_special_tokens=False)
        prompt_tokens += len(tok)
        if len(tok) > 7500:
            tok = tok[-7500:]
            t = tokenizer.decode(tok, skip_special_tokens=True)
        truncated.append(t)

    batch_dict = tokenizer(
        truncated, padding=True, truncation=True,
        max_length=8192, return_tensors="pt",
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    data = [
        EmbeddingData(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        usage=UsageInfo(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    global model, tokenizer, device
    device = args.device

    print(f"Loading model from {args.model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=False, local_files_only=True,
        torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    print(f"Model loaded. Starting server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
