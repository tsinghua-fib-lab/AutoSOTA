#!/usr/bin/env python3
"""OpenAI-compatible embedding server for gte-Qwen2-7B-instruct."""
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/models/iic--gte-Qwen2-7B-instruct/snapshots/master")
MODEL_NAME = os.environ.get("MODEL_NAME", "gte-Qwen2-7B-instruct")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="Embedding Server")

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {model.device}")


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction prefix per GTE model spec."""
    return f'Instruct: {task_description}\nQuery: {query}'


TASK_DESC = "Given a question, find relevant support questions that help estimate LLM performance on this question."


class EmbeddingRequest(BaseModel):
    input: str
    model: str = MODEL_NAME
    encoding_format: Optional[str] = "float"


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    global model, tokenizer
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Handle input: could be a single string or a list of strings
    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list):
        texts = [str(t) for t in request.input]
    else:
        texts = [str(request.input)]

    all_embeddings = []
    for text in texts:
        # Format with instruction prefix
        formatted = get_detailed_instruct(TASK_DESC, text)
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling of last hidden state
            attention_mask = inputs["attention_mask"]
            hidden_state = outputs.last_hidden_state  # [1, seq_len, dim]
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()  # [1, seq_len, 1]
            embedding = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)  # [1, dim]
            embedding = F.normalize(embedding, p=2, dim=-1)

        emb_list = embedding[0].cpu().tolist()
        all_embeddings.append(emb_list)

    data = [
        {"object": "embedding", "index": i, "embedding": emb}
        for i, emb in enumerate(all_embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage={"prompt_tokens": len(tokenizer.encode(request.input if isinstance(request.input, str) else request.input[0])), "total_tokens": 0},
    )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
