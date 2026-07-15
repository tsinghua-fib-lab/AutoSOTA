#!/usr/bin/env python3
"""Minimal OpenAI-compatible API server for serving a local model."""
import argparse
import json
import time
import os
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "fastapi", "uvicorn", "pydantic", "-q",
                           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                           "--trusted-host", "pypi.tuna.tsinghua.edu.cn"])
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

app = FastAPI(title="Local Model Server")

MODEL = None
TOKENIZER = None
DEVICE = "cuda"
MODEL_NAME = "local-model"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default="local-model")
    messages: list[Message]
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int = 0
    model: str = "local-model"
    choices: list[Choice]
    usage: Usage


def load_model(model_path: str):
    global MODEL, TOKENIZER
    print(f"Loading model from {model_path}...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    MODEL.eval()
    print(f"Model loaded. Device: {MODEL.device}")
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory used: {gpu_mem:.1f} GB")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


def build_prompt(messages: list[Message]) -> str:
    """Build a chat prompt from messages."""
    # Check if the model uses a chat template
    if hasattr(TOKENIZER, "apply_chat_template"):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        try:
            prompt = TOKENIZER.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            return prompt
        except Exception:
            pass

    # Fallback: simple concatenation
    parts = []
    for m in messages:
        role = m.role
        content = m.content
        if role == "system":
            parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
        elif role == "user":
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
        elif role == "assistant":
            parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prompt = build_prompt(request.messages)
        inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )

        # Decode only the generated part
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        response_text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)

        choice = Choice(
            message=Message(role="assistant", content=response_text),
            finish_reason="stop",
        )

        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model or MODEL_NAME,
            choices=[choice],
            usage=Usage(
                prompt_tokens=input_len,
                completion_tokens=len(generated_ids),
                total_tokens=input_len + len(generated_ids),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model)
    MODEL_NAME = os.path.basename(args.model.rstrip("/"))

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
