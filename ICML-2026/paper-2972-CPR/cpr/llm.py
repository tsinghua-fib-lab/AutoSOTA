"""LLM drivers for relation hints (vLLM / Ollama)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import ollama
except ImportError:
    ollama = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:7b", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature

    def _chat(self, prompt: str) -> str:
        if ollama is None:
            return ""
        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "include_thinking": False},
        )
        return resp["message"]["content"]

    def propose_relation_chains(
        self,
        question: str,
        max_hop: int = 2,
        num_chains: int = 8,
    ) -> List[List[str]]:
        prompt = f"""
You are helping a Freebase-style KGQA system.

Given a question, propose up to {num_chains} likely relation chains
(1 to {max_hop} hops). Relations must be in dot-separated format like "common.topic.image".

Output STRICT JSON ONLY in this exact format:
{{"chains":[["relation1"],["relationA","relationB"]]}}

Question:
{question}

JSON:
"""
        raw = self._chat(prompt)
        try:
            j = json.loads(self._extract_json(raw))
            chains = j.get("chains", [])
            return [
                [r.strip() for r in c if r.strip()]
                for c in chains
                if isinstance(c, list) and 1 <= len(c) <= max_hop
            ][:num_chains]
        except Exception:
            return []

    def answer_from_paths(
        self,
        question: str,
        topic_entities: List[str],
        paths: List[Dict[str, Any]],
        topk: int = 20,
    ) -> Dict[str, Any]:
        paths = paths[:topk]
        prompt = f"""
You are a KGQA answer generator.
Output STRICT JSON only:
{{"answers":[...], "confidence":0.0, "rationale":"short"}}

Question: {question}
Topic entities: {topic_entities}
Paths: {json.dumps(paths, ensure_ascii=False)}
JSON:
"""
        raw = self._chat(prompt)
        try:
            j = json.loads(self._extract_json(raw))
            return {
                "answers": [str(a) for a in j.get("answers", [])],
                "confidence": float(j.get("confidence", 0.0)),
                "rationale": j.get("rationale", ""),
            }
        except Exception:
            return {"answers": [], "confidence": 0.0, "rationale": ""}

    @staticmethod
    def _extract_json(text: str) -> str:
        l, r = text.find("{"), text.rfind("}")
        return text[l : r + 1] if l >= 0 and r > l else text


class VLLMLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
    ):
        if OpenAI is None:
            raise ImportError("openai package required for VLLMLLM. pip install openai")
        self.model = model or os.environ.get("CPR_LLM_MODEL", "Qwen/Qwen3-8B")
        base = base_url or os.environ.get("CPR_VLLM_URL", "http://127.0.0.1:8000/v1")
        self.client = OpenAI(api_key=api_key, base_url=base)
        self.temperature = temperature

    def _chat(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            return ""

    def propose_relation_chains(
        self,
        question: str,
        max_hop: int = 2,
        num_chains: int = 8,
    ) -> List[List[str]]:
        prompt = f"""
You are helping a Freebase-style KGQA system.

Given a question, propose up to {num_chains} likely relation chains
(1 to {max_hop} hops). Relations must be in dot-separated format like "common.topic.image".

Output STRICT JSON ONLY in this exact format:
{{"chains":[["relation1"],["relationA","relationB"]]}}

Question:
{question}

JSON:
"""
        raw = self._chat(prompt)
        try:
            j = json.loads(self._extract_json(raw))
            chains = j.get("chains", [])
            return [
                [r.strip() for r in c if r.strip()]
                for c in chains
                if isinstance(c, list) and 1 <= len(c) <= max_hop
            ][:num_chains]
        except Exception:
            return []

    def answer_from_paths(
        self,
        question: str,
        topic_entities: List[str],
        paths: List[Dict[str, Any]],
        topk: int = 20,
    ) -> Dict[str, Any]:
        prompt = f"""You are a KGQA answer generator.
Output STRICT JSON only:
{{"answers":[...], "confidence":0.0, "rationale":"short"}}

Question: {question}
Topic entities: {topic_entities}
Paths: {json.dumps(paths[:topk], ensure_ascii=False)}
JSON:"""
        raw = self._chat(prompt)
        try:
            j = json.loads(self._extract_json(raw))
            return {
                "answers": [str(a) for a in j.get("answers", [])],
                "confidence": float(j.get("confidence", 0.0)),
                "rationale": j.get("rationale", ""),
            }
        except Exception:
            return {"answers": [], "confidence": 0.0, "rationale": ""}

    @staticmethod
    def _extract_json(text: str) -> str:
        l, r = text.find("{"), text.rfind("}")
        return text[l : r + 1] if l >= 0 and r > l else text
