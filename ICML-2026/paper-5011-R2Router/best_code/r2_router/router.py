"""
R2-Router: LLM Router with Joint Model-Budget Optimization

Routes queries to the optimal (LLM, token_budget) pair using per-LLM Ridge
regression predictors, then generates a response via OpenAI-compatible API.

Each LLM has 17 Ridge regressors:
  - 15 limited budget quality predictors (budget=10,20,...,4000)
  - 1 unlimited quality predictor
  - 1 unlimited token count predictor

Usage:
    from r2_router import R2Router

    router = R2Router.from_pretrained(
        "./r2_router",
        embed_url="http://localhost:8000",
        llm_api_base="https://openrouter.ai/api/v1",
        llm_api_key="sk-or-...",
    )

    # End-to-end: embed → route → generate
    result = router.route_and_generate("Write a Python function to calculate factorial.")
    print(result["model"])      # e.g., "Qwen3-235B-A22B-Instruct-2507"
    print(result["budget"])     # e.g., 100
    print(result["response"])   # LLM's answer

    # Route only
    decision = router.route_text("Solve 2x + 5 = 13")
"""

import os
import json
import numpy as np
import joblib
from typing import Dict, List, Optional, Union


# Token budgets matching training data
TOKEN_LIMITS = [10, 20, 30, 40, 50, 80, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000]


class R2Router:
    """
    R2-Router: Routes queries to optimal (LLM, token_budget) pair.

    For each candidate (model, budget), computes:
        risk = (1 - lambda) * predicted_quality - lambda * predicted_cost
    where cost = token_count * model_size.

    Selects the (model, budget) pair with the highest risk score.
    """

    def __init__(
        self,
        predictors: Dict[str, Dict],
        model_configs: Dict[str, Dict],
        lambda_val: float = 0.99999,
        embed_url: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ):
        """
        Args:
            predictors: {model_name: {"limited": {budget_key: Ridge}, "unlimited_score": Ridge, "unlimited_token": Ridge}}
            model_configs: {model_name: {"input_price_per_million": float, "output_price_per_million": float, "openrouter_id": str, ...}}
            lambda_val: Cost-quality tradeoff. 0 = pure quality, 1 = pure cost minimization.
            embed_url: vLLM server URL for Qwen3-0.6B embedding
            llm_api_base: OpenAI-compatible API base for LLM generation
            llm_api_key: API key for LLM endpoint
        """
        self.predictors = predictors
        self.model_configs = model_configs
        self.lambda_val = lambda_val
        self.embed_url = embed_url
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self._embedder = None

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        lambda_val: float = 0.99999,
        embed_url: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        verbose: bool = False,
    ) -> "R2Router":
        """
        Load pre-trained Ridge checkpoints.

        Args:
            path: Local directory or HuggingFace repo ID
            lambda_val: Cost-quality tradeoff (default 0.99999)
            embed_url: vLLM server URL for embedding
            llm_api_base: OpenAI-compatible API base for generation
            llm_api_key: API key for LLM endpoint
        """
        if not os.path.isdir(path):
            path = cls._download_from_hf(path)

        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

        ckpt_dir = os.path.join(path, "checkpoints")
        predictors = {}

        for model_name, model_cfg in config["models"].items():
            ckpt_name = model_cfg["checkpoint_dir"]
            model_dir = os.path.join(ckpt_dir, ckpt_name)

            limited_path = os.path.join(model_dir, "limited_score_predictors.joblib")
            unlimited_score_path = os.path.join(model_dir, "unlimited_score_predictor.joblib")
            unlimited_token_path = os.path.join(model_dir, "unlimited_token_predictor.joblib")

            if not os.path.exists(limited_path):
                continue

            predictors[model_name] = {
                "limited": joblib.load(limited_path),
                "unlimited_score": joblib.load(unlimited_score_path),
                "unlimited_token": joblib.load(unlimited_token_path),
            }

        if not predictors:
            raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

        if verbose:
            print(f"R2-Router loaded {len(predictors)} models: {list(predictors.keys())}")

        return cls(
            predictors=predictors,
            model_configs=config["models"],
            lambda_val=lambda_val,
            embed_url=embed_url,
            llm_api_base=llm_api_base,
            llm_api_key=llm_api_key,
        )

    @staticmethod
    def _download_from_hf(repo_id: str) -> str:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id)
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download from HF. "
                "Install with: pip install huggingface_hub"
            )

    # ── Embedding ───────────────────────────────────────────────────────────

    def embed(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Embed queries using Qwen3-0.6B.

        Returns: numpy array of shape (N, 1024)
        """
        if isinstance(queries, str):
            queries = [queries]

        if self.embed_url:
            return self._embed_remote(queries)
        return self._embed_local(queries)

    def _embed_remote(self, queries: List[str]) -> np.ndarray:
        """Embed via a running vLLM server (OpenAI-compatible embeddings API)."""
        import urllib.request
        import urllib.error

        url = self.embed_url.rstrip("/") + "/v1/embeddings"
        payload = json.dumps({
            "model": "Qwen/Qwen3-0.6B",
            "input": queries,
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to embedding server at {self.embed_url}. "
                f"Start it with: vllm serve Qwen/Qwen3-0.6B --runner pooling --port 8000\n"
                f"Error: {e}"
            ) from e

        embeddings = [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]
        return np.array(embeddings)

    def _embed_local(self, queries: List[str]) -> np.ndarray:
        """Embed by loading Qwen3-0.6B locally via vLLM."""
        if self._embedder is None:
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError(
                    "vLLM is required for local embedding. "
                    "Install with: pip install 'r2-router[embed]'\n"
                    "Or start a remote vLLM server and pass embed_url."
                )
            self._embedder = LLM(
                model="Qwen/Qwen3-0.6B",
                runner="pooling",
                trust_remote_code=True,
                dtype="half",
            )

        outputs = self._embedder.embed(queries)
        return np.array([o.outputs.embedding for o in outputs])

    # ── Routing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_input_tokens(text: str) -> int:
        """Rough estimate: ~1.3 tokens per word for English text."""
        return max(1, int(len(text.split()) * 1.3))

    def route_text(
        self,
        query: Union[str, List[str]],
        lambda_val: Optional[float] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        End-to-end text routing: embed with Qwen3-0.6B, then route.

        Args:
            query: Single query string or list of queries
            lambda_val: Override default lambda

        Returns:
            Routing decision dict (single) or list of dicts (batch)
        """
        embeddings = self.embed(query)
        if isinstance(query, str):
            return self.route(embeddings[0], lambda_val,
                              input_tokens=self._estimate_input_tokens(query))
        return [self.route(embeddings[i], lambda_val,
                           input_tokens=self._estimate_input_tokens(q))
                for i, q in enumerate(
                    query if isinstance(query, list) else [query])]

    def route(
        self,
        embedding: np.ndarray,
        lambda_val: Optional[float] = None,
        input_tokens: int = 100,
    ) -> Dict:
        """
        Route a single query to the optimal (model, token_budget) pair.

        Cost = (input_tokens × input_price + output_tokens × output_price) / 1M

        For each (model, budget):
          - limited: output_tokens = budget value, quality = Ridge.predict(embedding)
          - unlimited: output_tokens = predicted by Ridge, quality = Ridge.predict(embedding)
          - risk = (1 - λ) × quality - λ × cost

        Args:
            embedding: Query embedding, shape (1024,) or (1, 1024)
            lambda_val: Override default lambda
            input_tokens: Estimated input token count (used for cost calculation)

        Returns:
            Dict with: model, openrouter_id, budget, predicted_quality, predicted_cost, risk, all_options
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        lam = lambda_val if lambda_val is not None else self.lambda_val
        all_options = []

        for model_name, pred in self.predictors.items():
            cfg = self.model_configs[model_name]
            in_price = cfg["input_price_per_million"]
            out_price = cfg["output_price_per_million"]
            input_cost = input_tokens * in_price / 1e6

            # Predict quality at each limited budget
            limited_predictors = pred["limited"]
            for budget_key, ridge in limited_predictors.items():
                budget_val = int(budget_key.replace("_score", ""))
                quality = float(np.clip(ridge.predict(embedding)[0], 0, 1))
                output_cost = budget_val * out_price / 1e6
                cost = input_cost + output_cost
                risk = (1 - lam) * quality - lam * cost

                all_options.append({
                    "model": model_name,
                    "openrouter_id": cfg["openrouter_id"],
                    "budget": budget_val,
                    "predicted_quality": round(quality, 4),
                    "predicted_output_tokens": budget_val,
                    "predicted_cost_usd": round(cost, 8),
                    "risk": round(risk, 8),
                })

            # Predict quality and tokens for unlimited
            q_unlimited = float(np.clip(pred["unlimited_score"].predict(embedding)[0], 0, 1))
            t_unlimited = float(max(0, pred["unlimited_token"].predict(embedding)[0]))
            output_cost = t_unlimited * out_price / 1e6
            cost_unlimited = input_cost + output_cost
            risk_unlimited = (1 - lam) * q_unlimited - lam * cost_unlimited

            all_options.append({
                "model": model_name,
                "openrouter_id": cfg["openrouter_id"],
                "budget": "unlimited",
                "predicted_quality": round(q_unlimited, 4),
                "predicted_output_tokens": round(t_unlimited, 1),
                "predicted_cost_usd": round(cost_unlimited, 8),
                "risk": round(risk_unlimited, 8),
            })

        if not all_options:
            raise RuntimeError("No valid routing options")

        best = max(all_options, key=lambda x: x["risk"])
        best["all_options"] = sorted(all_options, key=lambda x: -x["risk"])
        return best

    # ── Generation ──────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        model: str,
        budget: Union[int, str],
        temperature: float = 0.7,
    ) -> Dict:
        """
        Call a routed LLM via OpenAI-compatible API (e.g., OpenRouter).

        Budget is enforced via a system prompt instruction, matching the
        R2-Bench evaluation protocol.

        Args:
            query: The user query
            model: Model name (must be in model_configs)
            budget: Token budget (int) or "unlimited"
            temperature: Sampling temperature
        """
        if not self.llm_api_base:
            raise RuntimeError(
                "llm_api_base is required for generation. "
                "Pass llm_api_base and llm_api_key to from_pretrained()."
            )

        import urllib.request
        import urllib.error

        cfg = self.model_configs.get(model, {})
        model_id = cfg.get("openrouter_id", model)

        # Build messages with budget-constrained system prompt
        messages = []
        if budget != "unlimited":
            system_prompt = (
                f"You MUST respond in {budget} tokens or fewer. "
                "Directly give the answer follow the format WITHOUT ANY explanation."
            )
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        url = self.llm_api_base.rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }).encode()

        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot connect to LLM API at {self.llm_api_base}. Error: {e}"
            ) from e

        choice = result["choices"][0]
        usage = result.get("usage", {})

        return {
            "response": choice["message"]["content"],
            "model": model,
            "openrouter_id": model_id,
            "budget": budget,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        }

    def route_and_generate(
        self,
        query: str,
        lambda_val: Optional[float] = None,
        temperature: float = 0.7,
    ) -> Dict:
        """
        End-to-end: embed → route → generate response.

        Args:
            query: User query text
            lambda_val: Override default lambda
            temperature: Sampling temperature for generation

        Returns:
            Dict with: query, response, model, budget, predicted_quality, risk, usage
        """
        decision = self.route_text(query, lambda_val)

        gen = self.generate(
            query=query,
            model=decision["model"],
            budget=decision["budget"],
            temperature=temperature,
        )

        return {
            "query": query,
            "response": gen["response"],
            "model": decision["model"],
            "openrouter_id": decision["openrouter_id"],
            "budget": decision["budget"],
            "predicted_quality": decision["predicted_quality"],
            "predicted_cost_usd": decision["predicted_cost_usd"],
            "risk": decision["risk"],
            "usage": gen["usage"],
        }
