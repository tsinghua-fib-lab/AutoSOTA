"""Tests for core.model_loading helpers."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from cpc_llm.core.model_loading import (
    cleanup_model_clients,
    init_model_client_with_retry,
    preload_model_clients,
)


# ---------------------------------------------------------------------------
# init_model_client_with_retry
# ---------------------------------------------------------------------------


class TestInitModelClientWithRetry:
    @patch("cpc_llm.core.model_loading.ModelClient")
    @patch("cpc_llm.core.model_loading.torch.cuda.is_available", return_value=False)
    def test_success_on_first_try(self, _cuda, mock_mc_cls):
        client = MagicMock()
        mock_mc_cls.return_value = client

        result = init_model_client_with_retry("model/path", 128)

        assert result is client
        mock_mc_cls.assert_called_once_with(
            model_name_or_path="model/path",
            logger=pytest.importorskip("logging").getLogger(
                "cpc_llm.core.model_loading"
            ),
            max_generate_length=128,
            temperature=1.0,
            device="cuda",
        )

    @patch("cpc_llm.core.model_loading.ModelClient")
    @patch("cpc_llm.core.model_loading.torch.cuda.is_available", return_value=False)
    def test_retries_on_cuda_error(self, _cuda, mock_mc_cls):
        """Should retry on CUDA errors and succeed on a later attempt."""
        client = MagicMock()
        mock_mc_cls.side_effect = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("CUDA initialization failed"),
            client,
        ]

        result = init_model_client_with_retry("model/path", 128)

        assert result is client
        assert mock_mc_cls.call_count == 3

    @patch("cpc_llm.core.model_loading.ModelClient")
    @patch("cpc_llm.core.model_loading.torch.cuda.is_available", return_value=False)
    def test_falls_back_to_cpu(self, _cuda, mock_mc_cls):
        """After 5 CUDA failures, should fall back to CPU."""
        cpu_client = MagicMock()
        cuda_error = RuntimeError("CUDA error: device unavailable")
        mock_mc_cls.side_effect = [
            cuda_error,
            cuda_error,
            cuda_error,
            cuda_error,
            cuda_error,
            cpu_client,
        ]

        result = init_model_client_with_retry("model/path", 128)

        assert result is cpu_client
        # 5 CUDA attempts + 1 CPU fallback
        assert mock_mc_cls.call_count == 6
        last_call = mock_mc_cls.call_args
        assert last_call.kwargs["device"] == "cpu"

    @patch("cpc_llm.core.model_loading.ModelClient")
    @patch("cpc_llm.core.model_loading.torch.cuda.is_available", return_value=False)
    def test_raises_non_cuda_errors(self, _cuda, mock_mc_cls):
        """Non-CUDA errors should be raised immediately, not retried."""
        mock_mc_cls.side_effect = ValueError("bad config")

        with pytest.raises(ValueError, match="bad config"):
            init_model_client_with_retry("model/path", 128)

        assert mock_mc_cls.call_count == 1

    @patch("cpc_llm.core.model_loading.ModelClient")
    @patch("cpc_llm.core.model_loading.torch.cuda.is_available", return_value=False)
    def test_passes_temperature(self, _cuda, mock_mc_cls):
        mock_mc_cls.return_value = MagicMock()

        init_model_client_with_retry("model/path", 128, temperature=0.7)

        assert mock_mc_cls.call_args.kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# preload_model_clients
# ---------------------------------------------------------------------------


class TestPreloadModelClients:
    @pytest.fixture
    def cfg(self):
        return OmegaConf.create(
            {
                "compute_likelihooods_all_models": {
                    "args": {
                        "generation_config": {"max_new_tokens": 64},
                    },
                },
                "iterative_generation": {
                    "args": {"generation_config": {"max_new_tokens": 128}},
                },
                "temperature": 1.0,
                "conformal_policy_control": {
                    "constrain_against": "init",
                },
            }
        )

    @patch("cpc_llm.core.model_loading.init_model_client_with_retry")
    def test_unconstrained_uses_last_model_for_gen(self, mock_init, cfg):
        client_a = MagicMock(name="client_a")
        client_b = MagicMock(name="client_b")
        mock_init.side_effect = [client_a, client_b]

        gen_client, lik_clients = preload_model_clients(
            cfg, ["model_a", "model_b"], "unconstrained"
        )

        # Gen client should be the last model (reused from lik clients)
        assert gen_client is client_b
        assert lik_clients == {"model_a": client_a, "model_b": client_b}

    @patch("cpc_llm.core.model_loading.init_model_client_with_retry")
    def test_safe_init_uses_first_model_for_gen(self, mock_init, cfg):
        client_a = MagicMock(name="client_a")
        client_b = MagicMock(name="client_b")
        mock_init.side_effect = [client_a, client_b]

        gen_client, lik_clients = preload_model_clients(
            cfg, ["model_a", "model_b"], "safe"
        )

        # Gen client should be the first model (reused from lik clients)
        assert gen_client is client_a

    @patch("cpc_llm.core.model_loading.init_model_client_with_retry")
    def test_safe_sequential_no_gen_client(self, mock_init, cfg):
        cfg.conformal_policy_control.constrain_against = "sequential"
        client_a = MagicMock()
        mock_init.side_effect = [client_a]

        gen_client, lik_clients = preload_model_clients(cfg, ["model_a"], "safe")

        # For safe/sequential, gen client is None (recursion handles it)
        assert gen_client is None

    @patch("cpc_llm.core.model_loading.init_model_client_with_retry")
    def test_mixture_returns_dict_gen_client(self, mock_init, cfg):
        client_a = MagicMock(name="client_a")
        client_b = MagicMock(name="client_b")
        mock_init.side_effect = [client_a, client_b]

        gen_client, lik_clients = preload_model_clients(
            cfg, ["model_a", "model_b"], "mixture"
        )

        assert isinstance(gen_client, dict)
        assert gen_client["safe"] is client_a
        assert gen_client["unconstrained"] is client_b

    @patch("cpc_llm.core.model_loading.init_model_client_with_retry")
    def test_deduplicates_model_paths(self, mock_init, cfg):
        """Same model path should only be loaded once."""
        client = MagicMock()
        mock_init.return_value = client

        gen_client, lik_clients = preload_model_clients(
            cfg, ["same_model", "same_model"], "unconstrained"
        )

        # Should only call init once despite two entries
        assert mock_init.call_count == 1
        assert len(lik_clients) == 1


# ---------------------------------------------------------------------------
# cleanup_model_clients
# ---------------------------------------------------------------------------


class TestCleanupModelClients:
    def test_clears_dict_and_empties_cache(self):
        clients = {"a": MagicMock(), "b": MagicMock()}
        gen_client = MagicMock()

        with patch.object(torch.cuda, "empty_cache") as mock_empty:
            cleanup_model_clients(clients, gen_client)

        assert len(clients) == 0
        mock_empty.assert_called_once()

    def test_handles_none_gen_client(self):
        clients = {"a": MagicMock()}

        with patch.object(torch.cuda, "empty_cache"):
            cleanup_model_clients(clients, None)

        assert len(clients) == 0

    def test_handles_empty_dict(self):
        with patch.object(torch.cuda, "empty_cache"):
            cleanup_model_clients({}, None)
