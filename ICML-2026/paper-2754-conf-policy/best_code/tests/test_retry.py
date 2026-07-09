"""Tests for infrastructure.retry utilities."""

from unittest.mock import MagicMock, patch

import pytest

from cpc_llm.infrastructure.retry import cuda_retry, is_cuda_error


# ---------------------------------------------------------------------------
# is_cuda_error
# ---------------------------------------------------------------------------


class TestIsCudaError:
    def test_cuda_in_message(self) -> None:
        assert is_cuda_error(RuntimeError("CUDA out of memory"))

    def test_busy_in_message(self) -> None:
        assert is_cuda_error(RuntimeError("device is busy"))

    def test_unavailable_in_message(self) -> None:
        assert is_cuda_error(RuntimeError("GPU unavailable"))

    def test_accelerator_error_type(self) -> None:
        class AcceleratorError(Exception):
            pass

        assert is_cuda_error(AcceleratorError("something"))

    def test_unrelated_error(self) -> None:
        assert not is_cuda_error(ValueError("bad config"))

    def test_empty_message(self) -> None:
        assert not is_cuda_error(RuntimeError(""))


# ---------------------------------------------------------------------------
# cuda_retry
# ---------------------------------------------------------------------------


class TestCudaRetry:
    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_success_on_first_try(self, mock_sleep: MagicMock) -> None:
        result = cuda_retry(lambda: 42, stagger=False)
        assert result == 42

    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_retries_on_cuda_error_then_succeeds(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(
            side_effect=[RuntimeError("CUDA error"), RuntimeError("CUDA error"), 99]
        )

        result = cuda_retry(fn, stagger=False)

        assert result == 99
        assert fn.call_count == 3

    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_raises_non_cuda_error_immediately(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=ValueError("bad config"))

        with pytest.raises(ValueError, match="bad config"):
            cuda_retry(fn, stagger=False)

        assert fn.call_count == 1

    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=RuntimeError("CUDA error"))

        with pytest.raises(RuntimeError, match="CUDA error"):
            cuda_retry(fn, max_retries=3, stagger=False)

        assert fn.call_count == 3

    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(
            side_effect=[
                RuntimeError("CUDA error"),
                RuntimeError("CUDA error"),
                RuntimeError("CUDA error"),
            ]
        )

        with pytest.raises(RuntimeError):
            cuda_retry(fn, max_retries=3, base_delay=1.0, stagger=False)

        # Backoff sleeps: 1.0 (2^0), 2.0 (2^1) — no sleep after last failure
        sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0]

    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_on_retry_callback_called(self, mock_sleep: MagicMock) -> None:
        on_retry = MagicMock()
        fn = MagicMock(
            side_effect=[RuntimeError("CUDA error"), RuntimeError("CUDA error"), 1]
        )

        cuda_retry(fn, stagger=False, on_retry=on_retry)

        assert on_retry.call_count == 2

    @patch("cpc_llm.infrastructure.retry.random.uniform", return_value=0.3)
    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_stagger_delay(
        self, mock_sleep: MagicMock, mock_uniform: MagicMock
    ) -> None:
        cuda_retry(lambda: 42, stagger=True)

        # First sleep call is the stagger delay
        mock_sleep.assert_called_once_with(0.3)
        mock_uniform.assert_called_once_with(0.1, 0.5)

    @patch("cpc_llm.infrastructure.retry.time.sleep")
    def test_custom_logger_receives_messages(self, mock_sleep: MagicMock) -> None:
        log = MagicMock()
        fn = MagicMock(side_effect=[RuntimeError("CUDA error"), 1])

        cuda_retry(fn, stagger=False, logger=log, operation="test op")

        log.warning.assert_called_once()
        assert "test op" in log.warning.call_args.args[0]

    def test_max_retries_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            cuda_retry(lambda: 42, max_retries=0)
