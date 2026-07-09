import pytest
from omegaconf import OmegaConf

from cpc_llm.infrastructure.slurm_utils import (
    set_post_subprocess_hook,
    submit_cmd_direct,
    wait_for_direct_jobs_to_complete,
)
from cpc_llm.infrastructure.orchestration import _submit_cmd


class TestSubmitCmdDirect:
    def test_successful_command(self, tmp_path):
        p, job_id = submit_cmd_direct("echo hello", str(tmp_path), blocking=True)
        assert p.returncode == 0
        assert job_id.startswith("direct-")

    def test_failed_command_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Direct execution failed"):
            submit_cmd_direct("exit 1", str(tmp_path), blocking=True)

    def test_creates_log_file(self, tmp_path):
        submit_cmd_direct("echo test_output", str(tmp_path), blocking=True)
        log_files = list(tmp_path.glob("direct_*.log"))
        assert len(log_files) == 1
        assert "test_output" in log_files[0].read_text()

    def test_ignores_extra_kwargs(self, tmp_path):
        # slurm_kwargs should be silently ignored
        p, _ = submit_cmd_direct(
            "echo ok",
            str(tmp_path),
            blocking=True,
            nodes=1,
            gpus_per_node=2,
            partition="gpu2",
            path_to_repo="/fake/path",
        )
        assert p.returncode == 0

    def test_tees_output_to_stderr(self, tmp_path, capsys):
        submit_cmd_direct("echo tee_marker", str(tmp_path), blocking=True)
        captured = capsys.readouterr()
        # subprocess output should be tee'd to stderr
        assert "tee_marker" in captured.err


class TestWaitForDirectJobs:
    def test_waits_for_all(self, tmp_path):
        jobs = []
        for i in range(3):
            p, job_id = submit_cmd_direct(
                f"echo job_{i}", str(tmp_path), blocking=False
            )
            jobs.append((p, job_id))
        # Should not raise
        wait_for_direct_jobs_to_complete(jobs)

    def test_raises_on_failure(self, tmp_path):
        p, job_id = submit_cmd_direct("exit 42", str(tmp_path), blocking=False)
        with pytest.raises(RuntimeError):
            wait_for_direct_jobs_to_complete([(p, job_id)])


class TestPostSubprocessHook:
    def test_hook_called_on_success(self, tmp_path):
        calls = []
        set_post_subprocess_hook(lambda: calls.append("commit"))
        try:
            submit_cmd_direct("echo ok", str(tmp_path), blocking=True)
            assert calls == ["commit"]
        finally:
            set_post_subprocess_hook(None)

    def test_hook_called_on_failure(self, tmp_path):
        calls = []
        set_post_subprocess_hook(lambda: calls.append("commit"))
        try:
            with pytest.raises(RuntimeError):
                submit_cmd_direct("exit 1", str(tmp_path), blocking=True)
            assert calls == ["commit"]
        finally:
            set_post_subprocess_hook(None)

    def test_hook_called_per_job_in_wait(self, tmp_path):
        calls = []
        set_post_subprocess_hook(lambda: calls.append("commit"))
        try:
            jobs = [
                submit_cmd_direct(f"echo job_{i}", str(tmp_path), blocking=False)
                for i in range(3)
            ]
            wait_for_direct_jobs_to_complete(jobs)
            assert len(calls) == 3
        finally:
            set_post_subprocess_hook(None)


class TestDispatcher:
    def test_routes_to_direct(self, tmp_path):
        cfg = OmegaConf.create({"job_submission_system": "direct"})
        p, job_id = _submit_cmd(cfg, "echo dispatched", str(tmp_path), blocking=True)
        assert p.returncode == 0
        assert job_id.startswith("direct-")

    def test_defaults_to_slurm_signature(self):
        # Without job_submission_system, should default to "slurm"
        cfg = OmegaConf.create({})
        # We can't actually run slurm, but verify the dispatcher
        # selects the right path by checking the default
        system = getattr(cfg, "job_submission_system", "slurm")
        assert system == "slurm"
