import json
import os
import pickle
import argparse
import tempfile
from omegaconf import OmegaConf
from typing import Dict, Any, Optional, Tuple


def _mkstemp_path(suffix: str) -> str:
    """Atomically create a fresh, owner-only (0o600) temp file and return
    its path. The random name defeats co-tenant symlink-race attacks that
    target predictable shared-tmp names."""
    fd, path = tempfile.mkstemp(prefix="appfl-tes-", suffix=suffix)
    os.close(fd)
    return path


def _safe_open_for_write(path: str, binary: bool):
    """Open `path` for writing, refusing to follow a symlink at the final
    component. Truncates if the file already exists and is a regular file
    owned by the same caller. Returns a file object."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb" if binary else "w")


class TESClientCommunicator:
    """
    GA4GH TES client-side communicator for APPFL.

    This class handles the client-side execution within TES containers,
    following the same patterns as other APPFL client communicators.
    """

    def __init__(self, client_agent_config):
        self.client_agent_config = client_agent_config

    def load_model_from_path(self, model_path: str) -> Any:
        """Load model directly from file using torch.load."""
        if not model_path:
            return None
        try:
            import torch

            return torch.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def load_metadata_from_path(self, metadata_path: str) -> Dict:
        """Load metadata from JSON file path."""
        if not metadata_path:
            return {}
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {metadata_path}: {e}")

    def save_model_to_path(self, model: Any, output_path: str):
        """Save model to file path."""
        try:
            with _safe_open_for_write(output_path, binary=True) as f:
                pickle.dump(model, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {output_path}: {e}")

    def save_logs_to_path(self, logs: Dict, logs_path: str):
        """Save training logs to JSON file path."""
        try:
            with _safe_open_for_write(logs_path, binary=False) as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save logs to {logs_path}: {e}")

    def upload_results_to_s3(self, model: Any, logs: Dict, s3_upload_info: Dict):
        """Upload results directly to S3 using presigned URLs."""
        try:
            import requests
            import tempfile
            import torch
            import os

            # Extract presigned URLs
            model_upload_url = s3_upload_info["model_upload_url"]
            logs_upload_url = s3_upload_info["logs_upload_url"]

            # Upload model using presigned URL
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pkl"
            ) as temp_model_file:
                if isinstance(model, bytes):
                    temp_model_file.write(model)
                else:
                    torch.save(model, temp_model_file.name)
                temp_model_path = temp_model_file.name

            with open(temp_model_path, "rb") as f:
                response = requests.put(model_upload_url, data=f)
                response.raise_for_status()

            print("CLIENT: Uploaded model to S3 via presigned URL")
            os.remove(temp_model_path)

            # Upload logs using presigned URL
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as temp_logs_file:
                json.dump(logs, temp_logs_file, indent=2)
                temp_logs_path = temp_logs_file.name

            with open(temp_logs_path, "rb") as f:
                response = requests.put(logs_upload_url, data=f)
                response.raise_for_status()

            print("CLIENT: Uploaded logs to S3 via presigned URL")
            os.remove(temp_logs_path)

        except Exception as e:
            raise RuntimeError(f"Failed to upload results to S3: {e}")

    def execute_task(
        self,
        task_name: str,
        model_path: str = None,
        metadata_path: str = None,
        output_path: Optional[str] = None,
        logs_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Execute a federated learning task within the TES container.

        This method follows the same task execution patterns as Ray and Globus Compute
        client communicators in APPFL.

        Args:
            task_name: Name of the task to execute ('train', 'evaluate', etc.)
            model_path: Path to input model file
            metadata_path: Path to metadata JSON file
            output_path: Path for output model
            logs_path: Path for training logs

        Returns:
            Tuple of (output_path, logs_path) for the results
        """
        if output_path is None:
            output_path = _mkstemp_path(".pkl")
        if logs_path is None:
            logs_path = _mkstemp_path(".json")

        try:
            import time
            from appfl.comm.utils.executor import (
                get_sample_size_executor,
                data_readiness_report_executor,
                train_executor,
            )

            if task_name == "get_sample_size":
                model, metadata = get_sample_size_executor(
                    client_agent_config=self.client_agent_config
                )
            elif task_name == "data_readiness_report":
                model, metadata = data_readiness_report_executor(
                    client_agent_config=self.client_agent_config
                )
            elif task_name == "train":
                model, metadata = train_executor(
                    client_agent_config=self.client_agent_config,
                    model=self.load_model_from_path(model_path),
                    meta_data=self.load_metadata_from_path(metadata_path),
                )
            else:
                raise NotImplementedError(
                    f"Task {task_name} is not implemented in TES client."
                )
            # Check if we need to upload to S3 instead of saving locally
            loaded_metadata = (
                self.load_metadata_from_path(metadata_path) if metadata_path else {}
            )
            s3_upload_info = loaded_metadata.get("s3_upload_info")

            if s3_upload_info:
                # Upload results directly to S3
                self.upload_results_to_s3(model, metadata, s3_upload_info)
            else:
                # Save locally (for local storage or when no S3 info provided)
                self.save_model_to_path(model, output_path)
                self.save_logs_to_path(metadata, logs_path)

            return output_path, logs_path

        except Exception as e:
            import time

            # Save error information
            error_logs = {
                "task_name": task_name,
                "client_id": getattr(self.client_agent_config, "client_id", "unknown"),
                "status": "error",
                "timestamp": str(time.time()),
                "error": str(e),
                "metadata": metadata if "metadata" in locals() else {},
            }
            self.save_logs_to_path(error_logs, logs_path)
            raise


def tes_client_entry_point():
    """Main entry point for TES client execution."""
    parser = argparse.ArgumentParser(description="APPFL TES Client Runner")
    parser.add_argument(
        "--task-name", required=True, help="Name of the task to execute"
    )
    parser.add_argument("--client-id", required=True, help="Client ID")
    parser.add_argument(
        "--config-path", required=True, help="Path to client configuration file"
    )
    parser.add_argument("--model-path", help="Path to input model file")
    parser.add_argument("--metadata-path", help="Path to metadata JSON file")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path for output model. If omitted, a fresh tempfile is created.",
    )
    parser.add_argument(
        "--logs-path",
        default=None,
        help="Path for training logs. If omitted, a fresh tempfile is created.",
    )
    args = parser.parse_args()

    try:
        client_agent_config = OmegaConf.load(args.config_path)
        client_agent_config.client_id = args.client_id
        tes_client = TESClientCommunicator(client_agent_config=client_agent_config)
        tes_client.execute_task(
            task_name=args.task_name,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            output_path=args.output_path,
            logs_path=args.logs_path,
        )

        print(f"Task '{args.task_name}' completed successfully")
        return 0

    except Exception as e:
        print(f"Error executing TES client task: {e}")
        import traceback

        traceback.print_exc()
        return 1
