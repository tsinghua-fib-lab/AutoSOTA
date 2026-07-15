import os
import time
import torch
import threading
from google.colab import drive


class GoogleColabConnector:
    # A lock used only to protect the flush_and_unmount + mount steps.
    # We do NOT hold this lock for the entire load, so that multiple
    # clients can keep checking for the file concurrently.
    _flush_lock = threading.Lock()

    def __init__(self, drive_path):
        self.drive_path = drive_path
        self._mount_drive()
        self._create_directory_if_missing()
        self.file_paths = []

    def _mount_drive(self):
        """Mount Google Drive if not already mounted."""
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")

    def _create_directory_if_missing(self):
        """Create the directory in Drive if it doesn't exist."""
        if not os.path.exists(self.drive_path):
            os.makedirs(self.drive_path)
            print(f"Created directory: {self.drive_path}")
        else:
            print(f"Directory already exists: {self.drive_path}")

    def upload(self, model, filename):
        """
        Save a PyTorch model to the Google Drive path.
        `model` can be a full model or a state_dict.
        """
        full_path = os.path.join(self.drive_path, filename)
        torch.save(model, full_path)
        self.file_paths.append(full_path)
        print(f"PyTorch model saved to: {full_path}")
        return {"model_drive_path": full_path, "drive_path": self.drive_path}

    def load_model(
        self,
        filename,
        model_class=None,
        load_state_dict=False,
        timeout=150,
        flush_interval=15,
    ):
        """
        Load a PyTorch model or state_dict from the drive path, possibly flushing+remounting
        to force Google Drive sync. Only one client at a time will do the flush.

        Args:
            filename (str): The name of the file in the drive directory.
            model_class (nn.Module, optional): The class of the model (required if loading state_dict).
            load_state_dict (bool): If True, loads state_dict into model_class.
            timeout (int): How many seconds to wait for the file to appear before giving up.
            flush_interval (int): The minimum gap (seconds) before we try flush+mount again
                                  if the file still isn't present.

        Returns:
            nn.Module: The loaded model.
        """
        full_path = os.path.join(self.drive_path, filename)
        start_time = time.time()
        last_flush_time = start_time
        found = False

        # Loop until the file is found, or until we exceed timeout
        while not os.path.exists(full_path):
            found = False
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"File not found after {timeout} seconds: {full_path}"
                )

            # If it's time to do a forced flush/mount check
            if (time.time() - last_flush_time) >= flush_interval:
                # Attempt to acquire the flush_lock without blocking:
                # - If we get it, do the flush and mount
                # - If not, skip the flush (someone else is already doing it)
                acquired = self._flush_lock.acquire(blocking=False)
                if acquired:
                    try:
                        drive.flush_and_unmount()
                        drive.mount("/content/drive")
                        if os.path.exists(full_path):
                            # don't release lock yet, we do this so that other client do not interrupt the model loading
                            found = True
                        print("Drive forcibly re-mounted to sync.")
                    except Exception as e:
                        print("Exception during flush_and_unmount/mount:", e)
                    finally:
                        if not found:
                            self._flush_lock.release()

                last_flush_time = time.time()

            # Sleep briefly before checking again
            time.sleep(2)

        # If we reach here, the file is now visible; load the model
        if load_state_dict:
            if model_class is None:
                raise ValueError(
                    "model_class must be provided when loading state_dict."
                )
            model = model_class()
            state_dict = torch.load(full_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"State dict loaded into {model_class.__name__} from: {full_path}")
        else:
            model = torch.load(full_path, map_location="cpu")
            print(f"Full model loaded from: {full_path}")

        # check if its acquired then release it.
        if found and self._flush_lock.locked():
            self._flush_lock.release()

        return model

    def cleanup(self):
        for file_path in self.file_paths:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Google Drive cleaned!")

    def __getstate__(self):
        """Serialize only the drive_path."""
        return {"drive_path": self.drive_path}

    def __setstate__(self, state):
        """Reinitialize the object after deserialization."""
        self.drive_path = state["drive_path"]
        self._mount_drive()
        self._create_directory_if_missing()
