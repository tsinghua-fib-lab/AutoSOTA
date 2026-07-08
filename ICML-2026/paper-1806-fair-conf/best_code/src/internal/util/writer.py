import os
import datetime
import json
import sys
import torch
import numpy as np


class Tee:
    """This class allows for redirecting of stdout and stderr"""

    def __init__(self, primary_file, secondary_file):
        self.primary_file = primary_file
        self.secondary_file = secondary_file

        self.encoding = self.primary_file.encoding

    def isatty(self):
        return self.primary_file.isatty()

    def fileno(self):
        return self.primary_file.fileno()

    def write(self, data):
        # We get problems with ipdb if we don't do this:
        if isinstance(data, bytes):
            data = data.decode()

        self.primary_file.write(data)
        self.secondary_file.write(data)

    def flush(self):
        self.primary_file.flush()
        self.secondary_file.flush()


class Writer:
    _STDOUT = sys.stdout
    _STDERR = sys.stderr

    def __init__(self, logdir, make_subdir, tag_group):
        if make_subdir:
            os.makedirs(logdir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            logdir = os.path.join(logdir, timestamp)
            os.makedirs(logdir, exist_ok=True)

        self.logdir = logdir
        self._tag_group = tag_group

        sys.stdout = Tee(
            primary_file=self._STDOUT,
            secondary_file=open(os.path.join(logdir, "stdout"), "a"),
        )

        sys.stderr = Tee(
            primary_file=self._STDERR,
            secondary_file=open(os.path.join(logdir, "stderr"), "a", encoding="utf-8"),
        )

    def write_json(self, tag, data):
        text = json.dumps(data, indent=4)
        json_path = os.path.join(self.logdir, f"{tag}.json")
        with open(json_path, "w") as f:
            f.write(text)

    def write_textfile(self, tag, text):
        path = os.path.join(self.logdir, f"{tag}.txt")
        with open(path, "w") as f:
            f.write(text)

    def write_numpy(self, tag, arr):
        path = os.path.join(self.logdir, f"{tag}.npy")
        np.save(path, arr)
        print(f"Saved array to {path}")

    def write_pandas(self, tag, df):
        path = os.path.join(self.logdir, f"{tag}.csv")
        df.to_csv(path)
        print(f"Saved DataFrame to {path}")

    def write_plots(self, tag, fig):
        path = os.path.join(self.logdir, f"{tag}.png")
        fig.savefig(path)
        print(f"Saved plot to {path}")

    def write_checkpoint(self, tag, data):
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        checkpoint_path = self._checkpoint_path(tag)

        tmp_checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path), f"{os.path.basename(checkpoint_path)}.tmp"
        )

        torch.save(data, tmp_checkpoint_path)
        # replace is atomic, so we guarantee our checkpoints are always good
        os.replace(tmp_checkpoint_path, checkpoint_path)

    def load_checkpoint(self, tag, device):
        return torch.load(self._checkpoint_path(tag), map_location=device)

    def _checkpoint_path(self, tag):
        return os.path.join(self._checkpoints_dir, f"{tag}.pt")

    @property
    def _checkpoints_dir(self):
        return os.path.join(self.logdir, "checkpoints")

    def _tag(self, tag):
        return f"{self._tag_group}/{tag}"


def get_writer(dataset_name, **kwargs):
    cfg = kwargs["cfg"]
    writer = Writer(
        logdir=cfg["logdir_root"],
        make_subdir=True,
        tag_group=dataset_name,
    )

    writer.write_json(tag="config", data=kwargs["cfg"])

    return writer
