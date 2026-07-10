import datetime
import difflib
import json
import os
import re
import subprocess
import time
from typing import Any

import numpy as np
import yaml
from termcolor import colored


CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
}


def _detect_git_repo_root() -> str | None:
    """Best-effort lookup of the git repo root directory."""
    candidates = [os.getcwd()]
    try:
        this_file = os.path.abspath(__file__)
        # .../repo/src/utils/logger.py -> .../repo
        candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(this_file))))
    except Exception:
        pass

    for cwd in candidates:
        try:
            root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                text=True,
                errors="replace",
            ).strip()
            if root:
                return root
        except Exception:
            continue
    return None


def generate_git_diff_html(output_dir: str, *, context_lines: int = 5) -> str | None:
    """
    Write an HTML side-by-side diff of the current working tree vs HEAD.

    Returns the written path, or None if no git repo is available.
    """
    repo_root = _detect_git_repo_root()
    if repo_root is None:
        return None

    def _git(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            errors="replace",
        )

    try:
        head_sha = _git(["rev-parse", "HEAD"]).strip()
        head_short = _git(["rev-parse", "--short", "HEAD"]).strip()
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    except Exception:
        head_sha, head_short, branch = "", "", ""

    changed_files: list[str] = []
    try:
        changed_files = _git(["diff", "--name-only", "HEAD"]).splitlines()
    except Exception:
        changed_files = []

    # Include untracked files too (useful for experiments).
    try:
        untracked = _git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    except Exception:
        untracked = []

    # Stable order, no duplicates.
    all_files = sorted({p for p in (changed_files + untracked) if p})

    html_diff = difflib.HtmlDiff(wrapcolumn=120)
    sections: list[str] = []

    if not all_files:
        sections.append("<p><b>Working tree clean.</b></p>")
    else:
        toc_items = []
        for idx, filename in enumerate(all_files):
            anchor = f"file-{idx}"
            toc_items.append(f'<li><a href="#{anchor}">{filename}</a></li>')
        sections.append(
            '<h2>Files</h2>\n<ul class="toc">\n' + "\n".join(toc_items) + "\n</ul>"
        )

        for idx, filename in enumerate(all_files):
            anchor = f"file-{idx}"
            try:
                original_content = _git(["show", f"HEAD:{filename}"])
            except Exception:
                original_content = ""

            path_on_disk = os.path.join(repo_root, filename)
            if os.path.exists(path_on_disk):
                try:
                    with open(
                        path_on_disk, "r", encoding="utf-8", errors="replace"
                    ) as f:
                        modified_content = f.read()
                except Exception:
                    modified_content = ""
            else:
                modified_content = ""

            table = html_diff.make_table(
                original_content.splitlines(),
                modified_content.splitlines(),
                fromdesc=f"HEAD {head_short}",
                todesc="Working tree",
                context=True,
                numlines=int(context_lines),
            )
            sections.append(f'<h2 id="{anchor}">{filename}</h2>\n{table}')

    full_html = f"""\
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Git Diff (HEAD vs Working Tree)</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Arial, sans-serif; margin: 24px; color: #222; }}
      code, pre, table.diff {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }}
      h1 {{ margin: 0 0 8px 0; }}
      h2 {{ margin-top: 28px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
      .meta {{ color: #555; margin-bottom: 16px; }}
      ul.toc {{ columns: 2; -webkit-columns: 2; -moz-columns: 2; }}
      ul.toc li {{ break-inside: avoid; }}
      table.diff {{ border: 1px solid #e5e7eb; border-collapse: collapse; width: 100%; font-size: 12px; }}
      .diff_header {{ background-color: #f3f4f6; }}
      td.diff_header {{ text-align: right; padding: 2px 6px; color: #374151; }}
      .diff_next {{ background-color: #e5e7eb; }}
      td {{ padding: 2px 6px; vertical-align: top; }}
      .diff_add {{ background-color: #dcfce7; }}
      .diff_chg {{ background-color: #fef9c3; }}
      .diff_sub {{ background-color: #fee2e2; }}
    </style>
  </head>
  <body>
    <h1>Git Diff</h1>
    <div class="meta">
      <div><b>Repo:</b> <code>{repo_root}</code></div>
      <div><b>Branch:</b> <code>{branch}</code></div>
      <div><b>HEAD:</b> <code>{head_sha}</code></div>
      <div><b>Compared to:</b> <code>working tree</code></div>
    </div>
    {''.join(sections)}
  </body>
</html>
"""

    out_path = os.path.join(output_dir, "git_diff.html")
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(full_html)
    return out_path


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        # Expand user home and create absolute path to avoid creating literal '~' dirs
        expanded_path = os.path.abspath(os.path.expanduser(dir_path))
        os.makedirs(expanded_path, exist_ok=True)
    except OSError:
        pass
    return expanded_path if "expanded_path" in locals() else dir_path


def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs),
            _limstr(v),
        )

    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """
    Return a wandb-safe group name for logging.
    Optionally returns group name as list.
    """
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class Logger:
    """Primary logging object. Logs either locally or using wandb."""

    def __init__(
        self,
        work_dir,
        wandb_dir,
        seed,
        project,
        entity,
        tags,
        group,
        job_type,
        config,
        disable_wandb=False,
        save_agent=False,
        wandb_silent=False,
    ):
        self._root_dir = make_dir(work_dir)
        self._wandb_dir = make_dir(wandb_dir)
        self._save_agent = bool(save_agent)
        self._group = str(group)
        self._job_type = str(job_type)
        self._seed = int(seed)
        self._eval = []
        self.project = project
        self.entity = entity
        self.run_name = f"{job_type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_s{seed}"

        # Run directory: logs/<env>/<algo>/<run_name>/
        env_name = re.sub(r"[^0-9a-zA-Z._-]+", "-", self._group).strip("-")
        algo_name = re.sub(r"[^0-9a-zA-Z._-]+", "-", self._job_type.lower()).strip("-")
        self._run_dir = make_dir(
            os.path.join(self._root_dir, env_name, algo_name, self.run_name)
        )

        # Always write config and scalar logs locally.
        self._config_path = os.path.join(self._run_dir, "config.yaml")
        try:
            with open(self._config_path, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)
        except Exception as e:
            print(colored(f"Failed to write config.yaml: {e}", "red"))

        try:
            generate_git_diff_html(self._run_dir)
        except Exception as e:
            print(colored(f"Failed to write git_diff.html: {e}", "red"))

        self._metrics_path = os.path.join(self._run_dir, "metrics.jsonl")
        self._metrics_f = open(self._metrics_path, "a", buffering=1)

        self._wandb = None
        self._video = None

        if disable_wandb or (not self.project) or self.project == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            return
        os.environ["WANDB_SILENT"] = "true" if wandb_silent else "false"

        try:
            import wandb
        except Exception as e:
            print(colored(f"Failed to import wandb; disabling W&B logging: {e}", "red"))
            return

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            group=self._group,
            job_type=self._job_type,
            tags=tags,
            dir=self._wandb_dir,
            config=config,
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb

    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return os.path.join(self._run_dir, "checkpoints")

    @property
    def run_dir(self):
        return self._run_dir

    def _to_jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        try:
            arr = np.asarray(value)
            if arr.shape == ():
                return arr.item()
            return arr.tolist()
        except Exception:
            return str(value)

    def _write_metrics(self, record: dict[str, Any]) -> None:
        try:
            self._metrics_f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(colored(f"Failed to write metrics.jsonl: {e}", "red"))

    def save_agent(self, agent=None, step=None, identifier=None):
        if not (self._save_agent and agent):
            return None
        if not hasattr(agent, "save"):
            raise AttributeError(
                f"Agent of type {type(agent).__name__} does not implement save(path)"
            )

        if step is None:
            raise ValueError("save_agent requires 'step' to be provided")

        ckpt_dir = make_dir(self.model_dir)
        algo = self._job_type.lower()
        base = str(identifier) if identifier is not None else algo
        fp = os.path.join(ckpt_dir, f"{base}_step{int(step)}.pkl")

        agent.save(fp)
        return fp

    def finish(self, agent=None, step=None):
        if hasattr(self, "_metrics_f") and self._metrics_f:
            try:
                self._metrics_f.close()
            except Exception:
                pass
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise ValueError(f"invalid log format type: {ty}")

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def log(self, d, category="train"):
        """Generic log method for scalar metrics (requires 'step')."""
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        # Extract step key
        xkey = "step"
        if xkey not in d:
            raise KeyError(f"Missing required key '{xkey}' in log data")
        step = d[xkey]

        if "episode_reward" in d and "episode_length" in d:
            self._write_metrics(
                {
                    "step": int(step),
                    "episode_length": self._to_jsonable(d["episode_length"]),
                    "episode_reward": self._to_jsonable(d["episode_reward"]),
                }
            )

        wandb_data = {}
        for k, v in d.items():
            if k == xkey:
                continue
            key_name = f"{category}/{k}"
            wandb_data[key_name] = v

        if self._wandb:
            self._wandb.log(wandb_data, step=step)
