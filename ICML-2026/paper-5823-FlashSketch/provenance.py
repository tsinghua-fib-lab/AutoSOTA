from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path

import gitbud.gitbud as gitbud

import globals as g


def _configure_gitbud_globals():
    """Ensure gitbud uses repo-specific global constants."""
    gitbud.TIME_FORMAT = g.TIME_FORMAT
    gitbud.FILE_STORAGE_ROOT = g.FILE_STORAGE_ROOT


_configure_gitbud_globals()


def get_repo():
    """Return the current git repo or None."""
    return gitbud.get_repo()


def get_repo_root():
    """Return the repository root path as a Path or None."""
    repo = get_repo()
    if repo is None:
        return None
    return Path(repo.working_tree_dir)


def get_git_state():
    """Return commit hash and dirty flag for the current repo."""
    repo = get_repo()
    if repo is None:
        return {"commit": None, "dirty": None}
    return {
        "commit": gitbud.get_commit_hash(repo),
        "dirty": gitbud.is_dirty(repo),
    }


def ensure_clean_repo():
    """Raise a RuntimeError if the repo is dirty."""
    repo = get_repo()
    if repo is None:
        return
    if gitbud.is_dirty(repo):
        raise RuntimeError("Repository is dirty. Commit or stash before long runs.")


def get_tree_hashes(paths):
    """Return a dict mapping relative paths to git tree hashes."""
    repo = get_repo()
    if repo is None:
        return {path: None for path in paths}

    tree_hashes = {}
    for path in paths:
        tree_hashes[path] = gitbud.get_tree_hash(repo, path)
    return tree_hashes


def make_run_id():
    """Return a run id combining time, commit hash, and dirty flag."""
    repo = get_repo()
    if repo is None:
        return "no-repo"
    return gitbud.get_exp_info()


def run_dir(run_id):
    """Return the absolute run directory path for a run id."""
    return g.RUN_DIR(run_id)
