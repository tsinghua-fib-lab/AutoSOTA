from gitbud.gitbud import inject_repo_into_sys_path

REPO_ROOT_STR = inject_repo_into_sys_path()

from dataclasses import dataclass
from pathlib import Path

if not REPO_ROOT_STR:
    raise RuntimeError("Repository root not found.")

REPO_ROOT = Path(REPO_ROOT_STR)


@dataclass(frozen=True)
class AbstractMdConfig:
    """Configuration for exporting the abstract to Markdown."""

    abstract_tex_path: Path
    macros_tex_path: Path
    output_md_path: Path
    normalize_whitespace: bool = True


DEFAULT_CONFIG = AbstractMdConfig(
    abstract_tex_path=REPO_ROOT / "paper/sections/00_abstract.tex",
    macros_tex_path=REPO_ROOT / "paper/macros.tex",
    output_md_path=REPO_ROOT / "paper/abstract.md",
    normalize_whitespace=True,
)
