"""
Have not used in TabDDPM project.
"""

import datetime
import os
import shutil
import typing as ty
from pathlib import Path

PROJ = Path('tab-ddpm/').absolute().resolve()
EXP = PROJ / 'exp'
DATA = PROJ / 'data'


def get_path(path: ty.Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = PROJ / path
    return path.resolve()


def get_relative_path(path: ty.Union[str, Path]) -> Path:
    return get_path(path).relative_to(PROJ)


def duplicate_path(
    src: ty.Union[str, Path], alternative_project_dir: ty.Union[str, Path]
) -> None:
    src = get_path(src)
    alternative_project_dir = get_path(alternative_project_dir)
    dst = alternative_project_dir / src.relative_to(PROJ)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(
            dst.name + '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        )
    (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)