import os
import shutil
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

def make_file(bin_path: str, filesize: int):
	if os.path.exists(bin_path):
		os.remove(bin_path)
	path = Path(bin_path)
	usage = shutil.disk_usage(path.parent)
	old_size = path.stat().st_size if path.exists() else 0
	free_bytes = usage.free + old_size
	if free_bytes < filesize:
		raise RuntimeError(
			f"\x1b[31mNot enough disk space: need {filesize // 1_000_000:,}MB for {bin_path}, \x1b[0m"
			f"\x1b[31mbut only {free_bytes // 1_000_000:,}MB available\x1b[0m"
		)