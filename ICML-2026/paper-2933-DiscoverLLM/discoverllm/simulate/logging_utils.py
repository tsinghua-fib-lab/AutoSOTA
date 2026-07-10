import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

_error_log_file: Optional[object] = None
_error_log_path: Optional[str] = None


def init_error_log(output_dir: str) -> str:
    global _error_log_file, _error_log_path
    error_log_path = Path(output_dir) / "error.log"
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    _error_log_file = open(error_log_path, 'a', encoding='utf-8')
    _error_log_path = str(error_log_path)
    _error_log_file.write(f"\n{'='*80}\n")
    _error_log_file.write(f"Error Log - Experiment Started: {datetime.now().isoformat()}\n")
    _error_log_file.write(f"{'='*80}\n")
    _error_log_file.flush()
    return _error_log_path


def close_error_log():
    global _error_log_file
    if _error_log_file:
        _error_log_file.write(f"\n{'='*80}\n")
        _error_log_file.write(f"Error Log - Experiment Ended: {datetime.now().isoformat()}\n")
        _error_log_file.write(f"{'='*80}\n\n")
        _error_log_file.close()
        _error_log_file = None


def log_error(message: str, artifact_id: Optional[str] = None, assistant_id: Optional[str] = None, exception: Optional[Exception] = None, include_traceback: bool = True):
    global _error_log_file
    if not _error_log_file:
        return
    timestamp = datetime.now().isoformat()
    log_entry = f"\n[{timestamp}] ERROR"
    if artifact_id or assistant_id:
        context = []
        if artifact_id:
            context.append(f"artifact={artifact_id}")
        if assistant_id:
            context.append(f"assistant={assistant_id}")
        log_entry += f" ({', '.join(context)})"
    log_entry += f": {message}\n"
    if exception:
        log_entry += f"Exception Type: {type(exception).__name__}\n"
        log_entry += f"Exception Message: {str(exception)}\n"
        if include_traceback:
            log_entry += "Traceback:\n"
            log_entry += traceback.format_exc()
    log_entry += "-" * 80 + "\n"
    _error_log_file.write(log_entry)
    _error_log_file.flush()


def log_warning(message: str, artifact_id: Optional[str] = None, assistant_id: Optional[str] = None):
    global _error_log_file
    if not _error_log_file:
        return
    timestamp = datetime.now().isoformat()
    log_entry = f"\n[{timestamp}] WARNING"
    if artifact_id or assistant_id:
        context = []
        if artifact_id:
            context.append(f"artifact={artifact_id}")
        if assistant_id:
            context.append(f"assistant={assistant_id}")
        log_entry += f" ({', '.join(context)})"
    log_entry += f": {message}\n"
    _error_log_file.write(log_entry)
    _error_log_file.flush()

