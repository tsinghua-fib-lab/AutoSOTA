from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import json
import shutil
import logging
import sys
import os

# Add utils to path for token counter import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.token_counter import TokenCounter
except ImportError:
    # Fallback if utils not available
    class TokenCounter:
        def count_message_tokens(self, message): return 0
        def count_messages_tokens(self, messages): return 0


@dataclass
class TraceTrack:
    """
    One task -> one JSON trace file.

    - Internally keeps a list of "trace messages" that look like OpenAI messages,
      but each item has an extra field: "time_stamp".
    - This trace is for logging only. Do NOT send these items directly to the LLM.
    """
    root_dir: Path
    item_id: str
    agent_name: str = "run"

    run_dir: Path = field(init=False)
    trace_file: Path = field(init=False)
    _trace: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _token_counter: TokenCounter = field(default_factory=TokenCounter, init=False)
    _session_token_stats: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        safe_id = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in self.item_id)
        self.run_dir = Path(self.root_dir) / safe_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.trace_file = self.run_dir / f"{self.agent_name}_trace.json"
        if self.trace_file.exists():
            try:
                self._trace = json.loads(self.trace_file.read_text(encoding="utf-8")) or []
                if not isinstance(self._trace, list):
                    self._trace = []
            except Exception:
                self._trace = []
        else:
            self._trace = []
            self._flush()

        # Initialize session token stats
        self._session_token_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "turn_count": 0,
            "turns": []
        }

    def _ts(self) -> str:
        # Keep the original timestamp function, but store as string for JSON.
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _flush(self) -> None:
        self.trace_file.write_text(
            json.dumps(self._trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- core: record "messages-like" items --------------------------------

    def add_message(self, message: Dict[str, Any], token_stats: Optional[Dict[str, Any]] = None) -> None:
        """
        Append a message-like dict to trace with an extra "time_stamp" field.

        Example accepted inputs:
          {"role":"user","content":"hi"}
          {"role":"assistant","content":None,"tool_calls":[...]}
          {"role":"tool","tool_call_id":"...","name":"func","content":"..."}
        """
        item = dict(message)
        item["time_stamp"] = self._ts()

        # Add token count for this message
        item["message_tokens"] = self._token_counter.count_message_tokens(message)

        # Add turn-level token stats if provided (only for assistant messages typically)
        if token_stats:
            item["token_stats"] = token_stats

        self._trace.append(item)
        self._flush()

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        for m in messages:
            self.add_message(m)

    # ---- compatibility: keep step(), but now it becomes a trace message -----

    def step(self, msg: str, role: str = "trace") -> None:
        """
        Backward-compatible logger.

        Instead of writing a TXT line, it appends a trace entry:
          {"role":"trace","content":"...", "time_stamp":"..."}
        """
        self.add_message({"role": role, "content": msg.rstrip()})

    # ---- persistence helpers (unchanged) -----------------------------------

    def save_text(self, name: str, text: str):
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def save_json(self, name: str, obj):
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def copy_in(self, src_path: Union[str, Path], dst_name: Optional[str] = None):
        src = Path(src_path)
        dst = self.run_dir / (dst_name or src.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def attach_logging(self, level=logging.DEBUG, filename: str = "run.log"):
        log_path = self.run_dir / filename
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(handler)
        return log_path

    # ---- convenience getters -----------------------------------------------

    def get_trace(self) -> List[Dict[str, Any]]:
        """Return in-memory trace list."""
        return list(self._trace)

    def clear_trace(self) -> None:
        """Clear current trace (and overwrite file)."""
        self._trace = []
        self._flush()

    # ---- token tracking methods --------------------------------------------

    def start_turn(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Start a new turn and calculate prompt tokens.
        Returns turn info with prompt token count.
        """
        prompt_tokens = self._token_counter.count_messages_tokens(messages)

        turn_info = {
            "turn": len(self._session_token_stats["turns"]) + 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens
        }

        self._session_token_stats["turns"].append(turn_info)
        return turn_info

    def end_turn(self, completion_tokens: int) -> Dict[str, int]:
        """
        End current turn and update token counts.
        Returns complete turn statistics.
        """
        if not self._session_token_stats["turns"]:
            logging.error("end_turn called without start_turn")
            return {}

        current_turn = self._session_token_stats["turns"][-1]

        # Update current turn
        current_turn["completion_tokens"] = completion_tokens
        current_turn["total_tokens"] = current_turn["prompt_tokens"] + completion_tokens

        # Update session totals
        self._session_token_stats["total_prompt_tokens"] += current_turn["prompt_tokens"]
        self._session_token_stats["total_completion_tokens"] += completion_tokens
        self._session_token_stats["total_tokens"] += current_turn["total_tokens"]
        self._session_token_stats["turn_count"] = len(self._session_token_stats["turns"])

        return current_turn

    def get_session_token_stats(self) -> Dict[str, Any]:
        """Get complete session token statistics."""
        return self._session_token_stats.copy()

    def get_current_turn_prompt_tokens(self) -> int:
        """Get prompt tokens for current turn (if in progress)."""
        if self._session_token_stats["turns"]:
            return self._session_token_stats["turns"][-1]["prompt_tokens"]
        return 0

    def should_summarize_context(self, threshold: int = 8000) -> bool:
        """
        Check if context should be summarized based on token count.
        Returns True if current turn prompt tokens exceed threshold.
        """
        return self.get_current_turn_prompt_tokens() > threshold

    def save_token_stats(self) -> Path:
        """Save session token statistics to a separate file."""
        stats_file = self.run_dir / f"{self.agent_name}_token_stats.json"
        stats_file.write_text(
            json.dumps(self._session_token_stats, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return stats_file
