"""Logging helpers and live status rendering utilities used across training scripts."""

import base64
import io
import itertools
import logging
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.theme import Theme
from rich.text import Text

import numpy as np
import torch
from IPython import display
from tools.utils import IN_JUPYTER

precision = 8
width = precision+7
formatter = {"float_kind": lambda x: f"{x:<{width}.{precision}g}"}

monokai_theme = Theme({
    "repr.number": "bold magenta",
    "repr.string": "green",
    "repr.bool_true": "cyan",
    "repr.bool_false": "red",
    "repr.none": "dim",
    "logging.level.info": "green",
    "logging.level.warning": "yellow",
    "logging.level.error": "red",
    "logging.level.debug": "blue",
})


log_buffer = io.StringIO()
log_capture_handler = logging.StreamHandler(log_buffer)
log_capture_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
))
log_capture_handler.setLevel(logging.WARNING)  # Log everything to file

# log_console = Console(file=log_buffer, force_jupyter=IN_JUPYTER,  theme=monokai_theme, record=True)
display_console = Console(force_jupyter=IN_JUPYTER,  theme=monokai_theme)
terminal_console = Console()

fh = logging.handlers.RotatingFileHandler(
    "training.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB cap
    backupCount=5               # keep 5 old logs
)

fh.setLevel(logging.INFO)
run_log_handler = logging.FileHandler("run.log", mode="w")
run_log_handler.setLevel(logging.DEBUG)
run_log_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s'))

handlers = [run_log_handler, fh]
if IN_JUPYTER:
    handlers.append(log_capture_handler)
else:
    rh = RichHandler(rich_tracebacks=True)
    rh.setLevel(logging.INFO)
    handlers.append(rh)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    handlers=handlers
)

logger = logging.getLogger("training")


def set_log_level(level: str | int):
    """
    Set the logging level for both console and file handlers.
    
    Args:
        level: Logging level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               or as integer (logging.DEBUG, logging.INFO, etc.)
    
    Example:
        >>> from tools.feedback import set_log_level
        >>> set_log_level('DEBUG')  # Show debug messages
        >>> set_log_level('WARNING')  # Only show warnings and above
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Set level on training logger
    logger.setLevel(level)
    
    # Set level on root logger
    logging.getLogger().setLevel(level)
    
    # Update all root handlers (where our handlers are attached)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)


def make_status_panel(data: dict) -> Panel:
    color_cycle = itertools.cycle([
        "bold cyan", "bold red", "bold yellow", "bold magenta", 
        "bold green", "bold white", "dim"
    ])
    title = data.pop("desc", "")
    text = Text()
    text.append(f"saving to {title}: \n")
    for key, val in data.items():
        color = next(color_cycle)

        if isinstance(val, torch.Tensor):
            if val.dtype == torch.bool:
                val = f"{val.detach().numpy()}"
            elif val.numel() == 1:
                val = f"{val.item():<{width}.{precision}g}"
            elif val.numel() <= 10:
                val = f"[{', '.join([f'{v:<{width}.{precision}g}' for v in val.flatten().tolist()])}]"
            else:
                val = f"min={val.min().item():<{width}.{precision}g} mean={val.mean().item():<{width}.{precision}g} std={val.std().item():<{width}.{precision}g} max={val.max().item():<{width}.{precision}g}"
        elif isinstance(val, float):
            val = f"{val:<{width}.{precision}g}"
        elif isinstance(val, list):
            val = np.array2string(np.array(val), formatter=formatter, separator=", ")
        elif isinstance(val, np.ndarray):
            val = np.array2string(val, formatter=formatter, separator=", ")
        else:
            val = str(val)

        text.append(f"{key:<15}   ", style=color)
        text.append(f"{val}\n", style=color)

    return Panel(text, title=f"Training Status", border_style="bold white")

def render_panel_and_logs(panel: Panel) -> str:
    """Render the Rich panel and recent logs into a single HTML snippet."""
    panel_console = Console(record=True)
    panel_console.print(panel)
    panel_html = panel_console.export_html(inline_styles=True)

    # Render logs
    log_text = log_buffer.getvalue()
    log_panel = Panel(
        log_text or "[dim]No logs yet[/]",
        title="Logs",
        border_style="white",
    )
    t = Console(record=True)
    t.print(log_panel)
    logs_html = t.export_html(inline_styles=True)

    # Combine both HTML chunks
    full_html = f"""
    <html>
    <head><meta charset="utf-8"></head>
    <body style="background-color:#111; color:#ccc; font-family:monospace; padding:1em;">
        <div style="max-height: 250px; overflow-y: auto;">
            {logs_html}
        </div>
        <hr style="border: 1px solid #333;">
        <div style="max-height: 300px; overflow-y: auto;">
            {panel_html}
        </div>
    </body>
    </html>
    """

    # Encode the HTML as base64 to embed safely
    encoded = base64.b64encode(full_html.encode('utf-8')).decode('utf-8')

    # Wrap in iframe
    iframe = f"""
    <iframe
        src="data:text/html;base64,{encoded}"
        style="width:100%; height:600px; border:none; border-radius:8px; box-shadow:0 0 10px rgba(0,0,0,0.3);"
    ></iframe>
    """
    # Combine HTML
    return iframe

class LiveOrJupyter:
    """Context manager exposing a live status panel in terminal or Jupyter."""

    def __init__(self):
        self._live = None
        self._display_handle = None

    def __enter__(self):
        if IN_JUPYTER:
            self._display_handle = display.DisplayHandle()
            self._display_handle.display(display.HTML(""))
            return self
        else:
            panel = make_status_panel({})
            self._live = Live(panel, refresh_per_second=4, screen=False)
            self._live.__enter__()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not IN_JUPYTER and self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, panel: Panel):
        """Refresh the display with a new panel snapshot."""
        if IN_JUPYTER:
            html = render_panel_and_logs(panel)
            self._display_handle.update(display.HTML(html))
            display.clear_output(wait=True)
        else:
            self._live.update(panel, refresh=True)
