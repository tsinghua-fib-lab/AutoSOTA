import os
from typing import Annotated, Any, Dict, List, Optional, Union

import addict
import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from .io import get_ext, load_file


class BaseAPPLConfigs(BaseModel):
    """Base class for APPL configs."""

    @model_validator(mode="before")
    @classmethod
    def warn_extra_fields(cls, values):
        """Warn if extra fields are provided."""
        extra_fields = set(values.keys()) - cls.model_fields.keys()
        if extra_fields:
            logger.warning(
                f"Extra configs provided for {cls.__name__}: {extra_fields}. "
                "Ignoring them."
            )
        return values


# ===== BEGIN CONFIGS =====
class DisplayRichSettings(BaseAPPLConfigs):
    """Settings for displaying the logs."""

    lexer: str = Field(default="markdown", description="The lexer of the rich output")
    theme: str = Field(default="monokai", description="The theme of the rich output")
    line_numbers: bool = Field(
        default=False, description="Whether to display the line numbers"
    )
    word_wrap: bool = Field(default=True, description="Whether to wrap the words")
    refresh_per_second: int = Field(
        default=4, description="The refresh rate of the rich output"
    )


class LoggingDisplaySettings(BaseAPPLConfigs):
    """Settings for displaying the logs."""

    configs: bool = Field(default=False, description="Display the configurations")
    configs_update: bool = Field(
        default=False, description="Display the updates of the configurations"
    )
    docstring_warning: bool = Field(
        default=True, description="Display warning message when docstrings are excluded"
    )
    llm_raw_call_args: bool = Field(
        default=False, description="Display raw args for llm calls"
    )
    llm_raw_response: bool = Field(
        default=False, description="Display raw response of llm calls"
    )
    llm_raw_usage: bool = Field(
        default=False, description="Display raw usage of llm calls"
    )
    llm_call_args: bool = Field(default=False, description="Display args for llm calls")
    llm_response: bool = Field(
        default=True, description="Display response of llm calls"
    )
    llm_usage: bool = Field(default=False, description="Display usage of llm calls")
    llm_cache: bool = Field(default=False, description="Display cache info")
    llm_cost: bool = Field(default=True, description="Display cost of calls")
    tool_calls: bool = Field(default=True, description="Display tool calls")
    tool_results: bool = Field(
        default=True, description="Display results of tool calls"
    )
    streaming_mode: str = Field(
        default="print",
        description=(
            "Mode to display streaming output. choices are 'live', 'print', 'none'"
        ),
    )
    rich: DisplayRichSettings = Field(default_factory=DisplayRichSettings)


class LoggingFileSettings(BaseAPPLConfigs):
    """Settings for logging to a file."""

    path_format: str = Field(default="./logs/{basename}_{time:YYYY_MM_DD__HH_mm_ss}")
    log_level: Optional[str] = Field(
        default=None, description="Log level for file logging"
    )


class LoggingSettings(BaseAPPLConfigs):
    """Settings for logging."""

    enable_stderr: bool = Field(default=True, description="Enable logging to stderr")
    format: str = Field(
        default=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        description="Format of the log message",
        # change HH:mm:ss.SSS to HH:mm:ss for the default loguru format
    )
    log_level: str = Field(default="INFO", description="Level of log messages")
    max_length: int = Field(default=800, description="Maximum length of log message")
    suffix_length: int = Field(
        default=200, description="Length of suffix when truncated"
    )
    enable_file: bool = Field(default=True, description="Enable logging to file")
    log_file: LoggingFileSettings = Field(default_factory=LoggingFileSettings)
    display: LoggingDisplaySettings = Field(default_factory=LoggingDisplaySettings)


class CachingSettings(BaseAPPLConfigs):
    """Settings for caching."""

    enabled: bool = Field(default=True, description="Enable caching")
    folder: str = Field(
        default="~/.appl/caches", description="Folder to store cache files"
    )
    max_size: int = Field(
        default=100000, description="Maximum number of entries in cache"
    )
    time_to_live: int = Field(default=43200, description="Time-to-live in minutes")
    cleanup_interval: int = Field(
        default=1440, description="Cleanup interval in minutes"
    )
    allow_temp_greater_than_0: bool = Field(
        default=False, description="Allow caching with temperature > 0"
    )


class TracingSettings(BaseAPPLConfigs):
    """Settings for tracing."""

    enabled: bool = Field(default=False, description="Enable tracing")
    path_format: str = Field(
        default="./dumps/traces/{basename}_{time:YYYY_MM_DD__HH_mm_ss}"
    )
    patch_threading: bool = Field(
        default=True, description="Whether to patch threading.Thread"
    )
    strict_match: bool = Field(
        default=True,
        description="Include the gen_id into the key for cache operations",
    )
    trace_to_resume: Optional[str] = Field(
        default=None, description="Trace file to resume"
    )
    display_trace_info: bool = Field(default=True, description="Display trace info")


class ConcurrencySettings(BaseAPPLConfigs):
    """Settings for concurrency."""

    llm_max_workers: int = Field(default=10, description="Maximum LLM workers")
    thread_max_workers: int = Field(default=20, description="Maximum thread workers")
    process_max_workers: int = Field(default=10, description="Maximum process workers")


class MessageColorSettings(BaseAPPLConfigs):
    """Settings for message colors."""

    system: str = Field(default="red", description="Color for system messages")
    user: str = Field(default="green", description="Color for user messages")
    assistant: str = Field(default="cyan", description="Color for assistant messages")
    tool: str = Field(default="magenta", description="Color for tool messages")


class MessagesSettings(BaseAPPLConfigs):
    """Settings for messages."""

    colors: MessageColorSettings = Field(default_factory=MessageColorSettings)


class MiscSettings(BaseAPPLConfigs):
    """Settings for miscellaneous."""

    suppress_litellm_debug_info: bool = Field(default=True)


class SettingsConfigs(BaseAPPLConfigs):
    """Settings for the settings."""

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    caching: CachingSettings = Field(default_factory=CachingSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
    messages: MessagesSettings = Field(default_factory=MessagesSettings)
    misc: MiscSettings = Field(default_factory=MiscSettings)


class PromptsConfigs(BaseAPPLConfigs):
    """Settings for the prompts."""

    continue_generation: str = Field(
        default=(
            "The previous message was cut off due to length limit, please continue to "
            "complete the message by starting with the last line (marked with "
            "{last_marker}). Make sure the indentation is correct when continuing. "
            "Begin your continuation with {last_marker}."
        ),
        description="prompt for continuing generation by last line",
    )
    continue_generation_alt: str = Field(
        default=(
            "The previous message was cut off due to length limit, please continue to "
            "complete the message by starting with the last part (marked with "
            "{last_marker}). Make sure the newline and indentation are correct when "
            "continuing. Begin your continuation with {last_marker}."
        ),
        description="prompt for continuing generation when newline is not found",
    )


class DefaultServersConfigs(BaseAPPLConfigs):
    """Settings for the default servers."""

    default: Optional[str] = Field(default=None, description="Default server")
    small: Optional[str] = Field(
        default=None, description="Small-sized server, fallback to default"
    )
    large: Optional[str] = Field(
        default=None, description="Large-sized server, fallback to default"
    )


class APPLConfigs(BaseAPPLConfigs):
    """Settings for the APPL."""

    # static metadata in the config file
    metadata: Dict[str, Any] = Field(default_factory=lambda: {}, description="metadata")
    settings: SettingsConfigs = Field(default_factory=SettingsConfigs)
    prompts: PromptsConfigs = Field(default_factory=PromptsConfigs)
    default_servers: DefaultServersConfigs = Field(
        default_factory=DefaultServersConfigs
    )
    # TODO: backward-compatible for now, change to Dict[str, Dict[str, Any]]
    servers: Optional[Dict[str, Union[Dict[str, Any], str, None]]] = Field(
        default=None,
        description=(
            "set the default server and create aliases for models with default "
            "arguments"
        ),
    )


# ===== END CONFIGS =====


class ConfigsDict(addict.Dict):
    """A Nested Dictionary class."""

    def to_yaml(self) -> str:
        """Convert the Configs object to a YAML string."""
        return yaml.dump(self.to_dict())

    def __missing__(self, key: str) -> None:
        raise KeyError(key)


def load_config(file: str, *args: Any, **kwargs: Any) -> ConfigsDict:
    """Load a config file and return the data as a dictionary."""
    ext = get_ext(file)
    if ext not in [".json", ".yaml", ".yml", ".toml"]:
        raise ValueError(f"Unsupported config file type {ext}")
    content = load_file(file, *args, **kwargs)
    return ConfigsDict(content)


DEFAULT_CONFIGS = APPLConfigs()
"""The static default configs loaded from the default config file."""
