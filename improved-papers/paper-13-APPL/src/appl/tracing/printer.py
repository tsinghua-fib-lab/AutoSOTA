import copy
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import yaml
from deprecated import deprecated
from litellm import ModelResponse
from loguru import logger

from ..compositor import Tagged as OriginalTagged
from ..core.config import ConfigsDict
from ..core.globals import global_vars
from ..core.io import load_file
from ..core.printer import PromptRecords
from ..core.trace import (
    CompletionRequestEvent,
    CompletionResponseEvent,
    FunctionCallEvent,
    FunctionReturnEvent,
    GenerationInitEvent,
    GenerationResponseEvent,
    TraceEngineBase,
    TraceEventBase,
    TraceNode,
    TracePrinterBase,
)
from ..func import partial, ppl, records
from ..utils import get_meta_file
from .engine import TraceEngine

folder = os.path.dirname(__file__)

Tagged = partial(OriginalTagged, indent_inside=4)


def timestamp_to_iso(time_stamp: float) -> str:
    """Convert the timestamp to the ISO format."""
    return datetime.fromtimestamp(time_stamp, timezone.utc).isoformat()


@deprecated(
    reason="This printer is deprecated and will be removed in a future version."
)
class TraceHTMLPrinter(TracePrinterBase):
    """The printer used to print the trace in the format of HTML."""

    def __init__(self):
        """Initialize the printer."""
        super().__init__()
        self._generation_style = "text-success-emphasis bg-Success-subtle list-group-item d-flex justify-content-between align-items-center"
        self._time_style = "position-absolute top-0 start-100 translate-middle badge rounded-pill bg-info"
        self._cost_style = "position-absolute bottom-0 start-100 translate-middle badge rounded-pill bg-warning"
        self._longest_shown_output = 70

        self._head = load_file(os.path.join(folder, "header.html"))
        self._color_map = {
            "user": "text-bg-info",
            "assistant": "text-bg-warning",
            "system": "text-bg-success",
        }

    @ppl
    def print(
        self, trace: TraceEngineBase, trace_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Print the trace in the format of HTML."""
        with Tagged("html"):
            self._head
            with Tagged("body"):
                for node in trace.trace_nodes.values():
                    if node.parent is None:
                        self._print_node(node, trace.min_timestamp)
        if trace_metadata:
            with Tagged("table", attrs={"class": "table small"}):
                if start_time := trace_metadata.get("start_time", None):
                    self._make_line("Start Time", start_time)
                self._make_line(
                    "Full Configs", f"<pre>{yaml.dump(trace_metadata)}</pre>"
                )
        return str(records())

    @ppl
    def _print_messages(self, messages: List[Union[str, Dict]]) -> PromptRecords:
        def display(message: Union[str, Dict]) -> str:
            return f'<div style="white-space: pre-wrap;">{message}</div>'

        with Tagged("ul", attrs={"class": "list-group small"}):
            for message in messages:
                with Tagged("li", attrs={"class": "list-group-item"}):
                    if (
                        isinstance(message, dict)
                        and "role" in message
                        and "content" in message
                    ):
                        color = self._color_map.get(message["role"], "text-bg-info")
                        f"<span class='badge {color}'>{message['role']}</span>"
                        display(message["content"])
                    else:
                        display(message)
        return records()

    @ppl
    def _print_genargs(self, args: Dict, output: Optional[str] = None) -> PromptRecords:
        with Tagged("table", attrs={"class": "table small"}):
            for k, v in args.items():
                with Tagged("tr"):
                    with Tagged("th"):
                        f"{k}"
                    with Tagged("td"):
                        if k == "messages":
                            self._print_messages(v)
                        elif k in ["response_format"]:
                            f"<pre>{json.dumps(v, indent=2)}</pre>"
                        elif k in ["response_model"]:
                            f"<pre>{v}</pre>"
                        elif k == "stop":
                            repr(v)
                        else:
                            f"{v}"
            if output is not None:
                self._make_line("output", output)
        return records()

    @ppl
    def _print_gen(self, node: TraceNode, min_timestamp: float = 0.0) -> PromptRecords:
        name = node.name
        if node.args is not None:
            completion = node.children[0] if node.children else None
            with Tagged("ul", attrs={"class": "list-group m-1"}):
                li_attrs = {"class": self._generation_style}
                li_attrs.update(self._toggle_attrs(name))
                with Tagged("li", attrs=li_attrs):
                    f"<div><b>{name}:</b> {node.ret}</div>"
                    if completion and (runtime := completion.runtime) > 0:
                        f"<span class='{self._time_style}'>{runtime:.2e} s</span>"
                    if completion and (cost := completion.info.get("cost")):
                        f"<span class='{self._cost_style}'>$ {cost:.2e}</span>"
                with Tagged(
                    "li", attrs={"class": "list-group-item collapse", "id": name}
                ):
                    self._print_genargs(node.args, node.ret)
        else:
            li_attrs = {
                "class": "list-group-item text-warning-emphasis bg-warning-subtle"
            }
            with Tagged("ul", attrs={"class": "list-group m-1"}):
                with Tagged("li", attrs=li_attrs):
                    "Unfinished Generation"
        return records()

    @ppl
    def _print_func(self, node: TraceNode, min_timestamp: float = 0.0) -> PromptRecords:
        name = node.name
        with Tagged("ul", attrs={"class": "list-group m-2"}):
            li_attrs = {"class": "text-center bg-light list-group-item"}
            li_attrs.update(self._toggle_attrs(name, True))
            with Tagged("li", attrs=li_attrs):
                f"<b>{name}</b>"
            with Tagged("li", attrs={"class": "list-group-item show", "id": name}):
                # display details for the function
                # ? display time, args and kwargs
                # with Tagged("table", attrs={"class": "table small"}):
                #     start = node.start_time - min_timestamp
                #     end = node.end_time - min_timestamp
                #     runtime = end - start
                #     self._make_line(
                #         "Time", f"{runtime:.2e} s (from {start:.2e} to {end:.2e})"
                #     )
                #     # if node.args:
                #     #     func_args = node.args["args"]
                #     #     self._make_line("args", func_args)
                #     #     func_kwargs = node.args["kwargs"]
                #     #     for k, v in func_kwargs.items():
                #     #         self._make_line(k, v)
                for child in node.children:
                    self._print_node(child, min_timestamp)
        return records()

    def _print_node(self, node: TraceNode, min_timestamp: float = 0.0) -> Any:
        if node.type == "func":
            return self._print_func(node, min_timestamp)
        else:
            return self._print_gen(node, min_timestamp)

    def _toggle_attrs(self, name: str, expanded: bool = False) -> Dict:
        return {
            "data-bs-toggle": "collapse",
            "href": f"#{name}",
            "role": "button",
            "aria-controls": name,
            "aria-expanded": "true" if expanded else "false",
        }

    def _make_line(self, k: str, v: Any) -> str:
        return f"<tr><th>{k}</th><td>{v}</td></tr>"


class TraceLunaryPrinter(TracePrinterBase):
    """The printer used to log the trace to lunary."""

    def print(
        self, trace: TraceEngineBase, trace_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the trace to lunary."""
        import lunary

        project_id = os.environ.get("LUNARY_PUBLIC_KEY", None)
        if project_id is None:
            raise ValueError("LUNARY_PUBLIC_KEY is not set")
        url = os.environ.get("LUNARY_API_URL", "http://localhost:3333")
        logger.info(f"project_id: {project_id}, api url: {url}")
        lunary.config(app_id=project_id, api_url=url)

        suffix = f"_{uuid.uuid4().hex}"
        logger.info(f"suffix: {suffix}")

        user_id = None
        if trace_metadata:
            user_id = trace_metadata.get("user_id", None)

        def get_parent_run_id(node: TraceNode) -> Optional[str]:
            if node.parent is None:
                return None
            return node.parent.name + suffix

        """Log the trace to lunary."""
        for node in trace.trace_nodes.values():
            if node.type == "func":
                logger.info(
                    f"sending func event {node.name} to lunary with parent {get_parent_run_id(node)}"
                )
                lunary.track_event(
                    "chain",
                    "start",
                    run_id=node.name + suffix,
                    name=node.name,
                    parent_run_id=get_parent_run_id(node),
                    input=node.args,
                    timestamp=timestamp_to_iso(node.start_time),
                    user_id=user_id,
                )
                lunary.track_event(
                    "chain",
                    "end",
                    run_id=node.name + suffix,
                    output=node.ret,
                    timestamp=timestamp_to_iso(node.end_time),
                )

            elif node.type in ["gen", "raw_llm"]:
                merged_metadata: Dict[str, Any] = node.metadata or {}
                extra_info = copy.deepcopy(node.args or {})
                model_name = extra_info.pop("model", node.name)
                messages = extra_info.pop("messages", "")
                merged_metadata.update(extra_info)

                if node.type == "gen":
                    logger.info(
                        f"sending llm event {node.name} to lunary with parent {get_parent_run_id(node)}"
                    )

                    # skip the raw generation, support for legacy traces
                    # if node.name.endswith("_raw"):
                    #     continue
                    merged_metadata["gen_ID"] = node.name
                    lunary.track_event(
                        "llm",
                        "start",
                        run_id=node.name + suffix,
                        name=model_name,
                        parent_run_id=get_parent_run_id(node),
                        metadata=merged_metadata,
                        input=messages,
                        timestamp=timestamp_to_iso(node.start_time),
                        user_id=user_id,
                    )
                    lunary.track_event(
                        "llm",
                        "end",
                        run_id=node.name + suffix,
                        output={"role": "assistant", "content": node.ret},
                        timestamp=timestamp_to_iso(node.end_time),
                    )
                elif node.type == "raw_llm":
                    logger.info(
                        f"sending raw llm event {node.name} to lunary with parent {get_parent_run_id(node)}"
                    )
                    lunary.track_event(
                        "llm",
                        "start",
                        run_id=node.name + suffix,
                        name=model_name,
                        parent_run_id=get_parent_run_id(node),
                        metadata=merged_metadata,
                        input=messages,
                        timestamp=timestamp_to_iso(node.start_time),
                        user_id=user_id,
                    )
                    response: ModelResponse = node.ret  # complete response
                    message = response.choices[0].message  # type: ignore
                    output = {
                        "role": "assistant",
                        "content": message.content or "",  # type: ignore
                    }
                    if message.tool_calls:
                        output["tool_calls"] = [
                            {
                                "id": tool.id,
                                "type": "function",
                                "function": {
                                    "name": tool.function.name,
                                    "arguments": tool.function.arguments,
                                },
                            }
                            for tool in message.tool_calls
                            # TODO: support tool calls
                        ]
                    lunary.track_event(
                        "llm",
                        "end",
                        run_id=node.name + suffix,
                        output=output,
                        timestamp=timestamp_to_iso(node.end_time),
                    )


class TraceLangfusePrinter(TracePrinterBase):
    """The printer used to print the trace to langfuse."""

    def print(
        self, trace: TraceEngineBase, trace_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Print the trace to langfuse."""
        from langfuse import Langfuse
        from langfuse.client import StatefulTraceClient

        project_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", None)
        if project_public_key is None:
            raise ValueError("LANGFUSE_PUBLIC_KEY is not set")
        url = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        logger.info(f"project_public_key: {project_public_key}, api url: {url}")

        metadata = trace_metadata or {}
        user_id = metadata.get("user_id", None)
        if user_id is None:
            git_info: Dict[str, Any] = metadata.get("git_info", {})
            user_id = git_info.get("git_user_email", "unknown")

        session_id = metadata.get("start_time", str(uuid.uuid4()))
        if "exec_file_basename" in metadata:
            base_name = metadata["exec_file_basename"]
            # Use basename + start_time as the session_id
            session_id = f"[{base_name}] {session_id}"
        else:
            base_name = metadata.get("name", "main")

        root = Langfuse().trace(
            name=base_name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

        def visit_tree(node: TraceNode, trace_node: StatefulTraceClient) -> None:
            """Visit the trace node tree and send the trace event to langfuse."""
            start_time = datetime.fromtimestamp(node.start_time, timezone.utc)
            if node.end_time == 0.0:
                logger.warning(f"trace event {node.name} does not finish")
                end_time = start_time
            else:
                end_time = datetime.fromtimestamp(node.end_time, timezone.utc)

            logger.info(f"sending trace event {node.name} with type {node.type}")
            if node.type == "func":
                metadata = node.metadata or {}
                # metadata = metadata.copy()
                # if "source_code" in metadata:
                #     metadata["source_code"] = (
                #         "\n```python\n" + metadata["source_code"] + "\n```\n"
                #     )
                client = trace_node.span(
                    name=node.name,
                    start_time=start_time,
                    end_time=end_time,
                    input=node.args,
                    output=node.ret,
                    metadata=metadata,
                )
            elif node.type in ["gen", "raw_llm"]:
                inputs = (node.args or {}).copy()
                outputs = node.ret

                model_name = inputs.pop("model", None)
                metadata = inputs.pop("metadata", {})
                metadata = (node.metadata or {}) | metadata

                def extra_supported_model_parameters(inputs: Dict) -> Dict:
                    parameters = {}
                    for k, v in inputs.items():
                        if k not in [
                            "messages",
                            "tools",
                            "response_format",
                            "tool_choice",
                        ]:
                            parameters[k] = v

                    if tool_choice := inputs.get("tool_choice", None):
                        if isinstance(tool_choice, str):
                            name = tool_choice
                        elif isinstance(tool_choice, dict):
                            name = tool_choice.get("function", {}).get("name", None)
                        else:
                            logger.warning(f"unknown tool_choice: {tool_choice}")

                        if name:
                            parameters["tool_choice"] = name

                    for k in parameters.keys():
                        inputs.pop(k, None)

                    return parameters

                model_parameters = extra_supported_model_parameters(inputs)

                usage = None
                if outputs is not None:
                    if node.type == "gen":
                        model_name = None
                    elif not isinstance(outputs, str):  # for raw completion requests
                        outputs: ModelResponse = outputs  # type: ignore
                        usage = outputs.usage
                        message = outputs.choices[0].message  # type: ignore
                        outputs = message.content or ""
                        # Add reasoning content if exists
                        if provider_specific_fields := message.get(
                            "provider_specific_fields", None
                        ):
                            if reasoning_content := provider_specific_fields.get(
                                "reasoning_content", None
                            ):
                                outputs = (
                                    f"```reasoning\n{reasoning_content}\n```\n"
                                    + outputs
                                )
                        if message.tool_calls:
                            # replace the content with the tool calls
                            outputs = {
                                "content": message.content,
                                "tool_calls": [
                                    f"ToolCall(id={tool.id}, name={tool.function.name}, args={tool.function.arguments})"
                                    for tool in message.tool_calls
                                ],
                            }
                else:
                    outputs = "Not Finished."

                client = trace_node.generation(
                    name=node.name,
                    start_time=start_time,
                    end_time=end_time,
                    input=inputs,
                    output=outputs,
                    model=model_name,
                    model_parameters=model_parameters,
                    usage=usage,
                    metadata=metadata,
                )
            else:
                raise ValueError(f"Unknown node type: {node.type}")
            for child in node.children:
                visit_tree(child, client)

        for node in trace.trace_nodes.values():
            if node.parent is None:
                visit_tree(node, root)


class TraceYAMLPrinter(TracePrinterBase):
    """The printer used to print the trace in the format of YAML."""

    def print(
        self, trace: TraceEngineBase, trace_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Print the trace in the format of YAML."""
        # TODO: implement the YAML printer
        pass


class TraceProfilePrinter(TracePrinterBase):
    """The printer used to print the trace in the format of profile."""

    def __init__(self, display_functions: bool = False):
        """Initialize the printer.

        Args:
            display_functions: Whether to display the function calls.
        """
        self._display_functions = display_functions

    def build_event(self, event: TraceEventBase, min_timestamp: float) -> Dict:
        """Build the event for the trace."""
        ts = str((event.time_stamp - min_timestamp) * 1e6)
        data = {"pid": 0, "tid": 0, "name": event.name, "ts": ts}
        # TODO: add args to the trace
        if isinstance(event, CompletionRequestEvent):
            data["cat"] = "gen"
            data["ph"] = "b"
            data["id"] = event.name
        elif isinstance(event, CompletionResponseEvent):
            data["cat"] = "gen"
            data["ph"] = "e"
            data["id"] = event.name
            data["cost"] = event.cost
            data["output"] = event.ret.dict()
        elif self._display_functions:
            if isinstance(event, FunctionCallEvent):
                data["cat"] = "func"
                data["ph"] = "B"
                data["tid"] = "main"
            elif isinstance(event, FunctionReturnEvent):
                data["cat"] = "func"
                data["ph"] = "E"
                data["tid"] = "main"
        return data

    def print(
        self, trace: TraceEngineBase, trace_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Print the trace in the format of Chrome tracing."""
        events = []
        for event in trace.events:
            if data := self.build_event(event, trace.min_timestamp):
                events.append(data)
        return {"traceEvents": events}


def print_trace(
    printer: Optional[TracePrinterBase] = None,
    trace_file: Optional[str] = None,
) -> None:
    """Print to visualize the trace.

    Default printer is to the [langfuse](https://langfuse.com/) platform.
    You can also configured to the local hosted version.
    """
    if printer is None:
        printer = TraceLangfusePrinter()
    if trace_file is None:
        trace = global_vars.trace_engine
        if trace is None:
            raise ValueError("No trace found")
        trace_file = global_vars.metadata.trace_file
    else:
        trace = TraceEngine(trace_file)

    if trace_file is not None:
        meta_file = get_meta_file(trace_file)
        trace_metadata = load_file(meta_file)
    else:
        trace_metadata = None
    printer.print(trace, trace_metadata)
