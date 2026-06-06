import copy
import threading
from typing import Any, Dict, Optional

from litellm import model_list, provider_list
from loguru import logger

from ..core.globals import global_vars
from ..core.server import BaseServer, DummyServer


def _init_server(
    model: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseServer:
    """Initialize a server based on the model, provider and other arguments."""
    if provider is None:
        provider = model.split("/")[0]

    is_custom_llm = provider.split("/")[0] == "custom"
    custom_llm_provider = (
        provider[len("custom/") :] or "openai" if is_custom_llm else None
    )  # "openai" is the default provider for custom models

    if model == "_dummy":
        server: BaseServer = DummyServer()  # for testing purposes
    elif is_custom_llm or model in model_list or provider in provider_list:
        from .api import APIServer

        msg = f"Initializing APIserver for model {model}"
        if base_url is not None:
            msg += f" with address {base_url}"
        logger.info(msg)
        server = APIServer(
            model,
            base_url=base_url,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model {model}")

    return server


def _get_server_configs(name: str) -> Dict[str, Any]:
    servers = global_vars.configs.servers or {}
    server_configs: Dict[str, Any] = {}
    if name not in servers:
        if name != "_dummy":
            logger.info(
                f"Server {name} not found in configs, using the server name as model name"
            )
        server_configs["model"] = name
    else:
        cfgs = servers[name]
        if not isinstance(cfgs, dict):
            raise ValueError(f"The configs of server {name} is not a dictionary")
        server_configs = cfgs
    if server_configs is None:
        raise ValueError(f"The configs of server {name} is None")

    for _ in range(100):  # prevent infinite loop (max 100 templates)
        if "template" not in server_configs:
            break
        server_configs = copy.deepcopy(server_configs)
        template_name = server_configs.pop("template")
        if template_name not in servers:
            raise ValueError(f"Server config template {template_name} not found")
        template_config = servers[template_name]
        if not isinstance(template_config, dict):
            raise ValueError(
                f"The configs of server {template_name} is not a dictionary"
            )
        # override template config
        server_configs = {**template_config, **server_configs}
    if "template" in server_configs:
        raise ValueError(f"Template loop detected in server config {name}")
    return server_configs


class ServerManager:
    """The manager for all servers."""

    def __init__(self) -> None:
        """Initialize the server manager."""
        self._lock = threading.Lock()
        self._servers: Dict[str, BaseServer] = {}

    def register_server(self, name: str, server: BaseServer) -> None:
        """Register a server with a name."""
        self._servers[name] = server

    def close_server(self, name: str) -> None:
        """Close a server by name."""
        if name in self._servers:
            self._servers[name].close()
            del self._servers[name]

    def get_server(self, name: Optional[str]) -> BaseServer:
        """Get a server by name. If name is None, get the default server."""
        if name == "small":
            name = global_vars.configs.default_servers.small
        elif name == "large":
            name = global_vars.configs.default_servers.large
        if name is None:  # if still None, fall back to default server
            name = global_vars.configs.default_servers.default
            if name is None:  # backward-compatible for now
                name = global_vars.configs.servers.get("default", None)  # type: ignore
                # logger.warning(
                #     "Default server is moved to default_servers.default, "
                #     "please update your config file to set the default server. "
                #     "The current way will be deprecated in the future."
                # )
        if name is None:
            raise ValueError(
                "Default server is not configured. Please set the default server in your config file. "
                "See https://appl-team.github.io/appl/setup/#setup-appl-configuration for more details."
            )
        with self._lock:
            if name not in self._servers:
                server_configs = _get_server_configs(name)
                server = _init_server(**server_configs)
                self.register_server(name, server)
        return self._servers[name]


# Singleton server manager
server_manager = ServerManager()
