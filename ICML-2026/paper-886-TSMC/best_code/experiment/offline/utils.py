from typing import Any


def unpack_nested(config_dict: dict[str, Any], *keys) -> Any:
    x = config_dict
    for key in keys:
        x = x[key]
    return x.get("value", None)
