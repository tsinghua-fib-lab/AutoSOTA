import os

__all__ = ["dynamic_metadata", "get_requires_for_dynamic_metadata"]


def dynamic_metadata(
        field: str,
        settings: "dict[str, list[str] | str] | None" = None,
) -> "str | dict[str, str]":
    # print(os.environ)
    # TODO figure out how to handle torch versions
    if field == "dependencies":
        return ["torch"]
    return {}


def get_requires_for_dynamic_metadata(
        _settings: "dict[str, object] | None" = None,
) -> list[str]:
    # transformers 4.53 is broken with qwen3
    return ["torch", "transformers<=4.52"]
