import uuid
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def encode_to_uuid_v5(value: str, namespace: uuid.UUID = uuid.NAMESPACE_DNS) -> str:
    """Encode a string into a UUID version 5.

    Args:
        value: The input string to encode.
        namespace: The namespace for the UUID (default: DNS namespace).

    Returns:
        The generated UUID as a string.
    """
    return str(uuid.uuid5(namespace, value))


def pydantic_to_dict(obj: BaseModel) -> Dict[str, Any]:
    """Convert a Pydantic object to a dictionary."""
    return obj.model_dump()


def dict_to_pydantic(data: Dict[str, Any], cls: Type[M]) -> M:
    """Convert a dictionary to a Pydantic object."""
    return cls.model_validate(data)


if __name__ == "__main__":
    input_value = "example_value"
    encoded_uuid = encode_to_uuid_v5(input_value)

    print(f"Input Value: {input_value}")
    print(f"Encoded UUID: {encoded_uuid}")
