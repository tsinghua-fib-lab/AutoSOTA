import json
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.bias_pipeline.data_types.data_types import Persona


def load_data(path):
    extension = path.split(".")[-1]

    assert extension == "jsonl"

    with open(path, "r") as json_file:
        json_list = json_file.readlines()

    return load_data_from_lines(json_list)


def load_data_from_lines(json_list):
    from src.bias_pipeline.data_types.data_types import Persona

    data = []
    for json_str in json_list:
        pers = Persona.from_json(json.loads(json_str))
        data.append(pers)

    return data


def convert_dict_to_json_schema(attributes: dict[str, tuple[str, list[str]]]) -> dict:
    """
    Convert a dict of the form:
        {
          "someAttr": ("Description for this attribute", ["possibleVal1", "possibleVal2"]),
          "anotherAttr": ("Description for another attribute", ["valA", "valB"])
        }
    into a JSON Schema–like dictionary structure.
    """

    # Build the top-level schema wrapper.
    # You can tweak the name, description, and other fields as needed.
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "attributes",
            "description": "Attributes to generate a profile with",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }

    properties = {}
    required_fields = []

    for attr_name, (attr_desc, possible_values) in attributes.items():
        # Each attribute becomes a property with type "string" and an enum of possible values

        if len(possible_values) > 0:
            properties[attr_name] = {
                "type": "string",
                "description": attr_desc,
                "enum": possible_values,
            }
        else:
            properties[attr_name] = {
                "type": "string",
                "description": attr_desc,
            }
        required_fields.append(attr_name)

    schema["json_schema"]["schema"]["properties"] = properties
    schema["json_schema"]["schema"]["required"] = required_fields

    return schema


if __name__ == "__main__":
    attributes_example = {
        "color": ("Color of the item", ["red", "green", "blue"]),
        "size": ("Size options", ["small", "medium", "large"]),
    }

    json_schema_output = convert_dict_to_json_schema(attributes_example)
    print(json_schema_output)
