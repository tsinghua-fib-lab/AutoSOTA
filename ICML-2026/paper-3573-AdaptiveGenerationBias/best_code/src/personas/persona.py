from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, TextIO
import hashlib
import json
import random


@dataclass
class Persona:
    """
    Data model to represent a persona consisting of demographic and stylistic attributes.

    Attributes:
        name (str): A unique name or ID for the persona.
        demographics (Dict[str, Any]): Key demographic attributes like age, gender, location, etc.
        style (Dict[str, Any]): Stylistic attributes (e.g., short-authoritative, elaborative, friendly).
    """

    name: str
    id: int
    demographics: Dict[str, Any]
    style: Dict[str, Any]
    ancestor: Optional[Tuple[str, Tuple[str, str, str]]] = None

    def __repr__(self) -> str:
        return f"<Persona {self.name} | ID {self.id} | Ancestor {self.ancestor} | Demographics: {self.demographics} | Style: {self.style}>"

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.id,
            "ancestor": self.ancestor,
            "demographics": self.demographics,
            "style": self.style,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Persona":
        name = data["name"]
        id = data["id"]
        ancestor = data.get("ancestor", None)
        demographics = data["demographics"]
        style = data["style"]
        return cls(name, id, demographics, style, ancestor)

    def to_file(self, file: TextIO) -> None:
        file.write(json.dumps(self.to_json()) + "\n")
        file.flush()

    def __hash__(self) -> int:
        return int(hashlib.sha1(self.name.encode("utf-8")).hexdigest(), 16)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Persona):
            return False
        return self.id == other.id

    def create_counterfactual(self, attribute: str, new_val: Optional[str] = None) -> "Persona":
        """
        Creates a counterfactual persona by negating the demographics of the current persona.
        """

        # Update the attribute
        if new_val is None:
            # Randomly change the value within the domain
            domain = self.attributes[attribute][1].copy()
            # Remove existing value
            domain.remove(self.demographics[attribute])
            new_val = random.choice(domain)

        new_name = f"{self.name}_cf={attribute}"

        ancestor = (self.name, (attribute, self.demographics[attribute], new_val))

        # Copy the current persona
        new_persona = Persona(
            new_name,
            self.id,
            self.demographics.copy(),
            self.style.copy(),
            ancestor,
        )

        new_persona.demographics[attribute] = new_val

        return new_persona


def load_personas(file_path: str) -> List[Persona]:
    """
    Load personas from a file into a dictionary.

    Args:
        file_path (str): Path to jsonl file containing the personas.

    Returns:
        List[Persona]: A list of Persona instances.
    """
    personas = []

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            persona = Persona.from_json(data)
            personas.append(persona)
    return personas
