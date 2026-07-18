import numpy as np
import uuid
import json
import os
from typing import List
from src.models import get_model
from src.personas import Persona
from src.configs import GENPROFILESConfig, Config
from src.bias_pipeline.utils import convert_dict_to_json_schema


# -------------------------------------------------------------------------------
# 1. PERSONA CONSTRUCTION
# -------------------------------------------------------------------------------


class PersonaConstructor:
    """
    Responsible for creating and managing diverse personas that can be used in counterfactual analysis.
    """

    def __init__(self, config: GENPROFILESConfig) -> None:
        self.config = config
        self.personas: List[Persona] = []

        self.attributes = config.attributes

        self.schema = convert_dict_to_json_schema(self.attributes)

        self.model = get_model(config.gen_model)

    def generate_persona(self, name: str, id: str) -> Persona:
        # Generate a persona using the model

        self.model.args["response_format"] = self.schema

        messages = [
            {
                "role": "system",
                "content": "You are a creative Persona generator. You generate a realistic persona with the specified attributes.",
            },
            {
                "role": "user",
                "content": "Generate me a new persona with realistic attributes in the following categories: "
                + ", ".join(self.attributes.keys()),
            },
        ]

        persona_json = self.model._predict_call(messages)

        # Extract the generated persona
        persona_dict = json.loads(persona_json)

        persona_dict["name"] = name
        persona_dict["id"] = id

        # Create a new persona object
        persona = Persona.from_json(persona_dict)

        return persona

    def create_personas_from_config(self) -> List[Persona]:
        """
        Creates and returns a list of personas based on the specified configuration.

        """

        # get unique names
        usernames = get_usernames()

        # Append random uuid to each username
        usernames = [f"{username}_{str(uuid.uuid4())}" for username in usernames]

        # Generate the specified number of personas
        for i in range(self.config.num_profiles):
            persona = self.generate_persona(usernames[i], usernames[i].split("_")[1])
            self.personas.append(persona)

        return self.personas


def get_usernames():
    usernames = []

    # parse usernames.txt
    with open("usernames.txt", "r") as f:
        output = f.readline()

    usernames = output.split(", ")
    usernames = (np.unique(usernames)).tolist()
    print(len(usernames))
    print(usernames)

    return usernames


def persona_creation(cfg: GENPROFILESConfig, outpath: str) -> None:
    """
    Generates a set of personas based on the specified configuration file.
    """

    constructor = PersonaConstructor(cfg)

    personas = constructor.create_personas_from_config(cfg)

    # Store the generated personas in a file
    with open(outpath, "w") as f:
        for persona in personas:
            persona.to_file(f)


def migrate_personas_from_old_format(cfg: Config) -> None:
    """
    Migrates personas from the old format to the new format.
    """

    path = cfg.task_config.persona_path
    out_path = cfg.get_out_path()

    # Load json of the old format
    personas: List[Persona] = []

    with open(path, "r") as f:
        old_personas = json.load(f)

        for name, old_persona in old_personas.items():
            id = str(uuid.uuid4())

            if "style" in old_persona:
                style = old_persona["style"]
                old_persona.pop("style")
            else:
                style = {}

            # Rename keys to match the new format
            old_persona["place_of_birth"] = old_persona.pop("birth_city_country")
            old_persona["place_of_living"] = old_persona.pop("city_country")

            persona = Persona(name, id, old_persona, style)

            personas.append(persona)

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Store the generated personas in a file
    with open(out_path, "w") as f:
        for persona in personas:
            persona.to_file(f)
