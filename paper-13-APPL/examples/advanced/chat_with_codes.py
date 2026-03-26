# Usage: python examples/advanced/chat_with_codes.py

import glob
import os
from argparse import ArgumentParser

import seedir as sd
from loguru import logger
from prompt_toolkit.shortcuts import prompt
from rich.console import Console

from appl import AIRole, convo, gen, ppl
from appl.compositor import Tagged
from appl.core import load_file, make_panel
from appl.utils import get_num_tokens

parser = ArgumentParser()
parser.add_argument("--intro-file", type=str, default="./README.md")
parser.add_argument("--source", type=str, default="./examples")
parser.add_argument("--ext", type=str, default=".py")

args = parser.parse_args()


def read_file(file: str):
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


@ppl
def chat(intro_file: str, source_folder: str, ext: str = ".py"):
    with Tagged("introduction"):
        read_file(intro_file)

    with Tagged("directory", attrs={"name": source_folder}):
        "The source code is organized as follows:"
        (
            tree := sd.seedir(
                source_folder,
                style="spaces",
                printout=False,
                exclude_folders=["__pycache__"],
                exclude_files=[".DS_Store"],
            )
        )
        print(f"Chatting with {ext} files in the directory:\n{tree}")

    with Tagged("source"):
        "The contents of the source code are as follows:"

        for name in glob.glob(
            os.path.join(source_folder, "**", f"*{ext}"), recursive=True
        ):
            with Tagged("file", attrs={"name": name}):
                "```python"
                read_file(name)  # load the file contents and put in the prompt
                "```"

    "Now begin the chat about the project:\n"
    print(f"num_tokens: {get_num_tokens(str(convo()))}")

    console = Console()
    while True:
        (query := prompt("User: "))
        if query.startswith("exit"):
            break
        console.print(make_panel(query, title="User"))
        logger.info(f"User: {query}")
        with AIRole():
            (
                res := str(
                    gen(stream=True, max_relay_rounds=10).streaming(title="Assistant")
                )
            )
        logger.info(f"Assistant: {res}")


if __name__ == "__main__":
    chat(args.intro_file, args.source, args.ext)
