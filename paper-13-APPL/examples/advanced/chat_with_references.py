# Usage: python examples/advanced/chat_with_references.py

import glob
import os
from argparse import ArgumentParser

import seedir as sd
from prompt_toolkit.shortcuts import prompt
from rich.console import Console
from rich.prompt import Prompt

from appl import AIRole, convo, gen, ppl
from appl.compositor import Tagged
from appl.core import load_file, make_panel
from appl.utils import get_num_tokens

parser = ArgumentParser()
parser.add_argument("--intro-file", type=str, default="./README.md")
parser.add_argument("--source", type=str, default="./examples")
parser.add_argument("--ext", type=str, default=".py")
parser.add_argument("-rf", "--ref-folders", type=str, nargs="+", default=[])
parser.add_argument(
    "-ri",
    "--ref-info",
    type=str,
    nargs="+",
    default=[],
    help="The extra information for the reference.",
)

args = parser.parse_args()
if len(args.ref_info) != len(args.ref_folders):
    raise ValueError("The number of ref-info and ref-folders should be the same.")


def read_file(file: str):
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


@ppl
def chat(intro_file: str, source: str, ext: str = ".py"):
    with Tagged("introduction"):
        read_file(intro_file)

    for name in glob.glob(os.path.join(source, "**", f"*{ext}"), recursive=True):
        with Tagged("example", attrs={"file_path": name}):
            read_file(name)

    for folder, info in zip(args.ref_folders, args.ref_info):
        with Tagged("reference_folder", attrs={"file_path": folder}):
            info
            if os.path.isdir(folder):
                folder = os.path.join(folder, f"*")
            for name in glob.glob(folder, recursive=True):
                with Tagged("reference", attrs={"file_path": name}):
                    if os.path.isfile(name):
                        read_file(name)
        print(f"Folder: {folder}\n{info}")

    "Now begin the chat about the project along with the references.\n"

    print(f"Total tokens: {get_num_tokens(convo())}")
    console = Console()
    while True:
        (query := prompt("User: "))
        if query.startswith("exit"):
            break
        console.print(make_panel(query, title="User"))
        with AIRole():
            (response := str(gen(stream=True).streaming(title="Assistant")))
            print(response)


if __name__ == "__main__":
    chat(args.intro_file, args.source, args.ext)
