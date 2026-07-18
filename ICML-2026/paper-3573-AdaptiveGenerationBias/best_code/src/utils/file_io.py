import os
import shutil
import time
import threading
import queue
from tqdm import trange
from src.bias_pipeline.data_types.conversation import ConversationBatch


def check_and_overwrite(path):
    """
    Checks if 'path' exists. If so, prompts [Y/n] (Enter = Y) with a 60s timeout,
    then overwrites or cancels accordingly.
    """
    if not os.path.exists(path):
        print(f"'{path}' does not exist; nothing to overwrite.")
        os.makedirs(path, exist_ok=True)
        # Return filepath
        return path

    # Compose message based on file or non-empty folder
    if os.path.isdir(path):
        # If empty we can just return the path
        if not os.listdir(path):
            return path

        msg = "Directory is not empty"
    else:
        msg = "File exists"
    prompt = f"{msg}, overwrite? [Y/n]: "

    # Threaded input to allow timeout
    q = queue.Queue()

    def get_input():
        user_input = input().strip()
        q.put(user_input)

    print(prompt, end="", flush=True)
    threading.Thread(target=get_input, daemon=True).start()

    # Wait up to 60s, checking each second for user input
    answer = None
    for _ in trange(60, desc="Overwrite [Y/n]?", unit="s"):
        time.sleep(1)
        if not q.empty():
            answer = q.get()
            break

    # Default behaviors:
    if answer is None or answer == "":  # timed out or just pressed Enter
        answer = "y"

    if answer.lower() == "y":
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print("Overwritten.")
        os.makedirs(path, exist_ok=True)
        return path
    else:
        raise FileExistsError(
            f"'{path}' already exists and was not overwritten nor confirmed - Aborting"
        )


class SafeOpen:
    def __init__(self, path: str, mode: str = "a", ask: bool = False):
        self.path = path

        if ask and os.path.exists(path):
            if input("File already exists. Overwrite? (y/n)") != "y":
                raise Exception("File already exists")
        self.mode = "a+" if os.path.exists(path) else "w+"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.file = None
        self.lines = []

    def __enter__(self):
        self.file = open(self.path, self.mode)
        self.file.seek(0)  # move the cursor to the beginning of the file
        self.lines = self.file.readlines()
        # Remove last lines if empty
        while len(self.lines) > 0 and self.lines[-1] == "":
            self.lines.pop()
        return self

    def flush(self):
        self.file.flush()

    def write(self, content):
        self.file.write(content)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


def safe_append(path, content, constructor):
    # First check if the file exists
    if os.path.exists(path):
        # If it does, read it via the constructor
        elements = constructor(path)

        to_write = []

        if len(elements) > 0 and isinstance(elements[0], ConversationBatch):
            # Precompute the hashes of the elements
            hashes = [element.__hash__() for element in elements]
            # make efficient lookup
            hashes_set = set(hashes)
            elements = hashes_set

        # Append the new content whenever it is not already in the file
        for new_element in content:
            if isinstance(new_element, ConversationBatch):
                new_hash = new_element.__hash__()
                if new_hash not in hashes_set:
                    to_write.append(new_element)
            else:
                if new_element not in elements:
                    to_write.append(new_element)

        with SafeOpen(path) as f:
            for element in to_write:
                element.to_file(f)

    else:
        # If the file does not exist, just write the content
        with SafeOpen(path) as f:
            for element in content:
                element.to_file(f)
