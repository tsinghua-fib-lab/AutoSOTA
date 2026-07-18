import Levenshtein
import numpy as np
import re
import ast
import hashlib
import tiktoken
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

model = None  # lazy loaded


def hash_string(s: str, length: 8) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[0:length]


def get_norm_vector(vector):
    if len(vector.shape) == 1:
        return vector / np.linalg.norm(vector)
    else:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]


def dot_product(vectors, query_vector):
    similarities = np.dot(vectors, query_vector.T)
    return similarities


def cosine_similarity(vectors, query_vector):
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities


def str_is_close_any(a: str, b: List[str], min_sim: float = 0.75) -> bool:
    for b_str in b:
        if str_is_close(a, b_str, min_sim):
            return True
    return False


def str_is_close(a: str, b: str, min_sim: float = 0.75, strict=True) -> bool:
    if strict:
        return Levenshtein.jaro_winkler(a, b) > min_sim
    else:
        split_a = a.split(" ")
        if len(split_a) == 1:
            return Levenshtein.jaro_winkler(a, b) > min_sim
        elif len(split_a) > 4:
            return False
        else:
            for split in split_a:
                if Levenshtein.jaro_winkler(split, b) > min_sim:
                    return True

    return False


def select_closest(
    input_str: str,
    target_strings: List[str],
    dist: str = "jaro_winkler",
    return_sim: bool = False,
) -> Union[str, Tuple[str, float]]:
    best_sim = 0.0
    selected_str = ""

    for t_str in target_strings:
        if dist == "jaro_winkler":
            sim = Levenshtein.jaro_winkler(input_str, t_str)
        elif dist == "levenshtein":
            sim = Levenshtein.distance(input_str, t_str)
        elif dist == "embed":
            input_embed, t_embed = _get_model().encode([input_str, t_str])
            sim = cosine_similarity(input_embed, t_embed)
        if sim > best_sim:
            best_sim = sim
            selected_str = t_str

    if return_sim:
        return selected_str, best_sim

    return selected_str


def replace_parentheses(text: str) -> str:
    count = 1
    while "{" in text and "}" in text:
        text = re.sub(r"\{.*?\}", "<{}>".format(count), text, 1)
        count += 1
    return text


def string_to_dict(dict_string: str):
    if dict_string.startswith("{") and dict_string.endswith("}"):
        dict_string = dict_string[1:-1]
    # Split the string into pairs
    pairs = dict_string.split(",")
    # Split each pair into key-value and strip extra spaces
    dict_pairs = [pair.split(":") for pair in pairs]
    dict_pairs = [[key.strip(), value.strip()] for key, value in dict_pairs]
    # Convert values to appropriate types
    for pair in dict_pairs:
        try:
            pair[1] = ast.literal_eval(pair[1])
        except ValueError:
            pass
        except SyntaxError:
            pair[1] = str(pair[1])
    # Return as a dictionary
    return dict(dict_pairs)


def string_hash(s: str, len: int = 8) -> str:
    """Hashes a string using SHA256 and returns the first len characters of the hash."""

    assert isinstance(s, str), "Input should be of string type"
    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig[0:len]


def num_tokens_from_messages(messages: List[str]):
    """Returns the number of tokens used by a list of strings. Does not include formatting messages"""

    encoding = tiktoken.encoding_for_model("gpt-4")

    num_tokens = 0
    for message in messages:
        num_tokens += len(encoding.encode(message))

    return num_tokens


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def splice(text: str, start: int, end: int, offset: int, replacement: str) -> Tuple[str, int]:
    new_offset = offset + len(replacement) - (end - start)

    return text[: start + offset] + replacement + text[end + offset :], new_offset


def anonymize_str(
    input_str: str,
    entities: List[Tuple[str, int, int, str]] = [],
):
    """Anonymizes a string by replacing all entities with <ENTITY_TYPE> tags.

    Args:
        input_str (str): The string to anonymize
        include (bool, optional): Sorted list of entities to remove


    """

    text = input_str
    copy_str = text
    offset = 0
    last_end = -1
    for ent in entities:
        start = ent[1] + offset
        end = ent[2] + offset

        relevant_section = copy_str[start:end]
        # assert relevant_section == ent[0], "Entity does not match text"
        copy_str, offset = splice(
            copy_str,
            ent[1],
            ent[2],
            offset,
            ent[3],
        )

    return copy_str
