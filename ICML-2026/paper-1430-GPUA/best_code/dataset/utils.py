
import errno
import json
import os
from pathlib import Path
from typing import List, Union

from PIL import Image
from tqdm.auto import tqdm


def read_image(path: Union[str, Path]):
    return Image.open(path).convert("RGB")


def read_json(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def listdir_nohidden(path, sort=False):
    # From Dassl
    # https://github.com/KaiyangZhou/Dassl.pytorch
    # Lists non-hidden items in a directory
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def mkdir_if_missing(dirname):
    # From Dassl
    # https://github.com/KaiyangZhou/Dassl.pytorch
    # Creates dirname if it is missing
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise


def write_json(obj, fpath):
    # From Dassl
    # https://github.com/KaiyangZhou/Dassl.pytorch
    # Writes to a json file.

    mkdir_if_missing(os.path.dirname(fpath))

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def parse_video_paths(data_path: str, video_paths: List, label_names_path: str,
                      actions2names: str, extension: str):

    video_paths = list(sorted(video_paths))

    # pare video paths wich are a list of [class/video_path.extension]
    with open(label_names_path, "r", encoding="utf-8") as f:
        all_label_names = f.read().splitlines()
        all_label_names = [item.strip()
                           for item in all_label_names if len(item) > 0]

    with open(actions2names, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    file_paths, labels_names = [], []

    for idx, path_and_label in enumerate(tqdm(video_paths)):
        if len(path_and_label.split("/")[-1].split(" ")) > 1:
            # in ucf101 each line is: path label, rest is just label
            video_path, _ = path_and_label.split(" ")
        else:
            video_path = path_and_label

        video_path = Path(video_path)
        label_name = video_path.parent.name
        video_path = data_path / video_path

        assert label_name in all_label_names, \
            f"Class {label_name} is not one of the classes."

        if idx % 1000 == 0:
            # check every N iter, otherwise too slow
            assert os.path.isfile(
                video_path), f"Video {video_path} is not a valid file"

        assert is_valid_file(
            video_path.name, [extension]), f"Video {video_path} is not a valid file"

        file_paths.append(video_path)
        labels_names.append(label_name)

    # Create label name to class ID mapping
    label_names = list(set(labels_names))
    label_names = list(sorted(label_names))
    name2label = {j: i for i, j in enumerate(label_names)}

    label2name = {i: class_names[j] for i, j in enumerate(label_names)}
    label_names = list(label2name.values())
    labels = [name2label[label_name] for label_name in labels_names]

    return (
        file_paths,
        labels,
        label_names,
        label2name
    )


def list_files_in_dir(base_path: str, extensions: List):
    found_files = []

    for entry in os.scandir(base_path):
        if not entry.is_file():
            continue

        if is_valid_file(entry.name, extensions):
            found_files.append(entry.path)

    return found_files


def load_data_from_dir(data_path: str, class_names: List, extensions: List) -> tuple:
    # data_path is a dir and each folder contains instance
    # of one class

    data_path = os.path.expanduser(data_path)
    assert os.path.isdir(data_path), f"{data_path} is not a directory"

    paths, labels = [], []

    for class_name in tqdm(class_names):
        class_path = os.path.join(data_path, class_name)

        if os.path.isdir(class_path):
            found_files = list_files_in_dir(class_path, extensions)
            assert len(found_files) > 0, \
                f"No files found for class {class_name} in {class_path}"

            paths.extend(found_files)
            labels.extend([class_name] * len(found_files))

    return paths, labels


def check_path(path: Union[List, int]):

    def assert_path(path):
        assert isinstance(path, str) or isinstance(
            path, Path), "Invalid path"
        assert os.path.exists(path), f"Path {path} does not exist"

    if isinstance(path, list):
        for single_path in path:
            assert_path(single_path)
    else:
        assert_path(path)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def is_valid_file(filename: str, extensions: List):
    return any(
        [filename.lower().endswith(ext) for ext in extensions]
    )


def save_list_as_txt_file(file_path: str, to_save_list: List):
    with open(str(file_path), "w", encoding="utf-8") as f:
        for item in to_save_list:
            f.write(str(item) + "\n")
