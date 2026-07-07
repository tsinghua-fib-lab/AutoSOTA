import os
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.base_video import VideoDataset
from dataset.utils import parse_video_paths, save_list_as_txt_file
from dataset.video import VideoDatasetFromSplits

AVAIL_DATASETS = ["HMDB51FewShot", "HMDB51ZeroShot", "K400FewShot",
                  "K700ZeroShot", "UCF101FewShot", "UCF101ZeroShot"]


def np_random_sampling(original, n_samples):
    original = np.array(original)
    selected_indices = np.random.choice(
        len(original), size=n_samples, replace=False)
    selected = original[selected_indices]
    remaining = np.delete(original, selected_indices)

    return selected.tolist(), remaining.tolist()


def create_n_shot_c_way(
    n_shot: int,
    c_way: int,
    class_names: str,
    data_paths_per_class: str,
    seed: int,
    n_query_shot: int = 95,
    use_all_classes: bool = False
):

    # for consistency
    np.random.seed(seed)
    sampled_classes, _ = np_random_sampling(class_names, c_way)
    support_paths, query_paths = [], []

    for cls_name in sampled_classes:
        sampled_paths, remaining_paths = np_random_sampling(
            data_paths_per_class[cls_name], n_shot)

        support_paths.extend(sampled_paths)
        if not use_all_classes:
            query_paths.extend(np_random_sampling(
                remaining_paths, n_query_shot)[0])

    return support_paths, query_paths


def create_fewshot_splits(
    n_shot: int,
    c_way: int,
    train_split_path: str,
    val_split_path: str,
    few_shot_class_names_path: str,
    seed: int,
    n_query_shot: int = 95,
    use_all_classes: bool = False,
):

    with open(few_shot_class_names_path, "r", encoding="utf-8") as f:
        few_shot_class_names = f.read().splitlines()
        few_shot_class_names = [item.strip()
                                for item in few_shot_class_names if len(item) > 0]

    video_paths = []
    splits_to_use = [train_split_path] if use_all_classes else [
        train_split_path, val_split_path]

    for split_path in splits_to_use:
        with open(split_path, "r", encoding="utf-8") as f:
            paths = f.read().splitlines()
        video_paths += [item.strip()
                        for item in paths if len(item) > 0]

    per_cls_video_paths = {cls_name: [] for cls_name in few_shot_class_names}

    for cls_name in tqdm(few_shot_class_names):
        for path in video_paths:
            # cls_name_in_path = str(Path(path).parent
            #cls_name_in_path = cls_name_in_path.split("/")[-1]
            cls_name_in_path = path.split("/")[-2]

            if cls_name == cls_name_in_path:
                per_cls_video_paths[cls_name].append(path)

    support_paths, query_paths = create_n_shot_c_way(
        n_shot=n_shot,
        c_way=c_way,
        class_names=few_shot_class_names,
        data_paths_per_class=per_cls_video_paths,
        seed=seed,
        n_query_shot=n_query_shot,
        use_all_classes=use_all_classes
    )

    assert len(set(support_paths)) == c_way * n_shot
    if not use_all_classes:
        assert len(set(query_paths)) == c_way * n_query_shot
    else:
        assert len(query_paths) == 0
    return support_paths, query_paths


def load_or_create_few_shot_slip(n_shot,
                                 c_way, split, seed, train_split_path,
                                 test_split_path, few_shot_class_names_path,
                                 path_to_few_shot_splits, n_query_shot, extension,
                                 use_all_classes
                                 ):

    # create directory if it does not exist
    path_to_few_shot_splits.mkdir(parents=True, exist_ok=True)

    fewshot_split_path = path_to_few_shot_splits / \
        Path(f'{split}/split_{seed}.list')

    if fewshot_split_path.exists():
        logger.info("Fewshot split found, reusing it")

        with open(fewshot_split_path, "r", encoding="utf-8") as f:
            video_paths = f.read().splitlines()

        video_paths = [item for item in video_paths if len(item) > 0]

        # 3rd party splits do not have extension
        if len(Path(video_paths[0]).suffix) == 0:
            video_paths = [
                f"{str(item)}.{extension}" for item in video_paths]

    else:
        logger.info(f"Creating new fewshot split: {fewshot_split_path}")

        train_video_paths, test_video_paths = create_fewshot_splits(
            n_shot=n_shot,
            c_way=c_way,
            train_split_path=train_split_path,
            val_split_path=test_split_path,
            few_shot_class_names_path=few_shot_class_names_path,
            seed=seed,
            n_query_shot=n_query_shot,
            use_all_classes=use_all_classes
        )

        # save both for next time
        fewshot_train_path = path_to_few_shot_splits / Path('train')
        os.makedirs(fewshot_train_path, exist_ok=True)
        save_list_as_txt_file(fewshot_train_path /
                              Path(f'split_{seed}.list'), train_video_paths)

        if not use_all_classes:
            # we use the original test set, do not save it
            fewshot_test_path = path_to_few_shot_splits / Path('test')
            os.makedirs(fewshot_test_path, exist_ok=True)
            save_list_as_txt_file(fewshot_test_path /
                                  Path(f'split_{seed}.list'), test_video_paths)
        else:
            assert len(test_video_paths) == 0

        video_paths = train_video_paths if split == "train" else test_video_paths

    return video_paths


class VideoFewShot(VideoDataset):
    mode: str = None
    path_to_few_shot_splits: str = "dataset/fewshot_video_data"
    few_shot_class_names_path: str = None
    train_split_path: str = None
    test_split_path: str = None
    actions2names: str = None
    extension: str = None
    dataset_name: str = None
    label_names_path: str = None

    def __init__(
        self,
        data_path: str,
        n_shot: int,
        c_way: int,
        split: str,
        num_frames: int,
        temporal_strides: List,
        sampler_type: str,
        seed: int,
        num_clips: int = 1,
        target_fps: int = 30,
        transforms: Compose = None,
        n_query_shot: int = 95,
        use_all_classes: bool = False,
        **_,
    ):

        logger.info(f"Creating [fewshot] dataset {self.dataset_name} ....")

        assert split in ["train", "test"]

        if use_all_classes:
            assert c_way == -1, \
                "c_way must be set to -1 if use_all_classes is True"

        file_paths, labels, label_names, label2name = self.parse_data(
            data_path=Path(data_path),
            n_shot=n_shot,
            c_way=c_way,
            split=split,
            seed=seed,
            n_query_shot=n_query_shot,
            use_all_classes=use_all_classes
        )

        super().__init__(
            file_paths=file_paths,
            labels=labels,
            label_names=label_names,
            label2name=label2name,
            num_frames=num_frames,
            temporal_strides=temporal_strides,
            sampler_type=sampler_type,
            num_clips=num_clips,
            target_fps=target_fps,
            transforms=transforms
        )

    def parse_data(self, data_path: Path, n_shot: int, c_way: int,
                   split: str, seed: int, n_query_shot: int,
                   use_all_classes: bool):

        logger.info("Parsing data paths ....")

        if use_all_classes:
            few_shot_class_names_path = self.label_names_path
            # load self.label_names_path
            with open(self.label_names_path, "r", encoding="utf-8") as f:
                label_names = f.read().splitlines()
            c_way = len([i for i in label_names if len(i) > 0])
            path_to_few_shot_splits = Path(self.path_to_few_shot_splits) / \
                Path(f'{n_shot}_shot_C_way')
        else:
            few_shot_class_names_path = self.few_shot_class_names_path
            path_to_few_shot_splits = Path(self.path_to_few_shot_splits) / \
                Path(f'{n_shot}_shot_{c_way}_way')

        path_to_few_shot_splits = path_to_few_shot_splits / \
            Path(f'{self.dataset_name.lower()}')

        if (not use_all_classes) or (use_all_classes and split == "train"):
            video_paths = load_or_create_few_shot_slip(
                n_shot=n_shot,
                c_way=c_way,
                split=split,
                seed=seed,
                train_split_path=self.train_split_path,
                test_split_path=self.test_split_path,
                few_shot_class_names_path=few_shot_class_names_path,
                path_to_few_shot_splits=path_to_few_shot_splits,
                n_query_shot=n_query_shot,
                extension=self.extension,
                use_all_classes=use_all_classes
            )
        else:
            # n shot all classes, use standard test set
            with open(self.test_split_path, "r", encoding="utf-8") as f:
                video_paths = f.read().splitlines()
                video_paths = [item.strip()
                               for item in video_paths if len(item) > 0]

        return parse_video_paths(
            data_path=data_path,
            video_paths=video_paths,
            label_names_path=self.label_names_path,
            actions2names=self.actions2names,
            extension=self.extension
        )


class UCF101FewShot(VideoFewShot):
    mode = "video"
    dataset_name = "UCF101"
    few_shot_class_names_path = "dataset/video_data/ucf101/few_shot_actions.list"
    label_names_path = "dataset/video_data/ucf101/actions.list"
    train_split_path = "dataset/video_data/ucf101/trainlist01.txt"
    test_split_path = "dataset/video_data/ucf101/testlist01.txt"
    actions2names = "dataset/video_data/ucf101/actions2names.json"
    extension = "avi"


class HMDB51FewShot(VideoFewShot):
    mode = "video"
    dataset_name = "HMDB51"
    few_shot_class_names_path = "dataset/video_data/hmdb51/few_shot_actions.list"
    label_names_path = "dataset/video_data/hmdb51/actions.list"
    train_split_path = "dataset/video_data/hmdb51/train_split01.txt"
    test_split_path = "dataset/video_data/hmdb51/test_split01.txt"
    actions2names = "dataset/video_data/hmdb51/actions2names.json"
    extension = "avi"


class K400FewShot(VideoFewShot):
    mode = "video"
    dataset_name = "K400"
    few_shot_class_names_path = "dataset/video_data/k400/few_shot_actions.list"
    label_names_path = "dataset/video_data/k400/actions.list"
    train_split_path = "dataset/video_data/k400/train_split.txt"
    test_split_path = "dataset/video_data/k400/val_split.txt"
    actions2names = "dataset/video_data/k400/actions2names.json"
    extension = "h5"


#######################################################################################


class K700ZeroShot(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "K700"
    label_names_path = None
    label_names_train = "dataset/fewshot_video_data/zeroshot/k700/action_train.list"
    label_names_test = "dataset/fewshot_video_data/zeroshot/k700/action_test.list"
    train_split_path = "dataset/fewshot_video_data/zeroshot/k700/train.list"
    test_split_path = "dataset/fewshot_video_data/zeroshot/k700/test.list"
    actions2names = "dataset/video_data/k700/actions2names.json"
    extension = "mp4"

    def __init__(
        self,
        split: str,
        **kwargs
    ):

        self.label_names_path = self.label_names_train if \
            split == "train" else self.label_names_test

        super().__init__(split=split, **kwargs)


class UCF101ZeroShot(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "UCF101"
    label_names_path = None
    label_names_train = "dataset/fewshot_video_data/zeroshot/ucf101/action_train.list"
    label_names_test = "dataset/fewshot_video_data/zeroshot/ucf101/action_test.list"
    train_split_path = "dataset/fewshot_video_data/zeroshot/ucf101/train.list"
    test_split_path = "dataset/fewshot_video_data/zeroshot/ucf101/test.list"
    actions2names = "dataset/video_data/ucf101/actions2names.json"
    extension = "avi"

    def __init__(
        self,
        split: str,
        **kwargs
    ):

        self.label_names_path = self.label_names_train if \
            split == "train" else self.label_names_test

        super().__init__(split=split, **kwargs)


class HMDB51ZeroShot(VideoDatasetFromSplits):
    mode = "video"
    dataset_name = "HMDB51"
    label_names_path = None
    label_names_train = "dataset/fewshot_video_data/zeroshot/hmdb51/action_train.list"
    label_names_test = "dataset/fewshot_video_data/zeroshot/hmdb51/action_test.list"
    train_split_path = "dataset/fewshot_video_data/zeroshot/hmdb51/train.list"
    test_split_path = "dataset/fewshot_video_data/zeroshot/hmdb51/test.list"
    actions2names = "dataset/video_data/hmdb51/actions2names.json"
    extension = "avi"

    def __init__(
        self,
        split: str,
        **kwargs
    ):

        self.label_names_path = self.label_names_train if \
            split == "train" else self.label_names_test

        super().__init__(split=split, **kwargs)
        super().__init__(split=split, **kwargs)
