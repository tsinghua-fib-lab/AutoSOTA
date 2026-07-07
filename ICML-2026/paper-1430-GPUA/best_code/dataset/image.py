
import os
import pickle
import random
import re
from collections import OrderedDict, defaultdict

import scipy
import torchvision
from loguru import logger
from tqdm import tqdm

from .base_image import ImageDataset, ImageDatasets
from .utils import listdir_nohidden, read_json

AVAIL_DATASETS = [
    "Caltech101", "DescribableTextures", "EuroSAT", "Food101",
    "FGVCAircraft", "OxfordFlowers", "OxfordPets", "StanfordCars", "SUN397",
    "ImageUCF101", "ImageNet", "ImageNetA", "ImageNetR",
    "ImageNetSketch", "ImageNetV2"
]


class Caltech101(ImageDatasets):
    dataset_dir = "caltech-101"
    image_dir = "101_ObjectCategories"
    split_path = "split_zhou_Caltech101.json"
    split_fewshot_dir = "split_fewshot"
    ignored_classes = ["BACKGROUND_Google", "Faces_easy"]
    new_class_names = {
        "airplanes": "airplane",
        "Faces": "face",
        "Leopards": "leopard",
        "Motorbikes": "motorbike"
    }


class DescribableTextures(ImageDatasets):
    dataset_dir = "dtd"
    image_dir = "images"
    split_path = "split_zhou_DescribableTextures.json"
    split_fewshot_dir = "split_fewshot"


class EuroSAT(ImageDatasets):
    dataset_dir = "eurosat"
    image_dir = "2750"
    split_path = "split_zhou_EuroSAT.json"
    split_fewshot_dir = "split_fewshot"
    new_class_names = {
        "AnnualCrop": "Annual Crop Land",
        "Forest": "Forest",
        "HerbaceousVegetation": "Herbaceous Vegetation Land",
        "Highway": "Highway or Road",
        "Industrial": "Industrial Buildings",
        "Pasture": "Pasture Land",
        "PermanentCrop": "Permanent Crop Land",
        "Residential": "Residential Buildings",
        "River": "River",
        "SeaLake": "Sea or Lake"
    }


class Food101(ImageDatasets):
    dataset_dir = "food-101"
    image_dir = "images"
    split_path = "split_zhou_Food101.json"
    split_fewshot_dir = "split_fewshot"


class FGVCAircraft(ImageDatasets):
    dataset_dir = "fgvc_aircraft"
    image_dir = "images"
    split_path = ""
    split_fewshot_dir = "split_fewshot"

    def read_data(self):

        label_names = []
        variants_path = os.path.join(self.dataset_dir, "variants.txt")
        with open(variants_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                label_names.append(line.strip())

        name2label = {c: i for i, c in enumerate(label_names)}

        train = self.parse_txt_split(name2label, "images_variant_train.txt")
        val = self.parse_txt_split(name2label, "images_variant_val.txt")
        test = self.parse_txt_split(name2label, "images_variant_test.txt")

        return train, val, test

    def parse_txt_split(self, name2label, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        file_paths, labels, label_names = [], [], []

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                img_name = line[0] + ".jpg"
                label_name = " ".join(line[1:])
                path = os.path.join(self.image_dir, img_name)

                file_paths.append(path)
                labels.append(name2label[label_name])
                label_names.append(label_name)

        return self._lists_to_examples(file_paths, labels, label_names)


class OxfordFlowers(ImageDatasets):
    dataset_dir = "oxford_flowers"
    image_dir = "jpg"
    split_path = "split_zhou_OxfordFlowers.json"
    split_fewshot_dir = "split_fewshot"
    label2name_file = "cat_to_name.json"
    label_file = "imagelabels.mat"

    def read_data(self):
        self.label_file = os.path.join(self.dataset_dir, self.label_file)
        self.label2name_file = os.path.join(
            self.dataset_dir, self.label2name_file)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.create_split()
            self.save_split(train, val, test, self.split_path, self.image_dir)

        return train, val, test

    def create_split(self):
        tracker = defaultdict(list)
        label_file = scipy.io.loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        logger.info("Splitting data into 50% train, 20% val, and 30% test")

        label2name = read_json(self.label2name_file)
        train_filepaths, val_filepaths, test_filepaths = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        train_label_names, val_label_names, test_label_names = [], [], []

        for label, img_paths in tracker.items():
            random.shuffle(img_paths)
            n_total = len(img_paths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            label_name = label2name[str(label)]

            train_filepaths.extend(img_paths[:n_train])
            train_labels.extend([label] * n_train)
            train_label_names.extend([label_name] * n_train)

            val_filepaths.extend(img_paths[n_train: n_train + n_val])
            val_labels.extend([label] * n_val)
            val_label_names.extend([label_name] * n_val)

            test_filepaths.extend(img_paths[n_train + n_val:])
            test_labels.extend([label] * n_test)
            test_label_names.extend([label_name] * n_test)

        train_examples = self._lists_to_examples(
            train_filepaths, train_labels, train_label_names)
        val_examples = self._lists_to_examples(
            val_filepaths, val_labels, val_label_names)
        test_examples = self._lists_to_examples(
            test_filepaths, test_labels, test_label_names)

        return train_examples, val_examples, test_examples


class OxfordPets(ImageDatasets):
    dataset_dir = "oxford_pets"
    image_dir = "images"
    split_path = ""
    anno_dir = "annotations"
    split_path = "split_zhou_OxfordPets.json"
    split_fewshot_dir = "split_fewshot"

    def read_data(self):
        self.anno_dir = os.path.join(self.dataset_dir, self.anno_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.parse_split("trainval.txt")
            test = self.parse_split("test.txt")
            train, val = self.split_trainval(trainval)

            self.save_split(train, val, test, self.split_path, self.image_dir)

        return train, val, test

    def parse_split(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        file_paths, labels, label_names = [], [], []

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index

                file_paths.append(impath)
                labels.append(label)
                label_names.append(breed)

        return self._lists_to_examples(file_paths, labels, label_names)


class StanfordCars(ImageDatasets):
    dataset_dir = "stanford_cars"
    image_dir = ""
    split_path = "split_zhou_StanfordCars.json"
    split_fewshot_dir = "split_fewshot"

    def read_data(self):

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval_file = os.path.join(
                self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(
                self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(
                self.dataset_dir, "devkit", "cars_meta.mat")

            trainval = self.parse_split("cars_train", trainval_file, meta_file)
            test = self.parse_split("cars_test", test_file, meta_file)
            train, val = self.split_trainval(trainval)

            self.save_split(train, val, test, self.split_path, self.image_dir)

        return train, val, test

    def parse_split(self, image_dir, anno_file, meta_file):
        anno_file = scipy.io.loadmat(anno_file)["annotations"][0]
        meta_file = scipy.io.loadmat(meta_file)["class_names"][0]

        file_paths, labels, label_names = [], [], []

        for ann_f in anno_file:

            img_name = ann_f["fname"][0]
            path = os.path.join(self.dataset_dir, image_dir, img_name)
            label = ann_f["class"][0, 0]

            # convert to 0-based index
            label = int(label) - 1

            label_name = meta_file[label][0]
            names = label_name.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            label_name = " ".join(names)

            file_paths.append(path)
            labels.append(label)
            label_names.append(label_name)

        return self._lists_to_examples(file_paths, labels, label_names)


class SUN397(ImageDatasets):
    dataset_dir = "sun397"
    image_dir = "SUN397"
    split_path = "split_zhou_SUN397.json"
    split_fewshot_dir = "split_fewshot"

    def read_data(self):
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            label_names = []
            label_name_path = os.path.join(self.dataset_dir, "ClassName.txt")
            with open(label_name_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    # remove /
                    line = line.strip()[1:]
                    label_names.append(line)

            name2label = {c: i for i, c in enumerate(label_names)}
            trainval = self.parse_split(name2label, "Training_01.txt")
            test = self.parse_split(name2label, "Testing_01.txt")
            train, val = self.split_trainval(trainval)

            self.save_split(train, val, test, self.split_path, self.image_dir)

        return train, val, test

    def parse_split(self, name2label, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        file_paths, labels, label_names = [], [], []

        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                # remove /
                img_name = line.strip()[1:]
                label_name = os.path.dirname(img_name)
                label = name2label[label_name]
                path = os.path.join(self.image_dir, img_name)

                # remove 1st letter
                names = label_name.split("/")[1:]
                # put words like indoor/outdoor at first
                names = names[::-1]
                label_name = " ".join(names)

                file_paths.append(path)
                labels.append(label)
                label_names.append(label_name)

        return self._lists_to_examples(file_paths, labels, label_names)


class ImageUCF101(ImageDatasets):
    dataset_dir = "ucf101"
    image_dir = "UCF-101-midframes"
    split_path = "split_zhou_UCF101.json"
    split_fewshot_dir = "split_fewshot"

    def read_data(self):
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            name2label = {}
            filepath = os.path.join(
                self.dataset_dir, "ucfTrainTestlist/classInd.txt")

            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    label, classname = line.strip().split(" ")
                    # conver to 0-based index
                    label = int(label) - 1
                    name2label[classname] = label

            trainval = self.parse_split(
                name2label, "ucfTrainTestlist/trainlist01.txt")
            test = self.parse_split(
                name2label, "ucfTrainTestlist/testlist01.txt")
            train, val = self.split_trainval(trainval)

            self.save_split(train, val, test, self.split_path, self.image_dir)

        return train, val, test

    def parse_split(self, name2label, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        file_paths, labels, label_names = [], [], []

        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")[0]  # trainlist: filename, label
                action, filename = line.split("/")
                label = name2label[action]

                elements = re.findall("[A-Z][^A-Z]*", action)
                label_name = "_".join(elements)

                filename = filename.replace(".avi", ".jpg")
                path = os.path.join(self.image_dir, label_name, filename)

                file_paths.append(path)
                labels.append(label)
                label_names.append(label_name)

        return self._lists_to_examples(file_paths, labels, label_names)


class ImageNet(ImageDatasets):
    dataset_dir = "imagenet"
    image_dir = ""
    split_path = ""
    preprocessed = "preprocessed"
    split_fewshot_dir = "split_fewshot"
    ignored_classes = ["README.txt"]

    def read_data(self):
        # self.preprocessed = os.path.join(
        #     self.dataset_dir, "preprocessed_imagenet.pkl")
        self.preprocessed = os.path.join(
             self.new_path, "preprocessed_imagenet.pkl")
        print(f"数据集目录: {self.preprocessed}")
        if os.path.exists(self.preprocessed):
            logger.info(f"Reading splits from {self.preprocessed}")

            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
                val = test
        else:
            logger.info("Creating ImageNet splits.")
            
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            
            folder2label_names = self.read_label_names(text_file)
            
            train = self.parse_txt_split(folder2label_names, "train")
            print("*****")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.parse_txt_split(folder2label_names, "val")
            val = test

            preprocessed = {"train": train, "test": test}

            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        return train, val, test

    @staticmethod
    def read_label_names(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """

        label_names = OrderedDict()

        with open(text_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                label_name = " ".join(line[1:])
                label_names[folder] = label_name

        return label_names

    def parse_txt_split(self, folder2label_names, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())

        file_paths, labels, label_names = [], [], []

        tbar = tqdm(enumerate(folders))
        for label, folder in tbar:
            img_names = listdir_nohidden(os.path.join(split_dir, folder))
            label_name = folder2label_names[folder]

            for img_name in img_names:
                path = os.path.join(split_dir, folder, img_name)

                file_paths.append(path)
                labels.append(label)
                label_names.append(label_name)

            tbar.set_description(f"Reading ImageNet: {folder}")

        return self._lists_to_examples(file_paths, labels, label_names)


class ImageNetX(object):
    # For domain generatlization, changes the test set of ImagetNet
    # ImageNetR, ImageNetA, ImageNetSketch, ImageNetV2
    target_dataset_dir = None
    target_image_dir = None
    ignored_classes = []

    def __init__(self,
                 data_path: str,
                 test_transforms: torchvision.transforms.Compose = None,
                 **_
                 ):

        self.data_path = os.path.abspath(os.path.expanduser(data_path))

        self.target_dataset_dir = os.path.join(
            self.data_path, self.target_dataset_dir)

        self.target_image_dir = os.path.join(
            self.target_dataset_dir, self.target_image_dir)

        target_cls_file = os.path.join(
            self.target_dataset_dir, "classnames.txt")

        target_classnames = ImageNet.read_label_names(target_cls_file)
        target_test_examples = self.read_target_date(target_classnames)

        self.test_dataset = ImageDataset(
            **target_test_examples, transforms=test_transforms)

    def read_target_date(self, target_classnames):

        folders = listdir_nohidden(self.target_image_dir, sort=True)
        if len(self.ignored_classes) > 0:
            folders = [f for f in folders if f not in self.ignored_classes]

        file_paths, labels, label_names = [], [], []

        tbar = tqdm(enumerate(folders))
        for label, folder in tbar:
            img_names = listdir_nohidden(
                os.path.join(self.target_image_dir, folder))
            label_name = target_classnames[folder]

            for img_name in img_names:
                path = os.path.join(self.target_image_dir, folder, img_name)

                file_paths.append(path)
                labels.append(label)
                label_names.append(label_name)

            tbar.set_description(
                f"Reading {self.target_dataset_dir}: {folder}")

        return self._lists_to_examples(file_paths, labels, label_names)

    def _lists_to_examples(self, file_paths, labels, label_names):
        assert len(file_paths) == len(labels) == len(label_names), \
            "All lists must have the same length"

        dataset_examples = {
            "file_paths": file_paths,
            "labels": labels,
            "label_names": label_names
        }

        return dataset_examples


class ImageNetA(ImageNetX):
    target_dataset_dir = "imagenet-adversarial"
    target_image_dir = "imagenet-a"
    ignored_classes = ["README.txt"]


class ImageNetR(ImageNetX):
    target_dataset_dir = "imagenet-rendition"
    target_image_dir = "imagenet-r"
    ignored_classes = ["README.txt"]


class ImageNetSketch(ImageNetX):
    target_dataset_dir = "imagenet-sketch"
    target_image_dir = "images"
    ignored_classes = ["README.txt"]


class ImageNetV2(ImageNetX):
    target_dataset_dir = "imagenetv2"
    target_image_dir = "imagenetv2-matched-frequency-format-val"
    ignored_classes = ["README.txt"]

    def read_target_date(self, target_classnames):
        image_dir = self.target_image_dir
        folders = list(target_classnames.keys())

        file_paths, labels, label_names = [], [], []

        tbar = tqdm(range(1000))
        for label in tbar:
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = target_classnames[folder]

            for imname in imnames:
                impath = os.path.join(class_dir, imname)

                file_paths.append(impath)
                labels.append(label)
                label_names.append(classname)

            tbar.set_description(
                f"Reading {self.target_dataset_dir}: {folder}")

        return self._lists_to_examples(file_paths, labels, label_names)
