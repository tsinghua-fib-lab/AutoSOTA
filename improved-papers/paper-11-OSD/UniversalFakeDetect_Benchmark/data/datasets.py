import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from random import random, choice, shuffle
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os
import math
from skimage.io import imread
from copy import deepcopy
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073],
    "beitv2": [0.485, 0.456, 0.406],
    "siglip": [0.5, 0.5, 0.5],
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711],
    "beitv2": [0.229, 0.224, 0.225],
    "siglip": [0.5, 0.5, 0.5],
}


def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img


def recursively_read(rootdir, must_contain, classes=[], exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                if len(classes) == 0:
                    out.append(os.path.join(r, file))
                elif os.path.join(r, file).split('/')[-3] in classes:
                    out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain='', classes=[]):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain, classes)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        
        self.data_label  = opt.data_label
        if opt.data_mode == 'ours':
            pickle_name = "train.pickle" if opt.data_label=="train" else "val.pickle"
            real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
            fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
        elif opt.data_mode == 'wang2020':
            temp = 'train' if opt.data_label == 'train' else 'test/progan'
            if opt.data_label == 'train':
                # 20(all) classes supervision
                real_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
                fake_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )
            else:
                # 20(all) classes supervision
                real_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
                fake_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )
        elif opt.data_mode == 'ours_wang2020':
            pickle_name = "train.pickle" if opt.data_label=="train" else "val.pickle"
            real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
            fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
            
            temp = 'train' if opt.data_label == 'train' else 'test/progan'
            real_list += get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
            fake_list += get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )

        # setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.total_list = real_list + fake_list
        shuffle(self.total_list)
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)
        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        
        if opt.arch.lower().startswith("imagenet"):
            stat_from = "imagenet"
        elif opt.arch.lower().startswith("clip"):
            stat_from = "clip"
        elif opt.arch.lower().startswith("siglip"):
            stat_from = "siglip"
        elif opt.arch.lower().startswith("beitv2"):
            stat_from = "beitv2"

        print("mean and std stats are from: ", stat_from)
        if '2b' not in opt.arch:
            print ("using Official CLIP's normalization")
            self.transform = transforms.Compose([
                # rz_func,
                transforms.Lambda(lambda img: translate_duplicate(img, opt.loadSize)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            print ("Using CLIP 2B transform")
            self.transform = None # will be initialized in trainer.py

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
