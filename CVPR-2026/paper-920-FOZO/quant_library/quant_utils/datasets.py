"""
Reuse version v4
Author: Hahn Yuan
"""
import PIL
import torch
import argparse
import numpy as np
import os
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder,DatasetFolder
import torch.utils.data
import re
import warnings
from PIL import Image
from PIL import ImageFile
import random
import torch.nn.functional as F
from torch.utils.data import Dataset

def calculate_n_correct(outputs,targets):
    _, predicted = outputs.max(1)
    n_correct= predicted.eq(targets).sum().item()
    return n_correct

class SetSplittor():
    def __init__(self,fraction=0.2):
        self.fraction=fraction
    
    def split(self,dataset):
        pass

class LoaderGenerator():
    """
    """
    def __init__(self,root,dataset_name,train_batch_size=1,test_batch_size=1,num_workers=0,kwargs={}):
        self.root=root
        self.dataset_name=str.lower(dataset_name)
        self.train_batch_size=train_batch_size
        self.test_batch_size=test_batch_size
        self.num_workers=num_workers
        self.kwargs=kwargs
        self.items=[]
        self._train_set=None
        self._test_set=None
        self._calib_set=None
        self.train_transform=None
        self.test_transform=None
        self.train_loader_kwargs = {
            'num_workers': self.num_workers ,
            'pin_memory': kwargs.get('pin_memory',True),
            'drop_last':kwargs.get('drop_last',False)
            }
        self.test_loader_kwargs=self.train_loader_kwargs.copy()
        self.load()
    
    @property
    def train_set(self):
        pass
    
    @property
    def test_set(self):
        pass
    
    def load(self):
        pass
    
    def train_loader(self):
        assert self.train_set is not None
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)
    
    def test_loader(self,shuffle=False,batch_size=None):
        assert self.test_set is not None
        if batch_size is None:
            batch_size=self.test_batch_size
        return torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle,  **self.test_loader_kwargs)
    
    def val_loader(self):
        assert self.val_set is not None
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.test_batch_size, shuffle=False,  **self.test_loader_kwargs)
    
    def trainval_loader(self):
        assert self.trainval_set is not None
        return torch.utils.data.DataLoader(self.trainval_set, batch_size=self.train_batch_size, shuffle=True,  **self.train_loader_kwargs)

    def calib_loader(self,num=1024,seed=3):
        if self._calib_set is None:
            np.random.seed(seed)
            inds=np.random.permutation(len(self.train_set))[:num]
            self._calib_set=torch.utils.data.Subset(copy.deepcopy(self.train_set),inds)
            self._calib_set.dataset.transform=self.test_transform
        return torch.utils.data.DataLoader(self._calib_set, batch_size=num, shuffle=False,  **self.train_loader_kwargs)
        
class CIFARLoaderGenerator(LoaderGenerator):
    def load(self):
        if self.dataset_name=='cifar100':
            self.dataset_fn=datasets.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                             std=[0.2673, 0.2564, 0.2762])
        elif self.dataset_name=='cifar10':
            self.dataset_fn=datasets.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2470, 0.2435, 0.2616])
        else:
            raise NotImplementedError
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set=self.dataset_fn(self.root, train=True, download=True, transform=self.train_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set=self.dataset_fn(self.root, train=False, transform=self.test_transform)
        return self._test_set

class COCOLoaderGenerator(LoaderGenerator):
    def load(self):
        # download from https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh
        self.train_set = DetectionListDataset(os.path.join(self.root,'trainvalno5k.txt'),transform=augmentation_detection_tansforms)
        self.test_set = DetectionListDataset(os.path.join(self.root,'5k.txt'),transform=detection_tansforms,multiscale=False)
        self.train_loader_kwargs={"collate_fn":self.train_set.collate_fn}
        self.test_loader_kwargs={"collate_fn":self.test_set.collate_fn}
        
class DetectionListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = [path for path in file.readlines()]
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print(f"Could not apply transform.")
                return
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1
        # Drop invalid images
        batch = [data for data in batch if data is not None]
        
        paths, imgs, bb_targets = list(zip(*batch))
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0) for img in imgs])
        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

# def faster_im_loader(path):
#     with open(path,'rb') as f:
#         bgr_array = TurboJPEG().decode(f.read())
#     rgb_array=np.concatenate([bgr_array[:,:,2:3],bgr_array[:,:,1:2],bgr_array[:,:,0:1]],-1)
#     return torch.Tensor(rgb_array)/255

class ImageNetLoaderGenerator(LoaderGenerator):
    def load(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set=ImageFolder(os.path.join(self.root,'train'), self.train_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set=ImageFolder(os.path.join(self.root,'val'), self.test_transform)
        return self._test_set

class CacheDataset(Dataset):
    def __init__(self,datas,targets) -> None:
        super().__init__()
        self.datas=datas
        self.targets=targets
        
    def __getitem__(self,idx):
        return self.datas[idx],self.targets[idx]

    def __len__(self):
        return len(self.datas)

class FasterImageNetLoaderGenerator(ImageNetLoaderGenerator):
    def test_loader(self,shuffle=False,batch_size=None):
        cache='/dev/shm/imagenet.pkl'
        assert self.test_set is not None
        if batch_size is None:
            batch_size=self.test_batch_size
        if os.path.exists(cache):
            print("Loading the dataset from shared memory")
            datas,targets=torch.load(cache)
        else:
            print("Preprocessing the dataset and save it to shared memory")
            loader=torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle,  **self.test_loader_kwargs)
            datas=[]
            targets=[]
            for data,target in loader:
                datas.append(data)
                targets.append(target)
            datas=torch.cat(datas,0)
            targets=torch.cat(targets,0)
            torch.save([datas,targets],cache)
        dataset=CacheDataset(datas,targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,  **self.test_loader_kwargs)
            
class DebugLoaderGenerator(LoaderGenerator):

    def load(self):
        version=re.findall("\d+",self.dataset_name)[0]
        class DebugSet(torch.utils.data.Dataset):
            def __getitem__(self,idx):
                if version=='0':
                    return torch.ones([1,4,4]),0
                if version=='1':
                    return torch.ones([1,8,8]),0
                if version=='2':
                    return torch.ones([1,1,1]),0
                if version=='3':
                    return torch.ones([1,3,3]),0
                else:
                    raise NotImplementedError(f"version {version} of Debug dataset is not supported")
            def __len__(self): return 1
        self.train_set=DebugSet()
        self.test_set=DebugSet()

def get_dataset(args:argparse.Namespace):
    """ Preparing Datasets, args: 
        dataset (required): MNIST, cifar10/100, ImageNet, coco
        dataset_root: str, default='./datasets'
        num_workers: int
        batch_size: int
        test_batch_size: int
        val_fraction: float, default=0
        
    """
    dataset_name=str.lower(args.dataset)
    dataset_root=getattr(args,'dataset_root','./datasets') 
    num_workers=args.num_workers if hasattr(args,'num_workers') else 4
    batch_size=args.batch_size if hasattr(args,'batch_size') else 64
    test_batch_size=args.test_batch_size if hasattr(args,'test_batch_size') else batch_size
    val_fraction=args.val_fraction if hasattr(args,"val_fraction") else 0
    if "cifar" in dataset_name:
        # Data loading code
        g=CIFARLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    elif "coco" in dataset_name:
        g=COCOLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    elif "debug" in dataset_name:
        g=DebugLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    elif args.dataset=='ImageNet':
        g=ImageNetLoaderGenerator(dataset_root,args.dataset,batch_size,test_batch_size,num_workers)
    else:
        raise NotImplementedError
    return g.train_loader(),g.test_loader()
    

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class ViTImageNetLoaderGenerator(ImageNetLoaderGenerator):
    """
    DataLoader for Vision Transformer. 
    To comply with timm's framework, we use the model's corresponding transform.
    """
    def __init__(self, root, dataset_name, train_batch_size, test_batch_size, num_workers, kwargs={}):
        kwargs.update({"pin_memory":False})
        super().__init__(root, dataset_name, train_batch_size=train_batch_size, test_batch_size=test_batch_size, num_workers=num_workers, kwargs=kwargs)

    def load(self):
        model = self.kwargs.get("model", None)
        assert model != None, f"No model in ViTImageNetLoaderGenerator!"

        config = resolve_data_config({}, model=model)
        self.train_transform = create_transform(**config, is_training=True)
        self.test_transform = create_transform(**config)

