
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iGanFake, iCore50, iDomainNet
import torch


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        #class_order和提取数据集时的label数一样
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls] #[2,2,2,2,2],[50,50,..]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    
    @property
    def nb_tasks(self):
        return len(self._increments)#域数量、或者说是不同任务数量

    def get_task_size(self, task):
        return self._increments[task]#每个域的标签数量，任务大小

    #indices索引列表(标签范围)，source划分训练集还是测试集，当前transform模式是训练还是测试还是flip？
    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':#图片水平翻转
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])

        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)
        #加扩展数据
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

#indices索引列表(标签范围)，source划分训练集还是测试集，当前transform模式是训练还是测试还是flip？
#这里训练数据train模式时，返回DummyDatasetV1  
    def get_dataset_v1(self, indices, source, mode, appendent=None, ret_data=False):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            if source == 'train':
                train_trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
                test_trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        elif mode == 'flip':#图片水平翻转
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            data.append(class_data)
            targets.append(class_targets)
        #加扩展数据
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)
        
        if mode == 'train' and source == 'train':
            dataset=DummyDatasetV1(data, targets, train_trsf,test_trsf, self.use_path)
        else:
            dataset=DummyDataset(data, targets, trsf, self.use_path)
        if ret_data:
            return data, targets, dataset
        else:
            return dataset
    #只获取扩展的数据集
    def get_anchor_dataset(self, mode, appendent=None, ret_data=False):
        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        data, targets = [], []
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    #val_samples_per_class每个类采样多少作为评估数据集，训练集和评估集不重合，都返回
    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    #加载数据集
    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()#获取指定数据集的数据

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order排序
        #unique去除其中重复的元素 ，并按元素 由小到大 返回
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:#洗牌
            np.random.seed(seed)
            #随机排列序列
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order #按顺序，比如0-8 * 50，0-345*6，0-2*5
        self._class_order = order
        #logging.info(self._class_order)

        # Map indices 根据新的order去排序
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    #根据指定范围找x,y
    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.otherWLogits={}
        self.curWeights={}
        self.domainsLogits={}
        self.domainsWeights={}
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label
    def newDomainsLogits(self,taskNum):
        for taskid in range(taskNum):
            self.domainsLogits[taskid]={}
    def setDomainsLogits(self,taskId,wids,Logits):
        for index,wid in enumerate(wids): 
            self.domainsLogits[taskId][wid.item()]=Logits[index]
    def getDomainsLogits(self,taskId,wids):
        DomainsLogits=[]
        for id in wids: 
            DomainsLogits.append(self.domainsLogits[taskId][id.item()])
        DomainsLogits=torch.cat(DomainsLogits).view(len(DomainsLogits), -1).detach()
        return DomainsLogits
    def newDomainsWeights(self,taskNum):
        for taskid in range(taskNum):
            self.domainsWeights[taskid]={}
    def setDomainsWeights(self,taskId,wids,Weights):
        for index,wid in enumerate(wids): 
            self.domainsWeights[taskId][wid.item()]=Weights[index]
    def getDomainsWeights(self,taskId,wids):
        DomainsWeights=[]
        for id in wids: 
            DomainsWeights.append(self.domainsWeights[taskId][id.item()])
        DomainsWeights=torch.tensor(DomainsWeights)
        return DomainsWeights
    
    def setCurWeight(self,wids,curWeight):
        for index,wid in enumerate(wids): 
            self.curWeights[wid.item()]=curWeight[index]
    def setOtherLogits(self,wids,otherWLogits):
        for index,wid in enumerate(wids): 
            self.otherWLogits[wid.item()]=otherWLogits[index]
            
    def getCurWeight(self,wids):
        curWeight=[]
        for id in wids: 
            curWeight.append(self.curWeights[id.item()])
        curWeights=torch.cat(curWeight).view(len(curWeight), -1)
        return curWeights
    def getOtherLogits(self,wids):
        otherLogits=[]
        for id in wids: 
            otherLogits.append(self.otherWLogits[id.item()])
        otherLogits=torch.cat(otherLogits).view(len(otherLogits), -1)
        return otherLogits.detach()
class DummyDatasetV1(Dataset):
    def __init__(self, images, labels, train_trsf,test_trsf, use_path=False):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.train_trsf = train_trsf
        self.test_trsf=test_trsf
        self.use_path = use_path
        self.otherWLogits={}
        self.curWeights={}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            train_image = self.train_trsf(pil_loader(self.images[idx]))
            test_image = self.test_trsf(pil_loader(self.images[idx]))
        else:
            train_image = self.train_trsf(Image.fromarray(self.images[idx]))
            test_image = self.test_trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, train_image,test_image, label
    
    

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

#分类
def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cddb":
        return iGanFake(args)
    elif name == 'core50':
        return iCore50(args)
    elif name == 'domainnet':
        return iDomainNet(args)
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')#RGB读

#目前没有这种格式读
def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
