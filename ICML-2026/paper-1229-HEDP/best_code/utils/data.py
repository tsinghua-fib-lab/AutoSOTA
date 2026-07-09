import os
import numpy as np
from torchvision import transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        self.domain_names = self.args["task_name"]
        self.class_order  = np.arange(len(self.domain_names) * 2).tolist()
        
    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            #listdir返回指定的文件夹包含的文件或文件夹的名字的列表
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))
        #标签是01,23,...分组的
        self.train_data, self.train_targets = split_images_labels(train_dataset)#图片路径
        self.test_data, self.test_targets = split_images_labels(test_dataset)#标签


class iCore50(iData):

    use_path = False#处理后的x不是用图片路径,这里是向量
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # def getDomainName(self,index):
    #     return self.domain_names[index]
    def __init__(self, args):
        self.args = args
        self.domain_names = self.args["task_name"]
        self.class_order = np.arange(len(self.domain_names) * 50).tolist()#类标签范围，训练用8个域，每个域的标签有0-49，因此混合域是0-49，50-99...
        
    def download_data(self):
        datagen = CORE50(root=self.args["data_path"], scenario="ni")

        dataset_list = []
        for i, train_batch in enumerate(datagen):#根据next迭代得到批量数据（x,y）
            imglist, labellist = train_batch
            labellist += i*50 #为了划分开每个域标签差距50
            imglist = imglist.astype(np.uint8)
            
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list)[:, 0])
        train_y = np.concatenate(np.array(dataset_list)[:, 1])
        
        self.train_data = train_x
        self.train_targets = train_y
        #训练数据是全部批次，测试数据是其中最后一个批次
        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x#图片向量[128*128*3]
        self.test_targets = test_y#标签


class iDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        self.domain_names = self.args["task_name"]
        self.class_order  = np.arange(len(self.domain_names) * 345).tolist()
        
    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]#每个域标签划分345
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)#图片路径
        self.test_targets = np.array(train_y)#标签



