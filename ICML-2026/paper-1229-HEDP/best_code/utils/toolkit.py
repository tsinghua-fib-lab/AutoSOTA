import os
import numpy as np
import torch

#统计参数数量
def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

#生成one hot向量，根据数量去划分-1到1之间
def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    # Old accuracy nb_old上次任务识别的类数量；这里y>nb_old代表新任务的标签
    # idxes = np.where(y_true < nb_old)[0]
    # all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
    #                                                      decimals=2)

    # New accuracy
    # idxes = np.where(y_true >= nb_old)[0]
    # all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc

#划分 (图片路径，标签)
def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def accuracy_domain_total(y_pred, y_true, class_num=1):
    increment=class_num
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    #总准确率
    all_acc['total'] = np.around((y_pred%class_num == y_true%class_num).sum()*100 / len(y_true), decimals=2)

    return all_acc

#域准确率，这里%class_num 代表可能判定事物的标签对，但是事物所在域不同；increment是不同域的差
def accuracy_domain(y_pred, y_true, nb_old, increment=2, class_num=1):
    increment=class_num
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    #总准确率
    all_acc['total'] = np.around((y_pred%class_num == y_true%class_num).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy 测试集是全量的，包括不同域，这里old是旧数据在新网络的结果，new是新数据在新网络的结果
    #也按域去划分，每个域的数据在新网络里预测的准确率
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]#找到本域正确标签的索引
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))#域标签范围
        #本域的准确率
        all_acc[label] = np.around(((y_pred[idxes]%class_num) == (y_true[idxes]%class_num)).sum()*100 / len(idxes), decimals=2)

    # Old accuracy 小于旧标签值的索引
    # idxes = np.where(y_true < nb_old)[0]
    # all_acc['old'] = 0 if len(idxes) == 0 else np.around(((y_pred[idxes]%class_num) == (y_true[idxes]%class_num)).sum()*100 / len(idxes),decimals=2)

    # # New accuracy 大于旧标签值的索引
    # idxes = np.where(y_true >= nb_old)[0]
    # all_acc['new'] = np.around(((y_pred[idxes]%class_num) == (y_true[idxes]%class_num)).sum()*100 / len(idxes), decimals=2)

    return all_acc


#二元准确率
def accuracy_binary(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred%2 == y_true%2).sum()*100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0'))
        all_acc[label] = np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    # all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),decimals=2)
    all_acc['old'] = 0 if len(idxes) == 0 else np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes),decimals=2)

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around(((y_pred[idxes]%2) == (y_true[idxes]%2)).sum()*100 / len(idxes), decimals=2)

    return all_acc


