import numpy as np
import copy



def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

# 0.01:[5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
# 0.02:[5000, 3237, 2096, 1357, 878, 568, 368, 238, 154, 100]
# 0.05:[5000, 3584, 2569, 1842, 1320, 946, 678, 486, 348, 250]
def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes#5000
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type = 'exp'):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('Original number of samples of each label:')
    print(img_num_list)
    

    list_clients_indices = []
    classes = list(range(num_classes)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    return img_num_list, list_clients_indices

    #img_num_list :[5000, 3237, 2096, 1357, 878, 568, 368, 238, 154, 100]
    #list_clients_indices: 有5000张一类的索引 3237张二类的索引....................



