# set train and test set
import heapq

from sklearn.model_selection import train_test_split


# static
# 1. split train and test set of x and y


def split1(all_list, ratio):
    x_train, x_test, y_train, y_test = train_test_split(
        all_list[:, 2:-2], all_list[:, 1], ratio)
    return x_train, x_test, y_train, y_test


# 2. split train and test set of whole data set
# def split(all_list, shuffle=False, ratio=0.8):
def split2(all_list, ratio):
    #    part_list = all_list[:, :-1]
    num = len(all_list)
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    #   if shuffle:
    #       random.shuffle(all_list)  # random order
    train = all_list[:offset]
    test = all_list[offset:]
    return train, test


# split part
def split_part(data, partNo):
    part = data[data[:, -1] == partNo, :-1]
    return part


# dynamic all_list[:,-1] is the time factor
def split_time(all_list, train_length):
    last_date = max(all_list[:, -1])
    test = all_list[all_list[:, -1] == last_date, :-1]
    raw_train = all_list[all_list[:, -1] != last_date, :-1]
    time_order = heapq.nlargest(3, raw_train[:, -1])
    train = raw_train[raw_train[:, -1] == time_order, :-1]
    return train, test
