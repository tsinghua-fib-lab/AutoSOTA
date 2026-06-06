# Kitti parser based on https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/dataset/kitti/parser.py
# Modifications for torchsparse

import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

import torchsparse
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from .augmentation import augmentation

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class nuScenes(Dataset):

    def __init__(self, root,    # directory where data is
                sequences,     # sequences for this data (e.g. [1,3,4,6])
                labels,        # label dict: (e.g 10: "car")
                color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
                learning_map,  # classes to learn (0 to N-1 for xentropy)
                learning_map_inv,    # inverse of previous (recover labels)
                sensor,              # sensor to parse scans from
                voxel_size=0.05,   # voxel size for point cloud
                gt=True,             # send ground truth?
                aug=False):           # augmentation on point cloud
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.voxel_size = voxel_size
        self.gt = gt
        self.aug = aug

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        # make sure labels is a dict
        assert(isinstance(self.labels, dict))

        # make sure color_map is a dict
        assert(isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert(isinstance(self.learning_map, dict))

        # make sure sequences is a list
        assert(isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:04d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]

            # check all scans have labels
            if self.gt:
                assert(len(scan_files) == len(label_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()

        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        # open scan and labels
        scan = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
        if self.gt:
            labels = np.fromfile(label_file, dtype=np.uint32).reshape(-1)
            labels_orig = labels & 0xFFFF  # remove unused bits
            # map labels for training
            labels = self.map(labels_orig, self.learning_map)

        origin_len = len(scan)

        # augment point cloud
        if self.aug:
            scan = augmentation(scan, labels)

        ref_pc = scan[:, :3].copy()
        ref_index = np.arange(len(ref_pc))
        pc_ = np.round(scan[:, :3] / self.voxel_size)
        pc_ =  pc_ - pc_.min(axis=0, keepdims=True)
        feat_ = scan #np.concatenate((xyz, scan[:, 3:]), axis=1)
        if self.gt:
            ref_labels = labels.copy()

        _, inds, inverse_map = sparse_quantize(pc_,
                                               voxel_size=self.voxel_size,
                                               return_index=True,
                                               return_inverse=True)
        

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels[inds]
        num_voxel = len(inds)
        points = SparseTensor(ref_pc, pc_)
        ref_index = SparseTensor(ref_index, pc_)
        map = SparseTensor(inds, pc)
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_mapped = SparseTensor(ref_labels, pc_)
        labels_orig = SparseTensor(labels_orig, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        # get name and sequence
        patch_norm = os.path.normpath(scan_file)
        path_split = patch_norm.split(os.sep)
        path_seq = path_split[-3]  # e.g. 'sequences/00/velodyne/000000.bin'
        path_name = path_split[-1].replace('.bin', '.label')  # e.g. '000000.label'
        # print("path norm: ", patch_norm)
        # print("path seq: ", path_seq)
        # print("path name: ", path_name)

        data_dict = {}
        data_dict['lidar'] = lidar # input features
        data_dict['points'] = points # original point cloud
        data_dict['targets'] = labels # labels for input features
        data_dict['targets_mapped'] = labels_mapped # labels for original point cloud
        data_dict['targets_original'] = labels_orig # original labels with anomaly class
        data_dict['ref_index'] = ref_index # index of original point cloud
        data_dict['origin_len'] = origin_len # length of original point cloud
        data_dict['map'] = map # map from original point cloud to voxelized point cloud
        data_dict['num_voxel'] = num_voxel # number of voxels
        data_dict['inverse_map'] = inverse_map # inverse map from voxelized point cloud to original point cloud
        data_dict['seq'] = path_seq # sequence name
        data_dict['name'] = path_name # name of the scan

        return data_dict


    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               voxel_size,       # voxel size for point cloud
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               aug=False,         # point cloud augmentation for train
               shuffle_train=True):# shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.voxel_size = voxel_size
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    self.train_dataset = nuScenes(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       voxel_size=self.voxel_size,
                                       gt=self.gt,
                                       aug=aug)

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   collate_fn=sparse_collate_fn,
                                                   pin_memory=True,
                                                   drop_last=True)

    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = nuScenes(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       voxel_size=self.voxel_size,
                                       gt=self.gt)

    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   collate_fn=sparse_collate_fn,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = nuScenes(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        voxel_size=self.voxel_size,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    collate_fn=sparse_collate_fn,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return nuScenes.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return nuScenes.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = nuScenes.map(label, self.learning_map_inv)
    # put label in color
    return nuScenes.map(label, self.color_map)

  def get_resolution(self):
    try:
      H = self.train_dataset.sensor_img_H
      W = self.train_dataset.sensor_img_W
    except:
      H = self.train_dataset.dataset.sensor_img_H
      W = self.train_dataset.dataset.sensor_img_W
    return H, W