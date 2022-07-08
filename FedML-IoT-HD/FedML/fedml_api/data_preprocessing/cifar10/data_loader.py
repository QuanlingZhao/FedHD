import logging

import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.0):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        #AddGaussianNoise(),
    ])

    #train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        #AddGaussianNoise(),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


# def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
#                                             client_number, batch_size):
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
#                                                                                              data_dir,
#                                                                                              partition_method,
#                                                                                              client_number,
#                                                                                              partition_alpha)
#     class_num = len(np.unique(y_train))
#     logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
#     train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
#
#     # get global test data
#     if process_id == 0:
#         train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
#         logging.info("train_dl_global number = " + str(len(train_data_global)))
#         logging.info("test_dl_global number = " + str(len(test_data_global)))
#         train_data_local = None
#         test_data_local = None
#         local_data_num = 0
#     else:
#         # get local dataset
#         dataidxs = net_dataidx_map[process_id - 1]
#         local_data_num = len(dataidxs)
#         logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
#         # training batch size = 64; algorithms batch size = 32
#         train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
#                                                  dataidxs)
#         logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
#             process_id, len(train_data_local), len(test_data_local)))
#         train_data_global = None
#         test_data_global = None
#     return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num
