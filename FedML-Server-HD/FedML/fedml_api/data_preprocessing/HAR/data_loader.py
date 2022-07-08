import pandas as pd
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def read_data(train_data_dir,test_data_dir):
    users = []
    groups = []
    test_data = {}
    train_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        users.extend(cdata["users"])
        train_data.update(cdata["user_data"])


    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    return groups,users,train_data,test_data



def batch_data(data, batch_size):
    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x))
        batched_y = torch.from_numpy(np.asarray(batched_y))
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_HAR(batch_size,dataset_dir):
    train_path = dataset_dir+"/train"
    test_path = dataset_dir+"/test"
    groups, users, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    for u, g in zip(users, groups):
        user_train_data_num = len(train_data[u]["x"])
        user_test_data_num = len(test_data[u]["x"])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[u], batch_size)
        test_batch = batch_data(test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1
    client_num = client_idx
    output_dim = 6

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        output_dim,
    )



class HAR_loader(Dataset):
    def __init__(self,single_client_data):
        self.batch_num = len(single_client_data)
        self.sample_num = 0
        self.x = []
        self.y = []
        for batch_idx in range(self.batch_num):
            batch = single_client_data[batch_idx]
            self.x.append(batch[0])
            self.y.append(batch[1])
            self.sample_num += len(batch[0])


    def __len__(self):
        return self.sample_num

    def __getitem__(self,index):
        return (self.x[index],self.y[index])



def get_HAR_dataloader(batch_size,dataset_dir):
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        output_dim,
    ) = load_partition_data_HAR(batch_size,dataset_dir)

    train_loaders = []
    test_loaders = []

    test_data_global = HAR_loader(test_data_global)

    for loader_idx in range(client_num):
        train_loaders.append(HAR_loader(train_data_local_dict[loader_idx]))
        test_loaders.append(HAR_loader(test_data_local_dict[loader_idx]))

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_loaders,
        test_loaders,
        output_dim,
    )

























