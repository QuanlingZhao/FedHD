import argparse
import logging
import os
import sys
import time

import numpy as np
import requests

import torch
import torch.nn as nn
import torch_hd.hdlayers as hd
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.functional import accuracy
import torchvision.transforms as transforms

from pl_bolts.models.self_supervised import SimCLR
# from cifarDataModule import CifarData

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from FedML.fedml_api.distributed.fedhd.fedhd_ModelTrainer import MyModelTrainer
from FedML.fedml_api.distributed.fedhd.fedhd_Trainer import fedHD_Trainer
from FedML.fedml_api.distributed.fedhd.fedhd_ClientManager import FedHDClientManager

from FedML.fedml_api.data_preprocessing.load_data import load_partition_data
from FedML.fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from FedML.fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from FedML.fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare



def add_args(parser):
    parser.add_argument('--server_ip', type=str, default="http://127.0.0.1:5000",
                        help='IP address of the FedML server')
    parser.add_argument('--client_uuid', type=str, default="0",
                        help='number of workers in a distributed cluster')
    args = parser.parse_args()
    return args


def register(args, uuid):
    str_device_UUID = uuid
    URL = args.server_ip + "api/register"

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'device_id': str_device_UUID}

    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS)
    print(r)
    result = r.json()
    client_ID = result['client_id']
    # executorId = result['executorId']
    # executorTopic = result['executorTopic']
    training_task_args = result['training_task_args']

    class Args:
        def __init__(self):
            self.dataset = training_task_args['dataset']
            self.data_dir = training_task_args['data_dir']
            self.partition_method = training_task_args['partition_method']
            self.partition_alpha = training_task_args['partition_alpha']
            self.partition_secondary = training_task_args['partition_secondary']
            self.partition_label = training_task_args['partition_label']
            self.data_size_per_client = training_task_args['data_size_per_client']
            self.D = training_task_args['D']
            self.client_num_per_round = training_task_args['client_num_per_round']
            self.client_num_in_total = training_task_args['client_num_in_total']
            self.comm_round = training_task_args['comm_round']
            self.epochs = training_task_args['epochs']
            self.lr = training_task_args['lr']
            self.batch_size = training_task_args['batch_size']
            self.frequency_of_the_test = training_task_args['frequency_of_the_test']
            self.backend = training_task_args['backend']
            self.mqtt_host = training_task_args['mqtt_host']
            self.mqtt_port = training_task_args['mqtt_port']

    args = Args()
    return client_ID, args


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


def load_data(args, dataset_name):
    if dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num
    elif dataset_name == "cifar100" or dataset_name == "cinic10":
        if dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100 # Not tested
        else: # cinic10
            data_loader = load_partition_data_cinic10 # Not tested

        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        data_dir = './../../../data/' + args.dataset
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total,
                                args.batch_size)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))
    elif dataset_name == "mnist" or dataset_name == "fashionmnist" or \
        dataset_name == "cifar10":
        data_loader = load_partition_data
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        data_dir = './../../../data/' + args.dataset
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, data_dir, args.partition_method,
                                args.partition_label, args.partition_alpha, args.partition_secondary,
                                args.client_num_in_total, args.batch_size,
                                args.data_size_per_client)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset




if __name__ == '__main__':
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    client_ID, args = register(main_args, uuid)
    logging.info("client_ID = " + str(client_ID))
    logging.info("dataset = " + str(args.dataset))
    # logging.info("model = " + str(args.model))
    logging.info("client_num_per_round = " + str(args.client_num_per_round))
    client_index = client_ID - 1


    logging.info("client_ID = %d, size = %d" % (client_ID, args.client_num_per_round))
    device = init_training_device(client_ID - 1, args.client_num_per_round - 1, 4)
    # device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    net = SimCLR.load_from_checkpoint(
    "epoch=960.ckpt", strict=False, dataset='imagenet', maxpool1=False, first_conv=False, input_height=28)
    net.freeze()

    hd_projector = hd.RandomProjectionEncoder(2048, args.D)
    hd_projector.load_state_dict(torch.load("encoder.ckpt"))

    encoder = nn.Sequential(
    net,
    hd_projector
    )

    classifier = hd.HDClassifier(10, args.D)

    model = (encoder, classifier)  
    
     
    model_trainer = MyModelTrainer(model,args,device)
    model_trainer.set_id(client_index)
    
    # trash
    device = torch.device('cpu')

    # start training
    trainer = fedHD_Trainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, train_data_num, device,
                            args, model_trainer)

    size = args.client_num_per_round + 1
    
    print("mqtt port: ", args.mqtt_port)
    
#     client_manager = FedHDClientManager(args, trainer, rank=client_ID, size=size,
#                                          backend="MQTT",
#                                          mqtt_host=args.mqtt_host,
#                                          mqtt_port=args.mqtt_port)
    
    client_manager = FedHDClientManager(args.mqtt_port, args.mqtt_host, args, trainer, rank=client_ID, size=size,backend="MQTT")
                                       

    client_manager.run()
    client_manager.start_training()

    time.sleep(100000)
