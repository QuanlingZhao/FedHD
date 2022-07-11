# random proj

import argparse
import logging
import os
import sys
import time

import numpy as np
import copy

import torch
import torch.nn as nn
import torch_hd.hdlayers as hd
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.functional import accuracy
import torchvision.transforms as transforms

from FedML.fedml_api.data_preprocessing.load_data import load_partition_data
from FedML.fedml_api.data_preprocessing.load_data import load_partition_data_shakespeare
from FedML.fedml_api.data_preprocessing.load_data import load_partition_data_HAR
from FedML.fedml_api.data_preprocessing.load_data import load_partition_data_HPWREN


from FedML.fedml_api.distributed.fedhd.fedhd_ModelTrainer import MyModelTrainer
from FedML.fedml_api.distributed.fedhd.fedhdAggregator import FedHDAggregator

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle


from utils import *


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument('--D', type=int, default=10000,
                            choices=[10000],
                            help='dataset used for training')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'shakespeare','har','hpwren'],
                        help='dataset used for training')


    parser.add_argument('--partition_method', type=str, default='iid',
                        choices=['iid', 'noniid'],
                        help='how to partition the dataset on local clients')


    parser.add_argument('--client_num_in_total', type=int, default=30,
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=30,
                        help='number of workers')


    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epochs', type=int, default=1,
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=20,
                        help='how many round of communications we should use')


    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--partition_min_cls', type=int, default=1,
                            help='the min number of classes on each client used in noniid loader')

    parser.add_argument('--partition_max_cls', type=int, default=5,
                            help='the max number of classes on each client used in noniid loader')

    # Communication settings
    parser.add_argument('--backend', type=str, default='MQTT',
                        choices=['MQTT', 'MPI'],
                        help='communication backend')


    parser.add_argument('--mqtt_host', type=str, default='127.0.0.1',
                        help='host IP in MQTT')


    parser.add_argument('--mqtt_port', type=int, default=1883,
                        help='host port in MQTT')



    #
    parser.add_argument('--partition_alpha', type=float, default=0.5,
                        help='partition alpha (default: 0.5), used as the proportion'
                             'of majority labels in non-iid in latest implementation')

    parser.add_argument('--partition_secondary', type=bool, default=False,
                        help='True to sample minority labels from one random secondary class,'
                             'False to sample minority labels uniformly from the rest classes except the majority one')

    parser.add_argument('--partition_label', type=str, default='uniform',
                        choices=['uniform', 'normal'],
                        help='how to assign labels to clients in non-iid data distribution')

    parser.add_argument('--data_size_per_client', type=int, default=500,
                        help='input batch size for training (default: 64)')



    parser.add_argument('--rate', type=float, default=0.5,
                        help='HD learning rate')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='sgd optimizer momentum 0.9')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='weight_decay (default: 5e-4)')

    parser.add_argument('--method', type=str, default="fedsync",
                         choices=['fedsync', 'fedasync'],
                         help='fedmethod')


    args = parser.parse_args()
    return args







def load_data(args, dataset_name):
    if dataset_name == "shakespeare":
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size,"../FedML/data/shakespeare")
        #args.client_num_in_total = len(train_data_local_dict)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))

    elif dataset_name == "har":
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_HAR(args.batch_size,"FedML/data/HAR")
        #args.client_num_in_total = len(train_data_local_dict)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))


    elif dataset_name == "hpwren":
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_HPWREN(args.batch_size,"../FedML/data/HPWREN")
        #args.client_num_in_total = len(train_data_local_dict)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))


    elif dataset_name == "mnist" or dataset_name == "fashionmnist" or \
        dataset_name == "cifar10":
        data_loader = load_partition_data
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        data_dir = './../data/' + args.dataset
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, data_dir, args.partition_method,
                                args.partition_label, args.partition_alpha, args.partition_secondary,
                                args.partition_min_cls, args.partition_max_cls,
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
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)

    run_name = args.dataset + "_c_" + str(args.comm_round) + "_s_" + str(args.data_size_per_client) + "(hd_RandomProj)"
    log_path = "hist/" + run_name

    print("Logging to: " + log_path)

    os.mkdir(log_path)

    parameters_file = log_path + "/Parameters.txt"
    logging_file = log_path + "/log.txt"
    graph_file = log_path + "/graph.jpg"

    parameters_log = open(parameters_file, "w")
    parameters_log.write(str(args))
    parameters_log.close()

    info_log = open(logging_file, "w")


    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    #simclr
    '''
    cifar_feature_extractor = SimCLR.load_from_checkpoint(
        "epoch=960.ckpt", strict=False, dataset='imagenet', maxpool1=False, first_conv=False, input_height=28)
    cifar_feature_extractor.freeze()
    print("SimCLr module loaded")
    '''
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")



    if args.dataset == "mnist" or args.dataset == "fashionmnist":
        projector = hd.RandomProjectionEncoder(28*28, args.D)
        projector.load_state_dict(torch.load("mnist_fashionmnist_encoder.ckpt"))
        encoder = nn.Sequential(projector)
    elif args.dataset == "har":
        projector = hd.RandomProjectionEncoder(561, args.D)
        projector.load_state_dict(torch.load("har_encoder.ckpt"))
        encoder = nn.Sequential(projector)
    else:
        print("net implemented")
        exit(1)
    classifier = hd.HDClassifier(10, args.D)
    model = (encoder, classifier)



    # test batch idx selection
    test_batch_selection = []
    for i in range(5):
        test_batch_selection.append(i)

    clients = []
    for i in range(args.client_num_in_total):
        clients.append(MyModelTrainer(model,args,device))
        clients[i].set_id(i)

    
    global_model = MyModelTrainer(model,args,device)
    global_model.set_id("Server")
    global_model_aggregator = FedHDAggregator(train_data_global, test_data_global, train_data_num,
                                      train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                      args.client_num_per_round, device, args, global_model)

    print("init complete, training start")
    sd = 0

    start_time = time.time()


    acc_over_rounds=[]
    for c in range(args.comm_round):
        print("\n\n"+"Round#"+str(c))
        print("===================RoundStart====================")
        for i in range(len(clients)):
            print("++++++++++++++++++TrainClient: "+str(i))
            clients[i].train(train_data_local_dict[i],args)

            print("Add noise to hypervecs:")
            noisy_class_hypervecs = add_noise_to_vecs(clients[i].get_model_params(),sd=sd)
            print("Added")

            global_model_aggregator.add_local_trained_result(clients[i].get_id(), clients[i].get_model_params(), len(train_data_local_dict[i]))
        print("+++++++++++++++++++Aggregating")
        global_model_aggregator.aggregate()
        print("+++++++++++++++++++distribute")

        for i in range(len(clients)):
            clients[i].set_model_params(global_model.get_model_params())
        print("+++++++++++++++++++Test")
        if c % args.frequency_of_the_test == 0:
            acc = global_model_aggregator.test_on_server_for_all_clients(c,test_batch_selection).item()
            print("Round acc: "+str(acc))
            acc_over_rounds.append(acc)
            round_time = time.time()-start_time
            info_log.write(str(round_time)+","+str(acc)+"\n")
        else:
            print("skipped")
        print("===================RoundComplete====================")


    duration = time.time()-start_time
    print("duration: " + str(duration))
    info_log.close()

    print("-----Processing Graph-----")

    df = pd.read_csv(logging_file, sep=",", header=None,names=["Round", "Acc"])
    graph = sns.lineplot(x = "Round", y = "Acc", data = df).get_figure()
    graph.savefig(graph_file)


    print("Final Report:")
    print(acc_over_rounds)


    print("-----Finished-----")
    print("Training Time: ", duration)




'''
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)

    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

'''
'''
    if args.dataset == "mnist":
        input_length = 28*28
    elif args.dataset == "fashionmnist":
        input_length = 28*28
    elif args.dataset == "har":
            input_length = 561
    else:
        print("unimplemented")
        exit(1)

    # base vector, matrix 1x1000 1000x(single sample)
    base_matrix = []
    mu = 0.0
    sigma = 1.0
    for i in range(0, args.D):
    	base_matrix.append(np.random.normal(mu, sigma, input_length))
    base_vector = np.random.uniform(0, 2*math.pi, args.D)

    print("Input Length: " + str(input_length))
    print("Base vector: ")
    print(base_vector.shape)
    print("Base metrix: ")
    print(len(base_matrix))
    print(len(base_matrix[0]))
    # global, all client are the same

    with open('base_matrix_har_6_26.hd', 'wb') as file:
      pickle.dump(base_matrix, file)
    with open('base_vector_har_6_26.hd', 'wb') as file:
          pickle.dump(base_vector, file)
'''
'''


    # load base matrix / vector
    if args.dataset == "mnist" or args.dataset == "fashionmnist":
        with open('base/base_matrix_fmnist_mnist_6_26.hd', 'rb') as file:
            base_matrix = pickle.load(file)
        with open('base/base_vector_fmnist_mnist_6_26.hd', 'rb') as file:
            base_vector = pickle.load(file)
        print("28x28 loaded")
    elif args.dataset == "har":
        with open('base/base_matrix_har_6_26.hd', 'rb') as file:
            base_matrix = pickle.load(file)
        with open('base/base_vector_har_6_26.hd', 'rb') as file:
            base_vector = pickle.load(file)
        print("561 loaded")
    else:
        print("Unimplemented")
        exit(1)


    # test batch idx selection
    test_batch_selection = []
    for i in range(50):
        test_batch_selection.append(i)


    # below is train
    n_classes = 10
    class_hvs = np.zeros((n_classes, args.D))
    class_hvs_best = deepcopy(class_hvs)
    acc_max = -1

    for epoch in range(args.epochs):
        class_hvs = deepcopy(class_hvs_best)
        for batch_idx, (x, y) in enumerate(train_data_local_dict[0]):
            # encode x
            train_enc_hvs = encoding_nonlinear(x, base_matrix, base_vector)
            pickList = np.arange(0, len(train_enc_hvs))
            np.random.shuffle(pickList)
            for i in pickList:
                predict = max_match_nonlinear(class_hvs, train_enc_hvs[i])
                if predict != y[i]:
                    class_hvs[predict] = class_hvs[predict] - args.rate*train_enc_hvs[i]
                    class_hvs[y[i]]    = class_hvs[y[i]] + args.rate*train_enc_hvs[i]

        correct = 0
        for batch_idx, (x, y) in enumerate(test_data_local_dict[0]):
            # encode x
            if batch_idx >= (len(test_data_local_dict[0]) / args.batch_size):
                break
            test_enc_hvs = encoding_nonlinear(x, base_matrix, base_vector)
            for i in range(len(test_enc_hvs)):
                predict = max_match_nonlinear(class_hvs, test_enc_hvs[i])
                if predict == y[i]:
                    correct+=1

        print(correct)
        acc = correct/len(test_data_local_dict[0])
        print("[ Epoch {e} Acc {a} ]".format(e = epoch, a = acc))
        if acc > acc_max:
            print("Updating")
            class_hvs_best = deepcopy(class_hvs)
            acc_max = acc
        else:
            print("No update")



    print("training complete")
    print("testing start")

    correct = 0
    tested = 0
    for batch_idx, (x, y) in enumerate(test_data_global):
        if batch_idx not in test_batch_selection:
            continue
        test_enc_hvs = encoding_nonlinear(x, base_matrix, base_vector)
        for i in range(len(x)):
            predict = max_match_nonlinear(class_hvs_best, test_enc_hvs[i])
            tested+=1
            if predict==y[i]:
                correct+=1

    test_acc = correct/tested

    print("Final Acc: " + str(test_acc))
'''



































