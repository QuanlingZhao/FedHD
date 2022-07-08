import logging
import random
import numpy as np

from .MNIST.data_loader import load_mnist_data, get_dataloader_MNIST, get_dataloader_test_MNIST
from .FashionMNIST.data_loader import load_fashionmnist_data, get_dataloader_FashionMNIST, get_dataloader_test_FashionMNIST
from .cifar10.data_loader import load_cifar10_data, get_dataloader_CIFAR10, get_dataloader_test_CIFAR10

from .shakespeare.data_loader import get_shakespeare_dataloader
from .HAR.data_loader import get_HAR_dataloader
from .HPWREN.data_loader import get_HPWREN_dataloader


def uniform(N, k):
    """Uniform distribution of 'N' items into 'k' groups."""
    dist = []
    avg = int(N) / k
    # Make distribution
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))
    # Return shuffled distribution
    random.shuffle(dist)
    return dist


def normal(N, k):
    """Normal distribution of 'N' items into 'k' groups."""
    dist = []
    # Make distribution
    for i in range(k):
        x = i - (k - 1) / 2
        dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))
    # Add remainders
    remainder = N - sum(dist)
    dist = list(np.add(dist, uniform(remainder, k)))
    # Return non-shuffled distribution
    return dist


class Loader(object):
    """Load and pass IID data partitions."""

    def __init__(self, X_train, y_train, X_test, y_test):
        # Get data from generator
        self.labels = list(np.sort(np.unique(y_test)))
        self.trainset = {}
        for lb in self.labels:
            self.trainset[lb] = list(np.where(y_train == lb)[0])
            np.random.shuffle(self.trainset[lb])
        self.trainset_size = X_train.shape[0]

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []

    def extract(self, label, n):
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: {}'.format(label))
            logging.warning('Dumping used data for reuse')

            # Unmark data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

            # Extract replenished data
            return self.extract(label, n)

    def get_partition(self, partition_size):
        # Get an partition uniform across all labels

        # Use uniform distribution
        dist = uniform(partition_size, len(self.labels))
        logging.info('Label distribution on client:' + str(dist))

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_testset(self):
        # Return the entire testset
        return self.testset


class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, pref, alpha, secondary):
        # Get a non-uniform partition with a preference bias
        # Calculate sizes of majority and minority portions
        # The majority portion will include samples of only one class (pref)
        majority = int(partition_size * alpha)
        minority = partition_size - majority

        # Calculate number of minor labels
        len_minor_labels = len(self.labels) - 1

        if secondary:
            # Distribute to random secondary label
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # Distribute among all minority labels
            dist = uniform(minority, len_minor_labels)

        # Add majority data to distribution
        dist.insert(self.labels.index(pref), majority)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition


class NonIIDLoader(Loader):
    """Load and pass 'shard' data partitions."""

    def get_partition(self, partition_size, cls_num):
        # Get a non-uniform partition with a random number of classes for this client

        # Extract number of classes configuration
        cls_list = np.random.choice(np.arange(len(self.labels)), cls_num,
                                              replace=False)

        dist = [0] * len(self.labels)
        avg = partition_size / cls_num
        for i, c in enumerate(cls_list):
            dist[c] = int((i + 1) * avg) - int(i * avg)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def partition_data(dataset, datadir, partition_method, partition_label,
                   partition_alpha, partition_secondary,
                   partition_min_cls, partition_max_cls,
                   n_clients, data_size_per_client):
    """
    Partition data to IID or non-IID distribution on each client
    Assign a static dataset to each client
    Args:
        dataset: str, which dataset to use
        datadir: str, path to the directory of downloaded dataset
        partition_method: str, iid or non-iid
        partition_label: str, how to partition label among clients in non-iid, uniform or normal
        partition_alpha: float between 0 and 1, used in bias loader
            the ratio of majority of classes on one client in non-iid
        partition_secondary: True of False, used in bias loader
            whether to sample the minority sample from one class
            or uniformly from the rest classes in non-iid
        partition_min_cls: used in noniid loader, the min number of classes on one client
        partition_max_cls: used in noniid loader, the max number of classes on one client
        n_clients: int, the total number of clients
        data_size_per_client: int, the total number of samples on each client
    Return:
        X_train, y_train: the whole training dataset of samples and labels
        X_test, y_test: the whole testing dataset of samples and labels
        net_dataidx_map: a dictionary recording training sample indexes on each client
        traindata_cls_counts: a dictionary recording the number of training samples
            from each class on each client
    """
    logging.info("*********partition data***************")
    if dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == "fashionmnist":
        X_train, y_train, X_test, y_test = load_fashionmnist_data(datadir)
    elif dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    else:
        raise ValueError("dataset {} not supported!".format(dataset))

    if partition_method == "iid": # IID data distribution
        loader = Loader(X_train, y_train, X_test, y_test)
        net_dataidx_map = {}

        for j in range(n_clients):
            data_idx = loader.get_partition(data_size_per_client)
            net_dataidx_map[j] = data_idx

    elif partition_method == "bias":  # bias data distribution
        loader = BiasLoader(X_train, y_train, X_test, y_test)
        labels = list(np.sort(np.unique(y_test)))
        net_dataidx_map = {}

        # Create distribution for label preferences if non-IID
        dist = {
            "uniform": uniform(n_clients, len(labels)),
            "normal": normal(n_clients, len(labels))
        }[partition_label]
        random.shuffle(dist)  # Shuffle distribution
        logging.info('Label distribution:  %s' % str(dist))

        # Distribute bias data to each client and store in net_dataidx_map
        for j in range(n_clients):
            major_label = random.choices(labels, dist)[0]
            data_idx = loader.get_partition(data_size_per_client, major_label,
                                            partition_alpha, partition_secondary)
            net_dataidx_map[j] = data_idx

    elif partition_method == 'noniid':  # noniid data distribution
        loader = NonIIDLoader(X_train, y_train, X_test, y_test)
        partition_cls = np.random.randint(partition_min_cls,
                                          partition_max_cls,
                                          n_clients)

        net_dataidx_map = {}
        for j in range(n_clients):
            data_idx = loader.get_partition(data_size_per_client,
                                            partition_cls[j])
            net_dataidx_map[j] = data_idx

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset == "mnist":
        dataloader = get_dataloader_MNIST(datadir, train_bs, test_bs, dataidxs)
    elif dataset == "fashionmnist":
        dataloader = get_dataloader_FashionMNIST(datadir, train_bs, test_bs, dataidxs)
    elif dataset == "cifar10":
        dataloader = get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)
    else:
        raise ValueError("dataset {} not supported!".format(dataset))
    return dataloader


def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    if dataset == "mnist":
        testloader = get_dataloader_test_MNIST(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    elif dataset == "fashionmnist":
        testloader = get_dataloader_test_FashionMNIST(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    elif dataset == "cifar10":
        testloader = get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)
    else:
        raise ValueError("dataset {} not supported!".format(dataset))
    return testloader


def load_partition_data_shakespeare(batch_size,dataset_dir):

    logging.info("Loading shakespeare - "+dataset_dir)
    
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_loaders,
        test_loaders,
        output_dim,
    ) = get_shakespeare_dataloader(batch_size,dataset_dir)
        
    train_data_local_dict = train_loaders
    test_data_local_dict = test_loaders
    class_num = output_dim
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_data_HAR(batch_size,dataset_dir):

    logging.info("Loading HAR - "+dataset_dir)

    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_loaders,
        test_loaders,
        output_dim,
    ) = get_HAR_dataloader(batch_size,dataset_dir)

    train_data_local_dict = train_loaders
    test_data_local_dict = test_loaders
    class_num = output_dim
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num




def load_partition_data_HPWREN(batch_size,dataset_dir):

    logging.info("Loading HPWREN - "+dataset_dir)

    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_loaders,
        test_loaders,
        output_dim,
    ) = get_HPWREN_dataloader(batch_size,dataset_dir)

    train_data_local_dict = train_loaders
    test_data_local_dict = test_loaders
    class_num = output_dim
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
        
    
    



def load_partition_data(dataset, data_dir, partition_method, partition_label,
                        partition_alpha, partition_secondary,
                        partition_min_cls, partition_max_cls,
                        client_number, batch_size, data_size_per_client):
    # Partition the data
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = \
        partition_data(dataset, data_dir, partition_method, partition_label,
                       partition_alpha, partition_secondary,
                       partition_min_cls, partition_max_cls,
                       client_number, data_size_per_client)

    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # local training and testing dataset
        # train_data_local: data_size_per_client in total, each batch is of batch_size
        # test_data_local: the total test dataset, each batch is of batch_size
        train_data_local, test_data_local = get_dataloader(dataset, data_dir,
                                                           batch_size, batch_size,
                                                           dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
           
           
           
           
           
           
