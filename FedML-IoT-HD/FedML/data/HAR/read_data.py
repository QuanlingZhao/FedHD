#!/usr/bin/env python
# coding: utf-8

# In[63]:


import glob
import os
import json
import numpy as np

def z_norm(x):
    mean = np.nanmean(x, axis=0)
    var = np.nanstd(x, axis=0)
    return mean, var

def read_data(folder):
    X, y, subject = [], [], []

    X_file = glob.glob(os.path.join(folder, 'X_*.txt'))[0]
    with open(X_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            data = [float(d) for d in data]
            X.append(data)

    y_file = glob.glob(os.path.join(folder, 'y_*.txt'))[0]
    with open(y_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = int(line)
            y.append(label)

    subject_file = glob.glob(os.path.join(folder, 'subject_*.txt'))[0]
    with open(subject_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            subject.append(line.strip())

    return np.array(X), np.array(y).reshape((-1, 1)), np.array(subject).reshape((-1, 1))


def construct_dict(X, y, subject):
    # Convert from numpy array to list
    X = X.tolist()
    y = y.tolist()
    subject = subject.tolist()

    data = {}
    data['users'] = []
    data['user_data'] = {}
    data['num_samples'] = []

    for i in range(len(subject)):
        if subject[i] not in data['users']:
            data['users'].append(subject[i])
            data['user_data'][subject[i]] = {'x': [], 'y': []}

        user_idx = data['users'].index(subject[i])
        data['user_data'][subject[i]]['x'].append(X[i])
        data['user_data'][subject[i]]['y'].append(y[i])

    for i in range(len(data['users'])):
        user_name = data['users'][i]
        data['num_samples'].append(len(data['user_data'][user_name]['x']))

    return data


def save_data(data_dict, file_name):
    # Save the data dictionary with 'users', 'user_data' and 'num_samples' to json file
    with open(file_name, 'w') as fp:
        json.dump(data_dict, fp,  indent=4)


# In[64]:


X_train, y_train, subject_train = read_data('./UCI HAR Dataset/train')
print(X_train.shape, y_train.shape, subject_train.shape)

X_test, y_test, subject_test = read_data('./UCI HAR Dataset/test')
print(X_test.shape, y_test.shape, subject_test.shape)


# Because the original dataset partitions train and test dataset using diff users
# We here combine the original train and test data
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0).reshape((-1))
subject = np.concatenate((subject_train, subject_test), axis=0).reshape((-1))

print('Shape of data: ', X.shape, y.shape, subject.shape)

mean, var =  z_norm(X)
#print('Mean: {} Var: {}'.format(mean, var))

X = (X - mean) / var

data_dict = construct_dict(X, y, subject)


# We here load all data in the original train and test folders, and split it with tf portion for training
# and (1-tf) portion for testing
print('Users are: ', data_dict['user_data'].keys())
print('Num_samples are: ', data_dict['num_samples'])


# In[65]:


import copy
import argparse

# Randomly split the samples from each user as train and test portion
# tf portion for training and (1-tf) portion for testing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=float, default=0.9,
                        help='fraction of data in the training set, written as a decimal.')
    args = parser.parse_args()

    train_data_dict = copy.deepcopy(data_dict)
    test_data_dict = copy.deepcopy(data_dict)
    for user in data_dict['user_data']:
        total_len = len(data_dict['user_data'][user]['x'])
        select = np.random.choice(total_len, int(args.tf*total_len), replace=False)
        select.sort()
        not_select = list(set(np.arange(total_len)) - set(select))
        not_select.sort()
        train_data_dict['user_data'][user] = \
            {'x': [data_dict['user_data'][user]['x'][s] for s in select],
            'y': [data_dict['user_data'][user]['y'][s] for s in select]}
        test_data_dict['user_data'][user] = \
            {'x': [data_dict['user_data'][user]['x'][s] for s in not_select],
            'y': [data_dict['user_data'][user]['y'][s] for s in not_select]}

        user_idx = data_dict['users'].index(user)
        train_data_dict['num_samples'][user_idx] = len(train_data_dict['user_data'][user]['x'])
        test_data_dict['num_samples'][user_idx] = len(test_data_dict['user_data'][user]['x'])

    print('train users are: ', train_data_dict['user_data'].keys())
    print('train num_samples are: ', train_data_dict['num_samples'])

    print('test users are: ', test_data_dict['user_data'].keys())
    print('test num_samples are: ', test_data_dict['num_samples'])

    # Save data to json
    save_data(train_data_dict, 'train.json')
    save_data(test_data_dict, 'test.json')
    print('Data saved to json')


if __name__ == "__main__":
    main()
