#!/usr/bin/env python
# coding: utf-8

# In[22]:


import glob
import os
import pandas as pd
import json
import copy
import numpy as np
import argparse

FEATURES = ['Dn', 'Dm', 'Dx', 'Sn', 'Sm', 'Sx', 'Ta', 'Ua', 'Pa', 'Rc', 'Rd', 'Ri']

def z_norm(x):
    mean = np.nanmean(x, axis=0)
    var = np.nanstd(x, axis=0)
    return mean, var

# Compute mean and var
folder = './2021'
data = None
for file_name in glob.glob(os.path.join(folder, '*.csv')):
    new_df = pd.read_csv(file_name)
    new_data = np.array(new_df.dropna().values[:, 4:])  # Get rid of data info
    #print(new_data.shape)
    if data is None:
        data = new_data
    else:
        data = np.concatenate((data, new_data), axis=0)

print('Shape of all data in {}: {}'.format(folder, data.shape))
mean, var =  z_norm(data)
print('Mean: {} Var: {}'.format(mean, var))


def sliding_window(data, win_size=24):
    train_data = []
    for t in range(win_size, len(data)):
        train_data.append(data[t-win_size:t])
    train_data = np.stack(train_data)
    return train_data


def read_data(file_name, win_size=24):
    # Read raw data and preprocess
    df = pd.read_csv(file_name)
    data = np.array(df.dropna().values[:,4:]) # shape = (T,12)
    data = (data - mean) / var

    # Insufficient data, return None
    if data.shape[0] <= win_size:
        return None, None

    # Construct training data with sliding window
    train_data = sliding_window(data, win_size)  # shape = (num_samples, win_size, 12)
    print(train_data.shape)

    # Return training X and y as np array
    return train_data[:, :-1, :], train_data[:, -1, :]


def save_data(data_dict, file_name):
    # Save the data dictionary with 'users', 'user_data' and 'num_samples' to json file
    with open(file_name, 'w') as fp:
        json.dump(data_dict, fp,  indent=4)


# In[23]:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=float, default=0.9,
                        help='fraction of training data, written as a decimal.')
    parser.add_argument('--ws', type=int, default=24,
                        help='window size of training data')
    args = parser.parse_args()

    folder = './2021'
    data_dict = {}
    data_dict['users'] = []
    data_dict['user_data'] = {}
    data_dict['num_samples'] = []

    for file_name in glob.glob(os.path.join(folder, '*.csv')):
        new_X, new_y = read_data(file_name)

        if new_X is None:  # Skip the location with invalid data
            continue
        user_name = file_name.split('/')[-1].split('.')[0]

        data_dict['users'].append(user_name)
        data_dict['user_data'][user_name] = {'x': new_X, 'y': new_y}
        data_dict['num_samples'].append(new_X.shape[0])

        print('user name: {} num samples: {}'.format(user_name, data_dict['num_samples'][-1]))


    train_data_dict = copy.deepcopy(data_dict)
    test_data_dict = copy.deepcopy(data_dict)
    for user in data_dict['user_data']:
        total_len = data_dict['user_data'][user]['x'].shape[0]
        test_len = int((1 - args.tf)*total_len)
        train_data_dict['user_data'][user] = \
            {'x': train_data_dict['user_data'][user]['x'][:-test_len].tolist(),
            'y': train_data_dict['user_data'][user]['y'][:-test_len].tolist()}
        test_data_dict['user_data'][user] = \
            {'x': test_data_dict['user_data'][user]['x'][-test_len:].tolist(),
            'y': test_data_dict['user_data'][user]['y'][-test_len:].tolist()}

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
