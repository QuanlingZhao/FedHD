#!/usr/bin/env python
# coding: utf-8

# This notebook helps you query raw data from HPWREN, and reorganized queried data into csv files.
# The ultimate output consists of one csv file for each location in each year.
# 
# The data is available to public at http://hpwren.ucsd.edu/TM/Sensors/Data/

# In[1]:


import requests
from pprint import pprint #import pretty print
from bs4 import BeautifulSoup
import re
import csv
import datetime
import numpy as np
import os
import shutil


###############################
# Necessary functions
###############################
def get_sub_dir(url, pattern):
    '''
    Get a list of url of all the subdirectories under the given url
    that satisfy the given patterns
    '''
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    sub_dirs = [node.get('href') for node in soup.find_all('a')]
    sub_dirs = [node.strip('./') for node in sub_dirs] # filter out irrelevant characters

    # filter list with pattern
    # print(sub_dirs)
    # print(pattern)
    sub_dirs = [node for node in sub_dirs if (re.search(pattern, node) is not None)]
    return [node for node in sub_dirs]

def align_data(header, reading, value):
    '''
    Align the reading-value with the provided header, adding nans to non-appear headers
    '''
    fileData = []
    for i in range(len(header)):
        try: # try to find the index of current header in the reading list
            idx = reading.index(header[i])
            fileData.append(value[idx])
        except Exception as e: # current header not in the reading list
            print('Error: {}'.format(e))
            print('Current header {} not in the list of {}'.format(header[i], reading))
            fileData.append(float("nan"))
    return fileData


def parse_file(fileURL, header):
    '''
    Parse the time and data from file url
    Only take the data from the specified header
    '''
    from itertools import compress
    fileData = []
    page = requests.get(fileURL).text
    samples = page.strip().split('\n')
    fileTime = [int(sample.split('\t')[2]) for sample in samples]
    startTime = fileTime[0]
    fileTime = [t - startTime for t in fileTime]
    dataString = [sample.split('\t')[3] for sample in samples]
    for idx in range(len(dataString)):
        dataPhrase = dataString[idx].strip('0R0,').split(',')

        reading, value = [], []
        for ele in dataPhrase:
            try:
                record = ele.split('=')
                # extract headers in the record
                reading.append(record[0])
                # extract values in the record
                value.append(float(re.findall(r'[\d.-]*', record[1])[0]))
            except Exception as e:
                print('Error: {}'.format(e))
                print('At time {}'.format(fileTime[idx] + startTime))
                print('Original phrase: {}'.format(dataPhrase))
                print('Original record: {}'.format(record))

        reading.insert(0, 't') # insert time header at the beginning
        value.insert(0, fileTime[idx]) # insert time stamp at the beginning
        fileData.append(align_data(header, reading, value))

    return fileData


def write_csv(fileName, header, fileData):
    '''
    Write time and data to specified csv file
    '''
    with open(fileName, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv, delimiter=',')
        writer.writerow(header)
        for idx in range(len(fileData)):
            writer.writerow(fileData[idx])


# 1. Start querying data and store the raw data with hierarchy of directories:
# year - location - each day.csv

# In[2]:


###############################
# Start query
###############################
baseURL = "http://hpwren.ucsd.edu/TM/Sensors/Data/"
year = ['2021'] # year of data to request

filePattern = ":0R0:4:0"
header = ['t', 'Dn', 'Dm', 'Dx', 'Sn', 'Sm', 'Sx', 'Ta', 'Ua', 'Pa', 'Rc', 'Rd', 'Ri'] # type of sensor readings to request

for y in year:
    # remove existing folder (if exists) and create a new folder for the given year
    if os.path.exists(y):
        shutil.rmtree(y)
    os.makedirs(y)
    
    # query days from the year's url
    yearURL = baseURL + y + '/'
    dayPattern = y + '\d*'
    days = get_sub_dir(yearURL, dayPattern)
    if not len(days):
        print('yearURL {} is empty!'.format(yearURL))
        continue
    
    # query the files for each day
    for d in days:
        dayURL = yearURL + d + '/'
        files = get_sub_dir(dayURL, filePattern)
        if not len(files):
            print('dayURL {} is empty!'.format(dayURL))
            continue
            
        # query the data in each file
        for f in files:
            fileURL = dayURL + f
            print('querying file {}'.format(fileURL))
            fileData = parse_file(fileURL, header)
            
            # create a folder for the location if first-time appears
            loc = f.split(':')[1].split('-')[0]
            if not os.path.exists(y + '/' + loc):
                os.makedirs(y + '/' + loc)
                
            # write queried data to csv
            filepath = '{}/{}/{}.csv'.format(y, loc, d)
            write_csv(filepath, header, fileData)


# 2. Average and reorganze data. Store the formatted data under the year's
# directory, with the readings of each location in the corresponding year being in one csv file.

# In[11]:


###############################
# Necessary functions
###############################
# the superset of headers
all_header = ['year', 'month', 'day', 'hour', 'Dn', 'Dm', 'Dx', 'Sn',
              'Sm', 'Sx', 'Ta', 'Ua', 'Pa', 'Rc', 'Rd', 'Ri']

def read_file(file_path, interval, day_sample_num):
    """
    Read from one csv file

    Args:
        file_path: the name of the csv file to read from
        interval: number of minutes per sample
        day_sample_num: number of samples per day
    Returns:
        header: header of this file, list
        data: dictionary of export data, list
    """
    print('reading file {}'.format(file_path))
    data = {}
    with open(file_path, 'r', newline='') as incsv:
        reader = csv.reader(incsv, delimiter=',')
        header = next(reader)
        # add headers as keys in data dict
        for h in header:
            # init the readings to nan
            data[h] = np.empty(day_sample_num)
            data[h][:] = np.nan

        data_stack = []
        time_stamp = 0.0 # record the current time stamp for data_stack
        for row in reader:
            new_data = [float(d) for d in row]
            cur_time = new_data[0]
            # if the time exceeds one single day, the data becomes invalid
            if cur_time > 24*3600:
                break
            # append existing data_stack to dict if an interval is loaded
            if cur_time >= time_stamp + interval*60:
                # calculate the index to put this averaged data based on the time stamp
                idx = int(time_stamp/60/interval)
                data_stack = np.around(np.mean(np.array(data_stack), axis=0), 2)
                data_stack[0] = time_stamp # fill in the first time stamp
                # print('time stamp: {}'.format(time_stamp))
                for i in range(len(header)):
                    data[header[i]][idx] = data_stack[i]
                data_stack = []
                time_stamp += interval*60 # add seconds equivalent to interval in minutes

            # append the new data to the data_stack
            data_stack.append(new_data)

        # process the last chunk of data if not processed
        if len(data_stack) > 0 and time_stamp < 24*3600:
            idx = int(time_stamp/60/interval)
            data_stack = np.around(np.mean(np.array(data_stack), axis=0), 2)
            data_stack[0] = time_stamp
            for i in range(len(header)):
                data[header[i]][idx] = data_stack[i]

    # validation check on length of data
    for h in data:
        assert(data[h].shape[0] == day_sample_num), 'Incorrect length for ' \
            'file {} header {}: should be {} but is {}'.format(file_path, h,
            day_sample_num, data[h].shape[0])

    return header, data


def write_data_csv(file_path, header, data):
    """
    Write the aggregated data of a specific location in a specific year
    to one csv file

    Args:
        data: the aggregated dictionary
    """
    # if file already exists, remove it
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv, delimiter=',')
        writer.writerow(header)
        for idx in range(data['year'].shape[0]):
            # reorganize the data from each header into one row
            row = []
            for i in range(len(header)):
                row.append(data[header[i]][idx])
            #print(row)
            writer.writerow(row)


def daterange(start_date, end_date):
    """Date generator in a given range"""
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


# In[12]:


###############################
# Start reorganization
###############################
# basic information during data extraction
for y in year:
    dt_1st_day = datetime.datetime.strptime(y + '0101', '%Y%m%d')
    dt_last_day = datetime.datetime.strptime(y + '1231', '%Y%m%d')
    days_in_year = (dt_last_day - dt_1st_day).days + 1 # number of days in year
    interval = 60 # number of minutes per sample after averaging
    day_sample_num = int(24*60/interval)
    year_sample_num = int(days_in_year*24*60/interval)
    print('samples per day: {}'.format(day_sample_num))
    print('samples per year: {}'.format(year_sample_num))

    # preparation
    data = {}
    loc_list = [d for d in os.listdir(y) if os.path.isdir(os.path.join(y, d))]
    print(loc_list)

    for loc in loc_list:
        # add location as the first layer of keys in the data dict
        data[loc] = {}
        for h in all_header:
            # init the readings to nan
            data[loc][h] = np.empty(year_sample_num)
            data[loc][h][:] = np.nan

        # read the file of each day in an ascending order
        loc_path = os.path.join(y, loc)
        for single_date in daterange(dt_1st_day,
                                     dt_last_day+datetime.timedelta(1)):
            filename = single_date.strftime("%Y%m%d") + '.csv'
            file_path = os.path.join(loc_path, filename)
            yy, mm, dd = single_date.year, single_date.month, single_date.day

            # find the start and end index of this new data
            delta_days = (single_date - dt_1st_day).days
            st_idx = delta_days * day_sample_num
            ed_idx = (delta_days + 1) * day_sample_num
            # fill in year, month, day, hour into all slots associated with
            # this file
            data[loc]['year'][st_idx:ed_idx] = np.repeat(yy, day_sample_num)
            data[loc]['month'][st_idx:ed_idx] = np.repeat(mm, day_sample_num)
            data[loc]['day'][st_idx:ed_idx] = np.repeat(dd, day_sample_num)
            data[loc]['hour'][st_idx:ed_idx] = np.arange(0, 24, interval / 60)

            # do not try reading the file if file does not exist
            if not os.path.exists(file_path):
                continue

            # read from the csv file if file exists
            new_header, new_data = read_file(file_path, interval, day_sample_num)
            # pprint(new_data)

            # fill in the new data to corresponding header
            for h in new_header:
                if h != 't':
                    data[loc][h][st_idx:ed_idx] = new_data[h] # join two arrays

        # validation check on length of data
        for h in data[loc]:
            assert(data[loc][h].shape[0] == year_sample_num), \
                'Incorrect length for location {} header {}: should be {} ' \
                'but is {}'.format(loc, h, year_sample_num, data[loc][h].shape[0])

        # write the aggregated data of a location in a year to one csv file
        file_path = os.path.join(y, '{}.csv'.format(loc))
        write_data_csv(file_path, all_header, data[loc])
