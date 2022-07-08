import copy
import logging
import time

import numpy as np
import wandb

import torch

from .utils import transform_list_to_tensor


class FedHDAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                 worker_num, device, args, model_trainer):

        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.classifier = model_trainer
        self.worker_num = worker_num
        self.device = device
        self.args = args
        
        
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        
        
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.classifier.get_model_params()
    
    def set_global_model_params(self, model_parameters):
        self.classifier.set_model_params(model_parameters)



    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True




    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True




    def aggregate(self):
        
        num = len(self.model_dict)
        
        updated_params = self.classifier.get_model_params()
        #updated_params = torch.zeros(10,self.args.D)
        
        for model_idx in self.model_dict:
            updated_params += self.model_dict[model_idx]
            
        updated_params = updated_params / num
        
        self.set_global_model_params(updated_params)
        return updated_params


    '''
    # for local test only
    def aggregate(self,client_model_list):
        num = len(client_model_list)
        updated_params = self.classifier.get_model_params()
        for model in client_model_list:
            updated_params += model
        updated_params /= num
        self.set_global_model_params(updated_params)
        return updated_params
    '''



    def test_on_server_for_all_clients(self,round_idx,batch_selection=None):
        print("Round: ", round_idx)
        accuracy = self.classifier.test(self.test_global, self.args, batch_selection)
        return accuracy















