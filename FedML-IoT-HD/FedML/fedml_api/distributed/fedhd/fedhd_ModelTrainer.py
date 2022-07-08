import logging

import torch
from torch import nn
import copy
import numpy as np
from torchmetrics.functional import accuracy

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer




class MyModelTrainer(ModelTrainer):
    
    # assign initial model and encoder
    def __init__(self, model,args,device):
        self.encoder, self.classifier = copy.deepcopy(model)
        self.round = 0;
        self.device = device
        self.partition_method = args.partition_method

    # get hypervectors
    def get_model_params(self):
        return self.classifier.class_hvs.clone().to(self.device)

    # set hypervectors
    def set_model_params(self, model_parameters):
        self.classifier.class_hvs = nn.Parameter(model_parameters, requires_grad=False)
    
    def get_id(self):
        return self.id

    # train
    def train(self, train_data, args):

        self.classifier = self.classifier.to(self.device)
        encoder = self.encoder.to(self.device)
        
        self.classifier.train()

        for epoch in range(args.epochs):
            overall_acc = 0
            for batch_idx, (x, labels) in enumerate(train_data):
                print("TRAINING------------------", batch_idx)

                x = x.to(self.device).to(torch.float32)

                labels = labels.to(self.device).type(torch.int)
                x = encoder(x)

                labels_hat = self.classifier(x, labels)
                

                _, labels_hat = torch.max(labels_hat, dim=1)
                

                acc = accuracy(labels_hat,labels)
                

                overall_acc += acc

            overall_acc /= (batch_idx + 1)
            self.classifier.oneshot = True

        # if noniid, update lr
        self.round+=1
        if self.partition_method=="noniid":
            self.classifier.alpha = self.lr + (1 / (1 + self.round))

        self.classifier.oneshot=True
        print("\t Trainning round: {} \t=> client_id: {} accuracy: {}".format(self.round-1,self.id, overall_acc))



        
    
    # test
    def test(self, test_data, args, batch_selection):
        
        self.classifier = self.classifier.to(self.device)
        encoder = self.encoder.to(self.device)
        
        self.classifier.oneshot=True
        self.classifier.eval()

        overall_acc = 0
        for batch_idx, (x, target) in enumerate(test_data):
            if batch_selection!=None and batch_idx not in batch_selection:
                continue
	
            print("Testing: ",batch_idx)
            x = x.to(self.device).to(torch.float32)
            target = target.to(self.device)
            
            x = encoder(x)

            y_hat = self.classifier(x)
            
            _, y_hat = torch.max(y_hat, dim=1)
            
            acc = accuracy(y_hat, target)
            overall_acc += acc

        overall_acc /= len(batch_selection)

        print("\t=> client_id: {} accuracy: {}".format(self.id, overall_acc))

        return overall_acc




        

    # not used
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
