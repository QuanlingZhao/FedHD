import torch
import numpy as np


def transform_list_to_tensor(model_params_list):
    return torch.FloatTensor(model_params_list)


def transform_tensor_to_list(model_params):
    return model_params.tolist()
