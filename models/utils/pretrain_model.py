import copy
from typing import Tuple

import torch
from torch import nn

import backbone
from utils.conf import base_path
import os
from urllib import request


def download_model(dataset_type: str) -> Tuple[str, int, str]:
    if dataset_type == "cifar100":
        link = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/Eb4BgDZ5g_1Imuwz_PJAmdgBc8k9I_P5p0Y-A97edhsxIw?e=WmlZZc"
        file = "rs18_cifar100.pth"
        n_classes = 100
        model = 'resnet18'
    elif dataset_type == "tinyimgR":
        link = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EeWEOSls505AsMCTXAxWoLUBmeIjCiplFl40zDOCmB_lEw?download=1"
        file = "erace_pret_on_tinyr.pth"
        n_classes = 200
        model = ''
    else:
        raise ValueError
    local_path = os.path.join(base_path(), 'checkpoints')
    if not os.path.isdir(local_path):
        os.mkdir(local_path)
    if not os.path.isfile(os.path.join(local_path, file)):
        request.urlretrieve(link, os.path.join(local_path, file))
    return os.path.join(local_path, file), n_classes, model


def return_pretrained_model(dataset: str):
    local_path, n_classes, model_name = download_model(dataset)
    real_net = getattr(backbone, model_name)(n_classes)
    state_dict = torch.load(local_path)
    real_net.load_state_dict(state_dict)

    return real_net
