# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import socket
import os
import random
import torch
import numpy as np
import os
from utils import create_if_not_exists

def get_device(jobs_per_gpu=10) -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    if socket.gethostname() == 'goku' and torch.cuda.is_available():
        while True:
            lines = np.array(os.popen('nvidia-smi | grep " C " | awk \'{print $2}\'').read().splitlines()).astype(int)
            unique, counts = np.unique(lines, return_counts=True)
            if len(unique) > 1 and np.min(counts) < jobs_per_gpu:
                return torch.device('cuda:{}'.format(np.argmin(counts)))
            elif len(unique) == 0:
                return torch.device('cuda:0')
            elif len(unique) == 1:
                return torch.device('cuda:{}'.format('0' if unique.item() == 1 else '1'))
            sleep((random.random() + 1) * 5)
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    if 'GLADIO' in os.environ:
        path = '/shared/efrasca/mammoth-data/'
        create_if_not_exists(path)
        return path
    return '/nas/softechict-nas-2/efrascaroli/mammoth-data/' if 'SLURM_CONF' in os.environ else './data/'
    # return '/scratch/efrascaroli/mammoth/' if 'SLURM_CONF' in os.environ else './data/'


def base_path_dataset() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    if 'GLADIO' in os.environ:
        path = '/data/efrasca/datasets/'
        create_if_not_exists(path)
        return path
    uname = os.environ['LOGNAME']
    if uname == 'efrascaroli':
        return '/nas/softechict-nas-2/efrascaroli/datasets/'
    return f'/tmp/{uname}/mammoth-datasets/'
    # return '/nas/softechict-nas-2/efrascaroli/datasets/' if 'SLURM_CONF' in os.environ else './data/'
    # return '/scratch/efrascaroli/mammoth/datasets' if 'SLURM_CONF' in os.environ else './data/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
