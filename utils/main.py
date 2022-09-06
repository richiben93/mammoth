# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import numpy as np
import os
import sys
import socket
import wandb

conf_path = os.getcwd()
sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils import create_if_not_exists
from utils.distributed import make_dp
import torch

import uuid
import datetime


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]

    if args.set_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.set_device

    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset.replace('100', '10')][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # TODO remove
    if 'effnet_variant' in vars(args) and args.effnet_variant is not None:
        os.environ['EFF_VAR'] = args.effnet_variant
        print("WILL USE VARIANT ", os.environ['EFF_VAR'])
    # ---

    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    # args.conf_git_commit = os.popen(r"git log | head -1 | sed s/'commit '//").read().strip()
    # if not os.path.isdir('gitdiffs'):
    #     create_if_not_exists("gitdiffs")
    # os.system('git diff > gitdiffs/diff_%s.txt' % args.conf_jobnum)
    args.conf_host = socket.gethostname()

    dataset = get_dataset(args)
    if 'futures' in args and args.futures > 0:
        backbone = dataset.get_backbone(future_classes=args.futures)
    else:
        backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    if socket.gethostname().startswith('go') or socket.gethostname() == 'jojo':
        import setproctitle
        setproctitle.setproctitle(
            '{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    if args.distributed:
        model.net = make_dp(model.net)
        model.to('cuda:0')
    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
