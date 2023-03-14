# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
from argparse import ArgumentParser, Action
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--lr_decay_steps', type=lambda s: [] if s == '' else [int(v) for v in s.split(',')],
                        default='', help='Learning rate.')
    parser.add_argument('--lr_momentum', type=float, default=0,)                        
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multiple gpus.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wb_prj', type=str, default='rodo-super',
                        help='Wandb project')
    parser.add_argument('--wb_entity', type=str, default='regaz',
                        help='Watdb entity')
    parser.add_argument('--custom_log', action='store_true',
                        help='Enable log (custom for each model, must be implemented)')
    parser.add_argument('--save_checks', action='store_true',
                        help='Save checkpoints')
    parser.add_argument('--end_task', type=int, default=None, help='Last task to train on')
    parser.add_argument('--start_task', type=int, default=0,
                        help='First task to train on (evaluation on previous tasks done normally)')
    parser.add_argument('--load_check', type=str, default=None, help='Load checkpoint (insert path)')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--set_device', default=None, type=str)


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=None,
                        help='The batch size of the memory buffer (default=batch_size).')
    parser.add_argument('--load_buffer', type=str, default=None, help='Load buffer (insert path)')


def add_aux_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used to load initial (pretrain) checkpoint
    :param parser: the parser instance
    """
    parser.add_argument('--pre_epochs', type=int, default=200,
                        help='pretrain_epochs.')
    parser.add_argument('--pre_dataset', type=str, required=True,
                        choices=['cifar100', 'tinyimgR', 'imagenet'])
    parser.add_argument('--load_cp', type=str, default=f'/tmp/checkpoint_{datetime.now().timestamp()}.pth')
    parser.add_argument('--stop_after_prep', action='store_true')
