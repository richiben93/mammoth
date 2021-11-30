# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import torch
from torch.functional import F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.shapiry import MaskedShapiroShapingGaussianLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay with Shapiro.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--shap_weight', type=float, required=True,
                        help='Penalty weight.')
    return parser


class ErShapsoftFullClass(ContinualModel):
    NAME = 'er_shapsoft_full_class'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        # 'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(ErShapsoftFullClass, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS
        self.shapiro_loss = MaskedShapiroShapingGaussianLoss()
        if os.path.isfile('loss_task'):
            os.remove('loss_task')
        with open('loss_task', 'w') as obj:
            obj.write(f'task, loss, shapiro_loss\n')

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        if self.task+1 < self.n_task:
            masked_futures = outputs[:, (self.task+1)*self.classes: ]
            shap_loss = self.shapiro_loss(F.softmax(masked_futures))
        else:
            shap_loss = 0

        loss = self.loss(outputs, labels)
        # with open('loss_task', 'a') as obj:
        #     obj.write(f'{self.task}, {loss}, {shap_loss}\n')
        loss += shap_loss * self.args.shap_weight
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
