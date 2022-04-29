# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.shapiry import MaskedShapiroShapingGaussianLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay with equal head initialization.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErEqInitClass(ContinualModel):
    NAME = 'er_eqinit_class'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(ErEqInitClass, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        masked_outputs = outputs[:, :((self.task + 1) * self.classes)]

        loss = self.loss(masked_outputs, labels)
        # with open('loss_task', 'a') as obj:
        #     obj.write(f'{self.task}, {loss}, {shap_loss}\n')
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
        if self.task < self.n_task:
            with torch.no_grad():
                self.net.linear.weight[self.task*self.classes:(self.task + 1)*self.classes] = self.net.linear.weight[(self.task-1)*self.classes:self.task*self.classes].clone()
                self.net.linear.bias[self.task*self.classes:(self.task + 1)*self.classes] = self.net.linear.bias[(self.task-1)*self.classes:self.task*self.classes].clone()
