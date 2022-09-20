# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from models.utils.consolidation_pretrain import PretrainedConsolidationModel
from utils.conf import base_path
import numpy as np
from utils.wandbsc import WandbLogger


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    PretrainedConsolidationModel.add_consolidation_args(parser)
    return parser


class DerppConPre(PretrainedConsolidationModel):
    NAME = 'derpp_con_pre'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppConPre, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.args.name = 'DerppConPre' if self.args.con_weight > 0 else 'Derpp'
        self.wblog = WandbLogger(args, name=self.args.name)
        self.log_results = []

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'class_loss': None, 'con_loss': None, 'task': self.task}

        self.opt.zero_grad()
        con_loss = super().observe(inputs, labels, not_aug_inputs)
        wandb_log['con_loss'] = con_loss

        outputs = self.net(inputs)
        class_loss = self.loss(outputs, labels)
        if not self.buffer.is_empty() and self.args.buffer_size > 0:
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            class_loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            class_loss += self.args.beta * self.loss(buf_outputs, buf_labels)
        wandb_log['class_loss'] = class_loss.item()

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data)


        loss = class_loss
        if con_loss is not None and self.args.con_weight > 0:
            loss += self.args.con_weight * con_loss
        wandb_log['loss'] = loss

        loss.backward()
        self.opt.step()

        self.wblog({'training': wandb_log})

        return loss.item()


    def end_training(self):
        if self.args.custom_log:
            log_dir = f'{base_path()}logs/{self.dataset_name}/{self.NAME}'
            obj = {**vars(self.args), 'results': self.log_results}
            self.print_logs(log_dir, obj, name='results')
