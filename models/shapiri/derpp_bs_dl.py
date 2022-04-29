# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
from datasets import get_dataset
from utils.batch_shaping import BatchShapingLoss
from utils.diverse_loss import DiverseLoss


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

    parser.add_argument('--bs_weight', type=float, required=True,
                        help='Penalty weight of bs.')
    parser.add_argument('--dl_weight', type=float, required=True,
                        help='Penalty weight of dl.')
    parser.add_argument('--bs_alpha', type=float, default=0.6,
                        help='Alpha of the Beta distribution.')
    parser.add_argument('--bs_beta', type=float, default=0.4,
                        help='Beta of the Beta distribution.')

    parser.add_argument('--head_only', action='store_true',
                        help='BS does not propagate through the features')

    return parser


class DerppBsDl(ContinualModel):
    NAME = 'derpp_bs_dl'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppBsDl, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS

        self.name = 'DERPP'
        if self.args.bs_weight > 0:
            self.name += '_BS'
        else:
            self.args.alpha = 0
            self.args.beta = 0
        if self.args.dl_weight > 0:
            self.name += '_DL'

        if self.args.bs_weight > 0 or self.args.dl_weight > 0:
            self.name += '_detach' if self.args.head_only else ''
        else:
            self.args.head_only = False

        self.bs_loss = BatchShapingLoss(alpha=self.args.bs_alpha, beta=self.args.bs_beta)
        self.dl_loss = DiverseLoss()

    def begin_task(self, dataset):
        self.iters = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        features = self.net.features(inputs)
        outputs = self.net.linear(features)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        if self.task+1 < self.n_task and (self.args.bs_weight > 0 or self.args.dl_weight > 0):
            if self.args.head_only:
                outputs = self.net.linear(features.detach().clone())
            masked_futures = outputs[:, (self.task+1)*self.classes: (self.task+2)*self.classes]
            sigs = torch.sigmoid(masked_futures)
            if self.args.bs_weight > 0:
                bs_loss = self.bs_loss(sigs)
                loss += bs_loss * self.args.bs_weight
            if self.args.dl_weight > 0:
                dl_loss = self.dl_loss(masked_futures)
                loss += dl_loss * self.args.dl_weight

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
