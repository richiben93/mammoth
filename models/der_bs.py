# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle

import torch

from utils.batch_shaping import BatchShapingLoss
from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.shapiry import MaskedShapiroShapingGaussianLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay with batch shaping.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--bs_weight', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--alpha_der', type=float, default=0.3,
                        help='Alpha of the Beta distribution.')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha of the Beta distribution.')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Beta of the Beta distribution.')
    return parser


class DerBs(ContinualModel):
    NAME = 'der_bs'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(DerBs, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS
        self.bs_loss = BatchShapingLoss(alpha=self.args.alpha, beta=self.args.beta)

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        features = self.net.features(inputs)
        outputs = self.net.linear(features)
        # task_labels = labels // self.classes
        # eye = torch.eye(self.n_task).bool()
        # mask = torch.repeat_interleave(eye[task_labels], self.classes, 1)
        # masked_outputs = outputs[mask].reshape(inputs.shape[0], -1)
        loss = self.loss(outputs,  labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha_der * F.mse_loss(buf_outputs, buf_logits)

        if self.task + 1 < self.n_task:
            bs_outputs = self.net.linear(features.detach().clone())
            masked_futures = bs_outputs[:, (self.task + 1) * self.classes: (self.task + 2) * self.classes]
            bs_loss = self.bs_loss(F.sigmoid(masked_futures))
            loss += bs_loss * self.args.bs_weight

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data, labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
        # if self.task == self.n_task:
        #     with open(f'/homes/efrascaroli/output/logits_buffer_sw{self.args.shap_weight}.pkl', 'wb') as f:
        #         pickle.dump((self.buffer.logits.cpu().detach(), self.buffer.labels.cpu().detach()), f)
