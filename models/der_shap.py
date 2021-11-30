# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle

from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.shapiry import MaskedShapiroShapingGaussianLoss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--shap_weight', type=float, required=True,
                        help='Penalty weight.')
    return parser


class DerShap(ContinualModel):
    NAME = 'der_shap'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(DerShap, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS
        self.shapiro_loss = MaskedShapiroShapingGaussianLoss()

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        if self.task+1 < self.n_task:
            masked_futures = outputs[:, (self.task+1)*self.classes: ]
            loss += self.args.shap_weight * self.shapiro_loss(masked_futures)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data, labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
        if self.task == self.n_task:
            with open(f'/homes/efrascaroli/output/logits_buffer_sw{self.args.shap_weight}.pkl', 'wb') as f:
                pickle.dump((self.buffer.logits.cpu().detach(), self.buffer.labels.cpu().detach()), f)