# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.nn import functional as F
from utils.args import *
from models.utils.casper_model import CasperModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++ puls egin-mess.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    
    parser.add_argument('--grad_clip', default=0, type=float, help='Clip the gradient.')

    # --replay_mode, --replay_weight, --rep_minibatch, --knn_laplace
    CasperModel.add_replay_args(parser)
    
    return parser


class DerppCasper(CasperModel):
    NAME = 'derpp_casper'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppCasper, self).__init__(backbone, loss, args, transform)

    def get_name(self):
        return 'Derpp' + self.get_name_extension()

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        self.wb_log['class_loss'] = loss.item()


        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            self.wb_log['derpp_loss'] = derpp_loss.item()
            loss += derpp_loss

        if self.task > 0 and self.args.buffer_size > 0:
            if self.args.rep_minibatch > 0 and self.args.replay_weight > 0:
                replay_loss = self.get_replay_loss()
                self.wb_log['egap_loss'] = replay_loss.item()
                loss += replay_loss * self.args.replay_weight

        loss.backward()
        # clip gradients
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.opt.step()

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                        labels=labels,
                                        logits=outputs.data)

        return loss.item()
