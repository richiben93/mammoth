# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import os
import pickle


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        if self.args.load_buffer is not None:
            self.load_buffer(self.args.load_buffer)
            self.seen_so_far = torch.cat([self.seen_so_far, self.buffer.labels.unique()]).unique()

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.max() < (self.N_CLASSES - 1):
            mask[:, self.seen_so_far.max():] = 1
        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)

        if self.task > 0 and not self.buffer.is_empty():
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)
            self.wb_log['erace_loss'] = loss_re.item()
            loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()

    def save_checkpoint(self):
        log_dir = super().save_checkpoint()
        ## pickle the future_buffer
        with open(os.path.join(log_dir, f'task_{self.task}_buffer.pkl'), 'wb') as f:
            self.buffer.to('cpu')
            pickle.dump(self.buffer, f)
            self.buffer.to(self.device)

    def load_buffer(self, path):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
            # FIX for representative buffer
            # self.buffer.num_seen_examples *= self.args.n_epochs
            self.buffer.to(self.device)
