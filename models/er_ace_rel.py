# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from torch.functional import F
from models.utils.continual_model import ContinualModel
from models.er_ace import ErACE
from copy import deepcopy
from utils.no_bn import bn_track_stats
import os
import pickle
from models.icarl import baguette_fill_buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--n_anchors', type=int, default=100)
    parser.add_argument('--rel_weight', type=float, default=1)
    parser.add_argument('--herding', action='store_true')
    parser.add_argument('--double_transform', action='store_true')
    return parser


# a=dataset, b=vector
def cos_dist(a, b):
    return torch.sum(a * b, dim=1) / (torch.norm(a, dim=1) * torch.norm(b, dim=0))


# a,b=dataset
def cos_sim(a, b):
    return torch.mean(torch.sum(a * b, dim=1) / (torch.norm(a, dim=1) * torch.norm(b, dim=1)))


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class ErACERel(ErACE):
    NAME = 'er_ace_rel'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACERel, self).__init__(backbone, loss, args, transform)
        self.anchor_buffer = Buffer(self.args.n_anchors, self.device)
        self.fixed_net = None
        # self.anchors = None

    def begin_task(self, dataset):
        pass

    def end_task(self, dataset):
        if not self.buffer.is_empty():
            if len(self.buffer.get_all_data()) == 2:
                setattr(self.buffer, 'logits', self.buffer.get_all_data()[1])
            if self.args.herding:
                self.classes_so_far = self.seen_so_far
                with torch.no_grad():
                    baguette_fill_buffer(self, self.anchor_buffer, dataset, self.task)
            else:
                self.anchor_buffer.empty()
                x, y, logits = self.buffer.get_all_data()
                anchors = torch.randint(0, len(y), (self.args.n_anchors,))
                self.anchor_buffer.add_data(examples=x[anchors], labels=y[anchors], logits=logits[anchors])
            self.fixed_net = deepcopy(self.net)
        super().end_task(dataset)

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
            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)
            self.wb_log['erace_loss'] = loss_re.item()
            loss += loss_re

        if self.task > 0 and not self.anchor_buffer.is_empty():
            anchors = self.anchor_buffer.get_all_data(transform=self.transform)[0]
            buf_inputs, _, _ = self.buffer.get_data(self.args.minibatch_size)
            x1 = torch.stack([self.transform(buf_input.cpu()) for buf_input in buf_inputs]).to(self.device)
            if self.args.double_transform:
                x2 = torch.stack([self.transform(buf_input.cpu()) for buf_input in buf_inputs]).to(self.device)
            else:
                x2 = x1
            batch1 = torch.cat([x1, anchors])
            batch2 = torch.cat([x2, anchors])
            with torch.no_grad():
                with bn_track_stats(self.fixed_net, False):
                    embeds1 = self.fixed_net.features(batch1)
                    rel_space1 = sim_matrix(embeds1[:len(x1)], embeds1[len(x1):])

            embeds2 = self.net.features(batch2)
            rel_space2 = sim_matrix(embeds2[:len(x2)], embeds2[len(x2):])
            loss_re = F.mse_loss(rel_space2, rel_space1)
            self.wb_log['relative_loss'] = loss_re.item()
            loss += loss_re * self.args.rel_weight

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=labels)

        return loss.item()
