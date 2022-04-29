# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via ContinualProtoNets.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    return parser


class CPN(ContinualModel):
    NAME = 'cpn'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CPN, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.prototypes = None

        self.current_task = 0

    def forward(self, x):
        if self.prototypes is None:
            return None

        embeddings = self.net.features(x)
        embeddings = embeddings.unsqueeze(1)

        pred = (self.prototypes.unsqueeze(0) - embeddings).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None):
        real_batch_size = inputs.shape[0]
        # 	Fetch dal buffer (bilanciato??)
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        # 	Update w:
        # 		Media prototipi del batch -> c)
        # 		Loss: dist(wx, relativo c) - sum(dist(wx, altri c)) / n_batch
        embeddings = self.net.features(inputs)
        centroids = self.compute_prototypes(embeddings, labels)


        #   Update buffer
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        pass

    def end_task(self, dataset) -> None:
        self.current_task += 1
        # calcolo prototipi dal buffer

    def compute_prototypes(self, embeddings: torch.Tensor, targets: torch.Tensor):
        num_classes = len(targets.unique())

        batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

        num_samples = self.get_num_samples(targets, num_classes, dtype=embeddings.dtype)
        num_samples.unsqueeze_(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
        indices = targets.unsqueeze(-1).expand_as(embeddings)
        prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)
        return prototypes

    def get_num_samples(self, targets, num_classes, dtype=None):
        batch_size = targets.size(0)
        with torch.no_grad():
            ones = torch.ones_like(targets, dtype=dtype)
            num_samples = ones.new_zeros((batch_size, num_classes))
            num_samples.scatter_add_(1, targets, ones)
        return num_samples

    def proto_loss(self, prototypes, embeddings, targets):
        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                       - embeddings.unsqueeze(1)) ** 2, dim=-1)
        return F.cross_entropy(-squared_distances, targets)