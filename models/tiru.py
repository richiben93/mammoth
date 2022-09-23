# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from utils.args import *
from models.utils.continual_model import ContinualModel
from torch import nn

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--rank', type=int, default=1)

    return parser

class SingleModule(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(SingleModule, self).__init__()
        self.U = nn.Linear(in_features, rank)
        self.S = self.nn.Linear(rank, rank)
        self.V = nn.Linear(rank, out_features)

    def forward(self, x):
        return x @ (self.U(x) @ self.S @ self.V(x))


class Net(nn.Module):
    def __init__(self, rank, input_size, n_classes, n_layers=3):
        super(Net, self).__init__()
        self.activation = nn.ReLU()
        in_size = input_size
        for l in range(n_layers-1):
            self.add_module('mod_{}'.format(l), SingleModule(in_size, input_size//2, rank))
            in_size = in_size//2
        self.add_module('classifier'.format(n_layers-1), SingleModule(in_size, n_classes, rank))

    def forward(self, x):
        for mod in self.children():
            if 'classifier' in mod.name:
                x = mod(x)
            else:
                x = mod(x)
                x = self.activation(x)
        return x

    def update(self, rank):
        for mod in self.children():
            mod.U = nn.Linear(mod.U.in_features, rank)
            mod.S = nn.Linear(rank, rank)
            mod.V = nn.Linear(rank, mod.V.out_features)


class Tiru(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Tiru, self).__init__(backbone, loss, args, transform)

    def begin_task(self, dataset):
        self.net = Net(self.args.rank, dataset.input_size, dataset.n_classes)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr)


    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
