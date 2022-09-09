# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from utils.args import *
from models.utils.replay_model import ReplayModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math
from torchvision import transforms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size

    # --replay_mode, --replay_weight, --rep_minibatch,
    # --graph_sym, --heat_kernel, --cos_dist, --knn_laplace, --fmap_dim
    ReplayModel.add_replay_args(parser)
    return parser


class JointReplay(ReplayModel):
    NAME = 'joint_replay'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        args.buffer_size = args.batch_size
        args.rep_minibatch = -1
        super(JointReplay, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.task = 0

    def get_name(self):
        return 'Joint' + self.get_name_extension()

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)
            self.reset_scheduler()

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            self.transform = lambda xx: xx
            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())

                    self.buffer.empty()
                    self.buffer.add_data(examples=inputs, labels=labels, logits=outputs)
                    if self.args.rep_minibatch > 0 and self.args.replay_weight > 0:
                        replay_loss = self.get_replay_loss()
                        self.wb_log['replay_loss'] = replay_loss.item()
                        loss += replay_loss * self.args.replay_weight

                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS: return
            loader_caches = [[] for _ in range(len(self.old_data))]
            sources = torch.randint(5, (128,))
            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i+1) * bs]
                    labels = all_labels[order][i * bs: (i+1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

    def observe(self, inputs, labels, not_aug_inputs):
        return 0
