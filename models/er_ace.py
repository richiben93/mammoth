# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.wandbsc import WandbLogger


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
        dataset = get_dataset(args)
        self.n_tasks = dataset.N_TASKS
        self.num_classes = dataset.N_TASKS * dataset.N_CLASSES_PER_TASK
        self.task = 0
        self.wblog = WandbLogger(args)

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'task': self.task}
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        loss += loss_re
        wandb_log['loss'] = loss.item()

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        self.wblog({'training': wandb_log})

        return loss.item()

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()

        # running consolidation error
        con_error = None
        # if self.current_task > 2:
        #     with torch.no_grad():
        #         con_error = self.get_consolidation_error().item()

        log_obj = {
            'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': con_error,
            **{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
            **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])},
            'task': self.task,
        }
        self.wblog({'testing': log_obj})

        # self.log_results.append({'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': con_error})
