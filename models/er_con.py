# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_model import ConsolidationModel
import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    ConsolidationModel.add_consolidation_args(parser)
    return parser


class ErCon(ConsolidationModel):
    NAME = 'er_con'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErCon, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        if args.wandb:
            wandb.init(project="rodo-mammoth", entity="ema-frasca", config=vars(args))

    def observe(self, inputs, labels, not_aug_inputs):
        con_loss = super().observe(inputs, labels, not_aug_inputs)

        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels[:real_batch_size])

        outputs = self.net(inputs)
        class_loss = self.loss(outputs, labels)

        loss = class_loss
        if con_loss is not None and self.args.con_weight > 0:
            loss += self.args.con_weight * con_loss

        loss.backward()
        self.opt.step()

        if wandb.run:
            wandb.log({'loss': loss.item(), 'class_loss': class_loss, 'con_loss': con_loss})

        return loss.item()

    def end_task(self, dataset):
        super().end_task(dataset)

    def log_accs(self, accs):
        cil_acc, til_acc = accs

        self.net_eval()
        # running consolidation error
        con_error = None
        if self.cur_task > 1:
            with torch.no_grad():
                # self.buffer_evectors.append(self.compute_buffer_evects())
                cerr = self.get_consolidation_error(details=False)
                # self.buffer_evectors.pop()
        self.net_train()

        if wandb.run:
            wandb.log({'Class-IL': cil_acc, 'Task-IL': til_acc, 'Con-Error': con_error})
        if self.current_task > 3:
            exit()

    def end_training(self, dataset, print_latents=False):
        if print_latents:
            import os
            logs_dir = f'/nas/softechict-nas-2/efrascaroli/mammoth-data/logs/{dataset.NAME}/{self.NAME}'
            filename = 'latents.pyd'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            with open(os.path.join(logs_dir, filename), 'a') as f:
                f.write(str(self.custom_log) + '\n')
