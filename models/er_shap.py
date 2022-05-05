# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os.path

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.shapiry import MaskedShapiroShapingGaussianLoss
import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay with Shapiro.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--shap_weight', type=float, required=True,
                        help='Penalty weight.')
    return parser


class ErShap(ContinualModel):
    NAME = 'er_shap'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(ErShap, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS
        self.shapiro_loss = MaskedShapiroShapingGaussianLoss()
        if os.path.isfile('loss_task'):
            os.remove('loss_task')
        with open('loss_task', 'w') as obj:
            obj.write(f'task, loss, shapiro_loss\n')

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        task_labels = labels // self.classes
        eye = torch.eye(self.n_task).bool()
        mask = torch.repeat_interleave(eye[task_labels], self.classes, 1)
        masked_outputs = outputs[mask].reshape(inputs.shape[0], -1)
        if self.task+1 < self.n_task:
            masked_futures = outputs[:, (self.task+1)*self.classes: (self.task+2)*self.classes]
            shap_loss = self.shapiro_loss(masked_futures)
        else:
            shap_loss = 0

        loss = self.loss(masked_outputs, labels % self.classes)
        wandb.log({'loss': loss, 'shap_loss': shap_loss})
        # with open('loss_task', 'a') as obj:
        #     obj.write(f'{self.task}, {loss}, {shap_loss}\n')
        loss += shap_loss * self.args.shap_weight
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
        status = self.net.training
        self.net.eval()
        shapiri_test_total = torch.zeros(self.classes*self.n_task).to(self.device)
        for k, test_loader in enumerate(dataset.test_loaders):
            shapiri_test = torch.zeros(self.classes * self.n_task).to(self.device)
            for n, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if 'class-il' not in self.COMPATIBILITY:
                    outputs = self.net(inputs, k)
                else:
                    outputs = self.net(inputs)
                shapiri_test += self.shapiro_loss.shapiro_test(outputs)
            shapiri_test /= n+1
            shapiri_test_total += shapiri_test
        shapiri_test_total /= k+1
        # wandb.log({f"{num}": shapiri_test_total[num].item() for num in range(shapiri_test_total.shape[0])})
        # wandb.log({'logits': shapiri_test_total})
        self.net.train(status)
