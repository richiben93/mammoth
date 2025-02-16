# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import pickle

from torch.functional import F
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.batch_shaping import BatchShapingLoss
from utils.conf import base_path
import wandb

bp = base_path()

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay with BetaShaping.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--bs_weight', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha of the Beta distribution.')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Beta of the Beta distribution.')
    return parser


class ErBs(ContinualModel):
    NAME = 'er_bs'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(ErBs, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS
        self.bs_loss = BatchShapingLoss(alpha=self.args.alpha, beta=self.args.beta)
        # if os.path.isfile('loss_task'):
        #     os.remove('loss_task')
        # with open('loss_task', 'w') as obj:
        #     obj.write(f'task, loss, shapiro_loss\n')

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        # outputs = self.net(inputs)
        features = self.net.features(inputs)
        outputs = self.net.linear(features)

        task_labels = labels // self.classes
        eye = torch.eye(self.n_task).bool()
        mask = torch.repeat_interleave(eye[task_labels], self.classes, 1)
        masked_outputs = outputs[mask].reshape(inputs.shape[0], -1)
        if self.task+1 < self.n_task and self.args.bs_weight > 0:
            bs_outputs = self.net.linear(features.detach().clone())
            masked_futures = bs_outputs[:, (self.task+1)*self.classes: (self.task+2)*self.classes]
            # bs_loss = self.bs_loss(F.softmax(masked_futures, dim=1))
            bs_loss = self.bs_loss(F.sigmoid(masked_futures))
            # bs_loss = (1-F.sigmoid(masked_futures)).pow(2).mean()
        else:
            bs_loss = 0

        loss = self.loss(masked_outputs, labels % self.classes)
        # loss = torch.Tensor([0]).requires_grad_().to(outputs.device)

        # wandb.log({'futures': outputs[:, 6].cpu().detach().numpy(), 'sigmoid': F.sigmoid(outputs[:, 6]).cpu().detach().numpy()})
        # wandb.log({'futures': [x.item() for x in outputs[:, 6]], 'sigmoid': [x.item() for x in F.sigmoid(outputs[:, 6])]})
        # import pandas as pd
        # pd.DataFrame(F.sigmoid(outputs[:, 6]).cpu().detach().numpy()).T.to_csv('sigmoid.txt', mode='a',  header=False)
        loss += bs_loss * self.args.bs_weight
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
        # if self.task == 1:
        #     raise ValueError
        #     with torch.no_grad():
        #         with open(os.path.join(
        #             bp,
        #                 f'logits_buffer_bs{self.args.bs_weight}_a{self.args.alpha}_b{self.args.beta}.pkl'), 'wb') as f:
        #             buf_inputs, buf_labels = self.buffer.get_data(self.buffer.buffer_size, transform=self.transform)
        #             outputs = self.net(buf_inputs)
        #             pickle.dump((
        #                 outputs[:, :5].cpu().detach(),
        #                 outputs[:, 5:10].cpu().detach(),
        #                 F.softmax(outputs[:, :5], dim=1).cpu().detach(),
        #                 F.softmax(outputs[:, 5:10], dim=1).cpu().detach(),
        #                 buf_labels.cpu().detach()
        #             ), f)

        # status = self.net.training
        # self.net.eval()
        # shapiri_test_total = torch.zeros(self.classes*self.n_task).to(self.device)
        # for k, test_loader in enumerate(dataset.test_loaders):
        #     shapiri_test = torch.zeros(self.classes * self.n_task).to(self.device)
        #     for n, data in enumerate(test_loader):
        #         inputs, labels = data
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         if 'class-il' not in self.COMPATIBILITY:
        #             outputs = self.net(inputs, k)
        #         else:
        #             outputs = self.net(inputs)
        #         shapiri_test += self.shapiro_loss.shapiro_test(outputs)
        #     shapiri_test /= n+1
        #     shapiri_test_total += shapiri_test
        # shapiri_test_total /= k+1
        # # wandb.log({f"{num}": shapiri_test_total[num].item() for num in range(shapiri_test_total.shape[0])})
        # # wandb.log({'logits': shapiri_test_total})
        # self.net.train(status)
