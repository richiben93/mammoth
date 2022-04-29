# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import pickle
import sys

from torch.functional import F
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.batch_shaping import BatchShapingLoss
from utils.diverse_loss import DiverseLoss
import math
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier
import wandb

# wandb.init(project="batch-shaping", entity="ema-frasca")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay with BetaShaping and DiverseLoss.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--bs_weight', type=float, required=True,
                        help='Penalty weight of bs.')
    parser.add_argument('--dl_weight', type=float, required=True,
                        help='Penalty weight of dl.')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha of the Beta distribution.')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='Beta of the Beta distribution.')

    parser.add_argument('--futures', type=int, default=5,
                        help='Future classes to train')
    parser.add_argument('--reorder_futures', action='store_true',
                        help='Reorder step before each task')

    parser.add_argument('--knn_acc', action='store_true',
                        help='Evaluate a knn on future heads')
    parser.add_argument('--head_only', action='store_true',
                        help='BS does not propagate through the features')
    return parser


class ErBsDLCe(ContinualModel):
    NAME = 'er_bs_dl_ce'
    COMPATIBILITY = [
        'class-il',
        # 'domain-il',
        'task-il',
        # 'general-continual'
    ]

    def __init__(self, backbone, loss, args, transform):
        super(ErBsDLCe, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.classes = get_dataset(args).N_CLASSES_PER_TASK
        self.n_task = get_dataset(args).N_TASKS

        self.name = 'ER'
        if self.args.bs_weight > 0:
            self.name += '_BS'
        else:
            self.args.alpha = 0
            self.args.beta = 0
        if self.args.dl_weight > 0:
            self.name += '_DL'

        if self.args.bs_weight > 0 or self.args.dl_weight > 0:
            self.name += '_detach' if self.args.head_only else ''
        else:
            self.args.head_only = False

        self.bs_loss = BatchShapingLoss(alpha=self.args.alpha, beta=self.args.beta)
        self.dl_loss = DiverseLoss()

        # if os.path.isfile('loss_task'):
        #     os.remove('loss_task')
        # with open('loss_task', 'w') as obj:
        #     obj.write(f'task, loss, shapiro_loss\n')
        # self.res1 = {i: torch.zeros((5,), device='cuda') for i in range(self.classes)}

        # wandb.init(project="batch-shaping", entity="ema-frasca", name='diverse_loss', config=vars(self.args))

    def begin_task(self, dataset):
        self.iters = 0

        if self.args.reorder_futures and self.args.futures > 0 and self.task > 0:
            data, labels = [], []
            for batch in dataset.train_loader:
                data.append(batch[0])
                labels.append(batch[1])
            data = torch.cat(data).cuda()
            labels = torch.cat(labels).cuda()
            indices = labels.argsort()
            data = data[indices]
            labels = labels[indices] % self.classes

            with torch.no_grad():
                outputs = self.net(data)[:, self.task*self.classes:self.task*self.classes+self.args.futures]

            # angel_mx
            ce_mx = torch.zeros((self.classes, self.args.futures), device=outputs.device, dtype=outputs.dtype)
            for head in range(self.args.futures):
                for cl in range(self.classes):
                    ce_mx[cl, head] = torch.functional.F.binary_cross_entropy(torch.sigmoid(outputs[:, head]),
                                                                              (labels == cl).float())

            best_assignment = linear_sum_assignment(ce_mx.cpu().numpy(), maximize=False)[1].tolist()
            best_assignment += [n for n in range(self.args.futures) if n not in best_assignment]
            head_indices = torch.tensor(best_assignment)

            # head_indices = torch.arange(self.args.futures)
            # for i in range(self.classes):
            #     max_idx = means.argmax().item()
            #     max_class, max_head = max_idx // self.args.futures, max_idx % self.args.futures
            #
            #     means[max_class, :] = -math.inf
            #     means[:, max_head] = -math.inf
            #
            #     if max_head != max_class:
            #         head_indices[max_head] = max_class
            #         head_indices[max_class] = max_head

            head_indices += self.task*self.classes
            param_indices = torch.arange(self.n_task*self.classes+self.args.futures)
            param_indices[self.task * self.classes:self.task * self.classes + self.args.futures] = head_indices
            with torch.no_grad():
                self.net.linear.weight.data = self.net.linear.weight.data[param_indices]
                self.net.linear.bias.data = self.net.linear.bias.data[param_indices]

        if self.args.knn_acc and self.task == 1:
            # torch.save(self, f'/homes/efrascaroli/output/model_{self.name}_dl{self.args.dl_weight}_bs{self.args.bs_weight}_a{self.args.alpha}_b{self.args.beta}_dump.pt')
            # with open(f'/homes/efrascaroli/output/buffer_{self.name}_task1.pkl', 'wb') as f:
            #     pickle.dump(self.buffer.to('cpu'), f)
            # exit()
            log_dict = {"name": self.name, **vars(self.args), "knn_accs": []}
            print('\n', file=sys.stderr)
            for task in range(17):
                data, labels = [], []
                if task > 1:
                    dataset.get_data_loaders()
                for batch in dataset.test_loaders[task]:
                    data.append(batch[0])
                    labels.append(batch[1])
                data = torch.cat(data).cuda()
                labels = torch.cat(labels).cuda()
                indices = labels.argsort()
                data = data[indices]
                labels = labels[indices]

                with torch.no_grad():
                    output_bs = self.net(data)[:, 5:10]

                acc_bs = 0
                n_epochs = 50
                es_per_class = 10
                for epoch in range(n_epochs):
                    mask = torch.zeros((500,))
                    for i in range(5):
                        indices = torch.randperm(100)[:es_per_class]
                        mask[i * 100 + indices] = 1
                    mask = mask.bool()

                    knn_model_bs = KNeighborsClassifier(n_neighbors=3)
                    knn_model_bs.fit(output_bs[mask].cpu(), labels[mask].cpu())
                    predictions = knn_model_bs.predict(output_bs[~mask].cpu())
                    acc_bs += (torch.tensor(predictions) == labels[~mask].cpu()).float().mean() / n_epochs

                log_dict['knn_accs'].append(acc_bs.item())
                print(f'task: {task+1:2} acc: {acc_bs.item():0.4f}', file=sys.stderr, flush=True)
                # print(f'\r task: {task+1:2} / 17 |{"#" * task + "_" * (16 - task)}| acc: {acc_bs.item():0.4f}', end='', file=sys.stderr, flush=True)

            with open(f'/homes/efrascaroli/output/knn_results.pyd', 'a') as f:
                f.write(str(log_dict) + '\n')
                # print(f'{name.upper()}(a={self.args.alpha}, b={self.args.beta}) knn acc: ', acc_bs.item(), file=f)
            exit()


    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        # labels = labels.long()
        self.iters += 1

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
        masked_outputs = outputs[:, :self.n_task*self.classes][mask].reshape(inputs.shape[0], -1)
        loss = self.loss(masked_outputs, labels % self.classes)

        if self.task+1 < self.n_task and self.args.futures > 0 and (self.args.bs_weight > 0 or self.args.dl_weight > 0):
            if self.args.head_only:
                outputs = self.net.linear(features.detach().clone())
            masked_futures = outputs[:, (self.task+1)*self.classes: (self.task+1)*self.classes+self.args.futures]
            sigs = torch.sigmoid(masked_futures)
            if self.args.bs_weight > 0:
                bs_loss = self.bs_loss(sigs)
                loss += bs_loss * self.args.bs_weight
            if self.args.dl_weight > 0:
                dl_loss = self.dl_loss(masked_futures)
                loss += dl_loss * self.args.dl_weight
            # wandb.log({"loss_crossentr": loss, "loss_bs": bs_loss, "loss_dl": dl_loss})

        # wandb.log({'loss': loss, 'shap_loss': bs_loss})
        # with open('loss_task', 'a') as obj:
        #     obj.write(f'{self.task}, {loss}, {shap_loss}\n')
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        self.task += 1

        # if self.task == 1:
        #     with torch.no_grad():
        #         buf_inputs, buf_labels = self.buffer.get_data(self.buffer.buffer_size, transform=self.transform)
        #         buf_labels = buf_labels % self.classes
        #         outputs = self.net(buf_inputs)[:, (self.task)*self.classes: (self.task+1)*self.classes]
        #         sigs = torch.sigmoid(outputs)
        #         for i in range(self.classes):
        #             res = outputs[buf_labels == i].sum(0) / (buf_labels == i).sum()
        #             print(f'{i}: ' + ''.join([f'{v:15.3}' for v in res]))
        #
        #         for i in range(self.classes):
        #             res = sigs[buf_labels == i].sum(0) / (buf_labels == i).sum()
        #             print(f'{i}: ' + ''.join([f'{v:15.3}' for v in res]))
        #
        # if self.task == 2:
        #     for i in range(self.classes):
        #         res = self.res1[i] / self.iters
        #         print(f'{i}: ' + ''.join([f'{v:15.3}' for v in res]))
        #
        #     for i in range(self.classes):
        #         res = torch.sigmoid(self.res1[i] / self.iters)
        #         print(f'{i}: ' + ''.join([f'{v:15.3}' for v in res]))
        #     exit()

        # log logits in file
        # if self.task == 1:
        #     with torch.no_grad():
        #         with open(f'/homes/efrascaroli/output/logits_buffer_pre_bs{self.args.bs_weight}_a{self.args.alpha}_b{self.args.beta}.pkl', 'wb') as f:
        #         # with open(
        #         #         f'C:\\Users\\emace\\AImageLab\\SRV-Continual\\tmp\\logits_buffer_pre_bs{self.args.bs_weight}_a{self.args.alpha}_b{self.args.beta}.pkl',
        #         #         'wb') as f:
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
