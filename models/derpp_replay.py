# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
from datasets import get_dataset
import numpy as np
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs, calc_ADL_heat
from time import time
from utils.conf import base_path
from utils.wandbsc import WandbLogger
import os


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++ puls egin-mess.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')

    parser.add_argument('--grad_clip', default=0, type=float, help='Clip the gradient.')
    parser.add_argument('--rep_minibatch', type=int, default=-1,
                        help='Size of pre-dataset minibatch replay (for x, lats and dists).')
    parser.add_argument('--replay_mode', type=str, required=True, help='What you replay.',
                        choices=['none', 'features', 'dists', 'graph', 'laplacian', 'evec', 'fmap', 'eval', 'egap',
                                 'fmeval-0101', 'fmeval-0110', 'fmeval-1001', 'fmeval-1010',
                                 'evalgap', 'evalgap2', 'egap2'])

    parser.add_argument('--replay_weight', type=float, required=True, help='Weight of replay.')

    parser.add_argument('--graph_sym', action='store_true',
                        help='Construct a symmetric graph (only for methods without eigen computation).')
    parser.add_argument('--heat_kernel', action='store_true', help='Use heat kernel instead of knn.')
    parser.add_argument('--cos_dist', action='store_true', help='Use cosine distance.')
    parser.add_argument('--knn_laplace', type=int, default=10,
                        help='K of knn to build the graph for laplacian.')
    parser.add_argument('--fmap_dim', type=int, default=20,
                        help='Number of eigenvectors to take to build functional maps.')
    return parser


class DerppReplay(ContinualModel):
    NAME = 'derpp_replay'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        if args.rep_minibatch < 0:
            args.rep_minibatch = args.buffer_size
        if args.replay_mode == 'none' or args.replay_weight == 0:
            args.replay_mode = 'none'
            args.replay_weight = 0
        if args.replay_mode not in ['graph', 'laplacian']:
            args.graph_sym = True
        super(DerppReplay, self).__init__(backbone, loss, args, transform)

        self.task = 0
        dataset = get_dataset(args)
        self.N_TASKS = dataset.N_TASKS
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.dataset_name = dataset.NAME
        self.N_CLASSES = self.N_TASKS * self.N_CLASSES_PER_TASK

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.future_buffer = Buffer(self.args.buffer_size, self.device)

        self.args.name = self.get_name()
        self.wblog = WandbLogger(args, name=self.args.name, prj='rodo-super', entity='regaz')
        self.log_results = []
        self.temp_log = {}

    def get_name(self):
        name = 'Derpp' + self.args.replay_mode.capitalize()
        if self.args.graph_sym and self.args.replay_mode == 'graph':
            name += 'Sym'
        if self.args.cos_dist:
            name += 'Cos'
        if self.args.heat_kernel:
            name += 'Heat'
        return name

    def get_replay_loss(self):
        if self.args.replay_mode == 'none':
            return torch.tensor(0., dtype=torch.float, device=self.device)
        if self.args.rep_minibatch == self.args.buffer_size:
            if self.args.replay_mode in ['egap', 'egap2']:
                inputs, labels, _ = self.future_buffer.get_all_data(self.transform)
                features1 = self.net.features(inputs).detach()
            else:
                inputs, labels, features1 = self.buffer.get_all_data(self.transform)
        else:
            if self.args.replay_mode in ['egap', 'egap2']:
                inputs, labels, _ = self.future_buffer.get_data(self.args.rep_minibatch, self.transform)
                features1 = self.net.features(inputs).detach()
            else:
                inputs, labels, features1 = self.buffer.get_data(self.args.rep_minibatch, self.transform)
        features2 = self.net.features(inputs)

        if self.args.replay_mode == 'features':
            return F.mse_loss(features2, features1)

        dists1 = calc_cos_dist(features1) if self.args.cos_dist else calc_euclid_dist(features1)
        dists2 = calc_cos_dist(features2) if self.args.cos_dist else calc_euclid_dist(features2)
        if self.args.replay_mode == 'dists':
            # this loss (euclid) goes very high, needs to be clipped
            return F.mse_loss(dists2, dists1) / (1 if self.args.cos_dist else 1e7)

        if self.args.heat_kernel:
            A1, D1, L1 = calc_ADL_heat(dists1)
            A2, D2, L2 = calc_ADL_heat(dists2)
        else:
            A1, D1, L1 = calc_ADL_knn(dists1, k=self.args.knn_laplace, symmetric=self.args.graph_sym)
            A2, D2, L2 = calc_ADL_knn(dists2, k=self.args.knn_laplace, symmetric=self.args.graph_sym)

        if self.args.replay_mode == 'graph':
            return F.mse_loss(A2, A1)

        L1 = torch.eye(A1.shape[0]).to(A1.device) - normalize_A(A1, D1)
        L2 = torch.eye(A2.shape[0]).to(A2.device) - normalize_A(A2, D2)

        if self.args.replay_mode == 'laplacian':
            return F.mse_loss(L2, L1)

        evals1, evects1 = find_eigs(L1, n_pairs=self.args.fmap_dim)
        evals2, evects2 = find_eigs(L2, n_pairs=self.args.fmap_dim)
        gaps = evals2[1:] - evals2[:-1]
        self.temp_log['egap'] = torch.argmax(gaps).item()

        if self.args.replay_mode == 'evec':
            return F.mse_loss(evects2, evects1)
        if self.args.replay_mode == 'eval':
            return F.mse_loss(evals2, evals1)

        if self.args.replay_mode == 'fmap':
            c = evects2.T @ evects1
            return F.mse_loss(c.abs(), torch.eye(c.shape[0], device=c.device))

        if self.args.replay_mode == 'egap':
            n = self.N_CLASSES_PER_TASK * self.task
            return -gaps[n]

        if self.args.replay_mode == 'egap2':
            n = self.N_CLASSES_PER_TASK * self.task
            return evals2[:n+1].sum() - evals2[n+1]

        if self.args.replay_mode == 'evalgap':
            n = self.N_CLASSES_PER_TASK * self.task
            return -gaps[n] + F.mse_loss(evals2[:n], evals1[:n])

        if self.args.replay_mode == 'evalgap2':
            n = self.task
            return -gaps[n] + F.mse_loss(evals2[:n+1], evals1[:n+1])

        if self.args.replay_mode.startswith('fmeval'):
            codes = [int(c) for c in self.args.replay_mode.rsplit('-')[1]]
            assert len(codes) == 4
            evects = [evects2, evects1]
            evals = [evals2, evals1]
            return F.mse_loss((evects[codes[0]].T @ evects[codes[1]]) @ torch.diag(evals[0]),
                              torch.diag(evals[1]) @ (evects[codes[2]].T @ evects[codes[3]]))

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'class_loss': None, 'derpp_loss': None, 'replay_loss': None, 'task': self.task}

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        wandb_log['class_loss'] = loss.item()


        if not self.future_buffer.is_empty():
            buf_inputs, _, buf_logits = self.future_buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.future_buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            derpp_loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            wandb_log['derpp_loss'] = derpp_loss.item()
            loss += derpp_loss

        if self.task > 0 and self.args.buffer_size > 0:
            if self.args.rep_minibatch > 0 and self.args.replay_weight > 0:
                replay_loss = self.get_replay_loss()
                wandb_log['replay_loss'] = replay_loss.item()
                loss += replay_loss * self.args.replay_weight

        wandb_log['loss'] = loss.item()
        loss.backward()
        # clip gradients
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.opt.step()

        if self.args.buffer_size > 0:
            self.future_buffer.add_data(examples=not_aug_inputs,
                                        labels=labels,
                                        logits=outputs.data)

        # add temp_log to wblog
        for k, v in self.temp_log.items():
            wandb_log[k] = v
        self.temp_log = {}
        self.wblog({'training': wandb_log})

        return loss.item()


    def end_task(self, dataset):
        self.task += 1
        # buffer <- future_buffer (con logits)
        self.sync_buffers()

    @torch.no_grad()
    def sync_buffers(self):
        self.net.eval()
        inputs, labels, _ = self.future_buffer.get_all_data(self.transform)
        no_aug_inputs, _, _ = self.future_buffer.get_all_data()
        features = self.net.features(inputs)
        self.net.train()

        self.buffer.empty()
        self.buffer.add_data(no_aug_inputs, labels=labels, logits=features)

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()

        # running consolidation error
        with torch.no_grad():
            self.net.eval()
            replay_error = self.get_replay_loss().item()
            self.net.train()

        log_obj = {
            'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': replay_error,
            **{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
            **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])},
            'task': self.task,
        }
        self.log_results.append(log_obj)
        self.wblog({'testing': log_obj})
        self.save_checkpoint()

        # if self.task > 2 and self.args.save_checks:
        #     self.end_training()
        #     exit()

        if self.task == self.N_TASKS:
            self.end_training()

    def save_checkpoint(self):
        if self.args.save_checks:
            log_dir = os.path.join(base_path(), 'checkpoints', self.args.name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            torch.save(self.net.state_dict(), f'{log_dir}/task_{self.task}.pt')

    def end_training(self):
        if self.args.custom_log:
            log_dir = os.path.join(base_path(), 'logs', self.dataset_name, self.NAME)
            # obj = {**vars(self.args), 'results': self.log_results}
            # self.print_logs(log_dir, obj, name='results')
            obj = {**vars(self.args), 'results': self.log_results}
            self.print_logs(log_dir, obj, name='results')
