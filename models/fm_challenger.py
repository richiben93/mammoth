import torch
from torch.utils.data import DataLoader
from torch.functional import F

from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, calc_ADL_heat, normalize_A, find_eigs
from time import time
from utils.conf import base_path
from utils.wandbsc import WandbLogger, innested_vars
import os
import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Experiment: 2 steps of finetuning, then fmap on the first')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--fm_knn_k', type=int, default=20,
                        help='K of knn to build the graph for laplacian.')
    parser.add_argument('--fm_dim', type=int, default=20,
                        help='Number of eigenvectors to take to build functional maps.')
    parser.add_argument('--cos_dist', action='store_true', help='Use cosine disttance.')
    parser.add_argument('--fixed', action='store_true', help='Fixed minibatch = buffer size.')
    parser.add_argument('--heat_kernel', action='store_true', help='Use heat kernel instead of knn.')

    parser.add_argument('--fm_mode', type=str, required=True, help='What you replay.',
                        choices=['lats', 'dists', 'graph', 'laplacian', 'evec', 'fmap', 'eval', 'egap-5'])
    parser.add_argument('--fm_weight', type=float, required=True,
                        help='Weight of replay.')
    return parser


class FMChallenger(ContinualModel):
    NAME = 'fm_challenger'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        if args.fixed:
            args.minibatch_size = args.buffer_size
        super(FMChallenger, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.args.name = self.args.fm_mode.capitalize()
        if self.args.cos_dist:
            self.args.name += 'Cos'
        if self.args.heat_kernel:
            self.args.name += 'Heat'
        self.wblog = WandbLogger(args, name=self.args.name, prj='rodo-challenge', entity='regaz')
        self.task = 0

    def get_all_losses(self):
        if self.args.fixed:
            inputs, labels, features1 = self.buffer.get_all_data(self.transform)
        else:
            inputs, labels, features1 = self.buffer.get_data(self.args.minibatch_size, self.transform)
        features2 = self.net.features(inputs)
        losses = {}

        losses['lats'] = F.mse_loss(features2, features1)

        dists1 = calc_cos_dist(features1) if self.args.cos_dist else calc_euclid_dist(features1)
        dists2 = calc_cos_dist(features2) if self.args.cos_dist else calc_euclid_dist(features2)
        # this loss goes very high, needs to be clipped
        losses['dists'] = F.mse_loss(dists2, dists1) / 1e7

        if self.args.heat_kernel:
            A1, D1, L1 = calc_ADL_heat(dists1)
            A2, D2, L2 = calc_ADL_heat(dists2)
        else:
            A1, D1, L1 = calc_ADL_knn(dists1, k=self.args.fm_knn_k, symmetric=True)
            A2, D2, L2 = calc_ADL_knn(dists2, k=self.args.fm_knn_k, symmetric=True)
        losses['graph'] = F.mse_loss(A2, A1)

        L1 = torch.eye(A1.shape[0]).to(A1.device) - normalize_A(A1, D1)
        L2 = torch.eye(A2.shape[0]).to(A2.device) - normalize_A(A2, D2)
        losses['laplacian'] = F.mse_loss(L2, L1)

        evals1, evects1 = find_eigs(L1, n_pairs=self.args.fm_dim)
        evals2, evects2 = find_eigs(L2, n_pairs=self.args.fm_dim)
        losses['evec'] = F.mse_loss(evects2, evects1)
        losses['eval'] = F.mse_loss(evals2, evals1)

        c = evects2.T @ evects1
        losses['fmap'] = F.mse_loss(c.abs(), torch.eye(c.shape[0]).to(c.device))

        gaps = evals2[1:] - evals2[:-1]
        losses['egap'] = torch.argmax(gaps)

        if self.args.fm_mode.startswith('egap'):
            n = int(self.args.fm_mode.split('-')[1])
            # gaps[n] = -gaps[n]
            # losses[self.args.fm_mode] = gaps.sum()
            losses[self.args.fm_mode] = -gaps[n]
        return losses, c

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': 0, 'class_loss': 0, 'task': self.task}

        if self.args.buffer_size > 0 and self.task == 0:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        self.opt.zero_grad()

        if self.task < 2:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            wandb_log['class_loss'] = loss.item()
            if self.task == 1:
                with torch.no_grad():
                    fm_losses, c = self.get_all_losses()
                wandb_log['fm_loss'] = {k: v.item() for k, v in fm_losses.items()}
                # wandb_log['fm_c'] = wandb.Image(c.detach().cpu().numpy())
        else:
            fm_losses, c = self.get_all_losses()
            wandb_log['fm_loss'] = {k: v.item() for k, v in fm_losses.items()}
            # wandb_log['fm_c'] = wandb.Image(c.detach().cpu().numpy())
            loss = fm_losses[self.args.fm_mode] * self.args.fm_weight

        wandb_log['loss'] = loss

        loss.backward()
        # clip gradients
        # if self.args.grad_clip > 0:
        #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.opt.step()

        self.wblog({'training': wandb_log})
        return loss.item()

    def end_task(self, dataset):
        if self.task == 0:
            self.add_buf_latents()

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()

        log_obj = {
            'Class-IL mean': cil_acc, 'Task-IL mean': til_acc,
            **{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
            **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])},
            'task': self.task,
        }
        # self.log_results.append(log_obj)
        self.wblog({'test': log_obj})
        self.save_checkpoint()

        if self.task > 1:
            exit()

        self.task += 1

    @torch.no_grad()
    def add_buf_latents(self):
        self.net.eval()
        inputs, labels = self.buffer.get_all_data(self.transform)
        no_aug_inputs, _ = self.buffer.get_all_data()
        latents = self.net.features(inputs)
        self.net.train()

        self.buffer.empty()
        self.buffer.add_data(no_aug_inputs, labels=labels, logits=latents)


    def save_checkpoint(self):
        if self.args.save_checks:
            # f'{base_path()}checkpoints/{self.args.name}'
            log_dir = os.path.join(base_path(), 'checkpoints', f'{self.args.name}-{self.wblog.run_id}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # f'{log_dir}/task_{task}.pt'
            torch.save(self.net.state_dict(), os.path.join(log_dir, f'task_{self.task}.pt'))
            if self.task == 0:
                save_dict = innested_vars(self.args)
                filename = f'args.pyd'
                with open(os.path.join(log_dir, filename), 'w') as f:
                    f.write(str(save_dict))
