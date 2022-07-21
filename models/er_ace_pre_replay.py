import torch
from torch.utils.data import DataLoader

from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_pretrain import PretrainedConsolidationModel
import numpy as np
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs, calc_ADL_heat
from time import time
from utils.conf import base_path
from utils.wandbsc import WandbLogger
import os


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    PretrainedConsolidationModel.add_consolidation_args(parser)
    parser.add_argument('--pre_minibatch', type=int, default=-1,
                        help='Size of pre-dataset minibatch replay (for x, lats and dists).')
    parser.add_argument('--replay_mode', type=str, required=True, help='What you replay.',
                        choices=['none', 'x', 'lats', 'dists','graph', 'laplacian', 'evec', 'fmap', 'eval', 'fmheat',
                                 'fmeval-0101', 'fmeval-0110', 'fmeval-1001', 'fmeval-1010'])
    parser.add_argument('--replay_weight', type=float, required=True,
                        help='Weight of replay.')
    parser.add_argument('--graph_sym', action='store_true',
                        help='Construct a symmetric graph.')
    parser.add_argument('--grad_clip', default=0, type=float, help='Clip the gradient.')
    parser.add_argument('--cos_dist', action='store_true', help='Use cosine disttance.')
    return parser


class ErACEPreReplay(PretrainedConsolidationModel):
    NAME = 'er_ace_pre_replay'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        if args.pre_minibatch < 0 or args.replay_mode not in ['x', 'lats', 'dists']:
            args.pre_minibatch = args.spectral_buffer_size
        if args.replay_mode == 'none' or args.replay_weight == 0:
            args.replay_mode = 'none'
            args.replay_weight = 0
        super(ErACEPreReplay, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = self.N_TASKS * self.N_CLASSES_PER_TASK
        self.args.name = 'EracePre' + args.replay_mode.capitalize()
        self.wblog = WandbLogger(args, name=self.args.name, prj='rodo-pretrain', entity='regaz')
        self.log_results = []
        self.log_latents = []
        self.add_log_latents()
        self.spectral_memory = self.init_spectre()


    def get_spectre(self, x: torch.Tensor):
        spectre = self.net.features(x)
        if self.args.replay_mode != 'lats':
            if self.args.cos_dist:
                spectre = calc_cos_dist(spectre)
            else:
                spectre = calc_euclid_dist(spectre)
        if self.args.replay_mode == 'graph':
            spectre, _, _ = calc_ADL_knn(spectre, k=self.args.knn_laplace, symmetric=self.args.graph_sym)
        if self.args.replay_mode == 'laplacian':
            A, D, L = calc_ADL_knn(spectre, k=self.args.knn_laplace, symmetric=True)
            spectre = torch.eye(A.shape[0]).to(A.device) - normalize_A(A, D)
        if self.args.replay_mode in ('evec', 'fmap', 'fmheat'):
            if self.args.replay_mode == 'fmheat':
                A, D, L = calc_ADL_heat(spectre)
            else:
                A, D, L = calc_ADL_knn(spectre, k=self.args.knn_laplace, symmetric=True)
            L = torch.eye(A.shape[0]).to(A.device) - normalize_A(A, D)
            _, spectre = find_eigs(L, n_pairs=self.args.fmap_dim)
            if self.args.replay_mode == 'evec':
                spectre = spectre.abs()
        if self.args.replay_mode == 'eval':
            A, D, L = calc_ADL_knn(spectre, k=self.args.knn_laplace, symmetric=True)
            L = torch.eye(A.shape[0]).to(A.device) - normalize_A(A, D)
            spectre, _ = find_eigs(L, n_pairs=self.args.fmap_dim)
        if self.args.replay_mode.startswith('fmeval'):
            A, D, L = calc_ADL_knn(spectre, k=self.args.knn_laplace, symmetric=True)
            L = torch.eye(A.shape[0]).to(A.device) - normalize_A(A, D)
            eigenvalues, eigenvectors = find_eigs(L, n_pairs=self.args.fmap_dim)
            spectre = eigenvalues, eigenvectors

        return spectre

    @torch.no_grad()
    def init_spectre(self):
        inputs, labels = self.spectral_buffer.get_all_data()
        self.net.eval()
        spectre = self.get_spectre(inputs)
        self.net.train()
        # self.spectral_buffer.empty()
        # self.spectral_buffer.add_data(inputs, labels=labels, logits=spectre)
        return spectre

    def get_replay_loss(self):
        inputs, labels = self.spectral_buffer.get_all_data()
        targets = self.spectral_memory
        if self.args.pre_minibatch < self.args.spectral_buffer_size:
            choices = np.random.choice(len(labels), size=self.args.pre_minibatch, replace=False)
            inputs = inputs[choices]
            labels = labels[choices]
            targets = targets[choices]
            if self.args.replay_mode == 'dists':
                targets = targets[:, choices]

        if self.args.replay_mode == 'x':
            features = self.net.features(inputs)
            preds = self.pre_classifier(features)
            return self.loss(preds, labels)

        spectre = self.get_spectre(inputs)

        if self.args.replay_mode in ('fmap', 'fmheat'):
            spectre = (spectre.T @ targets).abs()
            targets = torch.eye(spectre.shape[0]).to(spectre.device)

        if self.args.replay_mode.startswith('fmeval'):
            evects = [targets[1], spectre[1]]
            evals = [targets[0], spectre[0]]
            codes = [int(c) for c in self.args.replay_mode.rsplit('-')[1]]
            assert len(codes) == 4
            spectre = (evects[codes[0]].T @ evects[codes[1]]) @ torch.diag(evals[0])
            targets = torch.diag(evals[1]) @ (evects[codes[2]].T @ evects[codes[3]])

        return torch.square(spectre - targets).sum()

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'class_loss': None, 'replay_loss': None, 'task': self.task}

        self.opt.zero_grad()
        # with torch.no_grad():
        #     self.net.eval()
        #     sploss = self.get_replay_loss()
        #     self.net.train()

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1
        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        class_loss = self.loss(logits, labels)
        if self.task > 0 and self.args.buffer_size > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            class_loss += self.loss(self.net(buf_inputs), buf_labels)
        loss = class_loss
        wandb_log['class_loss'] = class_loss.item()

        if self.args.pre_minibatch > 0 and self.args.replay_weight > 0:
            replay_loss = self.get_replay_loss()
            loss += replay_loss * self.args.replay_weight
            wandb_log['replay_loss'] = replay_loss.item()

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        wandb_log['loss'] = loss

        loss.backward()
        # clip gradients
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)

        self.opt.step()

        self.wblog({'training': wandb_log})
        return loss.item()

    def end_task(self, dataset):
        self.task += 1

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()
        pre_acc = self.pre_dataset_train_head()

        assert pre_acc > 0.5, f'Accuracy on pretrain too low: \n{pre_acc}'

        # running consolidation error
        with torch.no_grad():
            replay_error = self.get_replay_loss()

        log_obj = {
            'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': replay_error,
            'PreTrain-acc': pre_acc,
            **{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
            **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])},
            'task': self.task,
        }
        self.log_results.append(log_obj)
        self.wblog({'testing': log_obj})
        self.add_log_latents()
        self.save_checkpoint(self.task)

        # if self.task > 2 and self.args.save_checks:
        #     self.end_training()
        #     exit()

        if self.task == self.N_TASKS:
            self.end_training()

    @torch.no_grad()
    def add_log_latents(self):
        self.net.eval()
        lats, y = self.compute_buffer_latents()
        self.net.train()
        self.log_latents.append({'feat_buf': lats.tolist(), 'y_buf': y.tolist()})

    def save_checkpoint(self, task: int):
        if self.args.save_checks:
            log_dir = f'{base_path()}checkpoints/{self.args.name}'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            torch.save(self.net.state_dict(), f'{log_dir}/task_{task}.pt')

    def end_training(self):
        if self.args.custom_log:
            log_dir = f'{base_path()}logs/{self.dataset_name}/{self.NAME}'
            # obj = {**vars(self.args), 'results': self.log_results}
            # self.print_logs(log_dir, obj, name='results')
            obj = {**vars(self.args), 'results': self.log_results, 'latents': self.log_latents}
            self.print_logs(log_dir, obj, name='latents')
