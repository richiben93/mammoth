import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision
from argparse import Namespace

from utils.batch_norm_freeze import bn_untrack_stats
from utils.conf import get_device
from models.utils.continual_model import ContinualModel
from argparse import ArgumentParser
from utils.buffer import Buffer
from utils.spectral_analysis import laplacian_analysis
from time import time
from datasets import get_dataset


class ConsolidationModel(ContinualModel):

    @staticmethod
    def add_consolidation_args(parser: ArgumentParser):
        parser.add_argument('--spectral_buffer_size', type=int, default=100,
                            help='Size of the spectral buffer.')
        parser.add_argument('--con_weight', type=float, required=True,
                            help='Weight of consolidation.')
        parser.add_argument('--knn_laplace', type=int, default=10,
                            help='K of knn to build the graph for laplacian.')
        parser.add_argument('--fmap_dim', type=int, default=20,
                            help='Number of eigenvectors to take to build functional maps.')

        # parser.add_argument('--profiler', action='store_true', help='Log time of function.')

    @staticmethod
    def print_logs(path: str, obj: any, name='logs', extension='pyd'):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f'{name}.{extension}'
        with open(os.path.join(path, filename), 'a') as f:
            f.write(str(obj) + '\n')

    def __init__(self, backbone, loss, args, transform):
        super(ConsolidationModel, self).__init__(backbone, loss, args, transform)
        self.spectral_buffer = Buffer(args.spectral_buffer_size, self.device)
        self.task = 0
        self.buffer_evectors = []
        dataset = get_dataset(args)
        self.N_TASKS = dataset.N_TASKS
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.dataset_name = dataset.NAME

    def begin_task(self, dataset):
        pass

    def observe(self, inputs, labels, not_aug_inputs):
        if not self.spectral_buffer.is_full():
            self.spectral_buffer.add_data(examples=inputs, labels=labels)

        c_loss = None
        if self.task > 0 and self.args.con_weight > 0:
            with bn_untrack_stats(self.net):
                # t1 = time()
                evects = self.compute_buffer_evects()
                # t2 = time()
                self.buffer_evectors.append(evects)
                # t3 = time()
                c_loss = self.get_consolidation_error()
                # t4 = time()
                self.buffer_evectors.pop()
                # if self.args.profiler:
                #     print(f'\nCompute evects: {(t2-t1):.4f}'
                #           f'\nConsolidation error: {(t4-t3):.4f}'
                #           f'\n')

        return c_loss

    def end_task(self, dataset):
        self.task += 1
        with torch.no_grad():
            self.eval()
            evects = self.compute_buffer_evects()
            self.train()
        self.buffer_evectors.append(evects)

    def compute_buffer_evects(self):
        inputs, labels = self.spectral_buffer.get_all_data()
        latents = self.net.features(inputs)
        energy, eigenvalues, eigenvectors, L, _ = laplacian_analysis(latents, norm_lap=True, knn=self.args.knn_laplace,
                                                                     n_pairs=self.args.fmap_dim)
        return eigenvectors[:, :self.args.fmap_dim]

    def get_consolidation_error(self):
        evects = self.buffer_evectors
        n_vects = self.args.fmap_dim
        # c_0_last = evects[0][:, :n_vects].T @ evects[len(evects) - 1][:, :n_vects]
        c_product = torch.ones((n_vects, n_vects), device=self.device, dtype=torch.double)
        for i, ev in enumerate(evects[:-1]):
            c = ev[:, :n_vects].T @ evects[i + 1][:, :n_vects]
            if i == 0:
                c_product = c.clone()
            else:
                c_product = c_product @ c

        c = evects[-1][:, :n_vects].T @ evects[0][:, :n_vects]
        c_product = c_product @ c
        diff = (torch.eye(len(c_product), device=c_product.device) - c_product).abs()
        # diff = (c_0_last - c_product).abs()
        return diff.sum()


## old consolidation error with plt
    #
    # def get_consolidation_error(self):
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     evects = self.buffer_evectors
    #     n_vects = self.args.fmap_dim
    #
    #     ncols = len(evects) - 1
    #     figsize = (6*ncols, 6)
    #     fig, ax = plt.subplots(1, ncols, figsize=figsize)
    #     plt.suptitle(f'\nKnn Norm Laplacian | {n_vects} eigenvects | {len(evects[0])} data')
    #     mask = torch.eye(n_vects) == 0
    #     c_0_last = evects[0][:, :n_vects].T @ evects[len(evects) - 1][:, :n_vects]
    #     c_product = torch.ones((n_vects, n_vects), device=self.device, dtype=torch.double)
    #     for i, ev in enumerate(evects[:-1]):
    #         c = ev[:, :n_vects].T @ evects[i + 1][:, :n_vects]
    #         if i == 0:
    #             c_product = c.clone()
    #         else:
    #             c_product = c_product @ c
    #         oode = torch.square(c[mask]).sum().item()
    #         sns.heatmap(c.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[i], cbar=True if i + 1 == ncols else False)
    #         ax[i].set_title(f'FMap Task {i} => {i + 1} | oode={oode:.4f}')
    #
    #     if details: plt.show()
    #     else: plt.close()
    #
    #     figsize = (6 * 3, 8)
    #     fig, ax = plt.subplots(1, 3, figsize=figsize)
    #     plt.suptitle(f'\nCompare differences of 0->Last and consecutive product')
    #
    #     oode = torch.square(c_0_last[mask]).sum().item()
    #     sns.heatmap(c_0_last.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[0], cbar=False)
    #     ax[0].set_title(f'FMap Task 0 => {len(evects) - 1}\n oode={oode:.4f}')
    #     oode = torch.square(c_product[mask]).sum().item()
    #     sns.heatmap(c_product.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[1], cbar=False)
    #     ax[1].set_title(f'FMap Diagonal Product\n oode={oode:.4f}')
    #     diff = (c_0_last - c_product).abs()
    #     sns.heatmap(diff.detach().cpu(), cmap='bwr', vmin=-1, vmax=1, ax=ax[2], cbar=True)
    #     ax[2].set_title(f'Absolute Differences | sum: {diff.sum().item():.4f}')
    #     if details: plt.show()
    #     else: plt.close()
    #
    #     # if self.args.wandb:
    #     #     wandb.log({"fmap": wandb.Image(diff.cpu().detach().numpy())})
    #
    #     return diff.sum()