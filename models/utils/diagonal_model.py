import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from torch.optim import SGD
import torchvision
from argparse import Namespace
import torchvision.transforms as transforms
from utils.batch_norm_freeze import bn_untrack_stats
from utils.conf import get_device
from models.utils.continual_model import ContinualModel
from argparse import ArgumentParser
from utils.buffer import Buffer
from utils.spectral_analysis import laplacian_analysis
from time import time
from datasets import get_dataset


class DiagonalModel(ContinualModel):

    @staticmethod
    def add_consolidation_args(parser: ArgumentParser):
        parser.add_argument('--spectral_buffer_size', type=int, default=100,
                            help='Size of the spectral buffer.')
        parser.add_argument('--diag_weight', type=float, required=True,
                            help='Weight of consolidation.')
        parser.add_argument('--diag_losses', default=['d-err', 'od-err'],
                            nargs="+", help='Diagonal losses.')
        parser.add_argument('--knn_laplace', type=int, default=10,
                            help='K of knn to build the graph for laplacian.')
        parser.add_argument('--fmap_dim', type=int, default=20,
                            help='Number of eigenvectors to take to build functional maps.')
        parser.add_argument('--print_custom_log', action='store_true')
        parser.add_argument('--set_device', default=None)

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
        super().__init__(backbone, loss, args, transform)
        self.device = torch.device(f"cuda:{self.args.set_device}") if self.args.set_device is not None else get_device()
        self.spectral_buffer = Buffer(args.spectral_buffer_size, self.device)
        self.task = 0
        self.buffer_evectors = []
        dataset = get_dataset(args)
        self.spectral_buffer_not_aug_transf = transforms.Compose([dataset.get_normalization_transform()])
        self.N_TASKS = dataset.N_TASKS
        self.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
        self.dataset_name = dataset.NAME

    def begin_task(self, dataset):
        pass

    def observe(self, inputs, labels, not_aug_inputs):
        c_loss = None
        if self.task > 0 or self.args.pretrained_model is not None:
            if self.args.diag_weight > 0:
                with bn_untrack_stats(self.net):
                    evects = self.compute_buffer_evects()
                    self.buffer_evectors.append(evects)
                    c_loss = self.get_off_diagonal_error()
                self.buffer_evectors.pop()
        return c_loss

    def end_task(self, dataset):
        self.spectral_buffer = deepcopy(self.buffer)
        # if wandb.run:
        #     wandb.log({'spectral_buffer': self.spectral_buffer.get_all_data()[1]})
        with torch.no_grad():
            self.net.eval()
            evects = self.compute_buffer_evects()
            self.net.train()
        self.buffer_evectors.append(evects)

    def compute_buffer_evects(self, model=None):
        # add only normalization as transformation
        inputs, labels = self.spectral_buffer.get_all_data(transform=self.spectral_buffer_not_aug_transf)
        latents = self.net.features(inputs) if model is None else model.features(inputs)
        energy, eigenvalues, eigenvectors, L, _ = laplacian_analysis(latents, norm_lap=True, knn=self.args.knn_laplace,
                                                                     n_pairs=self.args.fmap_dim)
        return eigenvectors[:, :self.args.fmap_dim]

    def get_off_diagonal_error(self, return_c=False):
        oderr = 0
        derr = 0
        iderr = 0
        evects = self.buffer_evectors
        n_vects = self.args.fmap_dim
        c_0_last = evects[0][:, :n_vects].T @ evects[len(evects) - 1][:, :n_vects]
        # off diagonal error
        if 'od-err' in self.args.diag_losses:
            oderr = (c_0_last * ~(torch.eye(c_0_last.shape[0]).to(self.device) == 1)).pow(2).sum()
        # diagonal error
        if 'd-err' in self.args.diag_losses:
            derr = n_vects-(c_0_last * (torch.eye(c_0_last.shape[0]).to(self.device) == 1)).pow(2).sum()
        # identity error
        if 'id-err' in self.args.diag_losses:
            iderr = (torch.eye(c_0_last.shape[0]).to(self.device) - c_0_last).pow(2).sum()
        # import matplotlib.pyplot as plt
        # plt.imshow(c_0_last.detach().cpu())
        # plt.title(oderr)
        # plt.colorbar()
        # plt.show()
        loss = oderr+derr+iderr
        if return_c:
            return loss, c_0_last
        return loss

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
