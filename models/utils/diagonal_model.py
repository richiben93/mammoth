from copy import deepcopy

import torch
import torchvision.transforms as transforms
from utils.batch_norm_freeze import bn_untrack_stats
from utils.conf import get_device
from models.utils.continual_model import ContinualModel
from argparse import ArgumentParser
from utils.buffer import Buffer
from utils.spectral_analysis import laplacian_analysis
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
