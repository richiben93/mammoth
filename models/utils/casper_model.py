import torch

from models.utils.egap_model import EgapModel
from utils.spectral_analysis import calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs


class CasperModel(EgapModel):

    @staticmethod
    def add_replay_args(parser):
        parser.add_argument('--rep_minibatch', type=int, default=None,
                            help='Size of minibatch for casper.')
        parser.add_argument('--replay_mode', type=str, default='casper',
                            choices=['none', 'casper'])

        # replay_weight = rho in paper
        parser.add_argument('--replay_weight', type=float, default=0.01, help='Weight of casper.')
        # knn_laplace = k in paper
        parser.add_argument('--knn_laplace', type=int, default=10,
                            help='K of knn to build the graph for laplacian.')
        # b_nclasses = p in paper
        parser.add_argument('--b_nclasses', default=None, type=int, help='number of classes to be drawn from the buffer')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(CasperModel, self).__init__(backbone, loss, args, transform)

        self.nc = self.args.b_nclasses if self.args.b_nclasses is not None else self.N_CLASSES_PER_TASK

    def get_name_extension(self):
        name = '' if self.args.replay_mode == 'none' else 'Casper'
        if self.args.replay_weight == 0:
            return name
        name += f'NC{self.args.b_nclasses if self.args.b_nclasses is not None else self.N_CLASSES_PER_TASK}'
        name += f'K{self.args.knn_laplace}'
        return name

    def get_replay_loss(self):
        if self.args.replay_mode == 'none':
            return torch.tensor(0., dtype=torch.float, device=self.device)
        if self.args.rep_minibatch == self.args.buffer_size:
            buffer_data = self.buffer.get_all_data(self.transform)
        else:
            buffer_data = self.buffer.get_balanced_data(self.args.rep_minibatch, transform=self.transform,
                                                        n_classes=self.nc)
        inputs, labels = buffer_data[0], buffer_data[1]
        features = self.net.features(inputs)

        dists = calc_euclid_dist(features)
        A, D, L = calc_ADL_knn(dists, k=self.args.knn_laplace, symmetric=True)

        L = torch.eye(A.shape[0], device=A.device) - normalize_A(A, D)

        n = self.nc
        # evals = torch.linalg.eigvalsh(L)
        evals, _ = find_eigs(L, n_pairs=min(2*n, len(L)))

        #gaps = evals[1:] - evals[:-1]

        if self.args.replay_mode == 'casper':
            return evals[:n + 1].sum() - evals[n + 1]
