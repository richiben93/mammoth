import torch

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs, calc_ADL_heat
import os
import pickle


class EgapModel(ContinualModel):

    @staticmethod
    def add_replay_args(parser):
        parser.add_argument('--rep_minibatch', type=int, default=-1,
                            help='Size of pre-dataset minibatch replay (for x, lats and dists).')
        parser.add_argument('--replay_mode', type=str, required=True, help='What you replay.',
                            choices=['none', 'egap', 'egap2', 'egap2-1', 'egap2+1', 'egap3', 'egap2m',
                                     'egapB2', 'egapB2-1', 'gkd'])

        parser.add_argument('--replay_weight', type=float, required=True, help='Weight of replay.')

        parser.add_argument('--heat_kernel', action='store_true', help='Use heat kernel instead of knn.')
        parser.add_argument('--cos_dist', action='store_true', help='Use cosine distance.')
        parser.add_argument('--knn_laplace', type=int, default=10,
                            help='K of knn to build the graph for laplacian.')
        parser.add_argument('--b_nclasses', default=None, type=int, help='number of classes to be drawn in egap2b')
        return parser

    def __init__(self, backbone, loss, args, transform):
        if args.rep_minibatch < 0:
            args.rep_minibatch = args.buffer_size
        if args.replay_mode == 'none' or args.replay_weight == 0:
            args.replay_mode = 'none'
            args.replay_weight = 0
        super(EgapModel, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balancoir')

        if len(self.args.replay_mode) > 4 and self.args.replay_mode[4] == 'B':
            self.nc = self.args.b_nclasses if self.args.b_nclasses is not None else self.N_CLASSES_PER_TASK

    def get_name(self):
        return self.NAME.capitalize() + self.get_name_extension()

    def get_name_extension(self):
        name = self.args.replay_mode.capitalize()
        if self.args.replay_weight == 0:
            return name
        if len(self.args.replay_mode) > 4 and self.args.replay_mode[4] == 'B':
            name += f'NC{self.args.b_nclasses if self.args.b_nclasses is not None else self.N_CLASSES_PER_TASK}'
        if self.args.cos_dist:
            name += 'Cos'
        if self.args.heat_kernel:
            name += 'Heat'
        else:
            name += f'K{self.args.knn_laplace}'
        return name

    def get_replay_loss(self):
        if self.args.replay_mode == 'none':
            return torch.tensor(0., dtype=torch.float, device=self.device)
        if self.args.rep_minibatch == self.args.buffer_size:
            buffer_data = self.buffer.get_all_data(self.transform)
        elif len(self.args.replay_mode) > 4 and self.args.replay_mode[4] == 'B':
            buffer_data = self.buffer.get_balanced_data(self.args.rep_minibatch, transform=self.transform,
                                                        n_classes=self.nc)
        else:
            buffer_data = self.buffer.get_data(self.args.rep_minibatch, self.transform)
        inputs, labels = buffer_data[0], buffer_data[1]
        features = self.net.features(inputs)

        dists = calc_cos_dist(features) if self.args.cos_dist else calc_euclid_dist(features)

        if self.args.heat_kernel:
            A, D, L = calc_ADL_heat(dists)
        else:
            A, D, L = calc_ADL_knn(dists, k=self.args.knn_laplace, symmetric=True)

        if self.args.replay_mode == 'gkd':
            lab_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            return A[~lab_mask].sum()

        L = torch.eye(A.shape[0], device=A.device) - normalize_A(A, D)

        n = self.nc if len(self.args.replay_mode) > 4 and self.args.replay_mode[4] == 'B' else self.N_CLASSES_PER_TASK * self.task
        # evals = torch.linalg.eigvalsh(L)
        evals, _ = find_eigs(L, n_pairs=min(2*n, len(L)))

        gaps = evals[1:] - evals[:-1]
        self.wb_log['egap'] = torch.argmax(gaps).item()
        self.wb_log['egap-k-1'] = gaps[n-1].item()
        self.wb_log['egap-k']   = gaps[n].item()
        # self.wb_log['egap-k+1'] = gaps[n+1].item()
        # log evals
        # decode: pickle.loads(codecs.decode(evals.encode(), "base64"))
        # self.wb_log['evals'] = codecs.encode(pickle.dumps(evals2.detach().cpu()), "base64").decode()

        if self.args.replay_mode == 'egap':
            return -gaps[n]

        if self.args.replay_mode == 'egap2':
            return evals[:n + 1].sum() - evals[n + 1]

        if self.args.replay_mode == 'egap2m':
            return evals[:n + 1].mean() - evals[n + 1]

        if self.args.replay_mode == 'egap2-1':
            return evals[:n].sum() - evals[n]

        if self.args.replay_mode == 'egap2+1':
            return evals[:n + 2].sum() - evals[n + 2]

        if self.args.replay_mode == 'egap3':
            return evals[:n].mean()

        if self.args.replay_mode == 'egapB2-1':
            return evals[:n].sum() - evals[n]

        if self.args.replay_mode == 'egapB2':
            return evals[:n + 1].sum() - evals[n + 1]

    def save_checkpoint(self):
        log_dir = super().save_checkpoint()
        ## pickle the future_buffer
        with open(os.path.join(log_dir, f'task_{self.task}_buffer.pkl'), 'wb') as f:
            self.buffer.to('cpu')
            pickle.dump(self.buffer, f)
            self.buffer.to(self.device)
