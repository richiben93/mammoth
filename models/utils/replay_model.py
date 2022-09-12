import torch
from torch.functional import F
from torch.utils.data import DataLoader

from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs, calc_ADL_heat
import os
import pickle
import codecs
import wandb


class ReplayModel(ContinualModel):

    @staticmethod
    def add_replay_args(parser):
        parser.add_argument('--rep_minibatch', type=int, default=-1,
                            help='Size of pre-dataset minibatch replay (for x, lats and dists).')
        parser.add_argument('--replay_mode', type=str, required=True, help='What you replay.',
                            choices=['none', 'features', 'dists', 'graph', 'laplacian', 'evec', 'fmap', 'eval', 'egap',
                                     'fmeval-0101', 'fmeval-0110', 'fmeval-1001', 'fmeval-1010',
                                     'evalgap', 'evalgap2', 'egap2', 'egap2-1', 'egap2+1', 'egap3', 'egap2m'])

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

    def __init__(self, backbone, loss, args, transform):
        if args.rep_minibatch < 0:
            args.rep_minibatch = args.buffer_size
        if args.replay_mode == 'none' or args.replay_weight == 0:
            args.replay_mode = 'none'
            args.replay_weight = 0
        if args.replay_mode not in ['graph', 'laplacian']:
            args.graph_sym = True
        super(ReplayModel, self).__init__(backbone, loss, args, transform)

        self.fixed_buffer = Buffer(self.args.buffer_size, self.device)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def get_name(self):
        return self.NAME.capitalize() + self.get_name_extension()

    def get_name_extension(self):
        name = self.args.replay_mode.capitalize()
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
            if self.args.replay_mode.startswith('egap'):
                buffer_data = self.buffer.get_all_data(self.transform)
                inputs, labels = buffer_data[0], buffer_data[1]
                features1 = self.net.features(inputs).detach()
            else:
                inputs, labels, features1 = self.fixed_buffer.get_all_data(self.transform)
        else:
            if self.args.replay_mode.startswith('egap'):
                buffer_data = self.buffer.get_data(self.args.rep_minibatch, self.transform)
                inputs, labels = buffer_data[0], buffer_data[1]
                features1 = self.net.features(inputs).detach()
            else:
                inputs, labels, features1 = self.fixed_buffer.get_data(self.args.rep_minibatch, self.transform)
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
        self.wb_log['egap'] = torch.argmax(gaps).item()
        # log evals
        # decode: pickle.loads(codecs.decode(evals.encode(), "base64"))
        # self.wb_log['evals'] = codecs.encode(pickle.dumps(evals2.detach().cpu()), "base64").decode()

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
            return evals2[:n + 1].sum() - evals2[n + 1]

        if self.args.replay_mode == 'egap2m':
            n = self.N_CLASSES_PER_TASK * self.task
            return evals2[:n + 1].mean() - evals2[n + 1]

        if self.args.replay_mode == 'egap3':
            n = self.N_CLASSES_PER_TASK * self.task
            return evals2[:n].mean()

        if self.args.replay_mode == 'egap2-1':
            n = self.N_CLASSES_PER_TASK * self.task
            return evals2[:n].sum() - evals2[n]

        if self.args.replay_mode == 'egap2+1':
            n = self.N_CLASSES_PER_TASK * self.task
            return evals2[:n + 2].sum() - evals2[n + 2]

        if self.args.replay_mode == 'evalgap':
            n = self.N_CLASSES_PER_TASK * self.task
            return -gaps[n] + F.mse_loss(evals2[:n], evals1[:n])

        if self.args.replay_mode == 'evalgap2':
            n = self.task
            return -gaps[n] + F.mse_loss(evals2[:n + 1], evals1[:n + 1])

        if self.args.replay_mode.startswith('fmeval'):
            codes = [int(c) for c in self.args.replay_mode.rsplit('-')[1]]
            assert len(codes) == 4
            evects = [evects2, evects1]
            evals = [evals2, evals1]
            return F.mse_loss((evects[codes[0]].T @ evects[codes[1]]) @ torch.diag(evals[0]),
                              torch.diag(evals[1]) @ (evects[codes[2]].T @ evects[codes[3]]))

    def end_task(self, dataset):
        self.task += 1
        # buffer <- future_buffer (con logits)
        if not self.args.replay_mode.startswith('egap') and self.args.replay_mode != 'none':
            self.sync_buffers()

    @torch.no_grad()
    def sync_buffers(self):
        self.net.eval()
        buffer_data = self.buffer.get_all_data(self.transform)
        inputs, labels = buffer_data[0], buffer_data[1]
        no_aug_inputs = self.buffer.get_all_data()[0]
        features = self.net.features(inputs)
        self.net.train()

        self.fixed_buffer.empty()
        self.fixed_buffer.add_data(no_aug_inputs, labels=labels, logits=features)

    def save_checkpoint(self):
        log_dir = super().save_checkpoint()
        ## pickle the future_buffer
        with open(os.path.join(log_dir, f'task_{self.task}_buffer.pkl'), 'wb') as f:
            self.buffer.to('cpu')
            pickle.dump(self.buffer, f)
            self.buffer.to(self.device)
