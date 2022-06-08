import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_model import ConsolidationModel
import wandb
import numpy as np
from time import time
from utils.conf import base_path_dataset, base_path

from torchvision.datasets import CIFAR100
from datasets.seq_tinyimagenet import TinyImagenet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD

# name: path
checkpoints = {
    'cifar100': {
        'name': 'Cifar100',
        'ds_path': base_path_dataset(),
        'path': base_path() + 'checkpoints/rs18_cifar100_new.pth',
        'dataset': CIFAR100,
        'transform': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
        'n_logits': 100,
        'n_features': 512,
        'lr': 0.1,
    },
    'tinyimg': {
        'name': 'TinyImageNet',
        'ds_path': base_path_dataset() + 'TINYIMG',
        'path': base_path() + 'checkpoints/pre_tiny_fixed.pth',
        'dataset': TinyImagenet,
        'transform': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))]),
        'n_logits': 200,
        'n_features': 512,
        'lr': 0.1,
    },
}


class PretrainedConsolidationModel(ConsolidationModel):
    @staticmethod
    def add_consolidation_args(parser: ArgumentParser):
        ConsolidationModel.add_consolidation_args(parser)
        parser.add_argument('--pre_dataset', type=str, choices=list(checkpoints.keys()),
                            default=list(checkpoints.keys())[0],
                            help='Dataset of pre-training')
        parser.add_argument('--pretrain_epochs', type=int, default=1,
                            help='Head fine-tuning epochs')

    def __init__(self, backbone, loss, args, transform):
        super(PretrainedConsolidationModel, self).__init__(backbone, loss, args, transform)

        self.checkpoint_data = checkpoints[args.pre_dataset]
        saved_dict = torch.load(self.checkpoint_data['path'])
        pre_head_w = saved_dict.pop('classifier.weight')
        pre_head_b = saved_dict.pop('classifier.bias')
        self.pre_classifier = self.get_pre_classifier()
        self.pre_classifier.load_state_dict({'weight': pre_head_w, 'bias': pre_head_b})
        self.net.load_state_dict(saved_dict, strict=False)
        self.pre_classifier.to(self.device)
        self.net.to(self.device)

        self.pre_dataset_train_head(n_epochs=0)
        self.load_buffer()
        with torch.no_grad():
            self.eval()
            evects = self.compute_buffer_evects()
            self.train()
        self.buffer_evectors.append(evects)

    def load_buffer(self):
        ds = self.get_pre_dataset()
        dl = DataLoader(ds, self.args.spectral_buffer_size, shuffle=True)
        x, y = next(iter(dl))
        self.spectral_buffer.add_data(x, labels=y)

    def compute_buffer_latents(self):
        inputs, labels = self.spectral_buffer.get_all_data()
        latents = self.net.features(inputs)
        return latents, labels

    def get_pre_classifier(self):
        return torch.nn.Linear(self.checkpoint_data['n_features'], self.checkpoint_data['n_logits']).to(self.device)

    def get_pre_dataset(self, train=True):
        return self.checkpoint_data['dataset'](self.checkpoint_data['ds_path'], transform=self.checkpoint_data['transform'], train=train, download=True)

    def pre_dataset_train_head(self, n_epochs: int = None):
        if n_epochs is None:
            n_epochs = self.args.pretrain_epochs
        acc = self.pre_dataset_test()
        print(f'\n{self.checkpoint_data["name"]} [old head] accuracy: {acc}')
        if n_epochs < 1:
            return acc

        self.net.eval()
        ds = self.get_pre_dataset()
        dl = DataLoader(ds, 32, shuffle=True)
        self.pre_classifier = self.get_pre_classifier()
        opt = SGD(self.pre_classifier.parameters(), lr=self.checkpoint_data['lr'])
        for epoch in range(n_epochs):
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                with torch.no_grad():
                    features = self.net.features(x)
                logits = self.pre_classifier(features)
                loss = self.loss(logits, y)
                loss.backward()
                opt.step()
            acc = self.pre_dataset_test()
            print(f'{self.checkpoint_data["name"]} [epoch-{epoch}] accuracy: {acc}')

        self.net.train()
        return acc

    @torch.no_grad()
    def pre_dataset_test(self):
        ds = self.get_pre_dataset(train=False)
        dl = DataLoader(ds, 32)
        acc = 0
        self.net.eval()
        for x, y in dl:
            x, y = x.to(self.device), y.to(self.device)
            features = self.net.features(x)
            logits = self.pre_classifier(features)
            acc += (logits.argmax(1) == y).sum().item()
        acc /= len(ds)
        self.net.train()
        return acc
