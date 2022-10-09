# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18, lopeznet, resnet32
from backbone.RebuffiNet import resnet_rebuffi
import torch.nn.functional as F
import numpy as np
from utils.conf import base_path_dataset
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from torchvision import datasets
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize

from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100


class SequentialCIFAR100_10x10SS(ContinualDataset):
    NAME = 'seq-cifar100-10x10ss'
    DATASET_NAME = 'CIFAR100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761))])

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path_dataset() + self.DATASET_NAME, train=True,
                                   download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path_dataset() + self.DATASET_NAME, train=True,
                                   download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path_dataset() + self.DATASET_NAME, train=False,
                                     download=True, transform=test_transform)

        class_order = None
        train, test = store_masked_loaders(train_dataset, test_dataset, self, class_order)

        # ss treatment
        assert 'perc_labels' in self.args, "use with SS models only"
        lt = len(train.dataset.targets)
        for c in np.unique(train.dataset.targets):
            subcl = np.array(train.dataset.targets) == c
            lc = np.sum(subcl)
            ss_mask = np.random.permutation(lc)[int(lc * self.args.perc_labels):]
            train.dataset.targets[subcl][ss_mask] += 1000
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100_10x10SS.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        return resnet18(SequentialCIFAR100_10x10SS.N_CLASSES_PER_TASK
                        * SequentialCIFAR100_10x10SS.N_TASKS, hookme=hookme)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform
