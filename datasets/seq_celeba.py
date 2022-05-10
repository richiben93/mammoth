# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from itertools import product

import PIL
import numpy as np
import torch
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple, Any, List
from datasets.transforms.denormalization import DeNormalize


class MyCelebA(CelebA):
    NAME = 'celeba'
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, split='train', transform=None,
                 target_transform=None, download=False, target_idxes: List[int] = None,
                 target_names: List[str] = None,
                 delete_bad_attr=False, take_only_unique=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, split, download=download, transform=transform, target_transform=target_transform)
        assert target_idxes is not None or target_names is not None, 'Give one of target_names or target_idxes'
        assert (target_idxes is None and target_names is not None) or \
               (target_idxes is not None and target_names is None), 'Only one between target_idxes and target_names ' \
                                                                    'has to be not None'
        if target_names is not None:
            target_idxes = [np.where(np.array(self.attr_names) == a)[0][0] for a in target_names]

        if delete_bad_attr:
            bad_attr = ['Wearing_Hat', 'Bald', 'Receding_Hairline']
            bad_attr_ids = [np.where(np.array(self.attr_names) == a)[0][0] for a in bad_attr]
            ## images with a chosen attribute (only one per attribute to avoid ambiguity)
            proper_ids = np.where((self.attr[:, bad_attr_ids].sum(1) == 0))[0]
            self.filename = np.array(self.filename)[proper_ids].tolist()
            self.attr = self.attr[proper_ids]

        self.target_idxes = target_idxes
        targets = product([0, 1], repeat=len(target_idxes))
        if take_only_unique:
            targets = [x for x in targets if sum(x) == 1]

        proper_ids = torch.concat(
            [torch.where((torch.Tensor(tar)[None, :].broadcast_to(self.attr[:, target_idxes].shape)
                          == self.attr[:, target_idxes]).all(1))[0] for tar in targets])
        self.filename = np.array(self.filename)[proper_ids].tolist()
        self.attr = self.attr[proper_ids]
        self.trans_dict = {t: num for num, t in enumerate(targets)}
        self.targets = [self.trans_dict[tuple(x.tolist())] for x in self.attr[:, target_idxes]]

        if split == 'train':
            # class balancing for train
            unique, counts = np.unique(self.targets, return_counts=True)
            min_val = np.min(counts)
            t_arr = np.array(self.targets)
            idxes = []
            for v in unique:
                idxes.append(np.where(t_arr == v)[0][:min_val])
            proper_ids = np.concatenate(idxes)
            self.filename = np.array(self.filename)[proper_ids].tolist()
            self.attr = self.attr[proper_ids]
            self.targets = [self.trans_dict[tuple(x.tolist())] for x in self.attr[:, target_idxes]]

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        target = self.trans_dict[tuple(target[0][self.target_idxes].tolist())]
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class MyCelebATest(MyCelebA):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False,
                 target_idxes: List[int] = None, target_names: List[str] = None, delete_bad_attr=False,
                 take_only_unique=False):
        super().__init__(root, split, transform, target_transform, download, target_idxes, target_names,
                         delete_bad_attr, take_only_unique)

    def __getitem__(self, item):
        img, target = CelebA.__getitem__(self, item)
        return img, self.trans_dict[tuple(target[self.target_idxes].tolist())]


class SequentialCelebA(ContinualDataset):
    NAME = 'seq-celeba'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2
    IMG_SIZE = 64
    TRANSFORM = transforms.Compose(
        [transforms.Resize(IMG_SIZE),
         # transforms.RandomHorizontalFlip(),
         transforms.CenterCrop(IMG_SIZE),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    TARGET_IDXES = None
    TARGET_NAMES = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCelebA(base_path(), split='train', download=True, transform=transform,
                                 target_idxes=self.TARGET_IDXES, target_names=self.TARGET_NAMES,
                                 delete_bad_attr=True, take_only_unique=True)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = MyCelebATest(base_path(), split='test', download=True, transform=transform,
                                        target_idxes=self.TARGET_IDXES, target_names=self.TARGET_NAMES,
                                        delete_bad_attr=True, take_only_unique=True)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCelebA(base_path(), split='train', download=True, transform=transform,
                                 target_idxes=self.TARGET_IDXES)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCelebA.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCelebA.N_CLASSES_PER_TASK
                        * SequentialCelebA.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return transform
