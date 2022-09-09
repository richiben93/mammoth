# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18_imagenet
import torch.nn.functional as F
from utils.conf import base_path_dataset
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize
from urllib.request import urlopen


class Imagenet100(Dataset):
    N_CLASSES = 100
    classes = [f'class_{i}' for i in range(N_CLASSES)]

    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),transforms.ToTensor()])
            
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if not os.path.exists(root):
                # download file list
                print('Downloading file list...', end=' ')
                os.makedirs(root, exist_ok=True)
                with open(os.path.join(root, 'train_100.txt'), 'w') as f:
                    with urlopen('https://raw.githubusercontent.com/arthurdouillard/incremental_learning.pytorch/master/imagenet_split/train_100.txt') as urlf:
                        f.write(urlf.read().decode())

                with open(os.path.join(root, 'val_100.txt'), 'w') as f:
                    with urlopen('https://raw.githubusercontent.com/arthurdouillard/incremental_learning.pytorch/master/imagenet_split/val_100.txt') as urlf:
                        f.write(urlf.read().decode())
                print('Done.')
            else:
                print('Download not needed')
            if not os.path.exists(os.path.join(root, 'train')):
                raise NotImplementedError('TODO implement download')
            # if os.path.isdir(root) and len(os.listdir(root)) > 0:
            #     print('Download not needed, files already on disk.')
            # else:
            #     from onedrivedownloader import download

            #     ln = 'https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EbG7ptzea7pHg38XZif7mTsBnhmX_WXoD2WA-Q6eKRg9Hw?download=1'
            #     # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            #     print('Downloading dataset')
            #     download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root,
            #              clean=True)

        metadata_path = os.path.join(root, f'{"train" if self.train else "val"}_100.txt')
        self.data, self.targets = [], []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")
                self.data.append(os.path.join(root, path))
                self.targets.append(int(target))
        self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert("RGB")
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyImagenet100(Imagenet100):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False) -> None:
        super(MyImagenet100, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert("RGB")
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


def base_path():
    return "/tmp/mbosc/"


class SequentialImagenet100(ContinualDataset):
    NAME = 'seq-imgnet100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(brightness=63 / 255),
         transforms.ToTensor(),
         transforms.Normalize(MEAN,
                              STD)
                              ])
    TEST_TRANSFORM = transforms.Compose([
                transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
         transforms.Normalize(MEAN,
                              STD)])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = self.TEST_TRANSFORM if hasattr(self, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyImagenet100(base_path_dataset() + 'IMGNET100',
                                       train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = Imagenet100(base_path_dataset() + 'IMGNET100',
                                        train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self, dali=True)
        return train, test

    @staticmethod
    def get_backbone():
        return resnet18_imagenet(SequentialImagenet100.N_CLASSES_PER_TASK
                        * SequentialImagenet100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialImagenet100.MEAN,
                                         SequentialImagenet100.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialImagenet100.MEAN,
                                SequentialImagenet100.STD)
        return transform
