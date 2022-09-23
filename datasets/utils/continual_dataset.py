# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Iterator, Tuple, Union
from torchvision import datasets
import numpy as np
import socket
import torch
import os


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                         setting: ContinualDataset, class_order: Union[None, np.array] = None, dali=False) -> Tuple[DataLoader,
                                                                                                        DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :param class_order
    :return: train and test loaders
    """

    if class_order is not None:
        train_dataset.targets = class_order[np.array(train_dataset.targets)]
        test_dataset.targets = class_order[np.array(test_dataset.targets)]

    if 'seq-ilsvrc' not in setting.NAME:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                    np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
        test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                                   np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

        if not hasattr(train_dataset, 'lazy_load') or not train_dataset.lazy_load:
            train_dataset.data = train_dataset.data[train_mask]
        if not hasattr(test_dataset, 'lazy_load') or not test_dataset.lazy_load:
            test_dataset.data = test_dataset.data[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    if not dali:
        if 'MAMMOTH_RANK' not in os.environ:
            train_loader = DataLoader(train_dataset,
                                    batch_size=setting.args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset,
                                    batch_size=setting.args.batch_size,
                                    sampler=torch.utils.data.DistributedSampler(train_dataset, shuffle=True), num_workers=4)
        if not 'MAMMOTH_SLAVE' in os.environ:
            test_loader = DataLoader(test_dataset,
                                    batch_size=setting.args.batch_size, shuffle=False)
        else:
            test_loader = None
    else:
        try:
            from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
            from nvidia.dali.pipeline import pipeline_def
            import nvidia.dali.types as types
            import nvidia.dali.fn as fn
        except ImportError:
            raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
        @pipeline_def
        def get_dali_pipe(file_list, is_training):
            dali_cpu = True
            assert 'MAMMOTH_RANK' not in os.environ, "DALI is not supported with DDP"
            images, labels = fn.readers.file(file_list=file_list,
                                            shard_id=0,
                                            num_shards=1,
                                            random_shuffle=is_training,
                                            pad_last_batch=False,
                                            name="Reader")
            dali_device = 'cpu' if dali_cpu else 'gpu'
            decoder_device = 'cpu' if dali_cpu else 'mixed'
            # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
            device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
            host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
            # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
            preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
            preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
            if is_training:
                images = fn.decoders.image_random_crop(images,
                                                    device=decoder_device, output_type=types.RGB,
                                                    device_memory_padding=device_memory_padding,
                                                    host_memory_padding=host_memory_padding,
                                                    preallocate_width_hint=preallocate_width_hint,
                                                    preallocate_height_hint=preallocate_height_hint,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100)
                images = fn.resize(images,
                                device=dali_device,
                                resize_x=224,
                                resize_y=224,
                                interp_type=types.INTERP_TRIANGULAR)
                mirror = fn.random.coin_flip(probability=0.5)
            else:
                images = fn.decoders.image(images,
                                        device=decoder_device,
                                        output_type=types.RGB)
                images = fn.resize(images,
                                device=dali_device,
                                size=256,
                                mode="not_smaller",
                                interp_type=types.INTERP_TRIANGULAR)
                mirror = False

            images = fn.crop_mirror_normalize(images.gpu(),
                                            dtype=types.FLOAT,
                                            output_layout="CHW",
                                            crop=(224, 224),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                            mirror=mirror)
            labels = labels.gpu()
            return images, labels
        
        class DALIWrapper(object):
            def __init__(self, wrapped, dataset, train=False):
                self.wrapped = wrapped
                self.train = train
                self.dataset = dataset
                self.writer = None
            
            def __len__(self):
                return len(self.wrapped)

            def __iter__(self):
                self.writer = iter(self.wrapped)
                return self

            def __next__(self):
                data = next(self.writer)
                yeldo = (data[0]['data'], data[0]['label'].squeeze(1).long())
                if self.train:
                    yeldo += (data[0]['data'],)
                return yeldo

        current_files_file = os.path.join(train_dataset.root, f'_train_{setting.i}.txt')
        if not os.path.exists(current_files_file):
            with open(current_files_file, 'w') as f:
                f.write('\n'.join(f"{a.replace(train_dataset.root+'/', '')} {b}" for (a,b) in zip(train_dataset.data, train_dataset.targets)))
        # breakpoint()
        pipe = get_dali_pipe(current_files_file, True, num_threads=4, device_id=0,
                                seed=-1, batch_size=setting.args.batch_size)
        pipe.build()
        train_loader = DALIWrapper(DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL),
            train_dataset, True)

        current_files_file_te = os.path.join(test_dataset.root, f'_test_{setting.i}.txt')
        if not os.path.exists(current_files_file_te):
            bonifica_nome = lambda x: x.split('val/')[0] + 'val/ILSVRC2012_val_' + x.split('/ILSVRC2012_val_')[-1]
            with open(current_files_file_te, 'w') as f:
                f.write('\n'.join([f"{bonifica_nome(a.replace(train_dataset.root+'/', ''))} {b}" for (a,b) in zip(test_dataset.data, test_dataset.targets)]))

        pipe = get_dali_pipe(current_files_file_te, False, num_threads=2, device_id=0,
                                seed=-1, batch_size=2)
        pipe.build()
        test_loader = DALIWrapper(DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL),
            test_dataset, False)
        
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
