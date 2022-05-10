import copy
import os
from urllib import request

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from tqdm import tqdm

from backbone.ResNet18 import resnet18
from datasets import get_dataset
from datasets.seq_cifar100 import TCIFAR100, SequentialCIFAR100_20x5
from utils.buffer import Buffer
from utils.args import *
from models.utils.diagonal_model import DiagonalModel
import wandb
import numpy as np
from time import time

from utils.conf import base_path


def download_model(dataset_type: str):
    if dataset_type == "cifar100":
        link = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/Eb4BgDZ5g_1Imuwz_PJAmdgBc8k9I_P5p0Y-A97edhsxIw?e=WmlZZc"
        file = "rs18_cifar100.pth"
        n_classes = 100
    elif dataset_type == "tinyimgR":
        link = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EeWEOSls505AsMCTXAxWoLUBmeIjCiplFl40zDOCmB_lEw?download=1"
        file = "erace_pret_on_tinyr.pth"
        n_classes = 200
    else:
        raise ValueError
    local_path = os.path.join(base_path(), 'checkpoints')
    if not os.path.isdir(local_path):
        os.mkdir(local_path)
    if not os.path.isfile(os.path.join(local_path, file)):
        request.urlretrieve(link, os.path.join(local_path, file))
    return os.path.join(local_path, file), n_classes


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with functional mapping diagonal constraint.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    DiagonalModel.add_consolidation_args(parser)
    return parser


class ErACEDiag(DiagonalModel):
    NAME = 'er_ace_diag'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = self.N_TASKS * self.N_CLASSES_PER_TASK
        if args.wandb:
            wandb.init(project=self.args.experiment_name, entity="richiben", config=vars(args))
        self.log_results = []
        if self.args.pretrained_model is not None:
            local_path, n_classes = download_model(self.args.pretrained_model)
            real_net = resnet18(n_classes)
            state_dict = torch.load(local_path)
            real_net.load_state_dict(state_dict)
            real_net.to(self.device)
            start_classifier = self.net.classifier
            self.net = copy.deepcopy(real_net)
            self.new_classifier = real_net.classifier
            self.net.classifier = start_classifier

            self.load_buffer()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)
            self.pm_task(n_epochs=0)

    def load_buffer(self):
        if self.args.pretrained_model == 'cifar100':
            # todo: metto augmented o no?
            ds = CIFAR100(base_path() + 'CIFAR100'
                          , transform=SequentialCIFAR100_20x5.TRANSFORM, download=True)
            self.spectral_buffer_not_aug_transf = None
        else:
            raise ValueError
        dl = DataLoader(ds, self.args.spectral_buffer_size, shuffle=True)
        x, y = next(iter(dl))
        self.spectral_buffer.add_data(x, labels=y)

    def begin_task(self, dataset):
        pass

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'class_loss': None, 'diag_loss': None, 'task': self.task}

        self.opt.zero_grad()
        diag_loss = super().observe(inputs, labels, not_aug_inputs)
        wandb_log['diag_loss'] = diag_loss

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1
        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        class_loss = self.loss(logits, labels)
        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            class_loss += self.loss(self.net(buf_inputs), buf_labels)
        wandb_log['class_loss'] = class_loss.item()

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        loss = class_loss
        if diag_loss is not None and self.args.diag_weight > 0:
            loss += self.args.diag_weight * diag_loss
        wandb_log['loss'] = loss

        loss.backward()
        self.opt.step()

        if wandb.run:
            wandb.log(wandb_log)

        return loss.item()

    def end_task(self, dataset):
        if self.args.pretrained_model is not None:
            self.task += 1
            with torch.no_grad():
                self.eval()
                evects = self.compute_buffer_evects()
                self.train()
            self.buffer_evectors = [evects]
        else:
            super().end_task(dataset)

    def pm_task(self, n_epochs=1):
        if self.args.pretrained_model == 'cifar100':
            # freeze parameters that are not in the classifier
            pm_dataset_train = TCIFAR100(base_path() + 'CIFAR100', train=True,
                                         download=True, transform=SequentialCIFAR100_20x5.TRANSFORM)
            pm_dataset_test = TCIFAR100(base_path() + 'CIFAR100', train=False,
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      SequentialCIFAR100_20x5.get_normalization_transform()]))
            acc = self.pm_eval(pm_dataset_test)
            if n_epochs > 0:
                self.pm_train(pm_dataset_train, n_epochs)
            else:
                return acc

            acc = self.pm_eval(pm_dataset_test)
            return acc

    def pm_train(self, pm_dataset_train, n_epochs=1):
        self.new_classifier = nn.Linear(512, 100).to(self.device)
        train_loader = DataLoader(pm_dataset_train,
                                  batch_size=self.args.batch_size)
        opt = SGD(self.new_classifier.parameters(), lr=self.args.lr)
        for epoch in range(n_epochs):
            for i, data in tqdm(enumerate(train_loader)):
                opt.zero_grad()
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    out = self.net.features(inputs)
                out = self.new_classifier(out)
                loss = self.loss(out, labels)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def pm_eval(self, pm_dataset_test):

        test_loader = DataLoader(pm_dataset_test,
                                 batch_size=self.args.batch_size, shuffle=False)
        self.net.eval()
        correct, total = 0.0, 0.0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            out = self.net.features(inputs)
            out = self.new_classifier(out)
            _, pred = torch.max(out.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
        print(correct / total * 100)

        self.net.train()
        return correct / total * 100

    def log_accs(self, accs):
        log = {}
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()
        if self.args.pretrained_model is not None:
            pm_acc = self.pm_task(1)
            log.update({'Pretrained Model-acc': pm_acc})

        # running consolidation error
        diag_error = None
        c_0 = None
        if self.task > 1:
            with torch.no_grad():
                diag_error, c_0 = self.get_off_diagonal_error(return_c=True)
                diag_error = diag_error.item()
                print(f'diag err: {diag_error}')

        log.update({'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Diag-Error': diag_error})

        if wandb.run:
            log.update({**{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
                        **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])}})
            # wandb.log({"chart": plt})
            # log.update({})
            wandb.log(log)
        log.update({'c0': c_0.tolist() if c_0 is not None else c_0, 'task': self.task})
        self.log_results.append(log)
        #     exit()

        if self.task == self.N_TASKS:
            self.end_training()

    def end_training(self, print_latents=False):
        if self.args.print_custom_log:
            log_dir = os.path.join(base_path(), f'logs/{self.dataset_name}/{self.NAME}')
            obj = {**vars(self.args), 'results': self.log_results}
            self.print_logs(log_dir, obj, name='results')
        else:
            ...
        # if print_latents:
        #     logs_dir = f'/nas/softechict-nas-2/efrascaroli/mammoth-data/logs/{self.dataset_name}/{self.NAME}'
        #     self.print_logs(logs_dir, self.custom_log, name='latents')

