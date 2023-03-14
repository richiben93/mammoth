from copy import deepcopy

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from backbone.SupCon_Resnet import SupConResNet
from utils.augmentations import normalize
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.no_bn import bn_track_stats
import numpy as np

from utils.supconloss import SupConLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via SCR.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--temp', type=float, required=True,
                        help='Temperature for loss.')
    parser.add_argument('--head', type=str, required=False, default='mlp')
    return parser


input_size_match = {
    'cifar100': [3, 32, 32],
    'seq-cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50]
}


class SCR(ContinualModel):
    NAME = 'scr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        if args.dataset == 'mini_imagenet':
            backbone = SupConResNet(640, head=args.head)
        else:
            backbone = SupConResNet(head=args.head)
        super(SCR, self).__init__(backbone, loss, args, transform)
        self.class_means = None
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.args.dataset][1], input_size_match[self.args.dataset][2]),
                              scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            # ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            ColorJitter(0.4, 0.4, 0.4, 0.1),
            RandomGrayscale(p=0.2)

        )

        self.temp = args.temp
        self.loss = SupConLoss(temperature=self.args.temp)
        self.current_task = 0

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        # feats = self.net.features(x).squeeze()
        feats = self.net.features(x).float().squeeze()

        feats = feats.reshape(feats.shape[0], -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())
        loss = 0
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size)
            comb_inputs = torch.cat((not_aug_inputs, buf_inputs))
            comb_transformed_inputs = self.transform(comb_inputs)
            comb_labels = torch.cat((labels, buf_labels))
            pred = torch.cat([self.net(comb_inputs).unsqueeze(1), self.net(comb_transformed_inputs).unsqueeze(1)],
                             dim=1)
            loss = self.loss(pred, comb_labels)
            loss.backward()

            self.opt.step()
            loss = loss.item()
        self.buffer.add_data(not_aug_inputs, labels)

        return loss

    # def begin_task(self, dataset):
    #     if self.current_task > 0:
    #         dataset.train_loader.dataset.targets = np.concatenate(
    #             [dataset.train_loader.dataset.targets,
    #              self.buffer.labels.cpu().numpy()[:self.buffer.num_seen_examples]])
    #         if type(dataset.train_loader.dataset.data) == torch.Tensor:
    #             dataset.train_loader.dataset.data = torch.cat(
    #                 [dataset.train_loader.dataset.data, torch.stack([(
    #                     self.buffer.examples[i].type(torch.uint8).cpu())
    #                     for i in range(self.buffer.num_seen_examples)]).squeeze(1)])
    #         else:
    #             dataset.train_loader.dataset.data = np.concatenate(
    #                 [dataset.train_loader.dataset.data, torch.stack([((self.buffer.examples[i] * 255).type(
    #                     torch.uint8).cpu()) for i in range(self.buffer.num_seen_examples)]).numpy().swapaxes(
    #                     1, 3)])

    def end_task(self, dataset) -> None:

        # self.new_labels_zombie = deepcopy(self.new_labels)
        # self.new_labels.clear()
        self.net.train()
        if type(dataset.train_loader.dataset.data) == torch.Tensor:
            mem_x = torch.stack([(
                self.buffer.examples[i].type(torch.uint8).cpu())
                for i in range(self.buffer.num_seen_examples)]).squeeze(1)
        else:
            mem_x = self.buffer.examples[:self.buffer.num_seen_examples].cpu()
        mem_y = self.buffer.labels.cpu()[:self.buffer.num_seen_examples].float()
        # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        if mem_x.size(0) > 0:
            rv_dataset = TensorDataset(mem_x, mem_y)
            rv_loader = DataLoader(rv_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0,
                                   drop_last=True)
            for i, batch_data in enumerate(rv_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = torch.cat([self.net.forward(batch_x).unsqueeze(1),
                                    self.net.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                loss = self.loss(logits, batch_y)
                self.opt.zero_grad()
                loss.backward()
                params = [p for p in self.net.parameters() if p.requires_grad and p.grad is not None]
                grad = [p.grad.clone() / 10. for p in params]
                for g, p in zip(grad, params):
                    p.grad.data.copy_(g)
                self.opt.step()
        self.current_task += 1
        self.class_means = None

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                class_means.append(self.net.features(x_buf).mean(0).flatten())
        self.class_means = torch.stack(class_means)
