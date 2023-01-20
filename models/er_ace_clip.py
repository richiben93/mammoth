# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from torch.functional import F
from models.utils.continual_model import ContinualModel
from models.er_ace import ErACE
from copy import deepcopy
from utils.no_bn import bn_track_stats
import os
import pickle
from models.icarl import baguette_fill_buffer
from transformers import CLIPModel, CLIPTokenizer



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--clip_weight', type=float, default=1)
    parser.add_argument('--clip_stream', action='store_true')
    return parser

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class ErACEClip(ErACE):
    NAME = 'er_ace_clip'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACEClip, self).__init__(backbone, loss, args, transform)
        classes = [name.replace('_', ' ') for name in self.class_names]
        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(self.device)
        with torch.no_grad():
            self.class_features = torch.stack([clip_model.get_text_features(**tokenizer(text, return_tensors='pt').to(self.device)) for text in classes]).squeeze()

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.max() < (self.N_CLASSES - 1):
            mask[:, self.seen_so_far.max():] = 1
        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)

        if self.task > 0 and not self.buffer.is_empty():
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)
            self.wb_log['erace_loss'] = loss_re.item()
            loss += loss_re

        if self.task > 0 and not self.buffer.is_empty():
            buf_inputs, buf_labels, = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            if self.args.clip_stream:
                buf_inputs = torch.cat([buf_inputs, not_aug_inputs])
                buf_labels = torch.cat([buf_labels, labels])

            embeds = self.net.features(buf_inputs)
            clip_embeds = self.class_features[buf_labels]
            dist_space = sim_matrix(embeds, embeds)
            dist_clip = sim_matrix(clip_embeds, clip_embeds)
            loss_re = F.mse_loss(dist_space, dist_clip)
            self.wb_log['clip_loss'] = loss_re.item()
            loss += loss_re * self.args.clip_weight

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
