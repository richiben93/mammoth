import torch
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Replay of something else.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size

    parser.add_argument('--perc_labels', type=float, default=0.25, help='Percentage of labels to use for training.')

    
    return parser


class ErSS(ContinualModel):
    NAME = 'er_ss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErSS, self).__init__(backbone, loss, args, transform)
        
    def get_name(self):
        return 'ErSS' + self.get_name_extension()

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        sup_mask = labels < 1000

        logits = self.net(inputs)
        if sup_mask.sum():
            class_loss = self.loss(logits[sup_mask], labels[sup_mask])
        else:
            class_loss = torch.tensor(0.).to(self.device)
        self.wb_log['class_loss'] = class_loss.item()
        loss = class_loss
        if self.task > 0 and self.args.buffer_size > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            er_loss = self.loss(self.net(buf_inputs), buf_labels)
            self.wb_log['er_loss'] = er_loss.item()
            loss += er_loss

        if self.args.buffer_size > 0 and sup_mask.sum() > 0:
            self.buffer.add_data(examples=not_aug_inputs[sup_mask], labels=labels[sup_mask])

        if loss.requires_grad:
            loss.backward()
        self.opt.step()

        return loss.item()
