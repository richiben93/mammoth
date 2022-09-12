import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_model import ConsolidationModel
from utils.wandbsc import WandbLogger
import numpy as np
from time import time


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    ConsolidationModel.add_consolidation_args(parser)
    return parser


class ErCon(ConsolidationModel):
    NAME = 'er_con'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErCon, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.wblog = WandbLogger(args)
        self.log_results = []

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'class_loss': None, 'con_loss': None, 'task': self.task}
        con_loss = super().observe(inputs, labels, not_aug_inputs)
        wandb_log['con_loss'] = con_loss

        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels[:real_batch_size])

        outputs = self.net(inputs)
        class_loss = self.loss(outputs, labels)
        wandb_log['class_loss'] = class_loss

        # running accuracy on task 2
        if self.task > 1 and self.args.wandb:
            with torch.no_grad():
                count = 0
                acc = 0
                for l in [2, 3]:
                    count += (labels == l).sum().item()
                    acc += (outputs[labels == l].argmax(1) == l).sum().item()
                wandb_log['task-2 cl acc'] = acc / count


        loss = class_loss
        if con_loss is not None and self.args.con_weight > 0:
            loss += self.args.con_weight * con_loss
        wandb_log['loss'] = loss

        # t1 = time()
        loss.backward()
        # t2 = time()
        self.opt.step()

        # bw_time = None
        # if self.args.profiler:
        #     bw_time = t2-t1
        self.wblog({'training': wandb_log})

        return loss.item()

    def end_task(self, dataset):
        super().end_task(dataset)


    def end_training(self, print_latents=False):
        if print_latents:
            logs_dir = f'/nas/softechict-nas-2/efrascaroli/mammoth-data/logs/{self.dataset_name}/{self.NAME}'
            self.print_logs(logs_dir, self.custom_log, name='latents')
