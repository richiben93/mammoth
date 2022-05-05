import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_model import ConsolidationModel
import wandb
import numpy as np
from time import time


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Consolidation.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    ConsolidationModel.add_consolidation_args(parser)
    return parser


class ErACECon(ConsolidationModel):
    NAME = 'er_ace_con'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACECon, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = self.N_TASKS * self.N_CLASSES_PER_TASK
        if args.wandb:
            wandb.init(project="rodo-mammoth", entity="ema-frasca", config=vars(args))
        self.log_results = []

    def observe(self, inputs, labels, not_aug_inputs):
        wandb_log = {'loss': None, 'class_loss': None, 'con_loss': None, 'task': self.task}

        self.opt.zero_grad()
        con_loss = super().observe(inputs, labels, not_aug_inputs)
        wandb_log['con_loss'] = con_loss

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
        if wandb.run:
            wandb.log(wandb_log)

        return loss.item()

    def end_task(self, dataset):
        super().end_task(dataset)

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()

        # running consolidation error
        con_error = None
        if self.task > 1:
            with torch.no_grad():
                con_error = self.get_consolidation_error().item()
                print(f'con err: {con_error}')

        if wandb.run:
            wandb.log({'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': con_error,
                       **{f'Class-IL task-{i+1}': acc for i, acc in enumerate(accs[0])},
                       **{f'Task-IL task-{i+1}': acc for i, acc in enumerate(accs[1])}
                       })

        self.log_results.append({'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': con_error})

        # if self.task > 3:
        #     log_dir = f'/nas/softechict-nas-2/efrascaroli/mammoth-data/logs/{self.dataset_name}/{self.NAME}'
        #     obj = {**vars(self.args), 'results': self.log_results}
        #     self.print_logs(log_dir, obj, name='results')
        #     exit()

        if self.task == self.N_TASKS:
            self.end_training()

    def end_training(self, print_latents=False):
        if print_latents:
            logs_dir = f'/nas/softechict-nas-2/efrascaroli/mammoth-data/logs/{self.dataset_name}/{self.NAME}'
            self.print_logs(logs_dir, self.custom_log, name='latents')
