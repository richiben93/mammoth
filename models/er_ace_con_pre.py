import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_pretrain import PretrainedConsolidationModel
import numpy as np
from time import time
from utils.conf import base_path
from utils.wandbsc import WandbLogger
import os


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Consolidation.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    PretrainedConsolidationModel.add_consolidation_args(parser)
    return parser


class ErACEConPre(PretrainedConsolidationModel):
    NAME = 'er_ace_con_pre'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACEConPre, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = self.N_TASKS * self.N_CLASSES_PER_TASK
        self.args.name = 'EraceConPre' if self.args.con_weight > 0 else 'Erace'
        self.wblog = WandbLogger(args, name=self.args.name)
        self.log_results = []
        self.log_latents = []
        self.add_log_latents()

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
        if self.task > 0 and self.args.buffer_size > 0:
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
        self.wblog({'training': wandb_log})
        return loss.item()

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()
        pre_acc = self.pre_dataset_train_head()

        # running consolidation error
        con_error = None
        if self.task > 0:
            with torch.no_grad():
                con_error = self.get_consolidation_error().item()
                # print(f'con err: {con_error}')

        log_obj = {
            'Class-IL mean': cil_acc, 'Task-IL mean': til_acc, 'Con-Error': con_error,
            'PreTrain-acc': pre_acc,
            **{f'Class-IL task-{i + 1}': acc for i, acc in enumerate(accs[0])},
            **{f'Task-IL task-{i + 1}': acc for i, acc in enumerate(accs[1])},
            'task': self.task,
        }
        self.log_results.append(log_obj)
        self.wblog({'testing': log_obj})
        self.add_log_latents()
        self.save_checkpoint(self.task)

        if self.task > 2:
            self.end_training()
            exit()

        if self.task == self.N_TASKS:
            self.end_training()

    @torch.no_grad()
    def add_log_latents(self):
        lats, y = self.compute_buffer_latents()
        self.log_latents.append({'feat_buf': lats.tolist(), 'y_buf': y.tolist()})

    def save_checkpoint(self, task: int):
        if self.args.custom_log:
            log_dir = f'{base_path()}checkpoints/{self.args.name}'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            torch.save(self.net.state_dict(), f'{log_dir}/task_{task}.pt')

    def end_training(self):
        if self.args.custom_log:
            log_dir = f'{base_path()}logs/{self.dataset_name}/{self.NAME}'
            # obj = {**vars(self.args), 'results': self.log_results}
            # self.print_logs(log_dir, obj, name='results')
            obj = {**vars(self.args), 'results': self.log_results, 'latents': self.log_latents}
            # self.print_logs(log_dir, obj, name='latents')
