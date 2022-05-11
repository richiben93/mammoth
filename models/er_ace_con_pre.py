import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.consolidation_pretrain import PretrainedConsolidationModel
import wandb
import numpy as np
from time import time
from utils import random_id


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
        if args.wandb:
            name = 'EraceConPre' if self.args.con_weight > 0 else 'Erace'
            wandb.init(project="rodo-pretrain", entity="ema-frasca", config=vars(args),
                       name=f"{name}-{random_id(5)}")
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
        if wandb.run:
            wandb.log({'training': wandb_log})

        return loss.item()

    def log_accs(self, accs):
        cil_acc, til_acc = np.mean(accs, axis=1).tolist()
        pre_acc = self.pre_dataset_train_head(n_epochs=1)

        # running consolidation error
        con_error = None
        if self.task > 1:
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
        if wandb.run:
            wandb.log({'testing': log_obj})

        if self.task > 1:
            pass

        if self.task == self.N_TASKS:
            self.end_training()

    def end_training(self):
        if self.args.custom_log:
            log_dir = f'/nas/softechict-nas-2/efrascaroli/mammoth-data/logs/{self.dataset_name}/{self.NAME}'
            obj = {**vars(self.args), 'results': self.log_results}
            self.print_logs(log_dir, obj, name='results')
