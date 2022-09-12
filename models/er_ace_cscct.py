import torch
from utils.args import *
from models.utils.cscct_model import CscCtModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Replay of something else.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size
    parser.add_argument('--grad_clip', default=0, type=float, help='Clip the gradient.')
    parser.add_argument('--erace_weight', type=float, default=1., help='Weight of erace.')

    # --csc_weight, --ct_weight, --ct_temperature
    CscCtModel.add_replay_args(parser)
    
    return parser


class ErACECscCt(CscCtModel):
    NAME = 'er_ace_cscct'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACECscCt, self).__init__(backbone, loss, args, transform)
        self.seen_so_far = torch.tensor([], dtype=torch.long, device=self.device)

    def get_name(self):
        return 'Erace' + self.get_name_extension()

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

        class_loss = self.loss(logits, labels)
        self.wb_log['class_loss'] = class_loss.item()
        loss = class_loss
        if self.task > 0 and self.args.buffer_size > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            erace_loss = self.loss(self.net(buf_inputs), buf_labels)
            self.wb_log['erace_loss'] = erace_loss.item()
            loss += erace_loss * self.args.erace_weight

            if self.args.csc_weight > 0 and self.args.ct_weight > 0:
                # concatenate stream with buf
                full_inputs = torch.cat([inputs, buf_inputs], dim=0)
                full_targets = torch.cat([labels, buf_labels], dim=0)
                cscct_loss = self.get_cscct_loss(full_inputs, full_targets)
                self.wb_log['cscct_loss'] = cscct_loss.item()
                loss += cscct_loss

        if self.args.buffer_size > 0:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        loss.backward()
        # clip gradients
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.opt.step()

        return loss.item()
