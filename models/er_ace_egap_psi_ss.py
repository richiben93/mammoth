import torch
from utils.args import *
from models.utils.egap_model import EgapModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Replay of something else.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size
    parser.add_argument('--grad_clip', default=0, type=float, help='Clip the gradient.')
    parser.add_argument('--erace_weight', type=float, default=1., help='Weight of erace.')

    parser.add_argument('--perc_labels', type=float, default=0.25, help='Percentage of labels to use for training.')
    # --replay_mode, --replay_weight, --rep_minibatch, 
    # --heat_kernel, --cos_dist, --knn_laplace
    EgapModel.add_replay_args(parser)

    parser.add_argument('--stream_replay_weight', type=float, required=True, help='Weight of replay.')
    
    return parser


class ErACEEgapPsiSS(EgapModel):
    NAME = 'er_ace_egap_psi_ss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErACEEgapPsiSS, self).__init__(backbone, loss, args, transform)
        self.seen_so_far = torch.tensor([], dtype=torch.long, device=self.device)

    def get_name(self):
        return 'EraceSS' + self.get_name_extension()

    def pseudo_label(self, inputs, labels, not_aug_inputs, conf=5.5):
        self.net.eval()
        with torch.no_grad():
            psi_outputs = self.net(inputs)
            
            confs = psi_outputs[:, self.cpt * self.task: self.cpt * (self.task+1)].topk(2, axis=1)[0]
            confs = confs[:, 0] - confs[:, 1]
            conf_thresh = conf
            confidence_mask = confs > conf_thresh 
            _, psi_labels = torch.max(psi_outputs.data[:, self.cpt * self.task: self.cpt * (self.task+1)], 1)
            psi_labels += self.cpt * self.task
            
            out_labels = labels.clone()
            if confidence_mask.sum():
                out_labels[(labels > 999) & confidence_mask] = psi_labels[(labels > 999) & confidence_mask]
        self.net.train()
        return inputs, out_labels, not_aug_inputs

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        inputs, labels, not_aug_inputs = self.pseudo_label(inputs, labels, not_aug_inputs)
        sup_mask = labels < 1000

        present = labels[sup_mask].unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if len(self.seen_so_far) and self.seen_so_far.max() < (self.N_CLASSES - 1):
            mask[:, self.seen_so_far.max():] = 1
        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        if sup_mask.sum():
            class_loss = self.loss(logits[sup_mask], labels[sup_mask])
        else:
            class_loss = torch.tensor(0.).to(self.device)
        self.wb_log['class_loss'] = class_loss.item()
        loss = class_loss
        if self.task > 0 and self.args.buffer_size > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            erace_loss = self.loss(self.net(buf_inputs), buf_labels)
            self.wb_log['erace_loss'] = erace_loss.item()
            loss += erace_loss * self.args.erace_weight

            if self.args.rep_minibatch > 0 and self.args.replay_weight > 0:
                replay_loss = self.get_replay_loss()
                self.wb_log['egap_loss'] = replay_loss.item()
                loss += replay_loss * self.args.replay_weight

        # ------- STREAM BATCH EGAP -------
        if self.args.stream_replay_weight > 0 and len(inputs) > self.N_CLASSES_PER_TASK:
            stream_egap_loss = self.get_replay_loss(inputs, k=self.N_CLASSES_PER_TASK)
            self.wb_log['stream_egap_loss'] = stream_egap_loss.item()
            loss += stream_egap_loss * self.args.stream_replay_weight
        # ---------------------------------

        if self.args.buffer_size > 0 and sup_mask.sum() > 0:
            self.buffer.add_data(examples=not_aug_inputs[sup_mask], labels=labels[sup_mask])

        if loss.requires_grad:
            loss.backward()
        # clip gradients
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)
        self.opt.step()

        return loss.item()
