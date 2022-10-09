import torch
import torch.nn.functional as F
from utils.args import *
from models.utils.egap_model import EgapModel
from utils.ring_buffer import RingBuffer
from utils.triplet import batch_hard_triplet_loss, negative_only_triplet_loss
from torch.distributions.beta import Beta
from datasets import get_dataset
from torch.optim import Adam

def mixup(couples, alpha, force_lambda=None):
    lamda = Beta(alpha, alpha).rsample((len(couples[0][0]),)).to(couples[0][0].device)
    lamda = torch.max(lamda, 1 - lamda)

    if force_lambda is not None:
        lamda = torch.tensor(force_lambda).repeat((len(couples[0][0]),)).to(couples[0][0].device)

    returns = []

    for (i1, i2) in couples:
        lamda = lamda.view([lamda.shape[0]] + [1] * (len(i1.shape) - 1))
        assert i1.shape == i2.shape
        x_out = lamda * i1 + (1 - lamda) * i2
        returns.append(x_out)

    return tuple(returns) if len(returns) > 1 else returns[0]

def random_flip(x, enable=True):
    if not enable:
        return x
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x


def random_crop(x, padding):
    assert len(x.shape) == 4
    crop_x = torch.randint(-padding, padding, size=(x.shape[0],))
    crop_y = torch.randint(-padding, padding, size=(x.shape[0],))

    crop_x_start, crop_y_start = crop_x + padding, crop_y + padding
    crop_x_end, crop_y_end = crop_x_start + \
        x.shape[-1], crop_y_start + x.shape[-2]

    padded_in = F.pad(x, (padding, padding, padding, padding))
    mask_x = torch.arange(
        x.shape[-1] + padding * 2).repeat(x.shape[0], x.shape[-1] + padding * 2, 1)
    mask_y = mask_x.transpose(1, 2)
    mask_x = ((mask_x >= crop_x_start.unsqueeze(1).unsqueeze(2))
              & (mask_x < crop_x_end.unsqueeze(1).unsqueeze(2)))
    mask_y = ((mask_y >= crop_y_start.unsqueeze(1).unsqueeze(2))
              & (mask_y < crop_y_end.unsqueeze(1).unsqueeze(2)))
    return padded_in[mask_x.unsqueeze(1).repeat(1, x.shape[1], 1, 1) * mask_y.unsqueeze(1).repeat(1, 3, 1, 1)].reshape(x.shape[0], 3, x.shape[2], x.shape[3])


def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay ACE with Replay of something else.')
    add_management_args(parser)     # --wandb, --custom_log, --save_checks
    add_experiment_args(parser)     # --dataset, --model, --lr, --batch_size, --n_epochs
    add_rehearsal_args(parser)      # --minibatch_size, --buffer_size

    parser.add_argument('--perc_labels', type=float, default=0.25, help='Percentage of labels to use for training.')
    # --replay_mode, --replay_weight, --rep_minibatch, 
    # --heat_kernel, --cos_dist, --knn_laplace
    EgapModel.add_replay_args(parser)

    parser.add_argument('--stream_replay_weight', type=float, required=True, help='Weight of replay.')

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Unsupervised loss weight.')
    parser.add_argument('--k', type=int, default=1,
                        help='k of kNN.')
    parser.add_argument('--memory_penalty', type=float,
                        default=1.0, help='Unsupervised penalty weight.')
    parser.add_argument('--k_aug', type=int, default=3,
                        help='Number of augumentation to compute label predictions.')
    parser.add_argument('--lamda', type=float, default=0.5,
                        help='Regularization weight.')
    parser.add_argument('--sharp_temp', default=0.5,
                        type=float, help='Temperature for sharpening.')
    parser.add_argument('--mixup_alpha', default=0.75, type=float)
    
    return parser


class CCICEgapSS(EgapModel):
    NAME = 'ccic_egap_ss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CCICEgapSS, self).__init__(backbone, loss, args, transform)
        self.sup_virtual_batch = RingBuffer(
            self.args.batch_size, self.device, 1)
        self.unsup_virtual_batch = RingBuffer(
            self.args.batch_size, self.device, 1)
        denorm = get_dataset(args).get_denormalization_transform()
        self.cpt = self.N_CLASSES_PER_TASK
        self.normalization_transform = get_dataset(
            args).get_normalization_transform()
        self.n_tasks = self.N_TASKS
        self.dataset_mean, self.dataset_std = denorm.mean, denorm.std
        self.eye = torch.eye(self.cpt * self.n_tasks).to(self.device)
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)
        self.embeddings = None

    def get_name(self):
        return 'EraceSS' + self.get_name_extension()

    def begin_task(self, dataset):
        self.epoch_batches = len(dataset.train_loader)
        self.seen_batches = 0
        self.epoch = 0

    def end_task(self, dataset):
        super().end_task(dataset)
        
    
    def compute_embeddings(self):
        """
        Computes a vector representing mean features for each class.
        """
        with torch.no_grad():
            data = self.buffer.get_all_data(
                transform=self.normalization_transform)
            outputs = []
            while data[0].shape[0] > 0:
                inputs, labels = data[0][:self.args.batch_size], data[1][:self.args.batch_size]
                data = (data[0][self.args.batch_size:],
                        data[1][self.args.batch_size:])
                out = self.net.features(inputs)
                out = F.normalize(out, p=2, dim=1)
                outputs.append(out)

        self.embeddings = torch.cat(outputs)

    def forward(self, x):
        if self.embeddings is None:
            with torch.no_grad():
                self.compute_embeddings()
        buf_labels = self.buffer.labels[:self.buffer.num_seen_examples]
        feats = self.net.features(x)
        feats = F.normalize(feats, p=2, dim=1)
        distances = (self.embeddings.unsqueeze(0) -
                     feats.unsqueeze(1)).pow(2).sum(2)

        dist = torch.stack([distances[:, buf_labels == c].topk(1, largest=False)[0].mean(dim=1)
                            for c in range((self.task) * self.cpt)] +
                           [torch.zeros(x.shape[0]).to(self.device)] * ((self.n_tasks - (self.task)) * self.cpt)).T
        topkappas = self.eye[buf_labels[distances.topk(
            self.args.k, largest=False)[1]]].sum(1)
        return topkappas - dist * 10e-6

    def mng_epoch(self):
        # --------- EPOCH MGMT ----------
        self.seen_batches += 1
        if self.seen_batches == self.epoch_batches:
            self.epoch += 1
            self.seen_batches = 0
        # -------------------------------

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]
        sup_mask = labels < 1000
        unsup_mask = ~sup_mask
        
        # ------------ DATA PREPARATION ------------
        # PARTITION DATA
        sup_inputs, sup_labels, sup_not_aug_inputs = inputs[sup_mask], labels[sup_mask], not_aug_inputs[sup_mask]
        sup_inputs_for_buffer, sup_labels_for_buffer = sup_not_aug_inputs.clone(), sup_labels.clone()
        unsup_inputs, unsup_labels, unsup_not_aug_inputs = inputs[unsup_mask], labels[unsup_mask], not_aug_inputs[unsup_mask]

        # short-circuit if no labeled data
        if len(sup_inputs) == 0 and self.buffer.is_empty():
            self.mng_epoch()
            return 1.

        # VIRTUAL BATCHES
        self.sup_virtual_batch.add_data(sup_not_aug_inputs, sup_labels)
        sup_inputs, sup_labels = self.sup_virtual_batch.get_data(
            self.args.batch_size, transform=self.transform)
        if self.task > 0:
            self.unsup_virtual_batch.add_data(
                unsup_not_aug_inputs, unsup_labels)
            unsup_inputs = self.unsup_virtual_batch.get_data(
                self.args.batch_size, transform=self.transform)[0]

        # BUFFER RETRIEVAL
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            sup_inputs = torch.cat((sup_inputs, buf_inputs))
            sup_labels = torch.cat((sup_labels, buf_labels))
            if self.task > 0:
                masked_buf_inputs, masked_buf_labels = self.buffer.get_data(self.args.minibatch_size,
                                                                            mask_task=self.task, mask_task_cpt=self.cpt,
                                                                            transform=self.transform)
                unsup_labels = torch.cat((torch.zeros(unsup_inputs.shape[0]).to(self.device),
                                          torch.ones(masked_buf_labels.shape[0]).to(self.device))).long()
                unsup_inputs = torch.cat((unsup_inputs, masked_buf_inputs))
        # -------------------------------------------

        # ------------------ K AUG ---------------------
        real_mask = sup_mask[:real_batch_size]

        unsup_aug_inputs = normalize(
            random_flip(
                random_crop(not_aug_inputs[~real_mask].repeat_interleave(
                    self.args.k_aug, 0), 4)
            ), self.dataset_mean, self.dataset_std)

        # ------------------ PSEUDO LABEL ---------------------        
        self.net.eval()
        if len(unsup_aug_inputs):
            with torch.no_grad():
                unsup_aug_outputs = self.net(unsup_aug_inputs).reshape(
                    self.args.k_aug, -1, self.eye.shape[0]).mean(0)
                unsup_sharp_outputs = unsup_aug_outputs ** (
                    1 / self.args.sharp_temp)
                unsup_norm_outputs = unsup_sharp_outputs / \
                    unsup_sharp_outputs.sum(1).unsqueeze(1)
                unsup_norm_outputs = unsup_norm_outputs.repeat(
                    self.args.k_aug, 1)
        else:
            unsup_norm_outputs = torch.zeros(
                (0, len(self.eye))).to(self.device)
        self.net.train()

        # ------------------ MIXUP ---------------------
        
        W_inputs = torch.cat((sup_inputs, unsup_aug_inputs))
        W_probs = torch.cat((self.eye[sup_labels], unsup_norm_outputs))
        perm = torch.randperm(W_inputs.shape[0])
        W_inputs, W_probs = W_inputs[perm], W_probs[perm]
        sup_shape = sup_inputs.shape[0]

        sup_mix_inputs, sup_mix_targets = mixup(
            [(sup_inputs, W_inputs[:sup_shape]), (self.eye[sup_labels], W_probs[:sup_shape])], self.args.mixup_alpha)
        sup_mix_outputs = self.net(sup_mix_inputs)
        if len(unsup_aug_inputs):
            unsup_mix_inputs, unsup_mix_targets = mixup(
                [(unsup_aug_inputs, W_inputs[sup_shape:]),
                 (unsup_norm_outputs, W_probs[sup_shape:])],
                self.args.mixup_alpha)
            unsup_mix_outputs = self.net(unsup_mix_inputs)

        effective_mbs = min(self.args.minibatch_size,
                            self.buffer.num_seen_examples)
        if effective_mbs == 0:
            effective_mbs = -100

        # ------------------ CIC LOSS ---------------------
        if self.epoch < self.args.n_epochs / 10 * 9:
            loss_X = 0
            if real_mask.sum() > 0:
                loss_X += self.loss(sup_mix_outputs[:-effective_mbs],
                                    sup_labels[:-effective_mbs])
            if not self.buffer.is_empty():
                assert effective_mbs > 0
                loss_X += self.args.memory_penalty * \
                    self.loss(sup_mix_outputs[-effective_mbs:],
                            sup_labels[-effective_mbs:])

            if len(unsup_aug_inputs):
                loss_U = F.mse_loss(unsup_norm_outputs,
                                    unsup_mix_outputs) / self.eye.shape[0]
            else:
                loss_U = 0

        # CIC LOSS
        if self.epoch < self.args.n_epochs / 10 * 9:
            W_inputs = sup_inputs
            W_probs = self.eye[sup_labels]
            perm = torch.randperm(W_inputs.shape[0])
            W_inputs, W_probs = W_inputs[perm], W_probs[perm]

            sup_mix_inputs, sup_mix_targets = mixup(
                [(sup_inputs, W_inputs), (self.eye[sup_labels], W_probs)], 1)
        else:
            sup_mix_inputs = sup_inputs

        # STANDARD TRIPLET
        sup_mix_embeddings = self.net.features(sup_mix_inputs)
        loss = batch_hard_triplet_loss(sup_labels, sup_mix_embeddings, self.args.batch_size // 10,
                                       margin=1, margin_type='hard')
        if loss is not None:
            self.wb_log['bhard_loss'] = loss.item()
        self.wb_log['sup_cic_loss'] = loss_X.item()
        self.wb_log['unsup_cic_loss'] = loss_U.item()
        # sup_mix_outputs = self.net.linear(sup_mix_embeddings)
        if loss is None:
            if self.epoch >= self.args.n_epochs / 10 * 9:
                loss = 0
            else:
                loss = loss_X + self.args.lamda * loss_U
        else:
            if self.epoch >= self.args.n_epochs / 10 * 9:
                pass
            else:
                loss += loss_X + self.args.lamda * loss_U

        self.buffer.add_data(examples=sup_inputs_for_buffer,
                            labels=sup_labels_for_buffer)

        # SELF-SUPERVISED PAST TASKS NEGATIVE ONLY
        if self.task > 0 and self.epoch < self.args.n_epochs / 10 * 9:
            unsup_embeddings = self.net.features(unsup_inputs)
            loss_unsup = negative_only_triplet_loss(unsup_labels, unsup_embeddings, self.args.batch_size // 10,
                                                    margin=1, margin_type='hard')
            self.wb_log['unsup_loss'] = loss_unsup.item()
            if loss_unsup is not None:
                loss += self.args.alpha * loss_unsup


        # ------------ EGAP LOSS ----------------
        if self.task > 0 and self.args.buffer_size > 0:
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

        
        if loss.requires_grad:
            loss.backward()

        self.mng_epoch()
        return loss.item()
