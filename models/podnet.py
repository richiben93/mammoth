import copy
import math
import warnings

from models.utils.continual_model import ContinualModel
from copy import deepcopy
from utils.augmentations import normalize
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.no_bn import bn_track_stats
import numpy as np
from models.icarl import baguette_fill_buffer
from models.utils.podnet_utils import *

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via iCaRL.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--wd_reg', type=float, required=False,
                        help='L2 regularization applied to the parameters.')
    return parser


class Podnet(ContinualModel):
    NAME = 'podnet'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Podnet, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None
        self.current_task = 0
        self.num_classes = self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS
        self._examplars = {}
        self._metrics = {"nca": 0., "flat": 0., "pod": 0.}
        self._means = None

        self.old_net = None
        self._nca_config = {'margin': 0.6,
                            'scale': 1.,
                            'exclude_pos_denominator': True}
        self.classifier_config = {'type': 'cosine',
                                   'proxy_per_class': 10,
                                   'distance': 'neg_stable_cosine_distance', }
        self.classifier_config = {'type': 'cosine',
                                   'proxy_per_class': 10,
                                   'distance': 'neg_stable_cosine_distance', 'scaling': 3.0}

        self.postprocessor_config = {'type': 'learned_scaling',
                                     'initial_value': 1.0

                                     }
        self._pod_flat_config = {'scheduled_factor': 1.0}

        self._pod_spatial_config = {'scheduled_factor': 3.0, 'collapse_channels': 'spatial', }
        self._weight_generation = {'type': 'imprinted',
                                   'multi_class_diff': 'kmeans'}

        old_classifier = self.net.classifier
        self.net.classifier = CosineClassifier(
            old_classifier.in_features, device=self.device, **self.classifier_config
            )
        del old_classifier

        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr)


    def end_task(self, inc_dataset):
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            baguette_fill_buffer(self, self.buffer, inc_dataset, self.current_task)
        self.current_task += 1
        self.class_means = None

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x).float()
        return feats

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _ = self.buffer.get_all_data(transform)
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                class_means.append(self.net.features(x_buf).mean(0).flatten())
        self.class_means = torch.stack(class_means)

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels)
        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs, targets):

        outputs = self.net.forward_all(inputs)
        features, pre_logits, atts = outputs["raw_features"], outputs["features"], outputs["attention"]

        logits = self.net.classifier(pre_logits)
        # donnow why this is necessary
        # if self._post_processing_type is None:
        #     scaled_logits = self._network.post_process(logits)
        # else:
        #     scaled_logits = logits * self._post_processing_type

        if self.old_net is not None:
            with torch.no_grad():
                old_outputs = self.old_net.forward_all(inputs)
                old_features = old_outputs["raw_features"]
                old_atts = old_outputs["attention"]

        nca_config = copy.deepcopy(self._nca_config)
        if self._network.post_processor:
            nca_config["scale"] = self._network.post_processor.factor

        loss = nca(
            logits,
            targets,
            class_weights=self._class_weights,
            **nca_config
        )
        self._metrics["nca"] = loss.item()

        # --------------------
        # Distillation losses:
        # --------------------

        if self.old_net is not None:
            if self._pod_flat_config:
                if self._pod_flat_config["scheduled_factor"]:
                    factor = self._pod_flat_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size  # fixme
                    )
                else:
                    factor = self._pod_flat_config.get("factor", 1.)

                pod_flat_loss = factor * embeddings_similarity(old_features, features)
                loss += pod_flat_loss
                self._metrics["flat"] = pod_flat_loss.item()

            if self._pod_spatial_config:
                if self._pod_spatial_config.get("scheduled_factor", False):
                    factor = self._pod_spatial_config["scheduled_factor"] * math.sqrt(
                        self._n_classes / self._task_size
                    )
                else:
                    factor = self._pod_spatial_config.get("factor", 1.)

                pod_spatial_loss = factor * pod(
                    old_atts,
                    atts,
                    task_percent=(self.current_task + 1) / self.N_TASKS,
                    **self._pod_spatial_config
                )
                loss += pod_spatial_loss
                self._metrics["pod"] = pod_spatial_loss.item()

                self.old_net.zero_grad()
                self._network.zero_grad()

        return loss

    def _gen_weights(self):
        self._task = task_info["task"]
        self._total_n_classes = task_info["total_n_classes"]
        self._task_size = task_info["increment"]
        self._n_train_data = task_info["n_train_data"]
        self._n_test_data = task_info["n_test_data"]
        self._n_tasks = task_info["max_task"]
        if self._weight_generation:
            add_new_weights(
                self.net, self._weight_generation if self.current_task != 0 else "basic",
                self.N_CLASSES, self._task_size, self.inc_dataset
            )

    def begin_task(self, dataset):
        self._gen_weights()
        if self.current_task > 0:
            dataset.train_loader.dataset.targets = np.concatenate(
                [dataset.train_loader.dataset.targets,
                 self.buffer.labels.cpu().numpy()[:self.buffer.num_seen_examples]])
            if type(dataset.train_loader.dataset.data) == torch.Tensor:
                dataset.train_loader.dataset.data = torch.cat(
                    [dataset.train_loader.dataset.data, torch.stack([(
                        self.buffer.examples[i].type(torch.uint8).cpu())
                        for i in range(self.buffer.num_seen_examples)]).squeeze(1)])
            else:
                dataset.train_loader.dataset.data = np.concatenate(
                    [dataset.train_loader.dataset.data, torch.stack([((self.buffer.examples[i] * 255).type(
                        torch.uint8).cpu()) for i in range(self.buffer.num_seen_examples)]).numpy().swapaxes(
                        1, 3)])


#######WEIGHTS##############

def get_class_weights(dataset, log=False, **kwargs):
    targets = dataset.y

    class_sample_count = np.unique(targets, return_counts=True)[1]
    weights = 1. / class_sample_count

    min_w = weights.min()
    weights = weights / min_w

    if log:
        weights = np.log(weights)

    return np.clip(weights, a_min=1., a_max=None)


def extract_features(model, loader):
    targets, features = [], []

    state = model.training
    model.eval()

    for input_dict in loader:
        inputs, _targets = input_dict["inputs"], input_dict["targets"]

        _targets = _targets.numpy()
        _features = model.extract(inputs.to(model.device)).detach().cpu().numpy()

        features.append(_features)
        targets.append(_targets)

    model.train(state)

    return np.concatenate(features), np.concatenate(targets)


def add_new_weights(network, weight_generation, current_nb_classes, task_size, inc_dataset):
    print("Generating imprinted weights")

    network.add_imprinted_classes(
        list(range(current_nb_classes, current_nb_classes + task_size)), inc_dataset,
        **weight_generation
    )


########## LOSSES ##########
def pod(
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True,
        memory_flags=None,
        only_old=False,
        **kwargs
):
    """Pooled Output Distillation.
    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.
    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)


def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )


def nca(
        similarities,
        targets,
        class_weights=None,
        focal_gamma=None,
        scale=1,
        margin=0.,
        exclude_pos_denominator=True,
        hinge_proxynca=False,
):
    """Compute AMS cross-entropy loss.
    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.
    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


class BoundClipper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, module):
        if hasattr(module, "mtl_weight"):
            module.mtl_weight.data.clamp_(min=self.lower_bound, max=self.upper_bound)
        if hasattr(module, "mtl_bias"):
            module.mtl_bias.data.clamp_(min=self.lower_bound, max=self.upper_bound)
