import os.path

import torch
from torch import nn
from torch.nn.functional import avg_pool2d
from torchvision import models
from urllib import request
from utils.conf import base_path


class TrainedModel(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, dataset_type: str, num_classes: int) -> None:
        """
        Instantiates the layers of the network.
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super().__init__()
        self.resnet_type = dataset_type
        net = self.initialize_model(dataset_type)

        self.net = net
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        return self.resnet(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self.resnet._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress + torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def initialize_model(self, dataset_type: str) -> models.ResNet:
        if dataset_type == "cifar100":
            link = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/Eb4BgDZ5g_1Imuwz_PJAmdgBc8k9I_P5p0Y-A97edhsxIw?e=WmlZZc"
            file = "rs18_cifar100.pth"

        elif dataset_type == "tinyimgR":
            link = "https://unimore365-my.sharepoint.com/:u:/g/personal/215580_unimore_it/EeWEOSls505AsMCTXAxWoLUBmeIjCiplFl40zDOCmB_lEw?download=1"
            file = "erace_pret_on_tinyr.pth"
        else:
            raise ValueError
        local_path = os.path.join(base_path(), 'checkpoints')
        if not os.path.isdir(local_path):
            os.mkdir(local_path)
        if not os.path.isfile(os.path.join(local_path, file)):
            request.urlretrieve(link, os.path.join(local_path, file))
        model_ft = torch.load_state_dict(os.path.join(local_path, file))
        return model_ft