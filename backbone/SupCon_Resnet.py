from torch import nn
from torch.functional import F
import backbone.ResNet18 as res18
from backbone.EfficientNet import mammoth_efficientnet
from datasets.seq_miniimg import SequentialMiniImagenet


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128, backbone='resnet18'):
        super(SupConResNet, self).__init__()
        if backbone in ['resnet18', 'lopeznet']:
            self.encoder = getattr(res18, backbone)(100)
            dim_in = self.encoder.nf * 8 * self.encoder.block.expansion
        elif backbone == 'efficientnet':
            self.encoder = mammoth_efficientnet(SequentialMiniImagenet.N_CLASSES_PER_TASK * SequentialMiniImagenet.N_TASKS, 'efficientnet-b2')
            dim_in = self.encoder.classifier.in_features
        else:
            raise NotImplementedError(
                'backbone not supported: {}'.format(backbone))

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        return self.encoder.features(x)
