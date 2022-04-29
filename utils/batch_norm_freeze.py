import torch
import torch.nn as nn


# Use:
# with bn_track_stats(module)
class bn_untrack_stats:
    def __init__(self, module: nn.Module, condition=False):
        self.module = module
        self.enable = condition

    def __enter__(self):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = False

    def __exit__(self ,type, value, traceback):
        if not self.enable:
            for m in self.module.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.track_running_stats = True