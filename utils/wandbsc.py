import wandb
from argparse import Namespace
from utils import random_id


class WandbLogger:
    def __init__(self, args: Namespace, prj='rodo-pretrain', entity='ema-frasca', name=None):
        self.active = args.wandb
        if self.active:
            if name is not None:
                name += f'-{random_id(5)}'
            wandb.init(project=prj, entity=entity, config=vars(args), name=name)

    def __call__(self, obj: any):
        if wandb.run:
            wandb.log(obj)
