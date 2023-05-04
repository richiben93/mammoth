from re import sub
import torch
import numpy as np
import os
from tqdm import tqdm as el_tqdm
from argparse import Namespace, ArgumentParser
import pickle

import wandb

tqdm = lambda x: el_tqdm(x, leave=False)

def bbasename(path):
    return [x for x in path.split('/') if len(x)][-1]

def find_args(foldername):

    if 'args.pyd' in os.listdir(foldername):
        data = eval(open(os.path.join(foldername, 'args.pyd')).read())
        return data['model'], data['buffer_size'], 'egap' if ('egap' in data['name'].lower() or 'casper' in data['name'].lower()) else 'none'
    elif 'testfeats.pkl' in os.listdir(foldername):
        with open(os.path.join(foldername, 'testfeats.pkl'), 'rb') as f:
            a, b, c = list(pickle.load(f).keys())[0]
            return a, c, b
    elif 'bufeats.pkl' in os.listdir(foldername):
        with open(os.path.join(foldername, 'bufeats.pkl'), 'rb') as f:
            a, b, c = list(pickle.load(f).keys())[0]
            return a, c, b
    else:
        api = wandb.Api(timeout=180)
        entity = 'regaz'
        for project in ['casper-icml', 'rodo-istatsJIHAD', 'rodo-istats', 'rodo-istatsTEMP']:
            for runna in api.runs(f'{entity}/{project}'):
                if runna.name == bbasename(foldername).split('_2000')[0]:
                    print('-- Run found!')
                    return runna.config['model'], runna.config['buffer_size'], 'egap' if ('egap' in runna.config['name'].lower() or 'casper' in runna.config['name'].lower()) else 'none'        
        raise ValueError(f'Could not find run for {foldername}')

args = ArgumentParser()
args.add_argument('foldername', type=str)
args = args.parse_args()

torch.set_grad_enabled(False)
conf_path = os.getcwd()
while not 'mammoth' in bbasename(conf_path):
    args.foldername = os.path.join(bbasename(os.getcwd()), args.foldername)
    os.chdir("..")
    conf_path = os.getcwd()
os.environ['PYTHONPATH'] = f'{conf_path}'
os.environ['PATH'] += f':{conf_path}'

# print path
from backbone.ResNet18 import resnet18
from utils.conf import get_device
device = get_device()
# prepare dataset
from datasets.seq_cifar100 import SequentialCIFAR100_10x10
print('-- Searching run', args.foldername)
model, buf_size, reg = find_args(args.foldername)
# if os.path.exists(os.path.join(args.foldername, 'normfeatsvar.pkl')):
#     print("-- ALREADY DONE, ABORTING\n")
#     exit()

print('-- Loading Datasets')
foldername = args.foldername
args = Namespace(
            batch_size=64,
            dataset='seq-cifar100-10x10',
            validation=False,
)
dataset = SequentialCIFAR100_10x10(args)
dataset.get_data_loaders()
data_loaders = [dataset.get_data_loaders()[0] for _ in range(2)]


mymodel = "Derpp"#'Erace'
load = True

all_data = {}

all_data[(model, reg, buf_size)] = {}
path = foldername
if path[-1] != '/':
    path += '/'

id_task = 10
print('-- Loading models')
net = resnet18(100)
if model == 'scr_casper':
    from models.scr_casper import SCRCasper
    args.rep_minibatch = 64
    args.replay_mode = 'none'
    args.lr = 0.1
    args.model = model
    args.lr_momentum = 0
    args.wandb = False
    args.buffer_size= buf_size
    args.scheduler= None
    args.head='mlp'
    args.b_nclasses=16
    args.load_check=None
    args.backbone='resnet18'
    args.temp=0.1
    args.wb_prj, args.wb_entity = 'regaz', 'rodo-istatsTEMP'
    t_model = SCRCasper(net, lambda x:x, args, None)
    net = t_model.net
if model in ['scr_derpp', 'scr_xder_rpc']:
    from models.scr_derpp import SCRDerpp
    args.rep_minibatch = 64
    args.replay_mode = 'none'
    args.lr = 0.1
    args.model = model
    args.lr_momentum = 0
    args.wandb = False
    args.buffer_size= buf_size
    args.scheduler= None
    args.head='mlp'
    args.load_check=None
    args.backbone='resnet18'
    args.temp=0.1
    args.wb_prj, args.wb_entity = 'regaz', 'rodo-istatsTEMP'
    args.alpha = 0.1
    args.beta = 0.1
    t_model = SCRDerpp(net, lambda x:x, args, None)
    net = t_model.net
elif model == 'podnet_egap':
        from models.podnet_egap import PodNetEgap
        args.rep_minibatch = 64
        args.replay_mode = 'none'
        args.lr = 0.1
        args.load_check=None
        args.model = model
        args.lr_momentum = 0
        args.wandb = False
        args.buffer_size= buf_size
        args.scheduler= None
        args.k=10
        args.scaling=3
        args.eta =1
        args.delta=0.6
        args.wb_prj, args.wb_entity = 'regaz', 'rodo-istatsTEMP'
        t_model = PodNetEgap(net, lambda x: x, args, None)
        net = t_model.net
sd = torch.load(path + f'task_{id_task}.pt', map_location='cpu')
net.load_state_dict(sd)
net.eval()
all_data[(model, reg, buf_size)][id_task] = {}
all_data[(model, reg, buf_size)][id_task]['net'] = net

print('-- Computing projections')
net = all_data[(model, reg, buf_size)][id_task]['net']
net.to(device)    
# good_labels = torch.tensor([0, 2, 5, 8, 9, 11, 13, 14, 16, 18])
feats, labels = [], []
for d in dataset.test_loaders:
    for x, y in d:
        bx = x.to(device)
        # if torch.isin(y, good_labels).any():
        bproj = net.features(bx).cpu()
        feats.append(bproj)
        labels.append(y)
feats = torch.cat(feats)
labels = torch.cat(labels)

vars = []
mean_feat = feats.mean(0, keepdim=True)
norm_feat = feats.norm(2, 0, keepdim=True)
for l in labels.unique():
    # normalize features of class l and compute variance
    subfeats = feats[labels == l]
    subfeats = (subfeats - mean_feat) / norm_feat
    vars.append(subfeats.var(1).mean())

    # vars.append(feats[labels == l].var(1).mean())
vars = np.mean(vars)

all_data[(model, reg, buf_size)][id_task][f'fvar'] = vars
print('feature variance', vars)
net.to('cpu')
del all_data[(model, reg, buf_size)][id_task]['net']

print('-- Saving to', os.path.join(foldername, 'normfeatsvar.pkl'), '\n')
with open(os.path.join(foldername, 'normfeatsvar.pkl'), 'wb') as f:
    pickle.dump(all_data, f)