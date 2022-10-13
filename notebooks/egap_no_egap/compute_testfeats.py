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
    api = wandb.Api(timeout=180)
    entity = 'regaz'
    for project in ['rodo-istatsJIHAD', 'rodo-istats', 'rodo-istatsTEMP']:
        for runna in api.runs(f'{entity}/{project}'):
            if runna.name == bbasename(foldername).split('_')[0]:
                print('-- Run found!')
                return runna.config['model'], runna.config['buffer_size'], 'egap' if 'egap' in runna.config['name'].lower() else 'none'
    
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
if os.path.exists(os.path.join(args.foldername, 'testfeats.pkl')):
    print("-- ALREADY DONE, ABORTING\n")
    exit()

print('-- Loading Datasets')
foldername = args.foldername
args = Namespace(
            batch_size=64,
            dataset='seq-cifar100_10x10',
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

print('-- Loading models')
for id_task in range(1, 11):
    net = resnet18(100)
    sd = torch.load(path + f'task_{id_task}.pt', map_location='cpu')
    net.load_state_dict(sd)
    net.eval()
    all_data[(model, reg, buf_size)][id_task] = {}
    all_data[(model, reg, buf_size)][id_task]['net'] = net

print('-- Computing projections')
for id_task in tqdm(range(1, 11)):
    net = all_data[(model, reg, buf_size)][id_task]['net']
    net.to(device)    
    good_labels = torch.tensor([0, 2, 5, 8, 9, 11, 13, 14, 16, 18])
    feats, labels = [], []
    for d in dataset.test_loaders:
        for x, y in d:
            bx = x.to(device)
            if torch.isin(y, good_labels).any():
                bproj = net.features(bx[torch.isin(y, good_labels)]).cpu()
                feats.append(bproj)
                labels.append(y[torch.isin(y, good_labels)])
    all_data[(model, reg, buf_size)][id_task][f'bproj'] = torch.cat(feats)
    all_data[(model, reg, buf_size)][id_task][f'by'] = torch.cat(labels)
    
    net.to('cpu')

for id_task in tqdm(range(1, 11)):
    del all_data[(model, reg, buf_size)][id_task]['net']

print('-- Saving to', os.path.join(foldername, 'testfeats.pkl'), '\n')
with open(os.path.join(foldername, 'testfeats.pkl'), 'wb') as f:
    pickle.dump(all_data, f)