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
    entity, project = 'regaz', 'rodo-istats'
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
if os.path.exists(os.path.join(args.foldername, 'resubtest.pkl')):
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
# data_loaders = [dataset.get_data_loaders()[0] for _ in range(dataset.N_TASKS)]


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
    buf = pickle.load(open(path + f'task_{id_task}_buffer.pkl', 'rb'))

    all_data[(model, reg, buf_size)][id_task] = {}
    all_data[(model, reg, buf_size)][id_task]['net'] = net
    all_data[(model, reg, buf_size)][id_task]['buf'] = buf

print('-- Computing projections')
for id_task in tqdm(range(1, 11)):
    net = all_data[(model, reg, buf_size)][id_task]['net']
    all_data[(model, reg, buf_size)][id_task]['projs'] = []
    all_data[(model, reg, buf_size)][id_task]['labs'] = []
    net.to(device)    

    for j, dl in enumerate(dataset.test_loaders[:id_task]):
        corr, corrknn, tot = 0, 0, 0
        for x, y in dl:

            x = x.to(device)
            y = y
            proj = net.features(x).cpu()
            tot += len(y)
            
            all_data[(model, reg, buf_size)][id_task]['projs'].append(proj)
            all_data[(model, reg, buf_size)][id_task]['labs'].append(y)
            if tot > 200:
                break
            
    net.to('cpu')
    
    all_data[(model, reg, buf_size)][id_task]['projs'] = torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0)
    all_data[(model, reg, buf_size)][id_task]['labs'] = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)
    rando = torch.randperm(len(all_data[(model, reg, buf_size)][id_task]['projs']))
    all_data[(model, reg, buf_size)][id_task]['projs'] = all_data[(model, reg, buf_size)][id_task]['projs'][rando[:2000]]
    all_data[(model, reg, buf_size)][id_task]['labs'] = all_data[(model, reg, buf_size)][id_task]['labs'][rando[:2000]]

# knn
print('-- Computing bbs')
from utils.spectral_analysis import calc_cos_dist, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs, calc_ADL_heat
wrong_cons = []
for id_task in tqdm(range(1, 11)):
    features = all_data[(model, reg, buf_size)][id_task]['projs']
    labels = all_data[(model, reg, buf_size)][id_task]['labs']
    
    knn_laplace = 5 if buf_size == 500 else 4 #int(bbasename(foldername).split('-')[0].split('K')[-1])
    dists = calc_euclid_dist(features)
    A, _, _ = calc_ADL_knn(dists, k=knn_laplace, symmetric=True)
    lab_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    wrong_A = A[~lab_mask]
    wrong_cons.append(wrong_A.sum() / A.sum())

print('-- Saving to', os.path.join(foldername, 'resubtest.pkl'), '\n')
with open(os.path.join(foldername, 'resubtest.pkl'), 'wb') as f:
    pickle.dump((model, buf_size, reg, wrong_cons), f)

for id_task in tqdm(range(1, 11)):
    del all_data[(model, reg, buf_size)][id_task]['net']
    del all_data[(model, reg, buf_size)][id_task]['buf']

print('-- Saving to', os.path.join(foldername, 'resubtestfeats.pkl'), '\n')
with open(os.path.join(foldername, 'resubtestfeats.pkl'), 'wb') as f:
    pickle.dump(all_data, f)