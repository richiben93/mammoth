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
    try:
        entity, project = 'regaz', 'rodo-istats'
        for runna in api.runs(f'{entity}/{project}'):
            if runna.name == bbasename(foldername).split('_')[0]:
                print('-- Run found!')
                return runna.config['model'], runna.config['buffer_size'], 'egap' if 'egap' in runna.config['name'].lower() else 'none'
    except:
        entity, project = 'regaz', 'rodo-istatsTEMP'
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
# if os.path.exists(os.path.join(args.foldername, 'fmaps.txt')):
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
data_loaders = [dataset.get_data_loaders()[0] for _ in range(dataset.N_TASKS)]


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
    if model == 'podnet_egap':
        from models.podnet_egap import PodNetEgap
        args.rep_minibatch = 64
        args.replay_mode = 'none'
        args.lr = 0.1
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
    buf = pickle.load(open(path + f'task_{id_task}_buffer.pkl', 'rb'))

    all_data[(model, reg, buf_size)][id_task] = {}
    all_data[(model, reg, buf_size)][id_task]['net'] = net
    all_data[(model, reg, buf_size)][id_task]['buf'] = buf

print('-- Computing projections')
for id_task in tqdm(range(1, 11)):
    net = all_data[(model, reg, buf_size)][id_task]['net']
    all_data[(model, reg, buf_size)][id_task]['projs'] = []
    all_data[(model, reg, buf_size)][id_task]['preds'] = []
    all_data[(model, reg, buf_size)][id_task]['labs'] = []
    net.to(device)    

    for j, dl in enumerate(dataset.test_loaders[:id_task]):
        corr, corrknn, tot = 0, 0, 0
        projs, labs = [], []
        for x, y in dl:
            x = x.to(device)
            y = y
            pred = net(x).cpu()
            if model == 'podnet_egap':
                pred = torch.ones(pred.shape[0], 100)
            proj = net.features(x).cpu()
            corr += (pred.argmax(dim=1) == y).sum().item()
            tot += len(y)
            projs.append(proj)
            labs.append(y)
            
        all_data[(model, reg, buf_size)][id_task][f'projs_{j}'] = torch.cat(projs)
        all_data[(model, reg, buf_size)][id_task][f'labs_{j}'] = torch.cat(labs)
        
    net.to('cpu')

print('-- Computing FMAPS')
from utils.spectral_analysis import normalize_A, calc_euclid_dist, calc_ADL_knn, normalize_A, find_eigs, find_eigs
# TODO verifica se buffer troppo largo
knn_laplace = 5
ev_pairs = 20
subsamp = 500

fmaps = []
for id_task in tqdm(range(2, 11)):
    features_prev = torch.cat([all_data[(model, reg, buf_size)][id_task-1][f'projs_{k}'] for k in range(id_task-1)])
    features_next = torch.cat([all_data[(model, reg, buf_size)][id_task][f'projs_{k}'] for k in range(id_task-1)])
    filter = torch.randperm(len(features_prev))[:subsamp]
    features_prev = features_prev[filter]
    features_next = features_next[filter]
    
    dists_prev = calc_euclid_dist(features_prev)
    dists_next = calc_euclid_dist(features_next)

    A_prev, D_prev, _ = calc_ADL_knn(dists_prev, k=knn_laplace, symmetric=True)
    A_next, D_next, _ = calc_ADL_knn(dists_next, k=knn_laplace, symmetric=True)
    L_prev = torch.eye(A_prev.shape[0], device=A_prev.device) - normalize_A(A_prev, D_prev)
    L_next = torch.eye(A_next.shape[0], device=A_next.device) - normalize_A(A_next, D_next)

    _, evects_prev = find_eigs(L_prev, n_pairs=ev_pairs)
    _, evects_next = find_eigs(L_next, n_pairs=ev_pairs)

    fmap = evects_next.T @ evects_prev
    fmaps.append(fmap)
    
print('-- Saving to file', os.path.join(foldername, 'fmaps.pkl'))
with open(os.path.join(foldername, 'fmaps.pkl'), 'wb') as f:
    pickle.dump(fmaps, f)

exit()
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
a = pickle.load(open('../egap_no_egap/cps/EraceEgapb2NC16K4-DuNvm/fmaps.pkl', 'rb'))
for i in a:
    plt.figure()
    sns.heatmap(i, vmin = -1, vmax = 1, cmap='bwr')
