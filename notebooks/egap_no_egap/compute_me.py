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
if os.path.exists(os.path.join(args.foldername, 'responses.pkl')):
    print("-- ALREADY DONE, ABORTING\n")
    exit()

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

    buf = all_data[(model, reg, buf_size)][id_task]['buf']
    bufdata = buf.get_data(buf.buffer_size, transform=dataset.test_loaders[0].dataset.transform.transforms[1])
    bx, by = bufdata[0], bufdata[1]
    bx = bx.to(device)
    by = by
    bproj = net.features(bx).cpu()

    for j, dl in enumerate(dataset.test_loaders[:id_task]):
        corr, corrknn, tot = 0, 0, 0
        for x, y in dl:
            x = x.to(device)
            y = y
            pred = net(x).cpu()
            if model == 'podnet_egap':
                pred = torch.ones(pred.shape[0], 100)
            proj = net.features(x).cpu()
            corr += (pred.argmax(dim=1) == y).sum().item()
            tot += len(y)
            # bfilter = (by >= j*10) & (by < (j+1)*10)
            corrknn += (by[(bproj.unsqueeze(0) - proj.unsqueeze(1)).norm(dim=2).argmin(dim=1)] == y).sum().item()

            all_data[(model, reg, buf_size)][id_task]['projs'].append(proj)
            all_data[(model, reg, buf_size)][id_task]['preds'].append(pred)
            all_data[(model, reg, buf_size)][id_task]['labs'].append(y)
            
        all_data[(model, reg, buf_size)][id_task][f'acc_{j}'] = corr / tot
        all_data[(model, reg, buf_size)][id_task][f'acc_knn_{j}'] = corrknn / tot
    net.to('cpu')

# Kmeans
print('-- Computing Kmeans')
from sklearn.cluster import KMeans
kmscores = []
for id_task in tqdm(range(1, 11)):
    projs = torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0)
    labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)

    kmeans = KMeans(n_clusters=len(labs.unique())).fit_predict(projs)
    ents = []
    for i in np.unique(kmeans):
        _ , conf = labs[kmeans == i].unique(return_counts=True)
        # compute entropy
        probs = conf / conf.sum()
        entropy = -probs.mul(probs.log()).sum()
        ents.append(entropy.item())
        # print(f'cluster {i} -> {entropy}')
    kmscore = np.mean(ents)
    kmscores.append(kmscore)

# variance
def nize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())
stdscores = []
print('-- Computing std')
for id_task in tqdm(range(1, 11)):
    projs = nize(torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0))
    labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)
    varss = []
    for i in range(10 * id_task):
        varss.append(projs[labs == i].std(dim=0).mean().item())
    stdscore = np.mean(varss)
    stdscores.append(stdscore)

print('-- Saving to', os.path.join(foldername, 'responses.pkl'), '\n')
with open(os.path.join(foldername, 'responses.pkl'), 'wb') as f:
    pickle.dump((model, buf_size, reg, kmscores, stdscores), f)