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
    for project in ['casper-icml', 'rodo-istatsJIHAD', 'rodo-istats', 'rodo-istatsTEMP']:
        for runna in api.runs(f'{entity}/{project}'):
            if runna.name == bbasename(foldername):#.split('_')[0]:
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
if os.path.exists(os.path.join(args.foldername, 'bufeats.pkl')):
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
    net.to(device)    

    buf = all_data[(model, reg, buf_size)][id_task]['buf']
    bufdata = buf.get_data(buf.buffer_size, transform=dataset.test_loaders[0].dataset.transform.transforms[1])
    bx, by = bufdata[0], bufdata[1]
    bx = bx.to(device)
    by = by
    bproj = net.features(bx).cpu()
    all_data[(model, reg, buf_size)][id_task][f'bproj'] = bproj
    all_data[(model, reg, buf_size)][id_task][f'by'] = by
    
    net.to('cpu')

for id_task in tqdm(range(1, 11)):
    del all_data[(model, reg, buf_size)][id_task]['net']
    del all_data[(model, reg, buf_size)][id_task]['buf']

print('-- Saving to', os.path.join(foldername, 'bufeats.pkl'), '\n')
with open(os.path.join(foldername, 'bufeats.pkl'), 'wb') as f:
    pickle.dump(all_data, f)