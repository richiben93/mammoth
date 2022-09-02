import streamlit as st
from re import sub
import torch
import numpy as np
import os
from tqdm import tqdm
from argparse import Namespace

conf_path = os.getcwd()
while not 'mammoth' in os.path.basename(conf_path):
    os.chdir("..")
    conf_path = os.getcwd()
print('Config path: {}'.format(conf_path))

from backbone.ResNet18 import resnet18
from utils.conf import get_device
from datasets.seq_cifar100 import SequentialCIFAR100_10x10 as SequentialCIFAR100
from datasets.seq_cifar10 import SequentialCIFAR10

@st.cache(allow_output_mutation=True)
def load_model():
    # prepare dataset
    args = Namespace(
                batch_size=32,
                dataset='seq-cifar10',
                validation=False,
    )
    dataset = SequentialCIFAR10(args)
    data_loaders = [dataset.get_data_loaders()[0] for _ in range(dataset.N_TASKS)]

    # load the model
    all_data = {}

    path_base = 'notebooks/paths_an2/'

    for nid, cp in enumerate(sorted([x for x in os.listdir(path_base) if x.endswith('.pt')])):
        device = get_device()
        net = resnet18(10)
        sd = torch.load(path_base + cp)
        net.load_state_dict(sd)
        net.to(device)
        net.eval()
        
        all_data[cp] = {}
        all_data[cp]['net'] = net
        all_data[cp]['cp'] = cp

    # compute projections
    for nid in tqdm(all_data):
        net = all_data[nid]['net']
        with torch.no_grad():
            proj, labe, pred = [], [], []
            for dl in dataset.test_loaders:
                for x, y in dl:
                    x = x.to(device)
                    y = y.to(device)
                    proj.append(net.features(x).cpu())
                    pred.append(net(x).cpu())
                    labe.append(y.cpu())
            proj = torch.cat(proj)
            labe = torch.cat(labe)
            pred = torch.cat(pred)

            all_data[nid]['proj'] = proj
            all_data[nid]['labe'] = labe
            all_data[nid]['pred'] = pred

    # accuracy
    for nid in tqdm(all_data):
        preds, targets = all_data[nid]['pred'], all_data[nid]['labe']
        acc = torch.sum(torch.argmax(preds, dim=1) == targets).item() / preds.shape[0]
        all_data[nid]['acc'] = acc
        print('{} -?> Accuracy: {}'.format(nid, acc))

    return all_data

# %%
# compute k-means score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
all_data = load_model()
i = torch.rand(1) > 0.5
i = i.int().item()

def printme(ok):
    if ok:
        print('OK')
    else:
        print('FAIL')

a = st.button('<-', on_click=lambda : printme(i - 0))
b = st.button('->', on_click=lambda : printme(i - 1))

with open('results.txt', 'a') as f:
    if i == 0 and a or i == 1 and b:
        f.write('ok\n')
        print('ok')
    else:
        f.write('not ok\n')
        print('not ok')

sample_idx = torch.randperm(10000)[:1000]
from sklearn.manifold import TSNE
fig, ax = plt.subplots(1, 2, figsize=(6, 2.2))

for idd, nid in enumerate(all_data):
    
    idd = i - idd
    proj, labe = all_data[nid]['proj'], all_data[nid]['labe']
    proj_t = torch.tensor(TSNE(n_components=2).fit_transform(proj.numpy()))
    subsample = sample_idx
    proj_t = proj_t[subsample]
    labe_t = labe[subsample]
    for c in labe_t.unique():
        ax[idd].scatter(proj_t[labe_t == c, 0], proj_t[labe_t == c, 1], label=c.item(), s=0.7)
    # ax[idd].set_title(nid)
# ax[-1].legend()
st.pyplot(fig)

