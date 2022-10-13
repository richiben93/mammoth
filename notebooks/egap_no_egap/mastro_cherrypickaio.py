# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os
from sklearn.manifold import TSNE
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

def bbasename(path):
    return [x for x in path.split('/') if len(x)][-1]
conf_path = os.getcwd()
while not 'mammoth' in bbasename(conf_path):
    conf_path = os.path.dirname(conf_path)
print(conf_path)
tdir = os.getcwd()
os.environ['PYTHONPATH'] = f'{conf_path}'
os.environ['PATH'] += f':{conf_path}'
os.chdir(conf_path)
from utils.spectral_analysis import calc_ADL_knn, calc_euclid_dist
os.chdir(tdir)

dirs = [
    'EraceEgapb2NC10K6-9zwoG',
'EraceEgapb2NC10K6-wjhoC',
'EraceNone-OBTUH',
'EraceNone-WtdgD',
'DerppEgapb2NC10K10-Mn2MC', 'DerppEgapb2NC10K10-70FsB',
'DerppNone-jBBdP',
'ICarlEgapb2NC10K10-yG4S0',
'ICarlEgapb2NC10K10-a95uQ',
'ICarlNone-YAiFM',
'XDerRPCEgapb2NC16K4-XIL6B','XDerRPCEgapb2NC16K4-qqMmc',
'XDerRPCNone-zIOsf','XDerRPCNone-HdtRp'
]
# dirs = ['../sacred_cps/' + x for x in dirs]

plt.ioff()
os.makedirs('figsT', exist_ok=True)
stds, kms, OO, bbs = {}, {}, {}, {}
wcons = []
for dir in tqdm(dirs):#os.listdir('../sacred_cps')):
    #bufeats
    if os.path.isdir(os.path.join('../sacred_cps', dir)) and 'testfeats.pkl' in os.listdir(os.path.join('../sacred_cps', dir)):
        try:
            all_data = pickle.load(open(os.path.join('../sacred_cps', dir, 'testfeats.pkl'), 'rb'))
        except:
            print(f'Error in {dir}')
        
        labelle = torch.tensor([0, 2, 5, 8, 9, 11, 13, 14, 16, 18])
        fig, ax = plt.subplots(2, 5, figsize=(20, 8))
        ax = ax.flatten()
        for i, steppe in enumerate(list(all_data.values())[0]):
            bproj, by = list(all_data.values())[0][steppe]['bproj'], list(all_data.values())[0][steppe]['by']
            
            bproj = bproj[torch.isin(by, labelle)]
            by = by[torch.isin(by, labelle)]

            buf_size = list(all_data.values())[0][1]
            knn_laplace = 3 if buf_size == 500 else 4 #int(bbasename(foldername).split('-')[0].split('K')[-1])
            dists = calc_euclid_dist(bproj)
            A, _, _ = calc_ADL_knn(dists, k=knn_laplace, symmetric=True)
            lab_mask = by.unsqueeze(0) == by.unsqueeze(1)
            wrong_A = A[~lab_mask]
            wcons.append(f'{list(all_data.keys())[0]} - {dir} - {wrong_A.sum() / A.sum()}')

            bproj = TSNE(n_components=2).fit_transform(bproj)#perplexity=20
            myax = ax[i]
            for l in labelle:
                myax.scatter(bproj[by == l, 0], bproj[by == l, 1], label=l, c='C' + str(l.item()))

        plt.title(f'{list(all_data.keys())[0]} - {dir} - {wrong_A.sum() / A.sum():.2f}')
        plt.savefig(f'figsT/{"_".join([str(x) for x in list(all_data.keys())[0]])}_{dir}.pdf')
        plt.close()
# with open('results.txt', 'w') as f:
#     f.write('\n'.join(wcons))
# %%

import pandas as pd
# pd.read_csv('results.txt', header=None, sep='-').groupby(0).median(3)

