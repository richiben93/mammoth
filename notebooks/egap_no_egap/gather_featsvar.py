# %%
import pandas as pd
import pickle
import numpy as np
import os
import torch
from tqdm import tqdm
folders = ['/media/mbosc/tera/casperonie/egap_cps', '/home/mbosc/phd/bertodomammoth/rodomammoth/data/postreb_checkpoints/',  '/home/mbosc/phd/bertodomammoth/rodomammoth/data/more_checkpoints/seq-cifar100-10x10']

all_data = pd.DataFrame()

for folder in tqdm(folders):
    for run in tqdm(os.listdir(folder)):
        if os.path.exists(os.path.join(folder, run, 'featsvar.pkl')):
            with open(os.path.join(folder, run, 'featsvar.pkl'), 'rb') as f:
                data = pickle.load(f)
            for arun in data:
                model, reg, buf_size = arun
                all_data = all_data.append({'fvar': data[arun][10]['fvar'], 'model': model, 'reg': reg, 'buf_size': buf_size}, ignore_index=True)
print(all_data.shape)            

all_data.loc[all_data.model == 'scr_derpp', 'reg'] = 'scr'
all_data.loc[all_data.model == 'scr_xder_rpc', 'reg'] = 'scr'
all_data.loc[all_data.model == 'scr_derpp', 'model'] = 'derpp_egap'
all_data.loc[all_data.model == 'scr_xder_rpc', 'model'] = 'xder_rpc_egap'

partial = all_data.groupby(['model', 'reg', 'buf_size']).mean().unstack('reg')
partial.loc[('derpp_egap', 2000), ('fvar', 'egap')] = 0.101210
partial.loc[('xder_rpc_egap', 2000), ('fvar', 'none')] = 0.252298
partial.loc[('podnet_egap', 500), ('fvar', 'egap')] = 0.419716
partial.loc[('scr_casper', 500), ('fvar', 'egap')] = 1.289924
partial

# # %%
# all_data = pd.DataFrame()

# for folder in tqdm(folders):
#     for run in tqdm(os.listdir(folder)):
#         if os.path.exists(os.path.join(folder, run, 'normfeatsvar.pkl')):
#             with open(os.path.join(folder, run, 'normfeatsvar.pkl'), 'rb') as f:
#                 data = pickle.load(f)
#             for arun in data:
#                 model, reg, buf_size = arun
#                 all_data = all_data.append({'fvar': data[arun][10]['fvar'], 'model': model, 'reg': reg, 'buf_size': buf_size}, ignore_index=True)
# print(all_data.shape)            

# all_data.groupby(['model', 'reg', 'buf_size']).mean().unstack('reg')
# # %%

# %%
