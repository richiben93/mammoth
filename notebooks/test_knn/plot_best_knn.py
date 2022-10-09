# %%
import pandas as pd
import os

root = '/mnt/ext/egap_cps/'
# find all files in subfolders
all_data = pd.DataFrame(columns=['file', 'model', 'reg', 'buffer', 'task', 'accmean'])
files = []
for rd, _, f in os.walk(root):
    for file in f:
        if file == 'knn_5.txt':
            ffn = os.path.join(rd, file)
            with open(ffn, 'r') as f:
                ddict = eval(f.readlines()[0])
                for k, v in ddict.items():
                    model, reg, buffer = k
                    accmean = v[10]['knn_mean']
                    all_data = all_data.append({'file': os.path.basename(rd), 'model': model,
                        'reg': reg, 'buffer': buffer, 'task': 10,
                        'accmean': accmean}, ignore_index=True)

all_data.accmean = all_data.accmean.astype(float)

aggf = lambda x: x[:].mean() if len(x) > 1 else x.mean()
cursed_idxes = cursed = ['DerppEgapb2NC10K10-E8mrT', 'DerppEgapb2NC10K10-70FsB',
    'DerppEgapb2NC10K10-RQGuh', 'DerppEgapb2NC10K10-ysKP6', 'DerppNone-HPx0g',
    'DerppNone-Th2IC', 'DerppNone-yXRTL', 'DerppNone-ZjsSx_2000', 'EraceEgapb2NC10CosK5-VyQ5G_2000',
    'EraceNone-P4JAv_2000', 'EraceNone-Qzo4h', 'EraceNone-XSFZg', 'XDerRPCEgapb2NC16K4-XIL6B', 'XDerRPCNone-HdtRp',
    'ICarlNone-Mi7ND', 'ICarlNone-MYc7I', 'ICarlNone-DzvPO', 'ICarlNone-xJFGh', 'ICarlEgapb2NC10K10-v6EZ0',
    'PodnetEgapb2NC10K10-PZMe3', 'PodnetNone-Vw1GI']
display(all_data[~all_data.file.isin(cursed_idxes)].groupby(['model', 'reg', 'buffer']).agg(aggf))
print(all_data[~all_data.file.isin(cursed_idxes)].groupby(['model', 'reg', 'buffer']).agg(aggf).to_latex())
# %%
pd.set_option('display.max_rows', 100)
# (all_data.model == 'xder_rpc_egap') & (all_data.reg == 'egap') &
all_data[(all_data.model.isin(['podnet_egap', 'icarl_egap']))].sort_values(by=['model', 'reg', 'accmean'])
# %%
'''
good boys:
EraceEgapb2NC10CosK5-VyQ5G
EraceNone-2FkDp

EraceEgapb2NC16K4-DuNvm
EraceNone-37CZ2

ICarlEgapb2NC16K4-coegp
ICarlNone-9o8Br

ICarlEgapb2NC10K10-PMXxZ
ICarlNone-MYc7I

DerppEgapb2NC10K10-Mn2MC
DerppNone-Th2IC
'''