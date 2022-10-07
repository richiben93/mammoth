# %%
import pandas as pd
import os

root = '/mnt/ext/egap_cps/'
# find all files in subfolders
all_data = pd.DataFrame(columns=['file', 'model', 'reg', 'buffer', 'task', 'accmean'])
files = []
for rd, _, f in os.walk(root):
    for file in f:
        if file == 'acc.txt':
            ffn = os.path.join(rd, file)
            with open(ffn, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.count(' ') == 4:
                        model, reg, buffer, task, accmean = line.split(' ')
                        all_data = all_data.append({'file': os.path.basename(rd), 'model': model,
                            'reg': reg, 'buffer': buffer, 'task': task,
                            'accmean': accmean}, ignore_index=True)

all_data.accmean = all_data.accmean.astype(float)


# %%
pd.set_option('display.max_rows', 100)
all_data[(all_data.task == '10') & (all_data.model == 'icarl_egap') & (all_data.reg == 'egap') & (all_data.buffer == '2000')]
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