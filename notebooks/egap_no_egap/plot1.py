# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
from matplotlib import rc
plt.rcParams['text.usetex'] = True
rc('font', family='sans-serif')#, sans-serif='Times')
plt.rcParams.update({
    "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Times"]})
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

plt.rcParams.update({'font.size': 13})
from matplotlib import cm


stds, kms, OO, bbs, bbrs = {}, {}, {}, {}, {}
for dir in os.listdir('cps'):
    if os.path.isdir(os.path.join('cps', dir)) and 'responses.pkl' in os.listdir(os.path.join('cps', dir)):
        try:
            model, buffer, reg, km, std = pickle.load(open(os.path.join('cps', dir, 'responses.pkl'), 'rb'))
            model, buffer, reg, bb  = pickle.load(open(os.path.join('cps', dir, 'rebuf.pkl'), 'rb'))
            model, buffer, reg, bbr  = pickle.load(open(os.path.join('cps', dir, 'resubtest.pkl'), 'rb'))
        except:
            print(f'Error in {dir}')
        if (model, buffer, reg) not in stds:
            stds[(model, buffer, reg)] = []
            kms[(model, buffer, reg)] = []
            bbs[(model, buffer, reg)] = []
            bbrs[(model, buffer, reg)] = []
        stds[(model, buffer, reg)].append(std)
        kms[(model, buffer, reg)].append(km)
        bbs[(model, buffer, reg)].append(bb)
        bbrs[(model, buffer, reg)].append(bbr)

rbbs = {}
for k in stds:
    OO[k] = len(stds[k])
    stds[k] = np.stack(stds[k]).mean(0)
    kms[k] = np.stack(kms[k]).mean(0)
    # bbvar[k] = np.meadinp.stack(bbs[k]).std(0)
    rbbs[k] = np.median(np.stack(bbs[k]), axis=0)
    bbrs[k] = np.mean(np.stack(bbrs[k]), axis=0)

mods = ['er_ace_egap', 'icarl_egap']

plt.figure(figsize=(5, 2))
for b in [500]:
    for m in mods:
        for r in ['none']:
            plt.plot(range(9), np.array(rbbs[(m, b, r)][1:]), 
            '-' +
            ('*' if m == 'er_ace_egap' else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(r == 'egap')))
plt.grid()
plt.legend(['ER-ACE', 'iCaRL'])
plt.title('Clustering Error')

# %%

egap_exp = 'EraceEgapb2NC16K4-exSX7'
none_exp = 'EraceNone-37CZ2'