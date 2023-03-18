# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' + '\n' + r'\usepackage{mathpazo}' #for \text command
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
palette = '00340a-ace894-a8ba9a-4d82b7-6b6570-4a314d-1a090d-57120d-c6572c-f7934c'
from matplotlib.colors import ListedColormap
spookmap = ListedColormap(['#'+x for x in palette.split('-')])

# stds, kms, OO, bbs, bbrs = {}, {}, {}, {}, {}
cpdir = '/home/mbosc/phd/bertodomammoth/rodomammoth/data/more_checkpoints/seq-cifar100-10x10/'
OO, bbs = {}, {}
for dir in os.listdir(cpdir):
    if os.path.isdir(os.path.join(cpdir, dir)) and 'bufbagu.pkl' in os.listdir(os.path.join(cpdir, dir)):
        try:
            # model, buffer, reg, km, std = pickle.load(open(os.path.join(cpdir, dir, 'responses.pkl'), 'rb'))
            # model, buffer, reg, bb  = pickle.load(open(os.path.join(cpdir, dir, 'rebuf.pkl'), 'rb'))
            model, buffer, reg, bb, _  = pickle.load(open(os.path.join(cpdir, dir, 'bufbagu.pkl'), 'rb'))
        except:
            print(f'Error in {dir}')
        if (model, buffer, reg) not in bbs:
            bbs[(model, buffer, reg)] = []
        # stds[(model, buffer, reg)].append(std)
        # kms[(model, buffer, reg)].append(km)
        bbs[(model, buffer, reg)].append(bb)
        

rbbs = {}
for k in bbs:
    OO[k] = len(bbs[k])
    # stds[k] = np.stack(stds[k]).mean(0)
    # kms[k] = np.stack(kms[k]).mean(0)
    # bbvar[k] = np.meadinp.stack(bbs[k]).std(0)
    rbbs[k] = np.median(np.stack(bbs[k]), axis=0)

# print(rbbs[('xder_rpc_egap', 500, 'egap')].mean())
# print(rbbs[('xder_rpc_egap', 500, 'none')].mean())
# print(rbbs[('xder_rpc_egap', 2000, 'egap')].mean())
# print(rbbs[('xder_rpc_egap', 2000, 'none')].mean())
# del rbbs[('xder_rpc_egap', 500, 'egap')]    
# del rbbs[('xder_rpc_egap', 500, 'none')]
# rbbs[('xder_rpc_egap', 500, 'egap')] = rbbs[('xder_rpc_egap', 2000, 'egap')] / 2.03
# rbbs[('xder_rpc_egap', 500, 'none')] = rbbs[('xder_rpc_egap', 2000, 'none')] / 2.43
# import pandas as pd
# rbbs[('xder_rpc_egap', 500, 'egap')] = pd.Series(rbbs[('xder_rpc_egap', 500, 'egap')]).rolling(2, closed='both').mean().values
# rbbs[('xder_rpc_egap', 500, 'egap')][0] = 1520.39011390
# rbbs[('xder_rpc_egap', 500, 'egap')][1] = 1520.39011390
# rbbs[('xder_rpc_egap', 500, 'egap')][2] = 1843.39011390
# rbbs[('xder_rpc_egap', 500, 'egap')][-2] += 843.39011390
# rbbs[('xder_rpc_egap', 500, 'egap')][-1] += 843.39011390
# rbbs[('xder_rpc_egap', 500, 'none')] = pd.Series(rbbs[('xder_rpc_egap', 500, 'none')]).rolling(2, closed='both').mean().values
# rbbs[('xder_rpc_egap', 500, 'none')][0] = 1513.439150
# rbbs[('xder_rpc_egap', 500, 'none')][1] = 1513.439150


#'podnet_egap',
mods = [ 'icarl_egap', 'er_ace_egap', 'xder_rpc_egap']
# plt.figure(figsize=(5 * 1.5 * .9, 2 * 1.3 * .9)) # long
plt.figure(figsize=(5 * 1.5 * .9 * 2 * .25, 5 * 1.5 * .9 * .55)) # tall

# ----------------------------- MAGIC PLOTMAGIC -----------------------------
myax = plt.gca()
myax.xaxis.get_major_formatter()._usetex = False
myax.yaxis.get_major_formatter()._usetex = False

# myax.spines['top'].set_visible(False)
myax.spines['top'].set_color('#b0b0b0')
# myax.spines['right'].set_visible(False)
myax.spines['right'].set_color('#b0b0b0')

myax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) 
# myax.tick_params(axis='y', colors=pp[-1], labelcolor=pp[-1])

# axb.xaxis.get_major_formatter()._usetex = False
# axb.yaxis.get_major_formatter()._usetex = False
# axb.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     top=False) 
# axb.tick_params(axis='y', colors=pp[-3], labelcolor=pp[-3])
# --------------------------------------------------------------------------

markerdict = {
    'none': 'o',
    'egap': '^',
}

msdict = {
    'none': 7,
    'egap': 9,
}

cdict = {
    'icarl_egap': spookmap.colors[1],
    'er_ace_egap': spookmap.colors[3],
    'podnet_egap': spookmap.colors[9],
    'xder_rpc_egap': spookmap.colors[9],
}

for b in [500]:
    for m in mods:
        for r in ['none', 'egap']:
            plt.plot(range(9), np.array(rbbs[(m, b, r)][1:]), 
            ('-' if r == 'none' else '--') +
            markerdict[r], label=(m,b,r,f'[{OO[(m,b,r)]}]'), color=cdict[m],
            mew=1, mec='w', ms=msdict[r])

myax.xaxis.grid(True, which='major', linestyle=':', linewidth=1)
myax.yaxis.grid(True, which='major', linestyle='-', linewidth=1)
myax.set_axisbelow(True)

myax.set_xticks(range(9))
myax.set_xticklabels([f'$\\tau_{{{i+2}}}$' for i in range(9)]) # check convenzione ema

plt.legend()
handles, _ = myax.get_legend_handles_labels()
handles = np.array(handles)
plt.legend(#handles[[0,2,4,6,1,3,5,7]], ['PODNet', 'iCaRL', 'ER-ACE', 'X-DER'] +[' + CaSpeR']*4, 
handles[[0,2,4,1,3,5]], ['iCaRL', 'ER-ACE', 'X-DER'] +[' + CaSpeR']*3, 
edgecolor='k', framealpha=1, fancybox=False, loc='upper center',
    handletextpad=0.3, handlelength=1.7, ncol=2, columnspacing=0.4, labelspacing=0.15)#, bbox_to_anchor=(-0.015,1.03))

## OLD WAY
# myax.set_ylabel('Clustering Error')
# myax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# myax.set_yticklabels(['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1'])
# myax.set_ylim(0, 1)

myax.set_ylabel('Label-Signal Variation ($\\sigma$)')
myax.set_ylim(0, 4500)
myax.set_yticks(np.arange(10) * 500)
myax.set_xlim(-0.5, 8.5)
myax.set_xlabel('Task')
plt.savefig('plot1r.pdf', bbox_inches='tight')
# %%