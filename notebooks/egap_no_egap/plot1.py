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
            # model, buffer, reg, bb  = pickle.load(open(os.path.join('cps', dir, 'rebuf.pkl'), 'rb'))
            model, buffer, reg, bb, _  = pickle.load(open(os.path.join('cps', dir, 'bufbagu.pkl'), 'rb'))
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
        

rbbs = {}
for k in stds:
    OO[k] = len(stds[k])
    stds[k] = np.stack(stds[k]).mean(0)
    kms[k] = np.stack(kms[k]).mean(0)
    # bbvar[k] = np.meadinp.stack(bbs[k]).std(0)
    rbbs[k] = np.median(np.stack(bbs[k]), axis=0)
    

mods = ['icarl_egap', 'er_ace_egap']
# plt.figure(figsize=(5 * 1.5 * .9, 2 * 1.3 * .9)) # long
plt.figure(figsize=(5 * 1.5 * .9 * 2 * .25, 5 * 1.5 * .9 * .6)) # tall

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

for b in [500]:
    for m in mods:
        for r in ['none', 'egap']:
            plt.plot(range(9), np.array(rbbs[(m, b, r)][1:]), 
            ('-' if r == 'none' else '--') +
            ('o' if m == 'er_ace_egap' else '^'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(r == 'egap')))

myax.xaxis.grid(True, which='major', linestyle=':', linewidth=1)
myax.yaxis.grid(True, which='major', linestyle='-', linewidth=1)
myax.set_axisbelow(True)

myax.set_xticks(range(9))
myax.set_xticklabels([f'$\\tau_{{{i+2}}}$' for i in range(9)]) # check convenzione ema
handles, _ = myax.get_legend_handles_labels()
handles = np.array(handles)
plt.legend(handles[[0,2,1,3]], ['ER-ACE', 'iCaRL', ' + EP',  ' + EP'], edgecolor='k', framealpha=1, fancybox=False, loc='upper left',
    handletextpad=0.3, handlelength=0.8, ncol=2, columnspacing=0.4, labelspacing=0.15)#, bbox_to_anchor=(-0.015,1.03))

## OLD WAY
# myax.set_ylabel('Clustering Error')
# myax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# myax.set_yticklabels(['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1'])
# myax.set_ylim(0, 1)

myax.set_ylabel('Label-Signal Variation')
myax.set_ylim(0, 3400)
myax.set_xlim(-0.5, 8.5)

myax.set_xlabel('Task')
plt.savefig('plot1.pdf', bbox_inches='tight')

# %%
egap_exp = 'EraceEgapb2NC16K4-exSX7'
none_exp = 'EraceNone-37CZ2'
if not os.path.exists('scatter_meta.pkl'):
    from sklearn.manifold import TSNE, SpectralEmbedding
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

    sm = {}
    for dir in [none_exp, egap_exp]:
        sm[dir] = {}
        all_data = pickle.load(open(os.path.join('cps', dir, 'bufeats.pkl'), 'rb'))
        labelle = torch.tensor([0, 2, 5, 8, 9, 11, 13, 14, 16, 18])
        for i, steppe in enumerate([2,3,4,5,6,7,8,9,10]):
            bproj, by = list(all_data.values())[0][steppe]['bproj'], list(all_data.values())[0][steppe]['by']
            
            bproj = bproj[torch.isin(by, labelle)]
            by = by[torch.isin(by, labelle)]

            buf_size = list(all_data.values())[0][1]
            knn_laplace = 3 if buf_size == 500 else 4 #int(bbasename(foldername).split('-')[0].split('K')[-1])
            dists = calc_euclid_dist(bproj)
            A, _, _ = calc_ADL_knn(dists, k=knn_laplace, symmetric=True)
            lab_mask = by.unsqueeze(0) == by.unsqueeze(1)
            wrong_A = A[~lab_mask]
            # wcons.append(f'{list(all_data.keys())[0]} - {dir} - {wrong_A.sum() / A.sum()}')

            bproj = TSNE(n_components=2).fit_transform(bproj)#perplexity=20
            # bproj = SpectralEmbedding(n_components=2).fit_transform(bproj)
            sm[dir][steppe] = (bproj, by)
    
    with open('scatter_meta.pkl', 'wb') as f:
        pickle.dump(sm, f)
    print('Computed!')
else:
    sm = pickle.load(open('scatter_meta.pkl', 'rb'))
    print("Loaded!")

# %%
fig, ax = plt.subplots(2, 4, figsize=(5 * 1.5 * .9 * 2 * .7, 5 * 1.5 * .9 * .6))

def prep_ax(anax):
    anax.xaxis.get_major_formatter()._usetex = False
    anax.yaxis.get_major_formatter()._usetex = False

    anax.spines['top'].set_visible(False)
    anax.spines['right'].set_visible(False)
    anax.spines['left'].set_visible(False)
    anax.spines['bottom'].set_visible(False)

    anax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) 

data = sm[none_exp]
for myax, steppe in zip(ax[0], [2,3,4,5]):
    prep_ax(myax)
    myax.scatter(*data[steppe][0].T, c=data[steppe][1], s=5, cmap='tab10')
    # myax.set_title(f'Task {steppe}')
    myax.set_xticks([])
    myax.set_yticks([])

data = sm[egap_exp]
for myax, steppe in zip(ax[1], [2,3,4,5]):
    prep_ax(myax)
    myax.scatter(*data[steppe][0].T, c=data[steppe][1], s=5, cmap='tab10')
    myax.set_title(f'Task {steppe}')
    myax.set_xticks([])
    myax.set_yticks([])
    

ax[0,0].set_ylabel('ER-ACE')
ax[1,0].set_ylabel('ER-ACE + EP')
plt.savefig('plot2.pdf', bbox_inches='tight')
# %%
