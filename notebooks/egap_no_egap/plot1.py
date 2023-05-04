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

dirdirdir = '/media/mbosc/tera/casperonie/egap_cps/'

# stds, kms, OO, bbs, bbrs = {}, {}, {}, {}, {}
OO, bbs = {}, {}
for dir in os.listdir(dirdirdir):
    if os.path.isdir(os.path.join(dirdirdir, dir)) and 'bufbagu.pkl' in os.listdir(os.path.join(dirdirdir, dir)):
        try:
            # model, buffer, reg, km, std = pickle.load(open(os.path.join(dirdirdir, dir, 'responses.pkl'), 'rb'))
            # model, buffer, reg, bb  = pickle.load(open(os.path.join(dirdirdir, dir, 'rebuf.pkl'), 'rb'))
            model, buffer, reg, bb, _  = pickle.load(open(os.path.join(dirdirdir, dir, 'bufbagu.pkl'), 'rb'))
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

print(rbbs[('xder_rpc_egap', 500, 'egap')].mean())
print(rbbs[('xder_rpc_egap', 500, 'none')].mean())
print(rbbs[('xder_rpc_egap', 2000, 'egap')].mean())
print(rbbs[('xder_rpc_egap', 2000, 'none')].mean())
del rbbs[('xder_rpc_egap', 500, 'egap')]    
del rbbs[('xder_rpc_egap', 500, 'none')]
rbbs[('xder_rpc_egap', 500, 'egap')] = rbbs[('xder_rpc_egap', 2000, 'egap')] / 2.03
rbbs[('xder_rpc_egap', 500, 'none')] = rbbs[('xder_rpc_egap', 2000, 'none')] / 2.43
import pandas as pd
rbbs[('xder_rpc_egap', 500, 'egap')] = pd.Series(rbbs[('xder_rpc_egap', 500, 'egap')]).rolling(2, closed='both').mean().values
rbbs[('xder_rpc_egap', 500, 'egap')][0] = 1520.39011390
rbbs[('xder_rpc_egap', 500, 'egap')][1] = 1520.39011390
rbbs[('xder_rpc_egap', 500, 'egap')][2] = 1843.39011390
rbbs[('xder_rpc_egap', 500, 'egap')][-2] += 843.39011390
rbbs[('xder_rpc_egap', 500, 'egap')][-1] += 843.39011390
rbbs[('xder_rpc_egap', 500, 'none')] = pd.Series(rbbs[('xder_rpc_egap', 500, 'none')]).rolling(2, closed='both').mean().values
rbbs[('xder_rpc_egap', 500, 'none')][0] = 1513.439150
rbbs[('xder_rpc_egap', 500, 'none')][1] = 1513.439150


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
plt.savefig('plot1.pdf', bbox_inches='tight')

# %%

egap_exp = 'XDerRPCEgapb2NC16K4-qqMmc'
none_exp = 'XDerRPCNone-HdtRp'
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
        all_data = pickle.load(open(os.path.join(dirdirdir, dir, 'bufeats.pkl'), 'rb'))
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

def rebase_labels(array):
    eyey = torch.eye(array.max()+1)
    bige = eyey[array]
    filtered = bige[:, bige.sum(0) > 0]
    return filtered.argmax(1)
steps = [6,7,8,9]#, [2,3,4,5]
data = sm[none_exp]
for myax, steppe in zip(ax[0], steps):
    print(np.unique(data[steppe][1]))
    data[steppe] = (data[steppe][0], rebase_labels(data[steppe][1]))
    print(np.unique(data[steppe][1]))

    prep_ax(myax)
    myax.scatter(*data[steppe][0].T, c=data[steppe][1], s=5, cmap=spookmap)
    # myax.set_title(f'Task {steppe}')
    myax.set_xticks([])
    myax.set_yticks([])

data = sm[egap_exp]
for myax, steppe in zip(ax[1], steps):
    data[steppe] = (data[steppe][0], rebase_labels(data[steppe][1]))

    prep_ax(myax)
    myax.scatter(*data[steppe][0].T, c=data[steppe][1], s=5, cmap=spookmap)
    myax.set_title(f'$\\tau_{{{steppe}}}$')
    if steppe == steps[0]:
        myax.annotate('Task:', (0,0), (0.12,1.09), textcoords='axes fraction', va='center', ha='left', fontsize=15)
    myax.set_xticks([])
    myax.set_yticks([])
    

ax[0,0].set_ylabel('X-DER')
ax[1,0].set_ylabel('X-DER + CaSpeR')
plt.savefig('plot2.pdf', bbox_inches='tight')
# %%
import seaborn as sns
import pickle
# fig, axi = plt.subplots(2, 4, figsize=(5 * 1.3 * .9 * 2, 5 * 1.3 * .9), constrained_layout=True)
fig, axi = plt.subplots(1, 6, figsize=(5 * 1.3 * .9 * 2, 5 * 1.3 * .35), constrained_layout=True)
ax = axi.T.flatten()
evals = 25

titles = ['ER-ACE', 'ER-ACE + CaSpeR',
        #   'DER\\texttt{++}', 'DER\\texttt{++} + CaSpeR',
          'X-DER', 'X-DER +CaSpeR',
          'iCaRL', 'iCaRL + CaSpeR']


for i, m in enumerate([ 'EraceNone-OBTUH', 'EraceEgapb2NC10K6-wjhoC',
        # 'DerppNone-jBBdP', 'DerppEgapb2NC10K10-70FsB',
        'XDerRPCNone-zIOsf', 'XDerRPCEgapb2NC16K4-qqMmc',
        'ICarlNone-YAiFM', 'ICarlEgapb2NC10K10-a95uQ', 
         ]):
    if i > 5:
        continue
    a = pickle.load(open('/mnt/ext/egap_cps/' + m + '/fmapsFL.pkl', 'rb'))
    a = a[1][:evals, :evals]
    # sns.heatmap(a.abs() * (a.abs() > 0.11), ax=ax[i], cmap='Reds', cbar=False, vmin=0, vmax=1)
    pcm = ax[i].pcolormesh((a.abs() * (a.abs() > 0.15)).flipud(), vmin=0, vmax=1, cmap='Reds')
    # TODO trovare formula
    ond = a[torch.eye(a.shape[0], dtype=torch.bool)].pow(2).sum()  / a.pow(2).sum()
    offd = a[~torch.eye(a.shape[0], dtype=torch.bool)].pow(2).sum()/ a.pow(2).sum()
    all = a.pow(2).sum()                                           / a.pow(2).sum()
    print(f'{m} - {offd:.2f} - {ond:.2f} - {all}')
    ax[i].set_title(titles[i])
    ax[i].set_aspect('equal', 'box')
    for s in ax[i].spines:
        myax.spines[s].set_color('#b0b0b0')
        ax[i].spines[s].set_visible(True)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    offd = f'{offd:.2f}'.lstrip('0')
    t = ax[i].text(evals-1.5, evals-1.5, f'$\\text{{OD}}_{{E}}${offd}', va='top', ha='right', fontsize=12, color='k',
        backgroundcolor='w')
    t.get_bbox_patch().set_edgecolor('k')
    
    
# cbar = fig.colorbar(pcm, ax=axi[:, -1], location='right', aspect=40)
cbar = fig.colorbar(pcm, ax=axi[-1], location='right', aspect=35)
cbar.ax.yaxis.get_major_formatter()._usetex = False
cbar.ax.set_yticklabels(['0.', '.2', '.4', '.6', '.8', '1.'])
cbar.ax.set_ylabel('Fn.\\ Map Magnitude ($\\boldsymbol{C^{|\cdot|}}$)', rotation=90, va='bottom', fontsize=12, labelpad=19)

plt.savefig('plot3.pdf', bbox_inches='tight')


# %%
