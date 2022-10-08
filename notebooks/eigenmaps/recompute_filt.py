# %%
cursed_idxes = cursed = ['DerppEgapb2NC10K10-E8mrT', 'DerppEgapb2NC10K10-70FsB',
    'DerppEgapb2NC10K10-RQGuh', 'DerppEgapb2NC10K10-ysKP6', 'DerppNone-HPx0g',
    'DerppNone-Th2IC', 'DerppNone-yXRTL', 'DerppNone-ZjsSx_2000', 'EraceEgapb2NC10CosK5-VyQ5G_2000',
    'EraceNone-P4JAv_2000', 'EraceNone-Qzo4h', 'EraceNone-XSFZg', 'XDerRPCEgapb2NC16K4-XIL6B', 'XDerRPCNone-HdtRp',
    'ICarlNone-Mi7ND', 'ICarlNone-MYc7I', 'ICarlNone-DzvPO', 'ICarlNone-xJFGh', 'ICarlEgapb2NC10K10-v6EZ0',
    'PodnetEgapb2NC10K10-PZMe3', 'PodnetNone-Vw1GI']
# %%
good_hmaps = []
# %%
# ER 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['EraceEgapb2NC10CosK5-e5VM0', 'EraceEgapb2NC10CosK5-VyQ5G',
       'EraceEgapb2NC10CosK5-AsouZ']
cpbs = ['EraceNone-YdLI1', 'EraceNone-NHxIp' , 'EraceNone-P4JAv',
       'EraceNone-2FkDp', 'EraceNone-XSFZg', 'EraceNone-ZjRTT',
       'EraceNone-Qzo4h']
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')



plt.savefig('pdf.pdf', bbox_inches='tight')

# %%
# DER 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['DerppEgapb2NC10K10-Mn2MC', 'DerppEgapb2NC10K10-70FsB']
cpbs = ['DerppNone-Th2IC', 'DerppNone-HPx0g', 'DerppNone-ZjsSx']
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')




# %%
# icarl 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['ICarlEgapb2NC10K10-v6EZ0', 'ICarlEgapb2NC10K10-PMXxZ']
cpbs = ['ICarlNone-cpie7', 'ICarlNone-DzvPO', 'ICarlNone-xJFGh']
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')


# %%
# XDER 500
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['XDerRPCEgapb2NC16K4-V628Q', 'XDerRPCEgapb2NC16K4-oSqlA']
cpbs = ['XDerRPCNone-PIH1H', 'XDerRPCNone-g74UP']
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')



# %%
# Podnet 500
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['PodnetEgapb2NC16K4-eo9J2', 'PodnetEgapb2NC16K4-eyfIh']
cpbs = ['PodnetNone-UjftN', 'PodnetNone-0i7XX'] 
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')




# %%
# Podnet 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['PodnetEgapb2NC10K10-lMKWg','PodnetEgapb2NC10K10-PZMe3']
cpbs = ['PodnetNone-qjxuI','PodnetNone-Vw1GI'] 
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')


# %%
# Xder 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['XDerRPCEgapb2NC16K4-XIL6B','XDerRPCEgapb2NC16K4-qqMmc']
cpbs = ['XDerRPCNone-zIOsf','XDerRPCNone-HdtRp']
cpas = [x for x in cpas if x not in cursed_idxes]
cpbs = [x for x in cpbs if x not in cursed_idxes]


# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(4 * (len(cpas) + len(cpbs)), 8))

evals = 45

# a = pickle.load(open('/mnt/ext/egap_cps/EraceEgapb2NC10CosK5-VyQ5G/fmaps.pkl', 'rb'))
for j, cp in enumerate(cpas + cpbs):
    try:
        a = pickle.load(open('/mnt/ext/egap_cps/' + cp + '/fmapsFILT.pkl', 'rb'))
        print('found', cp)
    except:
        print('not found', cp)
        continue
    for ii, i in enumerate(a):
        x = ax[ii, j]
        sns.heatmap(i[:evals, :evals].abs(), vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')
        # good_hmaps.append(i[:evals, :evals])


# %%
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# cmap = matplotlib.cm.get_cmap('bwr')
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x = np.arange(0, 45, 1)
# y = np.arange(0, 45, 1)
# _xx, _yy = np.meshgrid(x, y)
# x, y = _xx.ravel(), _yy.ravel()
# top = good_hmaps[1].abs().cpu().numpy().ravel() - good_hmaps[3].abs().cpu().numpy().ravel()
# depth = width = 1
# bottom = np.zeros_like((top / 2)-0.5)
# ccc = cmap(top)
# ax.bar3d(x, y, bottom, width, depth, top, color=ccc, shade=True)
# ax.view_init(40, 20)
# ax.set_title('Shaded')
# %%
