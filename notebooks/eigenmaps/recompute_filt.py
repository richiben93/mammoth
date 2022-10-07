# %%
# ER 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['EraceEgapb2NC10CosK5-e5VM0', 'EraceEgapb2NC10CosK5-VyQ5G',
       'EraceEgapb2NC10CosK5-AsouZ'][1:2]
cpbs = ['EraceNone-YdLI1', 'EraceNone-P4JAv',
       'EraceNone-2FkDp', 'EraceNone-XSFZg', 'EraceNone-ZjRTT',
       'EraceNone-Qzo4h', 'EraceNone-NHxIp'][2:]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 4))

evals = 35

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
        sns.heatmap(i[:evals, :evals], vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')



plt.savefig('pdf.pdf', bbox_inches='tight')

# %%
# DER 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['DerppEgapb2NC10K10-Mn2MC', 'DerppEgapb2NC10K10-70FsB']#[0:1]
cpbs = ['DerppNone-Th2IC', 'DerppNone-HPx0g', 'DerppNone-ZjsSx'][:2]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 4))

evals = 35

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
        sns.heatmap(i[:evals, :evals], vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')




# %%
# icarl 2000
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['ICarlEgapb2NC10K10-v6EZ0', 'ICarlEgapb2NC10K10-PMXxZ'][0:1]
cpbs = ['ICarlNone-cpie7', 'ICarlNone-DzvPO', 'ICarlNone-xJFGh'][-1:]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 4))

evals = 35

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
        sns.heatmap(i[:evals, :evals], vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')


# %%
# XDER 500
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
cpas = ['XDerRPCEgapb2NC16K4-V628Q', 'XDerRPCEgapb2NC16K4-oSqlA'][:1]
cpbs = ['XDerRPCNone-PIH1H', 'XDerRPCNone-g74UP'][1:]

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 4))

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
        sns.heatmap(i[:evals, :evals], vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
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

# a = pickle.load(open('../egap_no_egap/cps/EraceNone-cc395/fmaps.pkl', 'rb'))
# fig, ax = plt.subplots(9, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 20))
fig, ax = plt.subplots(2, len(cpas) + len(cpbs), figsize=(2 * (len(cpas) + len(cpbs)), 4))

evals = 25

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
        sns.heatmap(i[:evals, :evals], vmin = -1, vmax = 1, cmap='bwr', ax=x, cbar=None)
        x.set_xticklabels([])
        x.set_yticklabels([])
        x.set_title('A' if j < len(cpas) else 'B')




# %%
