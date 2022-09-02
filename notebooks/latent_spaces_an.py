# %%
from re import sub
import torch
import numpy as np
import os
from tqdm import tqdm
from argparse import Namespace

import torchvision.transforms as transforms
cifar = 10

conf_path = os.getcwd()
while not 'mammoth' in os.path.basename(conf_path):
    os.chdir("..")
    conf_path = os.getcwd()
print('Config path: {}'.format(conf_path))

# prepare dataset
from datasets.seq_cifar100 import SequentialCIFAR100_10x10 as SequentialCIFAR100
from datasets.seq_cifar10 import SequentialCIFAR10
args = Namespace(
            batch_size=32,
            dataset=f'seq-cifar{cifar}',
            validation=False,
)
dataset = eval(f'SequentialCIFAR{cifar}(args)')
data_loaders = [dataset.get_data_loaders()[0] for _ in range(dataset.N_TASKS)]

from backbone.ResNet18 import resnet18
from utils.conf import base_path, get_device

# load the model
all_data = {}

path_base = 'notebooks/paths_erace_c10/'
import pickle
# sorted([x for x in os.listdir(path_base) if x.endswith('.pt')])
for nid, cp in enumerate(['task_4.pt', 'task_5.pt']):
    device = get_device()
    net = resnet18(cifar)
    sd = torch.load(path_base + cp)
    net.load_state_dict(sd)
    net.to(device)
    net.eval()
    
    all_data[cp] = {}
    all_data[cp]['net'] = net
    all_data[cp]['cp'] = cp

    buf = pickle.load(open(path_base + cp.replace('_', '').replace('.pt', '_buffer.pkl'), 'rb'))
    all_data[cp]['buf'] = buf

# %%
# compute projections
for nid in tqdm(all_data):
    net = all_data[nid]['net']
    with torch.no_grad():
        proj, labe, pred = [], [], []
        for dl in dataset.test_loaders[:int(nid.split('.')[0].split('_')[-1])]:
            for x, y in dl:
                x = x.to(device)
                y = y.to(device)
                proj.append(net.features(x).cpu())
                pred.append(net(x).cpu())
                labe.append(y.cpu())
        proj = torch.cat(proj)
        labe = torch.cat(labe)
        pred = torch.cat(pred)

        all_data[nid]['proj'] = proj
        all_data[nid]['labe'] = labe
        all_data[nid]['pred'] = pred

        buf = all_data[nid]['buf']
        proj, labe, pred = [], [], []
        data = buf.get_data(len(buf), transform=transforms.Compose([transforms.ToPILImage(), dataset.TRANSFORM]))
        x, y = data[0], data[1]
        x = x.to(device)
        y = y.to(device)
        proj.append(net.features(x).cpu())
        pred.append(net(x).cpu())
        labe.append(y.cpu())
        proj = torch.cat(proj)
        labe = torch.cat(labe)
        pred = torch.cat(pred)

        all_data[nid]['proj_buf'] = proj
        all_data[nid]['labe_buf'] = labe
        all_data[nid]['pred_buf'] = pred

# %%
# make buffer autovalorz
# import matplotlib.pyplot as plt
# from utils.spectral_analysis import laplacian_analysis
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# evals, evects = [], []
# with torch.no_grad():
#     for nid in tqdm(all_data):
   
#         # on buffer
#         projs, targets = all_data[nid]['proj_buf'], all_data[nid]['labe_buf']
#         print(projs.shape)

#         _, evalues, evectors, _, _ = laplacian_analysis(projs, knn=20, norm_lap=True, n_pairs=70)
#         evects.append(evectors)
#         evals.append(evalues)
    
# plt.figure()
# plt.title(f'Eigenvalues')
# for i, v in enumerate(evals):
#     plt.plot(v.detach().cpu().numpy(), label=i, marker='.', lw=0)
# plt.legend()
    
# plt.figure()
# plt.title(f'Eigengaps')
# for i, v in enumerate(evals):
#     plt.plot((v[1:] - v[:-1]).detach().cpu().numpy(), label=i, marker='_', lw=0)
#     print(f'{i} -> chiediamo {i*2}, giusto {i*2-1} -> {(v[1:] - v[:-1]).argsort(descending=True)[:5]}')


# %%
# accuracy
for nid in tqdm(all_data):
    preds, targets = all_data[nid]['pred'], all_data[nid]['labe']
    acc = torch.sum(torch.argmax(preds, dim=1) == targets).item() / preds.shape[0]
    all_data[nid]['acc'] = acc
    print('{} -?> Accuracy: {}'.format(nid, acc))

# %%
# # compute k-means score
# from sklearn.cluster import KMeans

# for nid in tqdm(all_data):
#     k = 10
#     feats, targets = all_data[nid]['proj'], all_data[nid]['labe']
#     kmeans = torch.tensor(KMeans(n_clusters=k, random_state=0).fit_predict(feats.numpy()))
#     ents = []
#     for i in kmeans.unique():
#         _ , conf = targets[kmeans == i].unique(return_counts=True)
#         # compute entropy
#         probs = conf / conf.sum()
#         entropy = -probs.mul(probs.log()).sum()
#         ents.append(entropy.item())
#         # print(f'cluster {i} -> {entropy}')
#     kmscore = np.mean(ents)
#     all_data[nid]['kmscore'] = kmscore
#     print(f'Task {nid} -> {kmscore}')
# %%
# Create Samplio
# sample_idx = torch.randperm(len(proj))[:2000]


# %%
# make graph
import matplotlib.pyplot as plt
from utils.spectral_analysis import laplacian_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
evals, evects = [], []
ssss = torch.randperm(10000)[:1000]
minlen = min([len(all_data[nid]['proj']) for nid in all_data])
with torch.no_grad():
    for nid in tqdm(all_data):
        projs, targets = all_data[nid]['proj'][ssss[ssss < minlen]], all_data[nid]['labe'][ssss[ssss < minlen]]
        print(projs.shape)
        _, evalues, evectors, _, _ = laplacian_analysis(projs, knn=5, norm_lap=True, n_pairs=20, cos_dist=True)
        evects.append(evectors)
        evals.append(evalues)
    c = evects[1].T @ evects[0]

for i, v in enumerate(evals):
    plt.plot(v.detach().cpu().numpy(), label=i)
plt.legend()
# we    
plt.figure()
sns.heatmap(c.detach().cpu().numpy(), vmin=-1, vmax=1, cmap='bwr')
plt.title(path_base)
plt.figure()
gap = evals[0][1:] - evals[0][:-1]
plt.plot(gap.detach().cpu().numpy(), marker='.', lw=0)
# plt.figure()
gap = evals[1][1:] - evals[1][:-1]
plt.plot(gap.detach().cpu().numpy(), marker='x', lw=0)
plt.grid()

plt.figure()
plt.plot(evals[0].detach().cpu().numpy(), marker='.', lw=0)
plt.plot(evals[1].detach().cpu().numpy(), marker='.', lw=0)
exit()
# %%
# check autovalori buffer ..
# %%
# plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
fig, ax = plt.subplots(1, 2, figsize=(6, 2.2))

for i, nid in enumerate(tqdm(all_data)):
    proj, labe = all_data[nid]['proj'], all_data[nid]['labe']
    proj_t = torch.tensor(TSNE(n_components=2).fit_transform(proj.numpy()))
    subsample = sample_idx
    proj_t = proj_t[subsample]
    labe_t = labe[subsample]
    for c in labe_t.unique():
        ax[i].scatter(proj_t[labe_t == c, 0], proj_t[labe_t == c, 1], label=c.item(), s=0.7)
    ax[i].set_title(nid)
# ax[-1].legend()


# %%


exit()
# TODO: compute the eigengap and color points by eigenvector
# compute eigengap
import matplotlib.pyplot as plt
from utils.spectral_analysis import laplacian_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
fig, ax = plt.subplots(2, 10, figsize=(30, 4.4))
rounds = 1
for nid in tqdm([1,2]):
    projs, targets = all_data[nid]['proj'], all_data[nid]['labe']
    egaps, evals = [], []
    for r in range(rounds):
        subsamp = torch.randperm(projs.shape[0])[:1000]
        projs, targets = projs[subsamp], targets[subsamp]
        _, evalues, evectors, _, _ = laplacian_analysis(projs, knn=10, norm_lap=True, n_pairs=20)
        # compute eigengap
        egap = evalues[1:] - evalues[:-1]
        evectors = evectors.detach()
        egaps.append(egap)
        evals.append(evalues)
    evals = torch.stack(evals).mean(0).detach().numpy()
    egaps = torch.stack(egaps).mean(0).detach().numpy()
    # plt.figure()
    # plt.title(id_task)
    # plt.plot(evals, marker='o')
    # plt.plot(egaps, marker='o')
    proj_t = torch.tensor(TSNE(n_components=2).fit_transform(projs.numpy()))
    labe_t = targets.detach()
    for c in labe_t.unique():
        ax[nid-1][0].scatter(proj_t[labe_t == c, 0], proj_t[labe_t == c, 1], label=c.item(), s=0.7)
    for iii, e in enumerate(range(1, 10)):
        ax[nid-1][iii+1].scatter(proj_t[:, 0], proj_t[:, 1], c=evectors[:, e], s=0.7)
        ax[nid-1][iii+1].set_title('eigenvector {} - gap {:.3f}'.format(e, egaps[e]))
    plt.tight_layout()
# %%
# compute k-means score
from sklearn.cluster import KMeans

for nid in tqdm(all_data):
    k = (nid + 1) * 2
    feats, targets = all_data[nid]['proj'], all_data[nid]['labe']
    kmeans = torch.tensor(KMeans(n_clusters=k, random_state=0).fit_predict(feats.numpy()))
    ents = []
    for i in kmeans.unique():
        _ , conf = targets[kmeans == i].unique(return_counts=True)
        # compute entropy
        probs = conf / conf.sum()
        entropy = -probs.mul(probs.log()).sum()
        ents.append(entropy.item())
        # print(f'cluster {i} -> {entropy}')
    kmscore = np.mean(ents)
    all_data[nid]['kmscore'] = kmscore
    print(f'Task {nid} -> {kmscore}')
# %%
# plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
fig, ax = plt.subplots(1, 5, figsize=(15, 2.2))

for nid in tqdm(all_data):
    proj, labe = all_data[nid]['proj'], all_data[nid]['labe']
    proj_t = torch.tensor(TSNE(n_components=2).fit_transform(proj.numpy()))
    subsample = torch.rand(proj_t.shape[0]) < 0.1
    proj_t = proj_t[subsample]
    labe_t = labe[subsample]
    for c in labe_t.unique():
        ax[nid].scatter(proj_t[labe_t == c, 0], proj_t[labe_t == c, 1], label=c.item(), s=0.7)

ax[-1].legend()
plt.savefig(path_base + 'tsne.png')

# %%
from sklearn.neighbors import KNeighborsClassifier

for nid in tqdm(all_data):
    proj, labe = all_data[nid]['proj'], all_data[nid]['labe']
    proj_train, labe_train = all_data[nid]['proj_train'], all_data[nid]['labe_train']
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(proj_train.numpy(), labe_train.numpy())
    pred = knn.predict(proj.numpy())
    knscore = (pred == labe.numpy()).mean()
    all_data[nid]['knscore'] = knscore
    print(f'Task {nid} -> {knscore}')
# %%

dump = [(x['kmscore'], x['knscore'], x['acc']) for x in all_data.values()]
with open(path_base + 'report.txt', 'w') as f:
    f.write('\n'.join(['{} {} {}'.format(x[0], x[1], x[2]) for x in dump]))