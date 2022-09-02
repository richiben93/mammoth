# %%
from re import sub
import torch
import numpy as np
import os
from tqdm import tqdm
from argparse import Namespace

conf_path = os.getcwd()
while not 'mammoth' in os.path.basename(conf_path):
    os.chdir("..")
    conf_path = os.getcwd()
print('Config path: {}'.format(conf_path))

# prepare dataset
from datasets.seq_cifar10 import SequentialCIFAR10
args = Namespace(
            batch_size=32,
            dataset='seq-cifar100',
            validation=False,
)
dataset = SequentialCIFAR10(args)
data_loaders = [dataset.get_data_loaders()[0] for _ in range(dataset.N_TASKS)]

from backbone.ResNet18 import resnet18
from utils.conf import get_device

# load the model
all_data = {}

path_base = 'notebooks/paths/gap2/'

for id_task, cp in enumerate(sorted([x for x in os.listdir(path_base) if x.endswith('.pt')])):
    device = get_device()
    net = resnet18(10)
    sd = torch.load(path_base + cp)
    net.load_state_dict(sd)
    net.to(device)
    net.eval()
    
    all_data[id_task] = {}
    all_data[id_task]['net'] = net
    all_data[id_task]['cp'] = cp

# %%
# compute projections
for id_task in tqdm(all_data):
    net = all_data[id_task]['net']
    with torch.no_grad():
        proj, labe, pred = [], [], []
        for dl in dataset.test_loaders[:id_task+1]:
            for x, y in dl:
                x = x.to(device)
                y = y.to(device)
                proj.append(net.features(x).cpu())
                pred.append(net(x).cpu())
                labe.append(y.cpu())
        proj = torch.cat(proj)
        labe = torch.cat(labe)
        pred = torch.cat(pred)

        all_data[id_task]['proj'] = proj
        all_data[id_task]['labe'] = labe
        all_data[id_task]['pred'] = pred

        proj, labe = [], []
        for dl in data_loaders[:id_task+1]:
            for x, y, _ in dl:
                x = x.to(device)
                y = y.to(device)
                proj.append(net.features(x).cpu())
                labe.append(y.cpu())
        proj = torch.cat(proj)
        labe = torch.cat(labe)

        all_data[id_task]['proj_train'] = proj
        all_data[id_task]['labe_train'] = labe

# %%
# accuracy
for id_task in tqdm(all_data):
    preds, targets = all_data[id_task]['pred'], all_data[id_task]['labe']
    acc = torch.sum(torch.argmax(preds, dim=1) == targets).item() / preds.shape[0]
    all_data[id_task]['acc'] = acc

# %%
# TODO: compute the eigengap and color points by eigenvector
# compute eigengap
import matplotlib.pyplot as plt
from utils.spectral_analysis import laplacian_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
fig, ax = plt.subplots(2, 10, figsize=(30, 4.4))
rounds = 1
for id_task in tqdm([1,2]):
    projs, targets = all_data[id_task]['proj'], all_data[id_task]['labe']
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
        ax[id_task-1][0].scatter(proj_t[labe_t == c, 0], proj_t[labe_t == c, 1], label=c.item(), s=0.7)
    for iii, e in enumerate(range(1, 10)):
        ax[id_task-1][iii+1].scatter(proj_t[:, 0], proj_t[:, 1], c=evectors[:, e], s=0.7)
        ax[id_task-1][iii+1].set_title('eigenvector {} - gap {:.3f}'.format(e, egaps[e]))
    plt.tight_layout()
# %%
# compute k-means score
from sklearn.cluster import KMeans

for id_task in tqdm(all_data):
    k = (id_task + 1) * 2
    feats, targets = all_data[id_task]['proj'], all_data[id_task]['labe']
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
    all_data[id_task]['kmscore'] = kmscore
    print(f'Task {id_task} -> {kmscore}')
# %%
# plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
fig, ax = plt.subplots(1, 5, figsize=(15, 2.2))

for id_task in tqdm(all_data):
    proj, labe = all_data[id_task]['proj'], all_data[id_task]['labe']
    proj_t = torch.tensor(TSNE(n_components=2).fit_transform(proj.numpy()))
    subsample = torch.rand(proj_t.shape[0]) < 0.1
    proj_t = proj_t[subsample]
    labe_t = labe[subsample]
    for c in labe_t.unique():
        ax[id_task].scatter(proj_t[labe_t == c, 0], proj_t[labe_t == c, 1], label=c.item(), s=0.7)

ax[-1].legend()
plt.savefig(path_base + 'tsne.png')

# %%
from sklearn.neighbors import KNeighborsClassifier

for id_task in tqdm(all_data):
    proj, labe = all_data[id_task]['proj'], all_data[id_task]['labe']
    proj_train, labe_train = all_data[id_task]['proj_train'], all_data[id_task]['labe_train']
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(proj_train.numpy(), labe_train.numpy())
    pred = knn.predict(proj.numpy())
    knscore = (pred == labe.numpy()).mean()
    all_data[id_task]['knscore'] = knscore
    print(f'Task {id_task} -> {knscore}')
# %%

dump = [(x['kmscore'], x['knscore'], x['acc']) for x in all_data.values()]
with open(path_base + 'report.txt', 'w') as f:
    f.write('\n'.join(['{} {} {}'.format(x[0], x[1], x[2]) for x in dump]))