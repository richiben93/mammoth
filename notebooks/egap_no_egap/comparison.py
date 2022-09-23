# %%
from re import sub
import torch
import numpy as np
import os
from tqdm import tqdm
from argparse import Namespace
import pickle
torch.set_grad_enabled(False)
conf_path = os.getcwd()
while not 'mammoth' in os.path.basename(conf_path):
    os.chdir("..")
    conf_path = os.getcwd()
print('Config path: {}'.format(conf_path))
from backbone.ResNet18 import resnet18
from utils.conf import get_device
device = get_device()
# prepare dataset
from datasets.seq_cifar100 import SequentialCIFAR100_10x10
args = Namespace(
            batch_size=64,
            dataset='seq-cifar100_10x10',
            validation=False,
)
dataset = SequentialCIFAR100_10x10(args)
data_loaders = [dataset.get_data_loaders()[0] for _ in range(dataset.N_TASKS)]

mymodel = "Derpp"#'Erace'
load = True

if not load:
    # load the model
    all_data = {}
    for model in [mymodel]:
        for reg in ['None', 'Egapb2']:
            for buf_size in [500, 2000]:
                all_data[(model, reg, buf_size)] = {}
                path = 'notebooks/egap_no_egap/cps/'
                folder = [x for x in os.listdir(path) if model+reg in x if ((buf_size == 2000) == ('2000' in x))][0]
                print(model, reg, buf_size, folder)
                path += folder + '/'
                for id_task in range(1, 11):
                    
                    net = resnet18(100)
                    sd = torch.load(path + f'task_{id_task}.pt', map_location='cpu')
                    net.load_state_dict(sd)
                    net.eval()
                    buf = pickle.load(open(path + f'task_{id_task}_buffer.pkl', 'rb'))

                    all_data[(model, reg, buf_size)][id_task] = {}
                    all_data[(model, reg, buf_size)][id_task]['net'] = net
                    all_data[(model, reg, buf_size)][id_task]['buf'] = buf

# %%
# compute accuracy
if not load:
    for id_task in tqdm(range(1, 11)):
        for buf_size in [500, 2000]:
            for model in [mymodel]:
                for reg in ['None', 'Egapb2']:
                    net = all_data[(model, reg, buf_size)][id_task]['net']
                    all_data[(model, reg, buf_size)][id_task]['projs'] = []
                    all_data[(model, reg, buf_size)][id_task]['preds'] = []
                    all_data[(model, reg, buf_size)][id_task]['labs'] = []
                    net.to(device)    

                    buf = all_data[(model, reg, buf_size)][id_task]['buf']
                    bufdata = buf.get_data(buf.buffer_size, transform=dataset.test_loaders[0].dataset.transform.transforms[1])
                    bx, by = bufdata[0], bufdata[1]
                    bx = bx.to(device)
                    by = by
                    bproj = net.features(bx).cpu()

                    for j, dl in enumerate(dataset.test_loaders[:id_task]):
                        corr, corrknn, tot = 0, 0, 0
                        for x, y in dl:
                            x = x.to(device)
                            y = y
                            pred = net(x).cpu()
                            proj = net.features(x).cpu()
                            corr += (pred.argmax(dim=1) == y).sum().item()
                            tot += len(y)
                            # bfilter = (by >= j*10) & (by < (j+1)*10)
                            corrknn += (by[(bproj.unsqueeze(0) - proj.unsqueeze(1)).norm(dim=2).argmin(dim=1)] == y).sum().item()

                            all_data[(model, reg, buf_size)][id_task]['projs'].append(proj)
                            all_data[(model, reg, buf_size)][id_task]['preds'].append(pred)
                            all_data[(model, reg, buf_size)][id_task]['labs'].append(y)
                            
                        all_data[(model, reg, buf_size)][id_task][f'acc_{j}'] = corr / tot
                        all_data[(model, reg, buf_size)][id_task][f'acc_knn_{j}'] = corrknn / tot
                net.to('cpu')

# %%
# save cache
if not load:
    with open(f"notebooks/egap_no_egap/tmp_{mymodel}.pkl", "wb") as f:
        pickle.dump(all_data, f)
else:
    with open(f"notebooks/egap_no_egap/tmp_{mymodel}.pkl", "rb") as f:
        all_data = pickle.load(f)
# %%
# Accuracy matrices
import matplotlib.pyplot as plt
import seaborn as sns

for buf_size in [500, 2000]:
    for model in [mymodel]:
        for reg in ['None', 'Egapb2']:
            am = np.ones((10, 11)) * float('nan')
            for i in range(1, 11):
                for j in range(i):
                    am[i-1, j] = all_data[(model, reg, buf_size)][i][f'acc_{j}'] * 100
            am[:, -1] = np.nanmean(am, axis=1)
            plt.figure()
            sns.heatmap(am, annot=True, fmt='.1f', vmin=0, vmax=100)
            plt.title(f'{model} {reg} {buf_size}')

for buf_size in [500, 2000]:
    for model in [mymodel]:
        for reg in ['None', 'Egapb2']:
            am = np.ones((10, 11)) * float('nan')
            for i in range(1, 11):
                for j in range(i):
                    am[i-1, j] = all_data[(model, reg, buf_size)][i][f'acc_knn_{j}'] * 100
            am[:, -1] = np.nanmean(am, axis=1)
            plt.figure()
            sns.heatmap(am, annot=True, fmt='.1f', vmin=0, vmax=100)
            plt.title(f'{model} {reg} {buf_size} - knn')

# %%
# test overfitting
for id_task in tqdm(range(1, 11)):
    for buf_size in [500, 2000]:
        for model in [mymodel]:
            for reg in ['None', 'Egapb2']:
                net = all_data[(model, reg, buf_size)][id_task]['net']
                net.to(device)    
                buf = all_data[(model, reg, buf_size)][id_task]['buf']
                data = buf.get_data(buf.buffer_size, transform=dataset.test_loaders[0].dataset.transform.transforms[1])
                x, y = data[0], data[1]
                x = x.to(device)
                y = y
                pred = net(x).cpu()
                corr = (pred.argmax(dim=1) == y).sum().item()
                tot = len(y)
                all_data[(model, reg, buf_size)][id_task]['acc_overfit'] = corr / tot
                net.to('cpu')

# %%
for buf_size in [500, 2000]:
    for model in [mymodel]:
        for reg in ['None', 'Egapb2']:
            ao = []
            for id_task in tqdm(range(1, 11)):
                ao.append(all_data[(model, reg, buf_size)][id_task]['acc_overfit'])
            plt.plot(ao, label=f'{model} {reg} {buf_size}', marker='o')
plt.legend()
plt.title('buffer overfitting')

plt.figure()
for buf_size in [500, 2000]:
    for model in [mymodel]:
        for reg in ['None', 'Egapb2']:
            ao = all_data[(model, reg, buf_size)][10]['acc_overfit']
            aa = np.mean([all_data[(model, reg, buf_size)][10][f'acc_{j}'] for j in range(10)])
            plt.scatter(ao, aa, label=f'{model} {reg} {buf_size}')
plt.legend()
plt.title('buffer overfitting vs accuracy')
plt.xlabel('buffer accuracy')
plt.ylabel('average accuracy')

# %%
# Kmeans
from sklearn.cluster import KMeans
model = mymodel
for buf_size in [500, 2000]:
    plt.figure()
    for o, reg in enumerate(['None', 'Egapb2']):
        kmscores = []
        for id_task in tqdm(range(1, 11)):
            projs = torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0)
            labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)

            kmeans = KMeans(n_clusters=len(labs.unique())).fit_predict(projs)
            ents = []
            for i in np.unique(kmeans):
                _ , conf = labs[kmeans == i].unique(return_counts=True)
                # compute entropy
                probs = conf / conf.sum()
                entropy = -probs.mul(probs.log()).sum()
                ents.append(entropy.item())
                # print(f'cluster {i} -> {entropy}')
            kmscore = np.mean(ents)
            kmscores.append(kmscore)
        plt.plot(kmscores, label=f'{model} {reg}', marker='o')
        plt.legend()

# %%
# varianza classi

def nize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

from sklearn.cluster import KMeans
for buf_size in [500, 2000]:
    plt.figure()
    model = mymodel
    for o, reg in enumerate(['None', 'Egapb2']):
        kmscores = []
        for id_task in tqdm(range(1, 11)):
            projs = nize(torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0))
            labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)
            varss = []
            for i in range(10 * id_task):
                varss.append(projs[labs == i].std(dim=0).mean().item())
            kmscore = np.mean(varss)
            kmscores.append(kmscore)
        plt.plot(kmscores, label=f'{model} {reg}', marker='o')
        plt.legend()
        plt.title(f'std all classes - {buf_size}')
    plt.figure()
    from sklearn.cluster import KMeans
    
    for o, reg in enumerate(['None', 'Egapb2']):
        kmscores = []
        for id_task in tqdm(range(1, 11)):
            projs = nize(torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0))
            labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)
            varss = []
            for i in range(10*(id_task-1)):
                varss.append(projs[labs == i].std(dim=0).mean().item())
            kmscore = np.mean(varss)
            kmscores.append(kmscore)
        plt.plot(kmscores, label=f'{model} {reg}', marker='o')
        plt.legend()
        plt.title(f'std past classes - {buf_size}')
    plt.figure()
    for o, reg in enumerate(['None', 'Egapb2']):
        kmscores = []
        for id_task in tqdm(range(1, 11)):
            projs = nize(torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0))
            labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)
            varss = []
            for i in range(10*(id_task-1), 10*id_task):
                varss.append(projs[labs == i].std(dim=0).mean().item())
            kmscore = np.mean(varss)
            kmscores.append(kmscore)
        plt.plot(kmscores, label=f'{model} {reg}', marker='o')
        plt.legend()
        plt.title(f'std cur classes - {buf_size}')

# %%
for buf_size in [500, 2000]:
    for model in [mymodel]:
        for reg in ['None', 'Egapb2']:
            am = np.ones((10, 11)) * float('nan')
            for id_task in range(1, 11):
                projs = torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'], dim=0)
                labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'], dim=0)
                for j in range(id_task):
                    am[id_task-1, j] = np.mean([projs[labs == l].std(dim=0).mean().item() for l in range(10*j, 10*(j+1))])
            am[:, -1] = np.nanmean(am, axis=1)
            plt.figure()
            sns.heatmap(am, annot=True, fmt='.1f' ,vmax=0.6, vmin=0)
            plt.title(f'{model} {reg} {buf_size}')

# %%
# Wandering points
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA as TSNE
model = 'Erace'
def rebase(array):
    array = array.clone()
    array -= array.min()
    eye = torch.eye(array.max()+1)
    reb = eye[array]
    return reb[:, reb.sum(0) != 0].argmax(axis=1)

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

for o, reg in enumerate(['None', 'Egapb2']):
    for h, id_task in tqdm(enumerate(range(7, 11))):
        axx = ax[o, h]
        projs = torch.cat(all_data[(model, reg, buf_size)][id_task]['projs'])
        labs = torch.cat(all_data[(model, reg, buf_size)][id_task]['labs'])
        preds = torch.cat(all_data[(model, reg, buf_size)][id_task]['preds'])
        filter = np.isin(labs, np.array([0, 1, 10, 11, 20, 21, 30, 31]))
        tsne = TSNE(n_components=2)#, perplexity=30, n_iter=1000)
        tsne_proj = tsne.fit_transform(projs[filter])
        key = (labs[filter] == preds[filter].argmax(1)).numpy()
        
        axx.scatter(tsne_proj[~key, 0], tsne_proj[~key, 1], c=rebase(labs[filter][~key]), marker='x', cmap='tab20', alpha=0.7)
        axx.scatter(tsne_proj[key, 0], tsne_proj[key, 1], c=rebase(labs[filter][key]), marker='o', cmap='tab20')
        axx.set_title(f'{model} {reg} task {id_task} labels')

    



# %%
exit()
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