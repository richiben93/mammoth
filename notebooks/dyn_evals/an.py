# %%
import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
os.chdir(os.path.dirname(__file__))

paths = {}
for d in os.listdir("eivals"):
    args = open(f"eivals/{d}/args.txt").read()
    lr = float(args.split("lr=")[1].split(",")[0])
    replay_weight = float(args.split("replay_weight=")[1].split(",")[0])
    if (lr, replay_weight) not in paths:
        paths[(lr, replay_weight)] = []
    paths[(lr, replay_weight)].append(f"eivals/{d}")


for (lr, replay_weight) in tqdm(paths):
    all_ev = {}
    
    # load data
    for path in paths[(lr, replay_weight)]:
        pickles = [x for x in os.listdir(path) if x.endswith(".pkl")]
        for f in sorted(pickles, key=lambda x: int(x.split(".")[0])):
            step = f.split('.')[0]
            if step not in all_ev:
                all_ev[step] = []
            ev = pickle.load(open(f"{path}/{f}", "rb"))
            all_ev[step].append(ev)

    # plot
    dp = f'plots/{lr}_{replay_weight}/'
    os.makedirs(dp, exist_ok=True)
    for s in all_ev:
        # to_plot = torch.stack(all_ev[s])[:, 1:] - torch.stack(all_ev[s])[:, :-1]
        to_plot = torch.stack(all_ev[s])
        to_plot = to_plot.mean(0)
        plt.figure()
        # plt.bar(np.arange(1, len(to_plot)+1), to_plot.cpu().numpy())
        plt.plot(np.arange(0, len(to_plot)), to_plot.cpu().numpy(), lw=0, marker='o')
        # plt.ylim(0, 1)
        plt.title(f'lr: {lr:.2f} - rw: {replay_weight:.2f} - {s}')
        plt.savefig(f'{dp}{s}.png')

# %%
