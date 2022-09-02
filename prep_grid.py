import os
import random
import itertools
import numpy as np

# LIP MEMORY

# --dataset seq-cifar10 --model er_ace_pre_replay --lr 0.1 --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --pretrain_epochs 3 --spectral_buffer_size 1000 --buffer_size 500 --con_weight 0 --pre_minibatch 32 --replay_mode fmeval-0101 --replay_weight 0.1
# --dataset seq-cifar10 --model er_ace_pre_replay --lr 0.1 --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --pretrain_epochs 3 --spectral_buffer_size 1000 --buffer_size 500 --con_weight 0 --pre_minibatch 32 --replay_mode fmeval-0101 --replay_weight 0.01
# --dataset seq-cifar10 --model er_ace_pre_replay --lr 0.1 --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --pretrain_epochs 3 --spectral_buffer_size 1000 --buffer_size 500 --con_weight 0 --pre_minibatch 32 --replay_mode fmeval-0101 --replay_weight 0.001
# --dataset seq-cifar10 --model er_ace_pre_replay --lr 0.1 --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --pretrain_epochs 3 --spectral_buffer_size 1000 --buffer_size 500 --con_weight 0 --pre_minibatch 32 --replay_mode fmeval-0101 --replay_weight 0.0001
# --dataset seq-cifar10 --model er_ace_pre_replay --lr 0.1 --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --pretrain_epochs 3 --spectral_buffer_size 1000 --buffer_size 500 --con_weight 0 --pre_minibatch 32 --replay_mode fmeval-0101 --replay_weight 0.00001

all_combos = [
    {
        'name': 'norepl_control',
        'combos':{
        'replay_mode': ['x'],
        'replay_weight': [0],
        'lr': [0.1, 0.01, 0.3]
    },
    'begin': '--dataset seq-cifar10 --model er_ace_pre_replay_nopre --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --spectral_buffer_size 500 --buffer_size 500 --con_weight 0 --pre_minibatch 32'
    },
    {
    'name': 'theboys',
    'combos':{
        'replay_mode': ['x', 'lats', 'dists','graph', 'laplacian','evec', 'fmap', 'eval', 'fmeval-0101','fmeval-0110', 'fmeval-1001', 'fmeval-1010'],
        'replay_weight': [0.1,0.01,0.001,0.0001,0.00001],
        'lr': [0.1, 0.01, 0.3]
    },
    'begin': '--dataset seq-cifar10 --model er_ace_pre_replay_nopre --wandb --minibatch_size 32 --batch_size 32 --n_epochs 10 --spectral_buffer_size 500 --buffer_size 500 --con_weight 0 --pre_minibatch 32'
    }

    
]

redund = 1
configs = []

for item in all_combos:
    filenam, combos = item['name'], item['combos']
    configs = list(itertools.product(*combos.values()))
    configs *= redund

    print(filenam, len(configs), 'items')

    # filenam = "merged"
    folder = 'data/jobs/'

    begin = item['begin']


    chances = [10, 0, 0, 0] #srv, boku, boten
    chances = np.array(chances) / np.sum(chances)

    print(f'{folder}list_{filenam}_full.txt')
    with open(f'{folder}list_{filenam}_full.txt', 'w') as f:
        for c in configs:
            f.write(begin)
            for k, v in zip(combos.keys(), c):
                if v is None:
                    continue
                if type(k) == tuple:
                    for i in range(len(k)):
                        f.write(f' --{k[i]}={v[i]}')
                else:
                    f.write(f' --{k}={v}')
            f.write('\n')
    try:
        os.remove(f'{folder}list_{filenam}_srv.txt')
    except:
        ...
    try:
        os.remove(f'{folder}list_{filenam}_bot.txt')    
    except:
        ...
    try:
        os.remove(f'{folder}list_{filenam}_bok.txt')
    except:
        ...
    try:
        os.remove(f'{folder}list_{filenam}_joj.txt')
    except:
        ...
    with open(f'{folder}list_{filenam}_full.txt', 'r') as f:
        lines = f.readlines()
    counts, countb, countt, countj = 0, 0, 0, 0

    for l in lines:
        r = random.random()
        if r < chances[0]:
            with open(f'{folder}list_{filenam}_srv.txt', 'a') as f:
                f.write(l)
            counts += 1
        elif r < chances[:2].sum():
            with open(f'{folder}list_{filenam}_bok.txt', 'a') as f:
                f.write(l)
            countb += 1
        elif r < chances[:3].sum():
            with open(f'{folder}list_{filenam}_bot.txt', 'a') as f:
                f.write(l)
            countt +=1
        else:
            with open(f'{folder}list_{filenam}_joj.txt', 'a') as f:
                f.write(l)
            countj +=1

    print("Total:",len(lines),"byfile", counts, countb, countt, countj)
    print('')
