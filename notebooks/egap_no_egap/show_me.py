# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os

stds, kms, OO, bbs = {}, {}, {}, {}
for dir in os.listdir('cps'):
    if os.path.isdir(os.path.join('cps', dir)) and 'responses.pkl' in os.listdir(os.path.join('cps', dir)):
        try:
            model, buffer, reg, km, std = pickle.load(open(os.path.join('cps', dir, 'responses.pkl'), 'rb'))
            model, buffer, reg, bb  = pickle.load(open(os.path.join('cps', dir, 'rebuf.pkl'), 'rb'))
        except:
            continue
        if (model, buffer, reg) not in stds:
            stds[(model, buffer, reg)] = []
            kms[(model, buffer, reg)] = []
        stds[(model, buffer, reg)].append(std)
        kms[(model, buffer, reg)].append(km)
        bbs[(model, buffer, reg)] = bb


for k in stds:
    OO[k] = len(stds[k])
    stds[k] = np.stack(stds[k]).mean(0)
    kms[k] = np.stack(kms[k]).mean(0)

buffy = 500#2000

plt.figure(figsize=(10, 6))
for b in [buffy]:
    for m in ['derpp_egap', 'er_ace_egap']:
        for r in ['none', 'egap']:
            plt.plot(kms[(m, b, r)][1:], ('-' if r == 'egap' else ':') +
            ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(m == 'er_ace_egap')))
plt.legend()
plt.title('KMS')

plt.figure(figsize=(10, 10))
for b in [buffy]:
    for m in ['derpp_egap', 'er_ace_egap']:
        for r in ['none', 'egap']:
            plt.plot(stds[(m, b, r)][1:], ('-' if r == 'egap' else ':') +
            ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(m == 'er_ace_egap')))
plt.legend()
plt.title('STD')

plt.figure(figsize=(10, 10))
for b in [buffy]:
    for m in ['derpp_egap', 'er_ace_egap']:
        for r in ['none', 'egap']:
            plt.plot(bbs[(m, b, r)][1:], ('-' if r == 'egap' else ':') +
            ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(m == 'er_ace_egap')))
plt.legend()
plt.title('bbs')
        
# %%
