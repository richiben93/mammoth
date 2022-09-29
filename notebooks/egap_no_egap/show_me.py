# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os

stds, kms, OO, bbs, bbrs = {}, {}, {}, {}, {}
for dir in os.listdir('cps'):
    if os.path.isdir(os.path.join('cps', dir)) and 'responses.pkl' in os.listdir(os.path.join('cps', dir)):
        try:
            model, buffer, reg, km, std = pickle.load(open(os.path.join('cps', dir, 'responses.pkl'), 'rb'))
            model, buffer, reg, bb  = pickle.load(open(os.path.join('cps', dir, 'rebuf.pkl'), 'rb'))
            model, buffer, reg, bbr  = pickle.load(open(os.path.join('cps', dir, 'resubtest.pkl'), 'rb'))
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
        bbrs[(model, buffer, reg)].append(bbr)


# plt.figure(figsize=(10, 10))
# for b in [500, 2000]:
#     for m in ['er_ace_egap']:
#         for r in ['none', 'egap']:
#             for i in range(len(bbs[(m, b, r)])):
#                 plt.scatter(range(10), np.stack(bbs[(model, buffer, reg)][i]), label=f'{m}-{b}-{r}')
#             plt.plot(np.array(bbs[(m, b, r)][1:]), ('-' if r == 'egap' else ':') +
#             ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(b == 500)))

rbbs = {}
for k in stds:
    OO[k] = len(stds[k])
    stds[k] = np.stack(stds[k]).mean(0)
    kms[k] = np.stack(kms[k]).mean(0)
    # bbvar[k] = np.meadinp.stack(bbs[k]).std(0)
    rbbs[k] = np.median(np.stack(bbs[k]), axis=0)
    bbrs[k] = np.median(np.stack(bbrs[k]), axis=0)


# buffy = 2000#2000

# plt.figure(figsize=(10, 6))
# for b in [buffy]:
#     for m in ['derpp_egap', 'er_ace_egap']:
#         for r in ['none', 'egap']:
#             plt.plot(kms[(m, b, r)][1:], ('-' if r == 'egap' else ':') +
#             ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(m == 'er_ace_egap')))
# plt.legend()
# plt.title('KMS')

# plt.figure(figsize=(10, 10))
# for b in [buffy]:
#     for m in ['derpp_egap', 'er_ace_egap']:
#         for r in ['none', 'egap']:
#             plt.plot(stds[(m, b, r)][1:], ('-' if r == 'egap' else ':') +
#             ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(m == 'er_ace_egap')))
# plt.legend()
# plt.title('STD')

mods = ['er_ace_egap', 'icarl_egap']


# plt.figure(figsize=(10, 10))
# for b in [500]:
#     for m in mods:
#         for r in ['none', 'egap']:
#             plt.plot(range(9), np.array(rbbs[(m, b, r)][1:]), 
#             ('-' if r == 'egap' else ':') +
#             ('*' if b == 2000 else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(b == 500)))

plt.figure(figsize=(7, 7))
for b in [500]:
    for m in mods:
        for r in ['none', 'egap']:
            plt.plot(range(9), np.array(rbbs[(m, b, r)][1:]), 
            '-' +
            ('*' if m == 'er_ace_egap' else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(r == 'egap')))
plt.grid()
plt.legend()
plt.title('bbs-buffer')

plt.figure(figsize=(7, 7))
for b in [500]:
    for m in mods:
        for r in ['none', 'egap']:
            plt.plot(range(9), np.array(bbrs[(m, b, r)][1:]), 
            '-' +
            ('*' if m == 'er_ace_egap' else 'o'), label=(m,b,r,f'[{OO[(m,b,r)]}]'), color='C' + str(int(r == 'egap')))
plt.grid()
plt.legend()
plt.title('bbs-test set')

# for b in [500, 2000]:
#     for m in [mod]:
#         for r in ['none', 'egap']:
#             for i in range(len(bbs[(m, b, r)])):
#                 plt.scatter(range(9), np.stack(bbs[(m, b, r)][i])[1:], marker='x' if r == 'egap' else '.', color='C' + str(int(b == 500)))

        
# %%
