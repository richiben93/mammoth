# %%
import os
import matplotlib.pyplot as plt
import torch
import pickle

egap_exp = 'EraceEgapb2NC16K4-exSX7'
none_exp = 'EraceNone-P4JAv'#'EraceNone-37CZ2'
if True:
# if not os.path.exists('scatter_meta.pkl'):
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
    for dir in [none_exp]:#, egap_exp]:
        sm[dir] = {}
        all_data = pickle.load(open(os.path.join('cps', dir, 'bufeats.pkl'), 'rb'))
        labelle = torch.tensor([0, 2, 5, 8, 9, 11, 13, 14, 16, 18])
        for ppl in [300]:
            for i, steppe in enumerate([2,3,4,5]):
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

                bproj = TSNE(n_components=2, perplexity=ppl).fit_transform(bproj)#perplexity=20
                # bproj = SpectralEmbedding(n_components=2).fit_transform(bproj)
                sm[dir][steppe] = (bproj, by)
        
                plt.figure()
                plt.scatter(*sm[dir][steppe][0].T, c=sm[dir][steppe][1], s=5, cmap='tab10')
                plt.title(f'{dir} - {steppe}')
    # with open('scatter_meta.pkl', 'wb') as f:
    #     pickle.dump(sm, f)
    # print('Computed!')
# else:
#     sm = pickle.load(open('scatter_meta.pkl', 'rb'))
#     print("Loaded!")
# %%
