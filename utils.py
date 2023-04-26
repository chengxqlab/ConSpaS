import copy
import faiss
import scipy
import pickle
import argparse
import scipy.stats
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

import sklearn.neighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import adjusted_rand_score,calinski_harabasz_score
from sklearn.cluster import KMeans

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages

from model.BuildPositivePair import BuildPositivePair
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dim", type=int, default=3000, help='ST data gene dimension')
    parser.add_argument("--num_hidden", type=int, default=512, help="hidden layer dimention")
    parser.add_argument('--out_dim', type=int, default=30, help='latent feature dimension')


    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.0004, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--knn", default=25, help="when searching positive sample,use k neariest spot as candidates")
    parser.add_argument("--num_centroids", type=int, default=20, help="the cluster number in kmeans when building positive sample")
    parser.add_argument("--clus_num_iters", type=int, default=20, help='max iteration in kmeans when building positive sample')
    
    parser.add_argument("--pool_percent", type=float, default=0.95, help='pool percent when sampling negative samples')
    parser.add_argument("--sample_percent", type=float, default=0.1, help='sampling percent in the sample pool')

    parser.add_argument("--alpha", type=int, default=0.2, help='the weight of contrastive loss')
    parser.add_argument("--temperature", type=float, default=0.5,help='the temperature parameter in the contrastive loss')

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--topkdevice", type=str, default='cuda:0')
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--save_loss", type=bool, default=True, help='wheather to save the loss')
    parser.add_argument("--n_cluster", type=int, default=0, help='prior cluster number, 0 means the number of clusters is not available')

    return parser.parse_known_args()


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def contrastive_loss(z, positive_emb, mask_nega, n_nega,device, temperature=0.5):

    device = device
    #embedding L2 normalization
    emb = F.normalize(z, dim=-1, p=2)
    similarity = torch.matmul(emb, torch.transpose(emb, 1, 0).detach())  # cosine similarity matrix
    e_sim = torch.exp((similarity / temperature))

    positive_emb_norm = F.normalize(positive_emb, dim=-1, p=2).to(device)
    positive_sim = torch.exp((positive_emb_norm * emb.unsqueeze(1)).sum(axis=-1) / temperature)

    x = mask_nega._indices()[0]
    y = mask_nega._indices()[1]
    N_each_spot = e_sim[x,y].reshape((-1,n_nega)).sum(dim=-1)

    N_each_spot = N_each_spot.unsqueeze(-1).repeat([1,positive_sim.shape[1]])

    contras = -torch.log(positive_sim / (positive_sim + N_each_spot))

    return torch.mean(contras)

def Transfer_pytorch_Data(adata):

    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def Cal_Spatial_Net(adata,rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index  
    coor.columns = ['imagerow', 'imagecol']  

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True) 
        KNN_list = []  
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list) 
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )) 
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], coor.shape[0]))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / coor.shape[0]))

    adata.uns['Spatial_Net'] = Spatial_Net 
    adata.uns['adj'] = KNN_df.iloc[:, 0:2]

    
#calculate align and uniform (L_au) metric
def cal_align_uniform(files, ratio):
    args, _ = parse_args()
    args.device = 'cuda:0'

    aligns_temp = []  
    uniforms_temp = []  

    for adata in files:
        
        emb = adata.obsm['Model_rep']

        
        positive = BuildPositivePair(args)
        positive_emb, _, _ = positive.fit(torch.Tensor(emb).to(args.device), args.knn)
        positive_emb = positive_emb.squeeze().cpu().numpy()
        
        emb_norm = emb / (np.sqrt((emb ** 2).sum(axis=1))[:, None])
        positive_emb_norm = positive_emb / (np.sqrt((positive_emb ** 2).sum(axis=1))[:, None])
        
        align = (((emb_norm - positive_emb_norm) ** 2).sum(axis=1)).mean()
        aligns_temp.append(-align)

        
        uniform = torch.pdist(torch.tensor(emb_norm), p=2).pow(2).mul(-2).exp().mean().log().item()
        uniforms_temp.append(-uniform)

    
    ratio = ratio
    au = ratio * np.array(aligns_temp) + np.array(uniforms_temp) 

    return au

def cal_distance_mat(adata):
    x_coor = np.array(adata.obsm['spatial'])[:, 0].reshape((-1, 1))
    y_coor = np.array(adata.obsm['spatial'])[:, 1].reshape((-1, 1))

    x_coor0 = np.repeat(x_coor, x_coor.shape[0], axis=1)
    y_coor0 = np.repeat(y_coor, y_coor.shape[0], axis=1)

    x_coor1 = np.repeat(x_coor.T, x_coor.shape[0], axis=0)
    y_coor1 = np.repeat(y_coor.T, y_coor.shape[0], axis=0)

    spa_dis = np.sqrt(((x_coor0 - x_coor1) ** 2) + ((y_coor0 - y_coor1) ** 2))
    return spa_dis

#calculate corr_spa_emb
def spa_emb_corr(adata, type='pearson'):
    
    print('Calculate corr_spa_emb...')
    #run pca
    sc.tl.pca(adata, n_comps=1500, return_info= True,zero_center=True)
    variance = 0
    for ind, var in enumerate(adata.uns['pca']['variance_ratio']):
        variance += var

        if variance >= 0.8:
            select_pc = ind + 1
            break
    sc.tl.pca(adata, return_info= True,n_comps=select_pc)

    X_pca = adata.obsm['X_pca']
    ##L2 normalization
    X_pca_norm = X_pca / (np.sqrt((X_pca ** 2).sum(axis=1)).reshape((-1, 1)))

    initial_corrmat = np.matmul(X_pca_norm, X_pca_norm.T) 
    spa_dis = cal_distance_mat(adata)

    corr_vector = initial_corrmat.reshape((1, -1))
    spa_vector = spa_dis.reshape((1, -1))

    if type == 'pearson':
        corr = scipy.stats.pearsonr(spa_vector.squeeze(), corr_vector.squeeze())[0]
    if type == 'spearman':
        corr = scipy.stats.spearmanr(spa_vector.squeeze(), corr_vector.squeeze())[0]
    print('Calculating correlation done.')
    return corr



#select temperature by L_au and silhouette_score
def select_temp(adatas, ratio=40, topk = 6, use_key = None):
    assert use_key is not None, 'You should run cluster algorithms first!'
    au = cal_align_uniform(adatas, ratio=ratio)
    cand_temps_ind = np.argsort(au)[10-topk:] 

    sics = []
    for cand_ind in cand_temps_ind:
        adata = adatas[cand_ind]
        y_pred = adata.obs[use_key]
        sics.append(silhouette_score(adata.obsm['rep_norm'], y_pred))
    best_temp = (cand_temps_ind[np.argmax(sics)] + 1) / 10 
    return best_temp

#calculate LISI metric
def cal_LISI(adata,pred_key,norm=False, ratio=None, n_cluster=None):

    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    lisis = []

    
    spatial = pd.DataFrame(adata.obsm['spatial'])
    lisi = LISI.compute_lisi(spatial, adata.obs,label_colnames = pred_key, perplexity=10)
    lisi_py = pandas2ri.rpy2py(lisi)
    lisis.append(lisi_py.values)

    return np.array(lisis)

