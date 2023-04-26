import copy
import pickle
import random
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score,calinski_harabasz_score,silhouette_score
from sklearn.cluster import KMeans

from model.GAE import GAE
from model.NegativeSample import NegativeSample
from model.BuildPositivePair import BuildPositivePair

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from torch_geometric.transforms import ToSparseTensor

from utils import contrastive_loss,Transfer_pytorch_Data

def train(adata, args):
    lr = args.lr
    knn = args.knn
    save_loss = args.save_loss
    device = args.device
    verbose = args.verbose
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    temp = args.temperature

    
    # seed_everything()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = GAE(args).to(device)

    data = data.to(device)
    data = ToSparseTensor()(data)  ##transfer data to sparse data which can ensure the reproducibility when seed fixed

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []
    print('Training start!')
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        z, h, out = model(data.x, data.adj_t)

        emb = z.clone()

        with torch.no_grad():
            # building positive sample--------------------------

            positive = BuildPositivePair(args)

            positive_emb, similarity, knn_neighbor = positive.fit(emb, knn)
            # sampling negative samples-------------------------

            nega_sampler = NegativeSample(knn_neighbor, args)

            nega_sampler.fit(similarity, seed, epoch)
        #--------------Loss of CL module--------------------

        loss_contra = args.alpha * contrastive_loss(z, positive_emb, nega_sampler.negasample, nega_sampler.n_nega,device, temperature=temp)

        #--------------Loss Function------------------------

        loss_recons = F.mse_loss(data.x, out)#reconstruction loss

        loss = loss_recons + loss_contra # loss of ConSpaS
        #--------------optimize-----------------------------
        ##
        loss_list.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) 
        optimizer.step()
        #----------------------------------------------------------------------------

        Model_rep = z.to('cpu').detach().numpy().copy()

        adata.obsm['Model_rep'] = Model_rep


    model.eval()
    z, h, out = model(data.x, data.adj_t)


    Model_rep = z.to('cpu').detach().numpy().copy()
    rep_norm = Model_rep / (np.sqrt((Model_rep ** 2).sum(axis=1))[:, None])
    adata.obsm['Model_rep'] = Model_rep  
    adata.obsm['rep_norm'] = rep_norm #L2 normalized emb
    adata.uns['args'] = args
    
    #clustering
    ##if the number of cluster is available, run kmeans
    if args.n_cluster:
        y_pred = KMeans(n_clusters=args.n_cluster, random_state=2022).fit_predict(rep_norm)
        adata.obs['kmeans'] = y_pred
        adata.obs['kmeans'] = adata.obs['kmeans'].astype('int') 
        adata.obs['kmeans'] = adata.obs['kmeans'].astype('category') 
    #run leiden
    else: 
        #select resolution by highest CH score
        chs = []
        resolutions = np.round(np.arange(0.1,1.05,0.05), 2)
        sc.pp.neighbors(adata, use_rep='rep_norm')
        for resolution in resolutions:
            sc.tl.leiden(adata, resolution=resolution,random_state=2022,key_added='leiden%.2f'%resolution)
            chs.append(calinski_harabasz_score(adata.obsm['rep_norm'], adata.obs['leiden%.2f'%resolution]))
        adata.uns['CH'] = np.array(chs)
        
    if save_loss:  
        adata.uns['loss'] = loss_list
    
    #reconstruct gene expression by decoder of GAE
    adata.obsm['out'] = out.cpu().detach().numpy()
    return adata


