import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import faiss
import random
class BuildPositivePair():
    def __init__(self, args):
        self.device = args.device
        self.num_centroids = args.num_centroids
        self.clus_num_iters = args.clus_num_iters
        self.args = args

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor(np.vstack((index, np.array(similar)))).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result
    
    #calculate kmeans centroids
    def cal_kmeans_centroid(self, emb, I_kmenas):
        centroids = []
        for i in range(self.args.num_centroids):
            ind_tmp = np.where(I_kmenas == i)[0] 
            cen_tmp = emb[ind_tmp,:].mean(axis=0) 
            centroids.append(cen_tmp)
        return np.array(centroids)

    #calculate positive sample embedding
    def cal_positive_emb(self, emb, I_knn, I_kmeans, centroids):
        I_knn = I_knn.cpu()
        spot_centroid = np.apply_along_axis(lambda x: centroids[x], 0, I_kmeans).squeeze() #get kmeans centroids

        spot_candi_emb = emb[I_knn.reshape(-1,1).squeeze().numpy(),:].reshape((-1,I_knn.shape[1],emb.shape[1])) #get candidate positive samples emb 
        expand_centroid = np.expand_dims(spot_centroid,axis=1).repeat(I_knn.shape[1],axis=1) 
        euc_distance = np.exp(np.sqrt(np.power((expand_centroid - spot_candi_emb), 2).sum(axis=2)))
        prob = nn.Softmax(dim=1)(1 / torch.tensor(euc_distance)).unsqueeze(1)

        #weighted sum to build positive sample
        positive_emb = torch.matmul(prob, torch.tensor(spot_candi_emb))
        return positive_emb

    def fit(self,  z, top_k):
        #seed everything
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        cudnn.benchmark = False
        cudnn.deterministic = True

        z = F.normalize(z, dim=-1, p=2) #node embedding L2 normalized
        n_data, d = z.shape

        similarity = torch.matmul(z, torch.transpose(z, 1, 0).detach()) #calculate cosine similarity matrix

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)  # knn-->construct knn candidate positive samples

        knn_neighbor = self.create_sparse(I_knn) #convert knn to sparse matrix

        #setting hyper-parameters in kmeans
        ncentroids = self.num_centroids
        niter = self.clus_num_iters


        emb = z.cpu().numpy()

        all_positive_emb = []
        
        ##kmeans clustering
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=0)
        kmeans.train(emb)

        _, I_kmeans = kmeans.index.search(emb, 1)
        centroids = self.cal_kmeans_centroid(emb, I_kmeans) #nparray shape:[n_centroids, embsize]
        positive_emb = self.cal_positive_emb(emb, I_knn, I_kmeans, centroids)

        if len(all_positive_emb) == 0:
            all_positive_emb = positive_emb
        else:
            all_positive_emb = np.concatenate((all_positive_emb,positive_emb),axis=1)
        return all_positive_emb, similarity, knn_neighbor



