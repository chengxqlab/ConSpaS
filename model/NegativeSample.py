import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


class NegativeSample():
    def __init__(self,  knn_neighbor, args):

        self.knn_neighbor = knn_neighbor #sparse tensor,shape=[n_spot,n_spot]
        self.device = args.device

        self.negapool = None #negative sampling pool
        self.negasample = None  # index of negative samples ; sparse tensor[n_spot,n_sample]

        self.pool_percent = args.pool_percent  
        self.sample_percent = args.sample_percent 

        self.topkdevice = args.topkdevice


    def fit(self,similarity,seed,epoch):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


        x = self.knn_neighbor.coalesce().indices()[0]
        y = self.knn_neighbor.coalesce().indices()[1]
        similarity[x, y] = 1.0


        self.pool_size = int(similarity.shape[0] * self.pool_percent) #the size of negative sampling pool
        self.n_sample = self.pool_size * self.sample_percent  #the number of negative samples


        _,self.negapool = similarity.topk(k=self.pool_size,largest=False,sorted=False)#construct the negative sampling pool

        #sampling in negative samplingn pool
        sample_ind = torch.tensor(random.sample(range(int(self.negapool.shape[1])),int(self.n_sample))).to(self.topkdevice)
        negasample = torch.index_select(self.negapool.to(self.topkdevice),1,sample_ind)
        self.negasample = self.create_sparse(negasample.cpu())#convert negasample to sparse tensor
        self.n_nega = sample_ind.shape[0]


    def create_sparse(self, I):

        similar = I.reshape(-1)
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor(np.vstack((index,np.array(similar))))
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)).cpu(), [I.shape[0], I.shape[0]])

        return result


