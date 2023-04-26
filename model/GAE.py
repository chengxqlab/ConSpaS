import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

from model.GCN import GCN

class GAE(torch.nn.Module):
    def __init__(self, args):
        super(GAE, self).__init__()

        self.in_dim = args.in_dim
        self.num_hidden = args.num_hidden
        self.out_dim = args.out_dim

        self.conv1 = GCN(self.in_dim, self.num_hidden)
        self.conv2 = GCN(self.num_hidden, self.out_dim)

        self.conv3 = GCN(self.out_dim, self.num_hidden)
        self.conv4 = GCN(self.num_hidden, self.in_dim)
        
        #prejection head
        self.head1 = Parameter(torch.Tensor(self.out_dim,self.out_dim))
        torch.nn.init.xavier_uniform_(self.head1)




    def forward(self, features, edge_index):

        h1 = self.conv1(features, edge_index)
        h2 = self.conv2(h1, edge_index) 
        
        #weight share
        self.conv3.conv.lin.weight.data = self.conv2.conv.lin.weight.transpose(0,1)
        self.conv4.conv.lin.weight.data = self.conv1.conv.lin.weight.transpose(0,1)
        
        #prejection head
        z = torch.matmul(h2,self.head1)
        
        h3 = self.conv3(z, edge_index)
        h4 = self.conv4(h3, edge_index)

        return z, h2, h4 


