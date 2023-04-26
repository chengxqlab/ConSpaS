import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class GCN(nn.Module):

    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):

        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x
