import torch
from torch.nn import Sigmoid, LazyLinear
from torch_geometric.nn import GCNConv, global_add_pool

from parameters import P_THRESHOLD
from utility import label, labelDistance

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = GCNConv(136, 136, improved=True)
        self.conv2 = GCNConv(136, 136, improved=True)
        self.lin = LazyLinear(1)
        self.pred = Sigmoid()

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        x = global_add_pool(x, batch)
        x = self.pred(x).flatten()

        return x
    
    def evaluate(self, x, edge_index, batch):
        pred = self.forward(x=x, edge_index=edge_index, batch=batch)
        
        label_dist = labelDistance(pred)
        labels = label(pred)

        return labels, label_dist
