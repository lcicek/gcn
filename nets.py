import torch
from torch.nn import Sigmoid, LazyLinear
from torch_geometric.nn import GCNConv, global_add_pool

from parameters import P_THRESHOLD
from utility import label, labelDistance

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(136, 68)
        self.conv2 = GCNConv(68, 34)
        self.conv3 = GCNConv(34, 17)
        self.lin = LazyLinear(3)
        self.lin2 = LazyLinear(1)
        self.pred = Sigmoid()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)

        y = self.lin2(x)
        y = global_add_pool(y, batch).flatten()
        y = self.pred(y)

        return y
    
    def evaluate(self, x, edge_index, batch):
        pred = self.forward(x=x, edge_index=edge_index, batch=batch)
        
        label_dist = labelDistance(pred)
        labels = label(pred)

        return labels, label_dist
