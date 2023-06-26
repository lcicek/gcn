import torch
from torch.nn import Sigmoid, LazyLinear, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool

from utility import label, labelDistance

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = GCNConv(136, 50, improved=True)
        self.conv2 = GCNConv(50, 50, improved=True)
        self.lin = LazyLinear(1)
        self.pred = Sigmoid()
        self.drop = Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch) # readout layer
        x = self.drop(x) # regularization layer
        x = self.lin(x)
        x = self.pred(x).flatten()

        return x
    
    def evaluate(self, x, edge_index, batch):
        pred = self.forward(x=x, edge_index=edge_index, batch=batch)
        
        label_dist = labelDistance(pred)
        labels = label(pred)

        return labels, label_dist
