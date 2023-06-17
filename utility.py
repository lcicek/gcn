import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree

from parameters import P_THRESHOLD

class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
    
def prepareNodes(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

def labelDistance(pred):
    one_pred = pred[pred > P_THRESHOLD]
    zero_pred = pred[pred <= P_THRESHOLD]

    label_dist = (1 - (one_pred)).sum() + (zero_pred).sum()
    
    return float(label_dist)

def label(pred):
    pred[pred > P_THRESHOLD] = 1
    pred[pred <= P_THRESHOLD] = 0

    return pred

@torch.no_grad()
def accuracy(loader, model):
    model.eval()

    total_acc = 0
    total_label_dist = 0
    false_positives = 0
    false_negatives = 0
    for data in loader:
        # data = data.to(device)
        labels, label_dist = model.evaluate(data.x, data.edge_index, data.batch)
        
        total_acc += int((labels == data.y).sum())
        false_positives += int((torch.logical_and(labels != data.y, labels == 1)).sum())
        false_negatives += int((torch.logical_and(labels != data.y, labels == 0)).sum())
        total_label_dist += label_dist

    total_acc /= len(loader.dataset)
    total_label_dist /= len(loader.dataset)
    false_positives /= len(loader.dataset)
    false_negatives /= len(loader.dataset)

    return total_acc, total_label_dist, false_positives, false_negatives