import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image, ImageDraw, ImageFont

from parameters import *

# Taken from: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/datasets.py
# Found in: https://github.com/pyg-team/pytorch_geometric/discussions/3334
class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data
    
def initializeNodes(dataset):
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

# NOTE: Everything below this point is NOT copied, unless specified

def label(pred):
    """Returns label of prediction. P_THRESHOLD = 0.5."""
    pred[pred > P_THRESHOLD] = 1
    pred[pred <= P_THRESHOLD] = 0

    return pred

@torch.no_grad()
def accuracy(loader, model):
    model.eval()

    total_acc = 0
    for data in loader:
        _, labels = model.evaluate(data.x, data.edge_index, data.batch)
        total_acc += int((labels == data.y).sum())

    total_acc /= len(loader.dataset)

    return total_acc

@torch.no_grad()
def printWrongBalance(loader, model):
    """Prints amount of incorrectly predicted 0-Graphs and 1-Graphs."""
    model.eval()

    total_wrong = 0
    sum_wrong = 0
    for data in loader:
        _, labels = model.evaluate(data.x, data.edge_index, data.batch)

        wrong = labels[labels != data.y]

        total_wrong += len(wrong)
        sum_wrong += int(wrong.sum())

    print(f'wrong: zero({total_wrong - sum_wrong}), one({sum_wrong})')

def printLabelBalance(loader):
    """Prints balance between 0-Graphs and 1-Graphs in dataset."""

    sum = 0

    for data in loader:
        sum += int(data.y.sum())

    print(f'Dataset training-graphs: zero({len(loader.dataset) - sum}), one({sum})')

def loadModel(final=True):
    """ Loads and returns saved model.
        If final=False is specified, model saved under saved-models/model.pt will be loaded.
    """
    
    from nets import GCN

    model = GCN()
    if final:
        model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    return model

def getFileIndex(dir):
    """Returns amount of files in directory."""

    return len([name for name in os.listdir(dir) if os.path.isfile(f"{dir}/{name}")])

def getFolderIndex():
    """Returns amount of folders in directory."""

    return len([i for i in os.listdir(RESULTS_FOLDER) if os.path.isdir(f"{RESULTS_FOLDER}/{i}")])

def loadDataset():
    dataset = TUDataset("./", "IMDB-BINARY")
    initializeNodes(dataset)

    return dataset

def createNewImageDirectory():
    """Creates a new folder in "results" to save images into later."""

    assert os.path.exists(RESULTS_FOLDER)
    
    idx = getFolderIndex()
    path = f"{RESULTS_FOLDER}/{idx}"

    assert not os.path.exists(path)
    os.mkdir(path)

    return path

def explain(data, visualize=True, see_change=False, dir=None):
    """ Creates explanation of graph, saves it under "text-files/explanation.txt" and 
        visualizes the graph in either "results/" or "images/".
    """

    model = loadModel()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EPOCHS),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        )
    )

    explanation = explainer(data.x, data.edge_index)

    # write explanation to file (will overwrite existing):
    torch.set_printoptions(threshold=100_000)
    file = open(EXPLANATION_FILE, "w")
    file.write(str(explanation.node_stores))
    file.close()

    pred = model(data.x, data.edge_index).item()

    if visualize:
        visualizeGraph(data, pred, explanation, see_change=see_change, dir=dir)
    
    print(f'pred: {pred:.2f}')

def visualizeGraph(data, pred, explanation, see_change=False, dir=dir):
    explanation.visualize_feature_importance(FEATURE_IMG_PATH, top_k=10)

    if not see_change: # just regular save to images/graph.png 
        img_path = GRAPH_PATH
    else: # save all images to test-graphs/idx to see the change
        idx = getFileIndex(dir)
        img_path = f"{dir}/{idx}-p{pred:.2f}.png" # working directory has been changed in modify-graphs, so no need to add it here
    
    explanation.visualize_graph(img_path)

    # load image that explanation saved:
    img = Image.open(img_path) 

    # draw label and prediction on top of image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 16)
    draw.text((90, 400),f"pred: {pred:.2f}", (255, 0, 0), font=font)
    if data.y is not None:
        draw.text((90, 64),f"label: {data.y.item()}", (0, 0, 255), font=font)

    # replace image:
    img.save(img_path)