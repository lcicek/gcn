import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer

from parameters import *
from nets import GCN
from utility import initializeNodes, accuracy, printLabelBalance, printWrongBalance

criterion = torch.nn.BCELoss()

def loadDataset():
    dataset = TUDataset("./", "IMDB-BINARY")
    initializeNodes(dataset)

    return dataset

def prepareLoaders(dataset):
    train_dataset = dataset[150:850]
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    printLabelBalance(train_loader) # assert that label balance is 350/350 

    test_dataset = dataset[0:150] + dataset[850:1000]
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def prepareTraining():
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer

def train(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.batch)
        loss = criterion(pred, data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss

def trainingLoop(model, train_loader, optimizer, log=True, logExtended = False, save_model=True):
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer)

        if log and epoch % 5 == 0:
                acc, label_dist, fp, fn = accuracy(train_loader, model)

                info = f'Epoch {epoch:>3} | Loss: {loss:.2f} | Accuracy: {acc:.2f}' 
                extended = f' | LD: {label_dist:.2f} | FP: {fp:.2f} | FN: {fn:.2f}'

                if (logExtended):
                     print(info + extended)
                else:
                     print(info)

    if (save_model):
        torch.save(model.state_dict(), MODEL_PATH)

def modelStats(model, test_loader, load=False, save=False):
    if load: 
        model.load_state_dict(torch.load(MODEL_PATH))

    acc, label_dist, fp, fn = accuracy(test_loader, model)

    if save:
        file = open("info.txt", "a")
        file.write(f'Model Accuracy: {acc:.2f} | LD: {label_dist:.2f} | FP: {fp:.2f} | FN: {fn:.2f}\n')
        file.close()
    else:
        print(f'Model Accuracy: {acc:.2f} | LD: {label_dist:.2f} | FP: {fp:.2f} | FN: {fn:.2f}')

    return acc

# Regular training is performed, stats are logged and model is saved.
def saveModel():
    dataset = loadDataset()
    train_loader, test_loader = prepareLoaders(dataset)
    model, optimizer = prepareTraining()

    trainingLoop(model, train_loader, optimizer)
    modelStats(model, test_loader)

# In each iteration a new model is trained on the uniquely shuffled dataset and the accuracy is saved in 'info.txt'.
def evaluateModel():
    count = 10
    train_loader, test_loader = prepareLoaders()

    avg = 0
    for i in range(count):
        model, optimizer = prepareTraining()

        print(f'Progress: {i}/{count}')
        trainingLoop(model, train_loader, optimizer, log=False, save_model=False)
        avg += modelStats(model, test_loader, save=True)

    avg /= count
    print(f"Average accuracy: {avg}")

def explain():
    dataset = loadDataset()
    data = dataset[0]
    model = GCN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

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

    torch.set_printoptions(threshold=100_000)
    explanation = explainer(data.x, data.edge_index)

    file = open("explanation.txt", "w")
    file.write(str(explanation.node_stores))
    file.close()

    explanation.visualize_feature_importance(FEATURE_IMG_PATH, top_k=20)
    explanation.visualize_graph(GRAPH_PATH)


# evaluateModel()
# saveModel()
explain()