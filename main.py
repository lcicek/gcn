import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from parameters import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH
from nets import GCN
from utility import initializeNodes, accuracy, printLabelBalance, printWrongBalance

# torch.manual_seed(12345)
criterion = torch.nn.BCELoss()

def loadDatasets():
    dataset = TUDataset("./", "IMDB-BINARY").shuffle()
    initializeNodes(dataset)

    train_dataset = dataset[0:700]
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    printLabelBalance(train_loader)

    test_dataset = dataset[700:1000]
    test_loader = DataLoader(test_dataset, BATCH_SIZE, False)

    return train_loader, test_loader

def prepareTraining():
    train_loader, test_loader = loadDatasets()
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return train_loader, test_loader, model, optimizer

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
        acc, label_dist, fp, fn = accuracy(train_loader, model)

        if log and epoch % 10 == 0:
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

# Regular training is performed, stats are logged and model is saved.
def saveModel():
    train_loader, test_loader, model, optimizer = prepareTraining()
    trainingLoop(model, train_loader, optimizer)
    modelStats(model, test_loader)

# In each iteration a new model is trained on the uniquely shuffled dataset and the accuracy is saved in 'info.txt'.
def evaluateModel():
    print("Evaluating model...")
    for i in range(30):
        train_loader, test_loader, model, optimizer = prepareTraining()

        print(f'Progress: {i}/30')
        trainingLoop(model, train_loader, optimizer, log=False, save_model=False)
        modelStats(model, test_loader, save=True)

saveModel()