import argparse
import torch
from torch_geometric.loader import DataLoader

from nets import GCN
from parameters import *
from utility import *

def prepareLoaders(dataset):
    """See https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader"""

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
        loss = loss_function(pred, data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss

def trainingLoop(model, train_loader, optimizer, log=True, logExtended = False, save_model=True):
    print("Training model...")

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer)

        if log and epoch % 5 == 0:
                avg_acc = accuracy(train_loader, model)
                print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Accuracy: {avg_acc:.2f}')

    if (save_model):
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model was saved to {MODEL_PATH}")

def modelAccuracy(model, test_loader, save=False):
    """ Returns and prints model accuracy on testset."""
    avg_acc = accuracy(test_loader, model)

    if save:
        file = open(INFO_FILE, "a")
        file.write(f'Model Accuracy: {avg_acc:.2f}')
        file.close()
    else:
        print(f'Model Accuracy: {avg_acc:.2f}')

    return avg_acc

# Regular training is performed, stats are logged and model is saved.
def saveModel():
    model, optimizer = prepareTraining()

    trainingLoop(model, train_loader, optimizer)
    modelAccuracy(model, test_loader)

def testModelAccuracy():
    """ Each model training contains randomness in the form of:
        1. How the model weights are initialized.
        2. How the dataset is shuffled every epoch.

        This function tests the impact of that randomness by training ten
        different models and logging their accuracy in "text-files/info.txt".
    """
    print("Testing different models:")

    count = 10
    avg_acc = 0
    for i in range(count):
        model, optimizer = prepareTraining()

        print(f'Testing model: {i+1}/{count}')
        trainingLoop(model, train_loader, optimizer, log=False, save_model=False)
        avg_acc += modelAccuracy(model, test_loader, save=True)

    avg_acc /= count
    print(f"Model accuracies were saved to {INFO_FILE}")
    print(f"Average accuracy across models: {avg_acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', action='store_true')  
    args = parser.parse_args()

    # global variables
    dataset = loadDataset()
    train_loader, test_loader = prepareLoaders(dataset)
    loss_function = torch.nn.BCELoss()

    if args.test:
        testModelAccuracy()
    else:
        saveModel()