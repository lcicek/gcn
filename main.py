from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch

from parameters import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH
from nets import GCN
from utility import initializeNodes, accuracy

torch.manual_seed(12345)

dataset = TUDataset("./", "IMDB-BINARY").shuffle()
initializeNodes(dataset)

train_dataset = dataset[0:750] # should contain roughly equal amount of 0 and 1 labels since dataset was shuffled
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

test_dataset = dataset[751:1000]
test_loader = DataLoader(test_dataset, BATCH_SIZE, False)

model = GCN()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def test(load=False):
    if load: 
        model.load_state_dict(torch.load(MODEL_PATH))

    acc, label_dist, fp, fn = accuracy(test_loader, model)
    print(f'Model Accuracy: {acc:.2f} | LD: {label_dist:.2f} | FP: {fp:.2f} | FN: {fn:.2f}')

def train():
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

for epoch in range(EPOCHS):
    loss = train()
    acc, label_dist, fp, fn = accuracy(train_loader, model)

    if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Accuracy: {acc:.2f} | LD: {label_dist:.2f} | FP: {fp:.2f} | FN: {fn:.2f}')
    
torch.save(model.state_dict(), MODEL_PATH)

test()