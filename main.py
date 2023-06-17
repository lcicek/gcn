from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch

from parameters import BATCH_SIZE, LEARNING_RATE, EPOCHS
from nets import GCN
from utility import prepareNodes, accuracy

torch.manual_seed(12345)

dataset = TUDataset("./", "IMDB-BINARY", use_node_attr=True).shuffle()
prepareNodes(dataset)

train_dataset = dataset[len(dataset) // 2:]
train_loader = DataLoader(train_dataset, BATCH_SIZE, False)

model = GCN()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    acc = accuracy(train_loader, model)

    if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Accuracy: {acc:.2f}')