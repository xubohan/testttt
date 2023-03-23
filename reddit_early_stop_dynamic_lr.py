from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# early stopping
patience = 5  # after 5 times, training will be stopped
min_delta = 0.001  # when the dif exceed this value, early stopping will consider to stop training

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_acc = -np.inf

    def check(self, val_acc):
        if self.best_val_acc - val_acc > self.min_delta:
            self.counter += 1
        else:
            self.counter = 0
        self.best_val_acc = max(self.best_val_acc, val_acc)

        if self.counter >= self.patience:
            return True
        return False

import copy
import os.path

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import datetime


c = datetime.datetime.now()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)

data = dataset[0].to(device, 'x', 'y')
batch_size = 2048
kwargs = {'batch_size': batch_size, 'num_workers':6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

del subgraph_loader.data.x, subgraph_loader.data.y
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        device = x_all.device
        for i, conv in enumerate([self.conv1, self.conv2]):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len([self.conv1, self.conv2]) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all


model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)


def train(epoch):
    model.train()

    total_loss = 0
    total_correct = 0
    total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        output = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((output.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test():
    model.eval()
    output = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(output.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((output[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

# def pyg_checktime():
a = datetime.datetime.now()
for epoch in range(1, 11):
    loss, acc = train(epoch)
    train_acc, val_acc, test_acc = test()
    scheduler.step(val_acc)
    print(f'Epoch:{epoch} Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    print(f'Epoch:{epoch} Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
    
    if early_stopping.check(val_acc):
        print("Early stopping triggered.")
        break
b = datetime.datetime.now()
print('time spend using pyg:\n compression time: ', a - c, ' train time: ', b - a, '  total time: ', b -c)

'''
Batch size 1024:

Epoch:1 Loss: 0.5056, Approx. Train: 0.8946
Epoch:1 Train: 0.9498, Val: 0.9500, Test: 0.9472
Epoch:2 Loss: 0.4290, Approx. Train: 0.9243
Epoch:2 Train: 0.9535, Val: 0.9483, Test: 0.9497
Epoch:3 Loss: 0.4936, Approx. Train: 0.9240
Epoch:3 Train: 0.9550, Val: 0.9514, Test: 0.9500
Epoch:4 Loss: 0.5082, Approx. Train: 0.9263
Epoch:4 Train: 0.9566, Val: 0.9505, Test: 0.9503
Epoch:5 Loss: 0.5539, Approx. Train: 0.9272
Epoch:5 Train: 0.9601, Val: 0.9524, Test: 0.9517
Epoch:6 Loss: 0.5820, Approx. Train: 0.9276
Epoch:6 Train: 0.9587, Val: 0.9498, Test: 0.9503
Epoch:7 Loss: 0.5495, Approx. Train: 0.9289
Epoch:7 Train: 0.9618, Val: 0.9504, Test: 0.9504
Epoch:8 Loss: 0.5641, Approx. Train: 0.9300
Epoch:8 Train: 0.9588, Val: 0.9490, Test: 0.9476
Epoch 00009: reducing learning rate of group 0 to 1.0000e-03.
Epoch:9 Loss: 0.6116, Approx. Train: 0.9303
Epoch:9 Train: 0.9618, Val: 0.9503, Test: 0.9489
Epoch:10 Loss: 0.4117, Approx. Train: 0.9419
Epoch:10 Train: 0.9714, Val: 0.9563, Test: 0.9567
time spend using pyg:
 compression time:  0:00:29.479854  train time:  0:05:42.766664   total time:  0:06:12.246518
 '''


'''
Batch Size 2048:
Epoch:1 Loss: 0.5600, Approx. Train: 0.8784
Epoch:1 Train: 0.9582, Val: 0.9568, Test: 0.9551
Epoch:2 Loss: 0.2755, Approx. Train: 0.9377
Epoch:2 Train: 0.9616, Val: 0.9573, Test: 0.9559
Epoch:3 Loss: 0.2753, Approx. Train: 0.9390
Epoch:3 Train: 0.9636, Val: 0.9575, Test: 0.9557
Epoch:4 Loss: 0.2881, Approx. Train: 0.9399
Epoch:4 Train: 0.9635, Val: 0.9546, Test: 0.9527
Epoch:5 Loss: 0.3008, Approx. Train: 0.9393
Epoch:5 Train: 0.9691, Val: 0.9580, Test: 0.9575
Epoch:6 Loss: 0.3051, Approx. Train: 0.9401
Epoch:6 Train: 0.9664, Val: 0.9528, Test: 0.9544
Epoch:7 Loss: 0.3832, Approx. Train: 0.9361
Epoch:7 Train: 0.9650, Val: 0.9539, Test: 0.9539
Epoch:8 Loss: 0.3321, Approx. Train: 0.9398
Epoch:8 Train: 0.9684, Val: 0.9551, Test: 0.9543
Epoch 00009: reducing learning rate of group 0 to 1.0000e-03.
Epoch:9 Loss: 0.3447, Approx. Train: 0.9417
Epoch:9 Train: 0.9708, Val: 0.9558, Test: 0.9557
Epoch:10 Loss: 0.2340, Approx. Train: 0.9519
Epoch:10 Train: 0.9774, Val: 0.9596, Test: 0.9604
time spend using pyg:
 compression time:  0:00:29.405730  train time:  0:04:03.515638   total time:  0:04:32.921368

'''
