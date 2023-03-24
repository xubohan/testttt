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
batch_size = 1024
num_workers = 12
kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'persistent_workers': True}
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
    print(f'Epoch:{epoch} Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    print(f'Epoch:{epoch} Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
    
b = datetime.datetime.now()
print('time spend using pyg:\n compression time: ', a - c, ' train time: ', b - a, '  total time: ', b -c)



'''
Epoch:1 Loss: 0.5088, Approx. Train: 0.8919
Epoch:1 Train: 0.9482, Val: 0.9472, Test: 0.9469
Epoch:2 Loss: 0.4421, Approx. Train: 0.9241
Epoch:2 Train: 0.9557, Val: 0.9519, Test: 0.9513
Epoch:3 Loss: 0.5044, Approx. Train: 0.9245
Epoch:3 Train: 0.9522, Val: 0.9467, Test: 0.9454
Epoch:4 Loss: 0.5085, Approx. Train: 0.9260
Epoch:4 Train: 0.9577, Val: 0.9511, Test: 0.9505
Epoch:5 Loss: 0.5284, Approx. Train: 0.9280
Epoch:5 Train: 0.9595, Val: 0.9510, Test: 0.9521
Epoch:6 Loss: 0.5540, Approx. Train: 0.9281
Epoch:6 Train: 0.9602, Val: 0.9513, Test: 0.9505
Epoch:7 Loss: 0.5769, Approx. Train: 0.9293
Epoch:7 Train: 0.9628, Val: 0.9533, Test: 0.9526
Epoch:8 Loss: 0.5415, Approx. Train: 0.9312
Epoch:8 Train: 0.9644, Val: 0.9505, Test: 0.9508
Epoch:9 Loss: 0.5327, Approx. Train: 0.9326
Epoch:9 Train: 0.9625, Val: 0.9501, Test: 0.9496
Epoch 010, Loss: 0.5642, Approx. Train: 0.9319
Epoch: 010, Train: 0.9642, Val: 0.9521, Test: 0.9517
time spend using pyg:
 compression time:  0:00:30.305174  train time:  0:04:58.503196   total time:  0:05:28.808370

'''