import copy
import os.path

import torch
import torch.nn.functional as F
# from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import datetime


c = datetime.datetime.now()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)

# Already send node features/labels to GPU for faster access during sampling:
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
for epoch in range(1, 10):
    loss, acc = train(epoch)
    train_acc, val_acc, test_acc = test()
    print(f'Epoch:{epoch} Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    print(f'Epoch:{epoch} Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')

loss, acc = train(epoch)
print(f'Epoch 010, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
train_acc, val_acc, test_acc = test()
print(f'Epoch: 010, Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
b = datetime.datetime.now()
print('time spend using pyg:\n compression time: ', a - c, ' train time: ', b - a, '  total time: ', b -c)


'''
Epoch:1 Loss: 0.5614, Approx. Train: 0.8771
Epoch:1 Train: 0.9564, Val: 0.9554, Test: 0.9552
Epoch:2 Loss: 0.2699, Approx. Train: 0.9380
Epoch:2 Train: 0.9622, Val: 0.9577, Test: 0.9576
Epoch:3 Loss: 0.2669, Approx. Train: 0.9408
Epoch:3 Train: 0.9643, Val: 0.9560, Test: 0.9565
Epoch:4 Loss: 0.2811, Approx. Train: 0.9402
Epoch:4 Train: 0.9649, Val: 0.9565, Test: 0.9549
Epoch:5 Loss: 0.2955, Approx. Train: 0.9396
Epoch:5 Train: 0.9681, Val: 0.9580, Test: 0.9572
Epoch:6 Loss: 0.2942, Approx. Train: 0.9413
Epoch:6 Train: 0.9674, Val: 0.9553, Test: 0.9551
Epoch:7 Loss: 0.3059, Approx. Train: 0.9417
Epoch:7 Train: 0.9677, Val: 0.9569, Test: 0.9556
Epoch:8 Loss: 0.3085, Approx. Train: 0.9427
Epoch:8 Train: 0.9694, Val: 0.9559, Test: 0.9555
Epoch:9 Loss: 0.3336, Approx. Train: 0.9413
Epoch:9 Train: 0.9716, Val: 0.9572, Test: 0.9556
Epoch 010, Loss: 0.3251, Approx. Train: 0.9437
Epoch: 010, Train: 0.9714, Val: 0.9542, Test: 0.9561
time spend using pyg:
 compression time:  0:00:29.491067  train time:  0:03:59.561323   total time:  0:04:29.052390




Epoch:1 Loss: 0.5490, Approx. Train: 0.8791
Epoch:1 Train: 0.9562, Val: 0.9548, Test: 0.9539
Epoch:2 Loss: 0.2785, Approx. Train: 0.9380
Epoch:2 Train: 0.9607, Val: 0.9560, Test: 0.9557
Epoch:3 Loss: 0.2688, Approx. Train: 0.9390
Epoch:3 Train: 0.9653, Val: 0.9577, Test: 0.9571
Epoch:4 Loss: 0.2697, Approx. Train: 0.9416
Epoch:4 Train: 0.9662, Val: 0.9564, Test: 0.9562
Epoch:5 Loss: 0.3009, Approx. Train: 0.9397
Epoch:5 Train: 0.9673, Val: 0.9575, Test: 0.9562
Epoch:6 Loss: 0.3332, Approx. Train: 0.9398
Epoch:6 Train: 0.9687, Val: 0.9585, Test: 0.9571
Epoch:7 Loss: 0.3122, Approx. Train: 0.9411
Epoch:7 Train: 0.9689, Val: 0.9545, Test: 0.9553
Epoch:8 Loss: 0.3176, Approx. Train: 0.9421
Epoch:8 Train: 0.9691, Val: 0.9531, Test: 0.9552
Epoch:9 Loss: 0.3223, Approx. Train: 0.9423
Epoch:9 Train: 0.9692, Val: 0.9545, Test: 0.9546
Epoch 010, Loss: 0.3185, Approx. Train: 0.9429
Epoch: 010, Train: 0.9713, Val: 0.9551, Test: 0.9560
time spend using pyg:
 compression time:  0:00:30.257770  train time:  0:04:01.416064   total time:  0:04:31.673834

'''
