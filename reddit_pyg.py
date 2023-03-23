# import copy
# import os.path

# import torch
# import torch.nn.functional as F
# # from tqdm import tqdm

# from torch_geometric.datasets import Reddit
# from torch_geometric.loader import NeighborLoader
# from torch_geometric.nn import SAGEConv
# import datetime


# c = datetime.datetime.now()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)

# # Already send node features/labels to GPU for faster access during sampling:
# data = dataset[0].to(device, 'x', 'y')

# kwargs = {'batch_size': 2048, 'num_workers': 6, 'persistent_workers': True}
# train_loader = NeighborLoader(data, input_nodes=data.train_mask,
#                               num_neighbors=[25, 10], shuffle=True, **kwargs)

# subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
#                                  num_neighbors=[-1], shuffle=False, **kwargs)

# # No need to maintain these features during evaluation:
# del subgraph_loader.data.x, subgraph_loader.data.y
# # Add global node index information.
# subgraph_loader.data.num_nodes = data.num_nodes
# subgraph_loader.data.n_id = torch.arange(data.num_nodes)


# class SAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

#     @torch.no_grad()
#     def inference(self, x_all, subgraph_loader):
#         device = x_all.device
#         for i, conv in enumerate([self.conv1, self.conv2]):
#             xs = []
#             for batch in subgraph_loader:
#                 x = x_all[batch.n_id.to(device)].to(device)
#                 x = conv(x, batch.edge_index.to(device))
#                 if i < len([self.conv1, self.conv2]) - 1:
#                     x = F.relu(x)
#                 xs.append(x[:batch.batch_size].cpu())
#             x_all = torch.cat(xs, dim=0)
#         return x_all


# model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# def train(epoch):
#     model.train()

#     total_loss = 0
#     total_correct = 0
#     total_examples = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         y = batch.y[:batch.batch_size]
#         output = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
#         loss = F.cross_entropy(output, y)
#         loss.backward()
#         optimizer.step

#         total_loss += float(loss) * batch.batch_size
#         total_correct += int((output.argmax(dim=-1) == y).sum())
#         total_examples += batch.batch_size

#     return total_loss / total_examples, total_correct / total_examples


# @torch.no_grad()
# def test():
#     model.eval()
#     output = model.inference(data.x, subgraph_loader).argmax(dim=-1)
#     y = data.y.to(output.device)

#     accs = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         accs.append(int((output[mask] == y[mask]).sum()) / int(mask.sum()))
#     return accs

# # def pyg_checktime():
# a = datetime.datetime.now()
# for epoch in range(1, 10):
#     loss, acc = train(epoch)
#     train_acc, val_acc, test_acc = test()
#     print(f'Epoch:{epoch} Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
#     print(f'Epoch:{epoch} Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')

# loss, acc = train(epoch)
# print(f'Epoch 010, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
# train_acc, val_acc, test_acc = test()
# print(f'Epoch: 010, Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
# b = datetime.datetime.now()
# print('time spend using pyg:\n compression time: ', a - c, ' train time: ', b - a, '  total time: ', b -c)


'''
Epoch 010, Loss: 3.8331, Approx. Train: 0.0190
Epoch: 010, Train: 0.0200, Val: 0.0152, Test: 0.0145
time spend using pyg:
 compression time:  0:00:29.400747  train time:  0:05:43.968320   total time:  0:06:13.369067






batch size = 1024导致模型无法收敛....



Epoch 010, Loss: 3.7796, Approx. Train: 0.0255
Epoch: 010, Train: 0.0256, Val: 0.0274, Test: 0.0273
time spend using pyg:
 compression time:  0:00:29.210769  train time:  0:04:01.784429   total time:  0:04:30.995198

但是准确率有问题

'''
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
batch_size = 1024
kwargs = {'batch_size': batch_size, 'num_workers':6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
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
Epoch:1 Loss: 0.5172, Approx. Train: 0.8944
Epoch:1 Train: 0.9441, Val: 0.9465, Test: 0.9445
Epoch:2 Loss: 0.4341, Approx. Train: 0.9241
Epoch:2 Train: 0.9554, Val: 0.9503, Test: 0.9514
Epoch:3 Loss: 0.5086, Approx. Train: 0.9245
Epoch:3 Train: 0.9539, Val: 0.9475, Test: 0.9471
Epoch:4 Loss: 0.5589, Approx. Train: 0.9253
Epoch:4 Train: 0.9585, Val: 0.9512, Test: 0.9497
Epoch:5 Loss: 0.5585, Approx. Train: 0.9277
Epoch:5 Train: 0.9579, Val: 0.9505, Test: 0.9493
Epoch:6 Loss: 0.5247, Approx. Train: 0.9288
Epoch:6 Train: 0.9605, Val: 0.9510, Test: 0.9491
Epoch:7 Loss: 0.5515, Approx. Train: 0.9300
Epoch:7 Train: 0.9612, Val: 0.9507, Test: 0.9508
Epoch:8 Loss: 0.5601, Approx. Train: 0.9304
Epoch:8 Train: 0.9630, Val: 0.9498, Test: 0.9498
Epoch:9 Loss: 0.5428, Approx. Train: 0.9315
Epoch:9 Train: 0.9642, Val: 0.9505, Test: 0.9513
Epoch 010, Loss: 0.5261, Approx. Train: 0.9330
Epoch: 010, Train: 0.9651, Val: 0.9518, Test: 0.9515
time spend using pyg:
 compression time:  0:00:29.348067  train time:  0:05:47.406936   total time:  0:06:16.755003


'''



