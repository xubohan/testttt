# quvier template code for testing, only for testing the performance
import os.path as osp
import datetime
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

import quiver

c = datetime.datetime.now()
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)


train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           drop_last=True) 
csr_topo = quiver.CSRTopo(data.edge_index) 
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[25, 10], device=0, mode='GPU')


subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())


            x_all = torch.cat(xs, dim=0)

        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


x = quiver.Feature(rank=0, device_list=[0], device_cache_size="4G", cache_policy="device_replicate", csr_topo=csr_topo) # Quiver
x.from_cpu_tensor(data.x) 

y = data.y.squeeze().to(device)


def train(epoch):
    model.train()
    total_loss = total_correct = 0

    for seeds in train_loader:
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results

a = datetime.datetime.now()
for epoch in range(1, 11):
    loss, acc = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

b = datetime.datetime.now()
print('time spend using pyg:\n compression time: ', a - c, ' train time: ', b - a, '  total time: ', b -c)



'''
quvier:

LOG>>> 100% data cached
LOG >>> Memory Budge On 0 is 534 MB
Epoch 01, Loss: 0.5110, Approx. Train: 0.8888
Train: 0.9512, Val: 0.9488, Test: 0.9487
Epoch 02, Loss: 0.4470, Approx. Train: 0.9187
Train: 0.9531, Val: 0.9504, Test: 0.9500
Epoch 03, Loss: 0.4863, Approx. Train: 0.9188
Train: 0.9575, Val: 0.9510, Test: 0.9500
Epoch 04, Loss: 0.5530, Approx. Train: 0.9209
Train: 0.9557, Val: 0.9499, Test: 0.9496
Epoch 05, Loss: 0.5833, Approx. Train: 0.9210
Train: 0.9572, Val: 0.9494, Test: 0.9476
Epoch 06, Loss: 0.5681, Approx. Train: 0.9218
Train: 0.9599, Val: 0.9488, Test: 0.9511
Epoch 07, Loss: 0.5396, Approx. Train: 0.9244
Train: 0.9611, Val: 0.9502, Test: 0.9517
Epoch 08, Loss: 0.5661, Approx. Train: 0.9250
Train: 0.9617, Val: 0.9511, Test: 0.9500
Epoch 09, Loss: 0.5953, Approx. Train: 0.9247
Train: 0.9631, Val: 0.9519, Test: 0.9508
Epoch 10, Loss: 0.5408, Approx. Train: 0.9273
Train: 0.9648, Val: 0.9521, Test: 0.9518
time spend using pyg:
 compression time:  0:00:17.397417  train time:  0:04:23.106671   total time:  0:04:40.504088

batch size 2048:
LOG>>> 100% data cached
LOG >>> Memory Budge On 0 is 534 MB
Epoch 01, Loss: 0.5538, Approx. Train: 0.8664
Train: 0.9570, Val: 0.9563, Test: 0.9554
Epoch 02, Loss: 0.2648, Approx. Train: 0.9276
Train: 0.9606, Val: 0.9564, Test: 0.9550
Epoch 03, Loss: 0.2687, Approx. Train: 0.9286
Train: 0.9642, Val: 0.9580, Test: 0.9575
Epoch 04, Loss: 0.2735, Approx. Train: 0.9298
Train: 0.9658, Val: 0.9575, Test: 0.9569
Epoch 05, Loss: 0.2961, Approx. Train: 0.9276
Train: 0.9670, Val: 0.9571, Test: 0.9579
Epoch 06, Loss: 0.3120, Approx. Train: 0.9285
Train: 0.9645, Val: 0.9539, Test: 0.9544
Epoch 07, Loss: 0.3386, Approx. Train: 0.9281
Train: 0.9681, Val: 0.9547, Test: 0.9560
Epoch 08, Loss: 0.3224, Approx. Train: 0.9289
Train: 0.9694, Val: 0.9569, Test: 0.9575
Epoch 09, Loss: 0.3340, Approx. Train: 0.9289
Train: 0.9677, Val: 0.9547, Test: 0.9545
Epoch 10, Loss: 0.3416, Approx. Train: 0.9302
Train: 0.9714, Val: 0.9567, Test: 0.9557
time spend using pyg:
 compression time:  0:00:19.715742  train time:  0:04:23.024798   total time:  0:04:42.740540
'''







