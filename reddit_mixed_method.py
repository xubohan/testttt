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
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import datetime


c = datetime.datetime.now()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)

data = dataset[0].to(device, 'x', 'y')
batch_size = 4096
kwargs = {'batch_size': batch_size, 'num_workers':24, 'persistent_workers': True}
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
scaler = GradScaler()

def train(epoch):
    model.train()

    total_loss = 0
    total_correct = 0
    total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        with autocast():
            output = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            loss = F.cross_entropy(output, y)
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

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
bs=1024 nw=12 no(dynamic lr, stop learn)
Epoch:1 Loss: 0.9767, Approx. Train: 0.7687
Epoch:1 Train: 0.9411, Val: 0.9447, Test: 0.9435
Epoch:2 Loss: 0.2883, Approx. Train: 0.9301
Epoch:2 Train: 0.9555, Val: 0.9564, Test: 0.9546
Epoch:3 Loss: 0.2278, Approx. Train: 0.9436
Epoch:3 Train: 0.9603, Val: 0.9597, Test: 0.9585
Epoch:4 Loss: 0.2011, Approx. Train: 0.9487
Epoch:4 Train: 0.9637, Val: 0.9619, Test: 0.9603
Epoch:5 Loss: 0.1828, Approx. Train: 0.9527
Epoch:5 Train: 0.9660, Val: 0.9615, Test: 0.9604
Epoch:6 Loss: 0.1699, Approx. Train: 0.9554
Epoch:6 Train: 0.9691, Val: 0.9635, Test: 0.9625
Epoch:7 Loss: 0.1588, Approx. Train: 0.9572
Epoch:7 Train: 0.9711, Val: 0.9632, Test: 0.9626
Epoch:8 Loss: 0.1481, Approx. Train: 0.9597
Epoch:8 Train: 0.9736, Val: 0.9642, Test: 0.9631
Epoch:9 Loss: 0.1393, Approx. Train: 0.9618
Epoch:9 Train: 0.9746, Val: 0.9653, Test: 0.9640
Epoch 010, Loss: 0.1342, Approx. Train: 0.9623
Epoch: 010, Train: 0.9760, Val: 0.9642, Test: 0.9634
time spend using pyg:
 compression time:  0:00:30.196367  train time:  0:03:34.731317   total time:  0:04:04.927684


bs=2048 nw=12  dynamic lr  stop learn
Epoch:1 Loss: 0.5617, Approx. Train: 0.8771
Epoch:1 Train: 0.9559, Val: 0.9546, Test: 0.9539
Epoch:2 Loss: 0.2748, Approx. Train: 0.9380
Epoch:2 Train: 0.9607, Val: 0.9564, Test: 0.9553
Epoch:3 Loss: 0.2615, Approx. Train: 0.9406
Epoch:3 Train: 0.9634, Val: 0.9570, Test: 0.9555
Epoch:4 Loss: 0.2707, Approx. Train: 0.9412
Epoch:4 Train: 0.9646, Val: 0.9542, Test: 0.9543
Epoch:5 Loss: 0.2980, Approx. Train: 0.9398
Epoch:5 Train: 0.9671, Val: 0.9559, Test: 0.9554
Epoch:6 Loss: 0.3098, Approx. Train: 0.9398
Epoch:6 Train: 0.9666, Val: 0.9548, Test: 0.9533
Epoch 00007: reducing learning rate of group 0 to 1.0000e-03.
Epoch:7 Loss: 0.3706, Approx. Train: 0.9387
Epoch:7 Train: 0.9670, Val: 0.9546, Test: 0.9533
Epoch:8 Loss: 0.2561, Approx. Train: 0.9495
Epoch:8 Train: 0.9755, Val: 0.9611, Test: 0.9603
Epoch:9 Loss: 0.2043, Approx. Train: 0.9555
Epoch:9 Train: 0.9781, Val: 0.9613, Test: 0.9613
Epoch:10 Loss: 0.1803, Approx. Train: 0.9577
Epoch:10 Train: 0.9796, Val: 0.9627, Test: 0.9618
time spend using pyg:
 compression time:  0:00:29.468756  train time:  0:03:37.699011   total time:  0:04:07.167767


bs=2048 nw=24  dynamic lr  stop learn
Epoch:1 Loss: 0.5424, Approx. Train: 0.8783
Epoch:1 Train: 0.9536, Val: 0.9526, Test: 0.9506
Epoch:2 Loss: 0.2793, Approx. Train: 0.9373
Epoch:2 Train: 0.9624, Val: 0.9567, Test: 0.9563
Epoch:3 Loss: 0.2724, Approx. Train: 0.9394
Epoch:3 Train: 0.9655, Val: 0.9572, Test: 0.9561
Epoch:4 Loss: 0.2735, Approx. Train: 0.9403
Epoch:4 Train: 0.9638, Val: 0.9548, Test: 0.9541
Epoch:5 Loss: 0.3067, Approx. Train: 0.9392
Epoch:5 Train: 0.9666, Val: 0.9562, Test: 0.9556
Epoch:6 Loss: 0.3815, Approx. Train: 0.9359
Epoch:6 Train: 0.9575, Val: 0.9499, Test: 0.9492
Epoch 00007: reducing learning rate of group 0 to 1.0000e-03.
Epoch:7 Loss: 0.4833, Approx. Train: 0.9327
Epoch:7 Train: 0.9666, Val: 0.9564, Test: 0.9547
Epoch:8 Loss: 0.2870, Approx. Train: 0.9487
Epoch:8 Train: 0.9735, Val: 0.9608, Test: 0.9605
Epoch:9 Loss: 0.2298, Approx. Train: 0.9536
Epoch:9 Train: 0.9762, Val: 0.9613, Test: 0.9602
Epoch:10 Loss: 0.2079, Approx. Train: 0.9559
Epoch:10 Train: 0.9779, Val: 0.9618, Test: 0.9609
time spend using pyg:
 compression time:  0:00:31.630198  train time:  0:03:18.858133   total time:  0:03:50.488331


bs=4096 nw=24  dynamic_lr  stop learn
Epoch:1 Loss: 0.7421, Approx. Train: 0.8375
Epoch:1 Train: 0.9529, Val: 0.9543, Test: 0.9528
Epoch:2 Loss: 0.2616, Approx. Train: 0.9403
Epoch:2 Train: 0.9633, Val: 0.9602, Test: 0.9586
Epoch:3 Loss: 0.2157, Approx. Train: 0.9475
Epoch:3 Train: 0.9682, Val: 0.9597, Test: 0.9599
Epoch:4 Loss: 0.1998, Approx. Train: 0.9495
Epoch:4 Train: 0.9721, Val: 0.9624, Test: 0.9608
Epoch:5 Loss: 0.1873, Approx. Train: 0.9521
Epoch:5 Train: 0.9733, Val: 0.9614, Test: 0.9603
Epoch:6 Loss: 0.1855, Approx. Train: 0.9530
Epoch:6 Train: 0.9757, Val: 0.9616, Test: 0.9613
Epoch:7 Loss: 0.1787, Approx. Train: 0.9544
Epoch:7 Train: 0.9764, Val: 0.9621, Test: 0.9603
Epoch 00008: reducing learning rate of group 0 to 1.0000e-03.
Epoch:8 Loss: 0.1752, Approx. Train: 0.9552
Epoch:8 Train: 0.9765, Val: 0.9608, Test: 0.9597
Epoch:9 Loss: 0.1411, Approx. Train: 0.9621
Epoch:9 Train: 0.9820, Val: 0.9646, Test: 0.9636
Epoch:10 Loss: 0.1230, Approx. Train: 0.9658
Epoch:10 Train: 0.9834, Val: 0.9650, Test: 0.9640
time spend using pyg:
 compression time:  0:00:29.407760  train time:  0:02:22.882771   total time:  0:02:52.290531


bs=4096 nw=24  dynamic_lr  stop learn amp
Epoch:1 Loss: 0.7302, Approx. Train: 0.8314
Epoch:1 Train: 0.9495, Val: 0.9522, Test: 0.9492
Epoch:2 Loss: 0.2695, Approx. Train: 0.9394
Epoch:2 Train: 0.9627, Val: 0.9598, Test: 0.9588
Epoch:3 Loss: 0.2196, Approx. Train: 0.9465
Epoch:3 Train: 0.9666, Val: 0.9603, Test: 0.9589
Epoch:4 Loss: 0.1961, Approx. Train: 0.9501
Epoch:4 Train: 0.9707, Val: 0.9607, Test: 0.9595
Epoch:5 Loss: 0.1848, Approx. Train: 0.9524
Epoch:5 Train: 0.9732, Val: 0.9608, Test: 0.9601
Epoch:6 Loss: 0.1797, Approx. Train: 0.9540
Epoch:6 Train: 0.9741, Val: 0.9612, Test: 0.9588
Epoch:7 Loss: 0.1804, Approx. Train: 0.9538
Epoch:7 Train: 0.9756, Val: 0.9590, Test: 0.9589
Epoch:8 Loss: 0.1917, Approx. Train: 0.9532
Epoch:8 Train: 0.9735, Val: 0.9591, Test: 0.9579
Epoch:9 Loss: 0.1896, Approx. Train: 0.9538
Epoch:9 Train: 0.9770, Val: 0.9593, Test: 0.9587
Epoch 00010: reducing learning rate of group 0 to 1.0000e-03.
Epoch:10 Loss: 0.1821, Approx. Train: 0.9545
Epoch:10 Train: 0.9777, Val: 0.9588, Test: 0.9591
time spend using pyg:
 compression time:  0:00:28.799253  train time:  0:02:21.578122   total time:  0:02:50.377375

二次测试:
Epoch:1 Loss: 0.8713, Approx. Train: 0.8237
Epoch:1 Train: 0.9479, Val: 0.9490, Test: 0.9479
Epoch:2 Loss: 0.2926, Approx. Train: 0.9367
Epoch:2 Train: 0.9620, Val: 0.9595, Test: 0.9579
Epoch:3 Loss: 0.2238, Approx. Train: 0.9462
Epoch:3 Train: 0.9660, Val: 0.9604, Test: 0.9585
Epoch:4 Loss: 0.2043, Approx. Train: 0.9499
Epoch:4 Train: 0.9689, Val: 0.9610, Test: 0.9595
Epoch:5 Loss: 0.1908, Approx. Train: 0.9515
Epoch:5 Train: 0.9727, Val: 0.9611, Test: 0.9598
Epoch:6 Loss: 0.1775, Approx. Train: 0.9538
Epoch:6 Train: 0.9756, Val: 0.9620, Test: 0.9610
Epoch:7 Loss: 0.1729, Approx. Train: 0.9551
Epoch:7 Train: 0.9763, Val: 0.9597, Test: 0.9596
Epoch:8 Loss: 0.1776, Approx. Train: 0.9549
Epoch:8 Train: 0.9768, Val: 0.9606, Test: 0.9595
Epoch:9 Loss: 0.1724, Approx. Train: 0.9559
Epoch:9 Train: 0.9786, Val: 0.9614, Test: 0.9606
Epoch 00010: reducing learning rate of group 0 to 1.0000e-03.
Epoch:10 Loss: 0.1690, Approx. Train: 0.9564
Epoch:10 Train: 0.9791, Val: 0.9613, Test: 0.9602
time spend using pyg:
 compression time:  0:00:28.272843  train time:  0:02:29.441323   total time:  0:02:57.714166

3次测试:
Epoch:1 Loss: 0.7912, Approx. Train: 0.8302
Epoch:1 Train: 0.9519, Val: 0.9523, Test: 0.9517
Epoch:2 Loss: 0.2720, Approx. Train: 0.9387
Epoch:2 Train: 0.9615, Val: 0.9591, Test: 0.9579
Epoch:3 Loss: 0.2191, Approx. Train: 0.9468
Epoch:3 Train: 0.9677, Val: 0.9604, Test: 0.9594
Epoch:4 Loss: 0.1939, Approx. Train: 0.9507
Epoch:4 Train: 0.9704, Val: 0.9608, Test: 0.9612
Epoch:5 Loss: 0.1857, Approx. Train: 0.9527
Epoch:5 Train: 0.9727, Val: 0.9621, Test: 0.9605
Epoch:6 Loss: 0.1799, Approx. Train: 0.9536
Epoch:6 Train: 0.9745, Val: 0.9609, Test: 0.9610
Epoch:7 Loss: 0.1746, Approx. Train: 0.9551
Epoch:7 Train: 0.9769, Val: 0.9613, Test: 0.9613
Epoch:8 Loss: 0.1741, Approx. Train: 0.9552
Epoch:8 Train: 0.9777, Val: 0.9605, Test: 0.9607
Epoch 00009: reducing learning rate of group 0 to 1.0000e-03.
Epoch:9 Loss: 0.1716, Approx. Train: 0.9559
Epoch:9 Train: 0.9764, Val: 0.9600, Test: 0.9595
Epoch:10 Loss: 0.1397, Approx. Train: 0.9622
Epoch:10 Train: 0.9822, Val: 0.9630, Test: 0.9628
time spend using pyg:
 compression time:  0:00:28.397538  train time:  0:02:29.843382   total time:  0:02:58.240920

'''






