from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import GCN, GraphUNet

dataset = Planetoid(root='data/Cora', name='Cora')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(dataset).to(device)
model = GraphUNet(in_channels=dataset.num_features,
                  hidden_channels=dataset.num_features//2,
                  out_channels=dataset.num_classes,
                  depth=3).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

writer = SummaryWriter('./runs/' + model._get_name())


def train(epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask + data.val_mask], data.y[data.train_mask + data.val_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    correct = sum(pred[data.train_mask + data.val_mask] == data.y[data.train_mask + data.val_mask])
    acc = int(correct) / int(sum(data.train_mask + data.val_mask))
    writer.add_scalar("train/loss", loss, epoch)
    writer.add_scalar("train/acc", acc, epoch)
    print(f"{epoch}: {acc}")


def test(epoch):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        writer.add_scalar("test/loss", loss, epoch)
        writer.add_scalar("test/acc", acc, epoch)


if __name__ == '__main__':
    for epoch in range(1, 201):
        train(epoch)
        test(epoch)
        writer.close()
