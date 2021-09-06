from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import GCN, GraphUNet

dataset = Planetoid(root='data/PubMed', name='PubMed')
best = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(dataset).to(device)
model = GraphUNet(in_channels=dataset.num_features,
                  hidden_channels=16,
                  out_channels=dataset.num_classes,
                  depth=3).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

writer = SummaryWriter('./runs/' + model._get_name())


def train(epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = F.normalize(out, dim=1)
    # mask = data.train_mask + data.val_mask
    mask = ~data.test_mask
    loss = F.nll_loss(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    correct = sum(pred[mask] == data.y[mask])
    acc = int(correct) / int(sum(mask))
    writer.add_scalar("train/loss", loss, epoch)
    writer.add_scalar("train/acc", acc, epoch)
    print(f"train {epoch}: {acc}")


def test(epoch, mode="val"):
    global best
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        out = F.normalize(out, dim=1)
        mask = data.val_mask
        if mode == "test":
            mask = data.test_mask
        loss = F.nll_loss(out[mask], data.y[mask])
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        writer.add_scalar(f"{mode}/loss", loss, epoch)
        writer.add_scalar(f"{mode}/acc", acc, epoch)
        print(f"{mode} {epoch}: {acc}")
        if mode == "test":
            if acc > best:
                torch.save(model.state_dict(), f"{model._get_name()}_{dataset.name}.pkl")
                best = acc
            print("best: ", best)


if __name__ == '__main__':
    for epoch in range(1, 201):
        train(epoch)
        # test(epoch)
        test(epoch, mode="test")
    writer.close()
    print(best)
