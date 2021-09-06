import argparse

from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import GCN, GraphUNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Cora",
                    help="Dataset: Cora, CiteSeer, PubMed.")
parser.add_argument('--isUseUNet', action='store_true', default=False,
                    help='Select model Graph U-net or origin GCN.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--depth', type=int, default=3,
                    help='Depth of Graph U-Nets.')

args = parser.parse_args()
dataset = Planetoid(root=f'data/{args.dataset}', name=f'{args.dataset}')
best = 0
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

if not args.isUseUNet:
    model = GCN(dataset).to(device)
else:
    model = GraphUNet(in_channels=dataset.num_features,
                      hidden_channels=args.hidden,
                      out_channels=dataset.num_classes,
                      depth=args.depth).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    print(f"train epoch: {epoch} - loss: {loss} - accuracy: {acc}")


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
        print(f"test epoch: {epoch} - loss: {loss} - accuracy: {acc}")

        if mode == "test":
            if acc > best:
                torch.save(model.state_dict(), f"best_{model._get_name()}_{dataset.name}.pkl")
                best = acc
            print("best: ", best)


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)
        test(epoch, mode="test")
    writer.close()
    torch.save(model.state_dict(), f"last_{model._get_name()}_{dataset.name}.pkl")
    print(best)
