from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphUNet
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

dataset = Planetoid(root='data/Cora', name='Cora')
# class GraphUNet(nn.Module):
#   def __init__(self, c_in, c_out):
#     super(GraphUNet, self).__init__()
#     self.gconv1 = GCNConv(in_channels=c_in, out_channels=16)
#     self.gpool1 = nn.Linear(in_features=16,out_features=8)

#     self.gconv2 = GCNConv(in_channels=8, out_channels=8)
#     self.gpool2 = nn.Linear(in_features=8, out_features=4)

#     self.gconv3 = GCNConv(in_channels=4, out_channels=4)

#     self.upool1 = nn.Linear(in_features=4, out_features=8)
#     self.gconv4 = GCNConv(in_channels=8, out_channels=8)

#     self.upool2 = nn.Linear(in_features=8, out_features=16)
#     self.gconv5 = GCNConv(in_channels=16, out_channels=c_out)

#   def forward(self, data):
#     X, A_0 = data.x, data.edge_index

#     X_0 = self.gconv1(X, A_0)
#     X_1 = self.gpool1(X_0)

#     X_1 = self.gconv2(X_1, A_0)
#     X_2 = self.gpool2(X_1)

#     X_3 = self.gconv3(X_2, A_0)
#     X_3 = F.relu(X_3)
#     # X_3 = F.dropout(X_3, training=self.training)

#     X_3 = self.upool1(X_3)
#     X_4 = self.gconv4(torch.cat((X_3, X_1), 1), A_0)

#     X_4 = self.upool2(X_4)
#     X_5 = self.gconv5(torch.cat((X_4, X_0), 1), A_0)

#     return F.log_softmax(X_5, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GraphUNet(c_in=dataset.num_node_features, c_out=dataset.num_classes).to(device)
model = GraphUNet(in_channels=dataset.num_features, hidden_channels=dataset.num_features//2, out_channels=dataset.num_classes, depth=3, pool_ratios=0.5, sum_res=False).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

writer = SummaryWriter('./runs/' + model._get_name())

def train(epoch):
    print("Epoch {}".format(epoch))
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = F.normalize(out, dim=1)
    loss = F.nll_loss(out[data.train_mask + data.val_mask], data.y[data.train_mask + data.val_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    correct = sum(pred[data.train_mask + data.val_mask] == data.y[data.train_mask + data.val_mask])
    acc = int(correct) / int(sum(data.train_mask + data.val_mask))
    # writer.add_scalar("train/loss", loss, epoch)
    # writer.add_scalar("train/acc", acc, epoch)

    print("Loss: {:7f}, Acc: {:2f}".format(loss.item(), acc))


def test(epoch):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        out = F.normalize(out, dim=1)
        loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        # writer.add_scalar("test/loss", loss, epoch)
        # writer.add_scalar("test/acc", acc, epoch)

        print("Test acc: {:4f}".format(acc))
        print("==============================")

if __name__ == '__main__':
    for epoch in range(1, 50):
        train(epoch)
        test(epoch)
        writer.close()
