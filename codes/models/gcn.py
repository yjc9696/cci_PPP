from dgl.nn.pytorch.conv import GraphConv
from torch import nn
import torch.nn.functional as F
import torch

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            GraphConv(in_feats,
                      out_feats=n_hidden,
                      activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden,
                          out_feats=n_hidden,
                          activation=activation))
        # output layer
        # self.linear1 = nn.Linear(n_hidden, n_hidden // 2)
        # self.linear2 = nn.Linear(n_hidden // 2, n_classes)

        self.linear2 = nn.Linear(n_hidden*3, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g, h, x1, x2):
        for layer in self.layers:
            h = layer(g, h)

        # h = self.linear1(h)
        # h = F.relu(h)
        # h = self.linear2(h)
        h = torch.cat([h[x1], h[x2], torch.abs(h[x1]-h[x2])], 1)
        h = self.linear2(h)

        return h
