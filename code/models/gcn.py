from dgl.nn.pytorch.conv import GraphConv
from torch import nn
import torch.nn.functional as F


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
        self.linear1 = nn.Linear(n_hidden, n_hidden // 2)
        self.linear2 = nn.Linear(n_hidden // 2, n_classes)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)

        h = self.linear1(h)
        h = F.relu(h)
        h = self.linear2(h)
        return h
