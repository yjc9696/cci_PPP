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
        self.linear1 = nn.Linear(n_hidden*3, n_hidden)
        # self.dense1_bn = nn.BatchNorm1d(n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_classes)
        self.linear_mse = nn.Linear(n_hidden, 1)

    def forward(self, g, h, x1, x2):
        for layer in self.layers:
            h = layer(g, h)

        h = self.linear1(torch.cat([h[x1], h[x2], torch.abs(h[x1]-h[x2])], 1))
        # h_src = self.linear1(h[x1] + h[x2])
        # mouse doesn't have bn
        # h = self.dense1_bn(h)
        h = F.relu(h)
        h_c = self.linear2(h)
        h_mse = self.linear_mse(h)
        return h_c, h_mse

